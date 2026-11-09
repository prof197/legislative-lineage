"""Size the semantic stage before embedding the corpus (dev only).

For each window scheme the stage could embed, this measures:
- how many units the corpus's matchable sections make, and their length in each model's tokens
  (a unit past a model's limit is truncated, its end never seen);
- each model's embedding rate on this machine, and from it the corpus's embedding time;
- the bytes of the vectors and of each index type, the release download, and the time an exact
  search over memory-mapped float16 vectors takes per query window;
- development recall: each development act's sections are embedded as queries against the
  enacting bills hidden among a seeded sample of other bills. An enactment counts as found when
  one of its bills has a unit closer to an act section's units than all but a fixed share of
  the (act section, sample bill) pairs. Fixing that share, not a cosine, compares schemes
  fairly, since a coarser unit shifts every cosine.

The sample leaves out every bill related to a development act that the corpus can name: the
enactments' bills, the near-duplicate stage's relatives of each act with their families, and
bills whose title names an act or, for Public Expression Protection, the lawsuits it addresses.
A related bill left in the sample would set the cutoff by itself.

Run after `uv run lineage index` and `uv run lineage model fetch`:

    uv run python -m benchmarks.semantic_units

and it writes reports/semantic_sizing.json. The held-out acts are never read.
"""

import hashlib
import json
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray
from tokenizers import Tokenizer

from benchmarks.minhash import machine
from lineage import queries
from lineage.acts import ACT_NAMES, ACTS, load_act
from lineage.boilerplate import boilerplate_kind
from lineage.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.embed import embed, load_model
from lineage.models import MODELS, require_model
from lineage.sections import normalize, split_sections
from lineage.windows import WINDOW_STRIDE, WINDOW_WORDS, display_windows, window_counts

COMMAND = "uv run python -m benchmarks.semantic_units"
REPORT = Path("reports/semantic_sizing.json")
TABLES = Path("data/tables")
RESULTS = Path("reports/results.json")
SEED = 20261108
CACHE = TABLES / "semantic_probe"

# (window words, stride): the scheme both stages use now, the same windows without overlap, and
# windows twice as long, which fit the longer model's 512 tokens and about the shorter's 256.
SCHEMES = {
    f"{WINDOW_WORDS}/{WINDOW_STRIDE}": (WINDOW_WORDS, WINDOW_STRIDE),
    f"{WINDOW_WORDS}/{WINDOW_WORDS}": (WINDOW_WORDS, WINDOW_WORDS),
    f"{2 * WINDOW_WORDS}/{WINDOW_WORDS}": (2 * WINDOW_WORDS, WINDOW_WORDS),
}
DISTRACTOR_BILLS = 3000
TOKEN_SAMPLE = 2000
PROBE_SECTIONS = 1000
# Shares of (act section, sample bill) pairs allowed above the cutoff.
FALSE_POSITIVE_RATES = (1e-3, 1e-4)
RELATED_TITLES = "|".join(
    [name.casefold() for act, name in ACT_NAMES.items() if act in ACTS]
    + ["slapp", "strategic lawsuit"]
)
GIB = 1 << 30
ASSET_LIMIT = 2 * GIB  # GitHub's limit per release asset (DR-2)
KEY_BYTES = 8  # per vector: its row in the windows table, as int64
HNSW_M = 32  # links per node; level 0 keeps 2M of them, 4 bytes each
PQ_BYTES = 48  # 8-bit codes for 48 subvectors of 8 dimensions
TOP_K = 20  # neighbors each window retrieves
SEARCH_ROWS = 1_000_000  # rows scanned to time an exact search
SEARCH_CHUNK = 1 << 17
QUERY_WINDOWS = 20


@dataclass(frozen=True, slots=True)
class Unit:
    bill: str
    section: int
    text: str


def _units(sections: list[tuple[str, int, str]], size: int, stride: int) -> list[Unit]:
    return [
        Unit(bill, section, text)
        for bill, section, whole in sections
        for text in display_windows(whole, size=size, stride=stride)
    ]


def _quantiles(values: NDArray[np.int64]) -> dict[str, int]:
    return {
        "median": int(np.quantile(values, 0.5, method="lower")),
        "p90": int(np.quantile(values, 0.9, method="lower")),
        "p99": int(np.quantile(values, 0.99, method="lower")),
        "max": int(values.max()),
    }


def sizes(units: int, dim: int, model_bytes: int) -> dict[str, int]:
    """Bytes of each stored form of `units` vectors of `dim` dimensions."""
    vectors = units * dim * 2
    download = vectors + units * KEY_BYTES + model_bytes
    return {
        "vectors_float16": vectors,
        "keys": units * KEY_BYTES,
        "hnsw_flat": units * (dim * 4 + 2 * HNSW_M * 4),
        "hnsw_float16": units * (dim * 2 + 2 * HNSW_M * 4),
        "ivf_pq": units * (PQ_BYTES + KEY_BYTES),
        "release_download": download,
        "release_assets": -(-download // ASSET_LIMIT),
    }


def bill_scores(
    queries_by_section: list[NDArray[np.float32]],
    pool: NDArray[np.float32],
    pool_bill: NDArray[np.int64],
    bills: int,
) -> NDArray[np.float32]:
    """(act section, bill): the best cosine between any of the section's units and the bill's."""
    out = np.full((len(queries_by_section), bills), -1.0, dtype=np.float32)
    for row, vectors in enumerate(queries_by_section):
        best = (vectors @ pool.T).max(axis=0)
        np.maximum.at(out[row], pool_bill, best)
    return out


def _act_sections() -> dict[str, list[tuple[str, int, str]]]:
    out: dict[str, list[tuple[str, int, str]]] = {}
    for act in ACTS:
        sections = split_sections(load_act(act)).sections
        out[act] = [
            (act, i, s.text)
            for i, s in enumerate(sections)
            if boilerplate_kind(normalize(s.text)) is None
        ]
    return out


def embed_once(name: str, model: TextEmbedding, texts: list[str]) -> NDArray[np.float32]:
    """Vectors for `texts`, embedding each distinct text once, kept in CACHE so a rerun reuses
    them."""
    key = hashlib.sha256("\0".join([name, *texts]).encode()).hexdigest()[:24]
    path = CACHE / f"{key}.npy"
    if path.exists():
        return np.load(path)
    distinct = sorted(set(texts))
    at = {text: i for i, text in enumerate(distinct)}
    vectors = embed(model, distinct)[[at[text] for text in texts]]
    CACHE.mkdir(parents=True, exist_ok=True)
    require_free_space(CACHE, needed=vectors.nbytes, floor=DEFAULT_FLOOR_BYTES)
    np.save(path, vectors)
    return vectors


def distinct_units(sections_glob: str, eligible: Path) -> dict[str, dict[str, int]]:
    """Per scheme, the units of every matchable section and how many distinct texts they hold:
    an identical text needs one vector, however many bills carry it."""
    hashes: dict[str, list[NDArray[np.uint64]]] = defaultdict(list)
    for path in sorted(Path().glob(sections_glob)):
        with queries.connect_quiet() as con:
            cursor = queries.matchable_sections(
                con, sections_path=path, eligible_path=eligible, display=True
            )
            while rows := cursor.fetchmany(100_000):
                for scheme, (size, stride) in SCHEMES.items():
                    digests = [
                        hashlib.blake2b(" ".join(w.split()).encode(), digest_size=8).digest()
                        for _, _, text, _ in rows
                        for w in display_windows(text, size=size, stride=stride)
                    ]
                    hashes[scheme].append(np.frombuffer(b"".join(digests), dtype=np.uint64))
    out: dict[str, dict[str, int]] = {}
    for scheme, parts in hashes.items():
        everything = np.concatenate(parts)
        out[scheme] = {"units": int(everything.size), "distinct": int(np.unique(everything).size)}
    return out


def _recall(
    model_name: str,
    sections: list[tuple[str, int, str]],
    acts: dict[str, list[tuple[str, int, str]]],
    targets: dict[str, list[tuple[str, list[str]]]],
    size: int,
    stride: int,
) -> dict[str, Any]:
    model = load_model(model_name)
    units = _units(sections, size, stride)
    bill_ids = sorted({u.bill for u in units})
    index = {b: i for i, b in enumerate(bill_ids)}
    pool = embed_once(model_name, model, [u.text for u in units])
    pool_bill = np.array([index[u.bill] for u in units], dtype=np.int64)
    per_act: dict[str, Any] = {}
    found: dict[str, list[bool]] = defaultdict(list)
    enacting_all = {b for rows in targets.values() for _, bills in rows for b in bills}
    distractor = np.array([b not in enacting_all for b in bill_ids])
    for act, act_sections in acts.items():
        act_units = _units(act_sections, size, stride)
        vectors = embed_once(model_name, model, [u.text for u in act_units])
        by_section: dict[int, list[int]] = defaultdict(list)
        for i, u in enumerate(act_units):
            by_section[u.section].append(i)
        scores = bill_scores(
            [vectors[rows] for rows in by_section.values()], pool, pool_bill, len(bill_ids)
        )
        background = scores[:, distractor].ravel()
        cutoffs = {rate: float(np.quantile(background, 1 - rate)) for rate in FALSE_POSITIVE_RATES}
        mean = scores.mean(axis=0)
        rows: list[dict[str, Any]] = []
        for state, bills in targets[act]:
            present = [index[b] for b in bills if b in index]
            best = float(scores[:, present].max()) if present else None
            # The bill's rank among the sample by its mean best cosine over the act's sections.
            rank = 1 + int((mean[distractor] > mean[present].max()).sum()) if present else None
            hits = {str(rate): best is not None and best > cut for rate, cut in cutoffs.items()}
            for rate, hit in hits.items():
                found[rate].append(hit)
            rows.append(
                {
                    "state": state,
                    "best_cosine": None if best is None else round(best, 4),
                    "rank": rank,
                    "found": hits,
                }
            )
        per_act[act] = {
            "units": len(act_units),
            "cutoffs": {str(r): round(c, 4) for r, c in cutoffs.items()},
            "enactments": rows,
        }
    return {
        "units_embedded": len(units),
        "found": {rate: sum(hits) for rate, hits in found.items()},
        "enactments": sum(len(t) for t in targets.values()),
        "acts": per_act,
    }


def exact_search_seconds(dim: int, scratch: Path) -> float:
    """Seconds per million float16 vectors to scan from a memory-mapped file, for one query of
    `QUERY_WINDOWS` windows keeping each window's `TOP_K` best, with the file in the page cache."""
    rng = np.random.default_rng(SEED)
    path = scratch / "vectors.npy"
    stored = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(SEARCH_ROWS, dim))
    for at in range(0, SEARCH_ROWS, SEARCH_CHUNK):
        block = rng.standard_normal((min(SEARCH_CHUNK, SEARCH_ROWS - at), dim), dtype=np.float32)
        stored[at : at + len(block)] = block / np.linalg.norm(block, axis=1, keepdims=True)
    stored.flush()
    del stored
    query = rng.standard_normal((QUERY_WINDOWS, dim), dtype=np.float32)
    vectors = np.load(path, mmap_mode="r")
    timings: list[float] = []
    for _ in range(3):  # the first pass reads the file into the page cache
        started = time.perf_counter()
        for at in range(0, SEARCH_ROWS, SEARCH_CHUNK):
            scores = query @ np.asarray(vectors[at : at + SEARCH_CHUNK], dtype=np.float32).T
            np.argpartition(scores, -TOP_K, axis=1)[:, -TOP_K:]
        timings.append(time.perf_counter() - started)
    return min(timings) * 1_000_000 / SEARCH_ROWS


def main() -> None:
    rng = np.random.default_rng(SEED)
    sections_glob = str(TABLES / "sections" / "*.parquet")
    eligible = TABLES / "eligible.parquet"
    commission = json.loads(RESULTS.read_text())["commission"]
    targets: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for e in commission["enactments"]:
        if e["act"] in ACTS and e["bills"]:
            targets[e["act"]].append((e["state"], [b["id"] for b in e["bills"]]))
    enacting = {b for rows in targets.values() for _, bills in rows for b in bills}
    relatives = [
        (r["state"], r["session"], r["identifier"])
        for act in ACTS
        for r in commission["queries"][act]["relatives"]
    ]
    with queries.connect_quiet() as con:
        words = np.array(
            queries.section_word_counts(con, sections_glob=sections_glob, eligible_path=eligible),
            dtype=np.int64,
        )
        pool_ids = queries.matchable_bill_ids(
            con, sections_glob=sections_glob, eligible_path=eligible
        )
        related = set(
            queries.related_bill_ids(
                con,
                bills_glob=str(TABLES / "bills" / "*.parquet"),
                families_path=TABLES / "families" / "families.parquet",
                listed=relatives,
                title_regex=RELATED_TITLES,
            )
        )
    others = sorted(set(pool_ids) - enacting - related)
    sample = sorted(rng.choice(others, size=DISTRACTOR_BILLS, replace=False).tolist())
    with queries.connect_quiet() as con:
        sections = queries.matchable_section_texts(
            con,
            sections_glob=sections_glob,
            eligible_path=eligible,
            ids=sorted(set(sample) | enacting),
        )
    acts = _act_sections()
    corpus = distinct_units(sections_glob, eligible)
    print("distinct", corpus, flush=True)

    tokenizers = {
        name: Tokenizer.from_file(str(require_model(name) / "tokenizer.json")) for name in MODELS
    }
    for tok in tokenizers.values():
        tok.no_truncation()
        tok.no_padding()
    probe_rows: list[int] = rng.choice(len(sections), size=PROBE_SECTIONS, replace=False).tolist()
    probe = [sections[i] for i in probe_rows]
    model_bytes = {name: sum(f.size for f in m.files) for name, m in MODELS.items()}

    scratch_root = TABLES.parent
    require_free_space(scratch_root, needed=SEARCH_ROWS * 384 * 2, floor=DEFAULT_FLOOR_BYTES)
    with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
        search = {name: exact_search_seconds(m.dim, Path(scratch)) for name, m in MODELS.items()}

    schemes: dict[str, Any] = {}
    for scheme, (size, stride) in SCHEMES.items():
        # Python's split, as the windows are cut; DuckDB's \\s+ can count words differently.
        units = corpus[scheme]["units"]
        counted = int(window_counts(words, size=size, stride=stride).sum())
        distinct = corpus[scheme]["distinct"]
        texts = [u.text for u in _units(probe, size, stride)]
        picked_rows: list[int] = rng.choice(len(texts), size=TOKEN_SAMPLE, replace=False).tolist()
        picked = [texts[i] for i in picked_rows]
        tokens: dict[str, Any] = {}
        for name, tok in tokenizers.items():
            counts = np.array([len(e.ids) for e in tok.encode_batch(picked)], dtype=np.int64)
            limit = MODELS[name].max_tokens
            tokens[name] = {
                **_quantiles(counts),
                "limit": limit,
                "over_limit_share": round(float((counts > limit).mean()), 4),
            }
        models: dict[str, Any] = {}
        for name in MODELS:
            model = load_model(name)
            embed(model, picked[:64])  # load and warm up before timing
            started = time.perf_counter()
            embed(model, picked)
            per_second = len(picked) / (time.perf_counter() - started)
            per_million = search[name]
            models[name] = {
                "units_per_second": round(per_second, 1),
                "corpus_hours": round(units / per_second / 3600, 1),
                "corpus_hours_distinct": round(distinct / per_second / 3600, 1),
                "exact_search_seconds_per_query": round(per_million * units / 1_000_000, 2),
                "bytes": sizes(units, MODELS[name].dim, model_bytes[name]),
                "development_recall": _recall(name, sections, acts, targets, size, stride),
            }
            print(
                scheme, name, units, models[name]["units_per_second"],
                models[name]["development_recall"]["found"], flush=True,
            )  # fmt: skip
        schemes[scheme] = {
            "window_words": size,
            "stride": stride,
            "units": units,
            "distinct_units": distinct,
            "units_from_sql_word_counts": counted,
            "tokens": tokens,
            "models": models,
        }

    doc = {
        "command": COMMAND,
        "seed": SEED,
        "machine": machine(),
        "sections": int(words.size),
        "distractor_bills": DISTRACTOR_BILLS,
        "related_bills_left_out": len(related - enacting),
        "related_titles": RELATED_TITLES,
        "false_positive_rates": list(FALSE_POSITIVE_RATES),
        "exact_search": {
            "rows_timed": SEARCH_ROWS,
            "query_windows": QUERY_WINDOWS,
            "top_k": TOP_K,
            "seconds_per_million": {k: round(v, 4) for k, v in search.items()},
            "note": "page cache warm; a cold read adds the file size over the disk's read rate",
        },
        "schemes": schemes,
    }
    REPORT.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
