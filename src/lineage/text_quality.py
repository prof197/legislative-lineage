"""`lineage text sample`: the quality test that decides, per state and format, what to fetch.

A seeded sample from the fetch plan, about 50 documents per state and link format: up to 30
records whose stored text is unusable, which the fetch would fill, and at least 20 that already
have Open States text, which the fetch would be compared against. Each document is fetched and
extracted as the full run would, then judged by the clean-text rate: the share whose text splits
at section headings. For records with Open States text, the same test runs on their text, and
the two are compared for version (which version each text came from) and content (word 5-gram
overlap). The documents stay in `data/text/sample/`; the counts go to
`reports/text_quality.json`. See DESIGN.md#dr-6.
"""

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lineage import queries
from lineage.sections import clean, normalize, split_sections
from lineage.text_fetch import Job, fetch_all, host_of
from lineage.text_rules import RULES, TextRules, load_rules, usable

COMMAND = "uv run lineage text sample"
REPORT = Path("reports/text_quality.json")
STORE = Path("data/text/sample")
SEED = 42
PER_CELL = 50
WITH_TEXT = 20  # of each cell, records that already have Open States text
FORMATS = ("html", "pdf")


def draw_sample(
    plan: list[dict[str, Any]], *, seed: int = SEED, per_cell: int = PER_CELL
) -> list[tuple[str, Job]]:
    """(group, job) per sampled document; group is "without_text" or "with_text"."""
    cells: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in plan:
        if row["outcome"] != "planned":
            continue
        for fmt in FORMATS:
            if row[f"{fmt}_url"]:
                cells[(row["state"], fmt, row["kind"] == "usable")].append(row)
    sample: list[tuple[str, Job]] = []
    for state, fmt in sorted({(s, f) for s, f, _ in cells}):
        have = sorted(cells[(state, fmt, True)], key=lambda r: r["id"])
        need = sorted(cells[(state, fmt, False)], key=lambda r: r["id"])
        n_need = min(len(need), per_cell - min(len(have), WITH_TEXT))
        n_have = min(len(have), per_cell - n_need)
        rng = random.Random(f"{seed}:{state}:{fmt}")
        for group, rows, n in (("without_text", need, n_need), ("with_text", have, n_have)):
            for row in rng.sample(rows, n):
                job = Job(row["id"], row[f"{fmt}_url"], fmt, row["note"] or "", row["date"] or "")
                sample.append((group, job))
    return sample


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = normalize(clean(text)).split()
    return {tuple(words[i : i + 5]) for i in range(max(0, len(words) - 4))}


def overlap(ours: str, theirs: str) -> float:
    """Jaccard similarity of the two texts' word 5-grams."""
    a, b = _shingles(ours), _shingles(theirs)
    return len(a & b) / len(a | b) if a | b else 1.0


def _read_store(store: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """The latest stored result per (id, format)."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(store.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                latest[(row["id"], row["format"])] = row
    return latest


def _version_note(versions: list[dict[str, Any]], url: str | None) -> str | None:
    """The note of the version whose link Open States extracted its text from."""
    for version in versions:
        links: list[dict[str, str | None]] = version["links"] or []
        if any(link["url"] == url for link in links):
            return version["note"]
    return None


def _rate(hits: int, n: int) -> float | None:
    return round(hits / n, 3) if n else None


def containment(ours: str, theirs: str) -> float:
    """The share of their word 5-grams that are also in ours.

    Jaccard overlap penalizes text that only one side keeps, such as the digest or analysis
    Oregon, Wisconsin and Louisiana print before the bill; containment asks whether their text
    is in ours.
    """
    a, b = _shingles(ours), _shingles(theirs)
    return len(a & b) / len(b) if b else 1.0


def _readings(row: dict[str, Any], fmt: str) -> dict[str, str]:
    texts = {"plain": row["text"]}
    if fmt == "pdf" and row.get("layout_text"):
        texts["layout"] = row["layout_text"]
    return texts


def evaluate(
    sample: list[tuple[str, Job]],
    fetched: dict[tuple[str, str], dict[str, Any]],
    theirs: dict[str, tuple[str | None, str | None, list[dict[str, Any]]]],
    states: dict[str, str],
    rules: TextRules,
) -> dict[str, Any]:
    """Per state and format: outcomes; per reading, usable and clean rates; and the comparison
    with Open States on the records that have both."""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for group, job in sample:
        cell = cells.setdefault(
            (states[job.id], job.format),
            {"hosts": set(), "sampled": Counter(), "outcomes": Counter(), "rows": []},
        )
        cell["sampled"][group] += 1
        if host_of(job.url) in rules.retired:
            # A retired host's links are left out of the fetch, so they are not judged.
            cell["outcomes"]["retired_host"] += 1
            continue
        cell["hosts"].add(host_of(job.url))
        row = fetched.get((job.id, job.format))
        cell["outcomes"][row["outcome"] if row else "not_fetched"] += 1
        if row and row["outcome"] == "ok":
            cell["rows"].append((group, job, row))
    out: dict[str, Any] = {}
    for (state, fmt), cell in sorted(cells.items()):
        ok = cell["rows"]
        need = [r for g, _, r in ok if g == "without_text"]
        have = [(job, r) for g, job, r in ok if g == "with_text" and theirs[job.id][0]]
        readings: dict[str, Any] = {}
        everything = [r for _, _, r in ok]
        for reading in ("plain", "layout") if fmt == "pdf" else ("plain",):
            texts = [t for r in need if (t := _readings(r, fmt).get(reading))]
            pairs = [t for _, r in have if (t := _readings(r, fmt).get(reading))]
            every = [t for r in everything if (t := _readings(r, fmt).get(reading))]
            readings[reading] = {
                # Over every extracted document: the bar applies to all fetched text.
                "usable_rate": _rate(sum(usable(t, rules) for t in every), len(everything)),
                "clean_rate": {
                    "all": _rate(sum(_clean(t) for t in every), len(everything)),
                    "without_text": _rate(sum(_clean(t) for t in texts), len(need)),
                    "with_text": _rate(sum(_clean(t) for t in pairs), len(have)),
                },
            }
        their_texts = [theirs[job.id][0] or "" for job, _ in have]
        notes = [_version_note(theirs[job.id][2], theirs[job.id][1]) for job, _ in have]
        known = [n == job.note for n, (job, _) in zip(notes, have, strict=True) if n is not None]
        out.setdefault(state, {})[fmt] = {
            "hosts": sorted(cell["hosts"]),
            "sampled": dict(cell["sampled"]),
            "outcomes": dict(sorted(cell["outcomes"].items())),
            "readings": readings,
            "compared": len(have),
            "theirs": {
                "usable_rate": _rate(sum(usable(t, rules) for t in their_texts), len(have)),
                "clean_rate": _rate(sum(_clean(t) for t in their_texts), len(have)),
            },
            "same_version_rate": _rate(sum(known), len(known)),
            "overlap_median": _median(
                [overlap(r["text"], t) for (_, r), t in zip(have, their_texts, strict=True)]
            ),
            "containment_median": _median(
                [containment(r["text"], t) for (_, r), t in zip(have, their_texts, strict=True)]
            ),
            "text_bytes_mean": (
                round(statistics.mean(len(r["text"].encode()) for _, _, r in ok)) if ok else None
            ),
            "body_bytes_mean": (
                round(statistics.mean(r["body_bytes"] for _, _, r in ok)) if ok else None
            ),
            "seconds_median": _median([r["seconds"] for _, _, r in ok], 2),
        }
    return out


def _clean(text: str) -> bool:
    return split_sections(text).by_heading


def _median(values: list[float], digits: int = 3) -> float | None:
    return round(statistics.median(values), digits) if values else None


def run_sample(
    tables_dir: Path,
    *,
    plan_path: Path,
    store: Path,
    report_path: Path,
    fetch: bool = True,
    rules_path: Path = RULES,
) -> dict[str, Any]:
    rules = load_rules(rules_path)
    with queries.connect_quiet() as con:
        plan = queries.fetch_plan(con, path=plan_path)
    sample = draw_sample(plan)
    if fetch:
        # PDFs are also read by word position, so the report can choose the reading per state.
        fetch_all((job for _, job in sample), store, layout_too=True, containers=rules.containers)
    ids = sorted({job.id for _, job in sample})
    with queries.connect_quiet() as con:
        theirs = queries.stored_text(
            con, bills_glob=str(tables_dir / "bills" / "*.parquet"), ids=ids
        )
    states = {row["id"]: row["state"] for row in plan}
    doc: dict[str, Any] = {
        "command": COMMAND,
        "seed": SEED,
        "per_cell": PER_CELL,
        "with_text_per_cell": WITH_TEXT,
        "usable_means": (
            f"at least {rules.min_words} words, at least {rules.min_real_word_share:.0%} of them "
            "real words (lineage.text_rules.usable)"
        ),
        "clean_means": "the text splits at section headings (lineage.sections.split_sections)",
        "states": evaluate(sample, _read_store(store), theirs, states, rules),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc
