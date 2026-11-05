"""The document-frequency stoplist: shingles found in more than a set share of all sections.

Shared clauses that no pattern list anticipates (standard definitions, amending formulas,
appropriation language) show up as shingles present in a large share of sections. Dropping
them before MinHash keeps them from creating matches. The list comes from the data, so it
follows each state's habits without upkeep, and it is rebuilt with the corpus. Frequencies are
counted over matchable sections only: those of eligible document types (DESIGN.md#dr-5) not
flagged as boilerplate, so excluded resolutions cannot shift the list. See
DESIGN.md#boilerplate.
"""

import hashlib
import json
import multiprocessing
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.shingles import SHINGLE_WORDS, section_shingles, shingle_hashes

# Fork is Linux's default, and forking after DuckDB has started threads can deadlock the
# child. Spawn starts each worker clean on every platform.
_SPAWN = multiprocessing.get_context("spawn")

COMMAND = "uv run lineage boilerplate"
# The shares the report tabulates, so the threshold can be chosen from the data.
REPORT_SHARES = (0.01, 0.003, 0.001, 0.0003, 0.0001)
_TOP = 50
_FETCH_SECTIONS = 5_000
# Hashes are split by their top bits before counting, so no single sort holds the corpus.
_PARTITION_BITS = 3


class StoplistError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class StoplistResult:
    sections: int
    distinct_shingles: int
    occurrences: int
    stoplisted: int
    occurrences_removed: int
    seconds: float


def document_frequencies(
    section_sets: Sequence[NDArray[np.uint32]],
) -> tuple[NDArray[np.uint32], NDArray[np.int64]]:
    """Each distinct hash and the number of sections whose set contains it."""
    if not section_sets:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.int64)
    hashes, counts = np.unique(np.concatenate(section_sets), return_counts=True)
    return hashes, counts.astype(np.int64)


def over_threshold(
    hashes: NDArray[np.uint32], counts: NDArray[np.int64], *, sections: int, max_share: float
) -> NDArray[np.uint32]:
    """Hashes present in more than `max_share` of `sections`."""
    return hashes[counts > max_share * sections]


def apply_stoplist(
    section_sets: Sequence[NDArray[np.uint32]], stoplist: NDArray[np.uint32]
) -> list[NDArray[np.uint32]]:
    """Each section's shingle set without the stoplisted hashes; `stoplist` must be sorted."""
    return [s[~np.isin(s, stoplist, assume_unique=True)] for s in section_sets]


def load_stoplist(tables_dir: Path) -> NDArray[np.uint32]:
    """The stoplisted hashes, sorted."""
    path = tables_dir / "stoplist.parquet"
    if not path.exists():
        raise StoplistError(f"no stoplist in {tables_dir}; run `lineage boilerplate` first")
    with queries.connect_quiet() as con:
        return np.array(queries.read_stoplist(con, parquet_path=path), dtype=np.uint32)


def _part_shingles(part: Path, eligible: Path) -> tuple[int, list[NDArray[np.uint32]]]:
    """All matchable section shingle sets of one sections file, split into hash partitions."""
    sets: list[NDArray[np.uint32]] = []
    sections = 0
    with queries.connect_quiet() as con:
        scan = queries.matchable_section_texts(con, sections_path=part, eligible_path=eligible)
        while batch := scan.fetchmany(_FETCH_SECTIONS):
            texts = [t for (t,) in batch]
            sections += len(texts)
            sets.extend(section_shingles(texts))
    flat = np.concatenate(sets) if sets else np.empty(0, dtype=np.uint32)
    top = flat >> np.uint32(32 - _PARTITION_BITS)
    return sections, [flat[top == p] for p in range(1 << _PARTITION_BITS)]


def _texts_of(parts: list[Path], eligible: Path, wanted: set[int]) -> dict[int, str]:
    """Recover one example text for each wanted hash, scanning sections until all are found."""
    found: dict[int, str] = {}
    wanted_array = np.array(sorted(wanted), dtype=np.uint32)
    with queries.connect_quiet() as con:
        for part in parts:
            scan = queries.matchable_section_texts(con, sections_path=part, eligible_path=eligible)
            while batch := scan.fetchmany(_FETCH_SECTIONS):
                texts = [t for (t,) in batch]
                hashes, owners = shingle_hashes(texts)
                first = np.searchsorted(owners, np.arange(len(texts)))
                for i in np.flatnonzero(np.isin(hashes, wanted_array)).tolist():
                    h = int(hashes[i])
                    if h not in found:
                        owner = int(owners[i])
                        start = i - int(first[owner])
                        found[h] = " ".join(texts[owner].split()[start : start + SHINGLE_WORDS])
                if len(found) == len(wanted):
                    return found
    return found


def build_stoplist(
    tables_dir: Path,
    *,
    max_share: float,
    workers: int,
    report_path: Path,
    manifest_sha256: str,
    rules_sha256: str,
) -> StoplistResult:
    """Write `<tables>/stoplist.parquet` and the report from the sections table."""
    started = time.monotonic()
    parts = sorted((tables_dir / "sections").glob("*.parquet"))
    if not parts:
        raise StoplistError(f"no sections table in {tables_dir}; run `lineage split` first")
    eligible = tables_dir / "eligible.parquet"
    if not eligible.exists():
        raise StoplistError(f"no eligibility table in {tables_dir}; run `lineage eligible` first")
    with ProcessPoolExecutor(max_workers=workers, mp_context=_SPAWN) as pool:
        results = list(pool.map(_part_shingles, parts, [eligible] * len(parts)))
    sections = sum(n for n, _ in results)

    stop_hashes: list[NDArray[np.uint32]] = []
    stop_counts: list[NDArray[np.int64]] = []
    distinct = occurrences = 0
    by_share = {s: [0, 0] for s in (*REPORT_SHARES, max_share)}
    for p in range(1 << _PARTITION_BITS):
        hashes, counts = document_frequencies([r[1][p] for r in results])
        distinct += hashes.size
        occurrences += int(counts.sum())
        for share, tally in by_share.items():
            over = counts > share * sections
            tally[0] += int(over.sum())
            tally[1] += int(counts[over].sum())
        keep = counts > max_share * sections
        stop_hashes.append(hashes[keep])
        stop_counts.append(counts[keep])
    del results
    stoplist = np.concatenate(stop_hashes)
    dfs = np.concatenate(stop_counts)

    with queries.connect_quiet() as con:
        # Written beside the old file and renamed, so a failure never leaves half a stoplist.
        partial = tables_dir / "stoplist.parquet.partial"
        queries.write_stoplist(
            con, parquet_path=partial, hashes=stoplist.tolist(), sections=dfs.tolist()
        )
        partial.replace(tables_dir / "stoplist.parquet")
        boilerplate = queries.boilerplate_counts(
            con, sections_glob=str(tables_dir / "sections" / "*.parquet")
        )

    # Highest document frequency first; the hash breaks ties so the choice is reproducible.
    top = np.lexsort((stoplist, -dfs))[:_TOP]
    top_df = {int(stoplist[i]): int(dfs[i]) for i in top}
    texts = _texts_of(parts, eligible, set(top_df))
    top_rows = sorted(
        (
            {"shingle": texts[h], "sections": df, "share": round(df / sections, 6)}
            for h, df in top_df.items()
        ),
        key=lambda r: (-int(r["sections"]), str(r["shingle"])),
    )
    removed = int(dfs.sum())
    report = {
        "command": COMMAND,
        "manifest_sha256": manifest_sha256,
        "rules_sha256": rules_sha256,
        "sections": sections,
        "distinct_shingles": distinct,
        "shingle_occurrences": occurrences,
        "max_share": max_share,
        "max_sections": int(max_share * sections),
        "stoplisted_shingles": int(stoplist.size),
        "occurrences_removed": removed,
        "by_threshold": [
            {"max_share": s, "stoplisted_shingles": n, "occurrences_removed": o}
            for s, (n, o) in sorted(by_share.items(), reverse=True)
        ],
        "boilerplate_sections": boilerplate,
        "top_shingles": top_rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return StoplistResult(
        sections=sections,
        distinct_shingles=distinct,
        occurrences=occurrences,
        stoplisted=int(stoplist.size),
        occurrences_removed=removed,
        seconds=time.monotonic() - started,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
