"""`lineage query`: find a bill's near-duplicates in the built index, offline.

The query text goes through the same steps as the corpus: cleaned, split, normalized, with
boilerplate-only sections set aside, then shingled, stoplisted and signed. Each signature's band
keys are looked up in the memory-mapped band index by binary search, and a bucket over the
build's cap yields nothing here either. Every candidate is verified against its stored signature
at the build's threshold. Matches are grouped by family, so a state's copies of a bill count
once, and a family is scored from the query's side: the share of the query's signed sections it
matches, so a provision carried inside a larger bill still scores high. Nothing here needs a
network connection or an API key. See DESIGN.md#near-duplicate-stage.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.bands import load_band_index
from lineage.boilerplate import boilerplate_kind
from lineage.lsh import BANDS, BUCKET_CAP, band_keys
from lineage.minhash import NUM_PERM, permutations, signatures
from lineage.sections import normalize, split_sections
from lineage.shingles import section_shingles
from lineage.signatures import open_signatures, signature_bill_ids, signature_values
from lineage.stoplist import apply_stoplist, load_stoplist
from lineage.verify import THRESHOLD, min_agreements


class IndexMissingError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Match:
    """One family of bills that shares text with the query, shown through its best bill."""

    family: str
    family_bills: int
    bill_id: str
    state: str
    session: str
    identifier: str
    title: str
    status: str | None
    sections_matched: int  # of the query's signed sections
    share: float
    best_estimate: float  # the highest estimated similarity of any matched section pair


@dataclass(frozen=True, slots=True)
class QueryResult:
    sections: int
    signed: int
    matches: list[Match]


def _sign(raw: str, tables_dir: Path) -> tuple[int, NDArray[np.uint32]]:
    split = split_sections(raw)
    texts = [normalize(section.text) for section in split.sections]
    matchable = [text for text in texts if boilerplate_kind(text) is None]
    sets = [
        s for s in apply_stoplist(section_shingles(matchable), load_stoplist(tables_dir)) if s.size
    ]
    if not sets:
        return len(texts), np.empty((0, NUM_PERM), dtype=np.uint32)
    return len(texts), signatures(sets, permutations())


def _candidates(
    keys: NDArray[np.uint64], tables_dir: Path
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(query section, signature row) for every bucket a query section lands in."""
    index = load_band_index(tables_dir)
    found: list[NDArray[np.int64]] = []
    for band in range(BANDS):
        column = index.keys[band]
        starts = np.searchsorted(column, keys[:, band], side="left")
        stops = np.searchsorted(column, keys[:, band], side="right")
        for q, (start, stop) in enumerate(zip(starts.tolist(), stops.tolist(), strict=True)):
            if 0 < stop - start <= BUCKET_CAP:
                rows = index.rows[band, start:stop].astype(np.int64)
                found.append((np.int64(q) << 32) | rows)
    if not found:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    codes = np.unique(np.concatenate(found))
    return codes >> 32, codes & 0xFFFF_FFFF


def _require_index(tables_dir: Path) -> None:
    needed = [tables_dir / "bands" / "keys.npy", tables_dir / "families" / "families.parquet"]
    if not all(path.exists() for path in needed):
        raise IndexMissingError(f"no index in {tables_dir}; run `lineage index` first")


def query_text(raw: str, tables_dir: Path) -> QueryResult:
    """Every family that shares verified text with `raw`, best first."""
    _require_index(tables_dir)
    sections, signed = _sign(raw, tables_dir)
    if not signed.shape[0]:
        return QueryResult(sections, 0, [])
    query_rows, rows = _candidates(band_keys(signed), tables_dir)
    files = open_signatures(tables_dir)
    counts = np.count_nonzero(signature_values(files, rows) == signed[query_rows], axis=1)
    kept = counts >= min_agreements(THRESHOLD)
    query_rows, rows, counts = query_rows[kept], rows[kept], counts[kept]

    bill_ids = signature_bill_ids(files, rows)
    with queries.connect_quiet() as con:
        summaries = queries.bill_summaries(
            con,
            bills_glob=str(tables_dir / "bills" / "*.parquet"),
            families_path=tables_dir / "families" / "families.parquet",
            ids=sorted(set(bill_ids)),
        )
    by_id = {s.id: s for s in summaries}
    matched: dict[str, set[int]] = defaultdict(set)  # bill id -> query sections it matches
    best: dict[str, int] = defaultdict(int)
    for bill_id, q, count in zip(bill_ids, query_rows.tolist(), counts.tolist(), strict=True):
        matched[bill_id].add(q)
        best[bill_id] = max(best[bill_id], count)

    members: dict[str, list[str]] = defaultdict(list)
    for bill_id in matched:
        members[by_id[bill_id].family].append(bill_id)
    results: list[Match] = []
    for family, bills in members.items():
        # The family is shown through the bill that matches the most query sections.
        shown = by_id[min(bills, key=lambda b: (-len(matched[b]), -best[b], b))]
        covered = set[int]().union(*(matched[b] for b in bills))
        results.append(
            Match(
                family=family,
                family_bills=shown.family_bills,
                bill_id=shown.id,
                state=shown.state,
                session=shown.session,
                identifier=shown.identifier,
                title=shown.title,
                status=shown.status,
                sections_matched=len(covered),
                share=len(covered) / signed.shape[0],
                best_estimate=max(best[b] for b in bills) / NUM_PERM,
            )
        )
    results.sort(key=lambda m: (-m.share, -m.best_estimate, m.state, m.identifier, m.bill_id))
    return QueryResult(sections, signed.shape[0], results)
