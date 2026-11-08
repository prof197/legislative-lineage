"""`lineage query`: find a bill's near-duplicates in the built index, offline.

The query text goes through the same steps as the corpus: cleaned, split, normalized, with
boilerplate-only sections set aside, then cut into windows, shingled, stoplisted and signed. A
query section matches a bill when any of its windows matches any of the bill's. Each band
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
from lineage.lsh import BANDS, BUCKET_CAP, BandIndex, band_keys
from lineage.minhash import NUM_PERM, permutations, signatures
from lineage.sections import normalize, split_sections
from lineage.shingles import section_shingles
from lineage.signatures import open_signatures, signature_bill_ids, signature_values
from lineage.stoplist import apply_stoplist, load_stoplist
from lineage.verify import THRESHOLD, min_agreements
from lineage.windows import normalized_windows


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
    signed: int  # sections with at least one signed window
    matches: list[Match]


@dataclass(frozen=True, slots=True)
class SignedQuery:
    sections: int
    values: NDArray[np.uint32]  # one signature per signed window
    section: NDArray[np.int64]  # each window's signed section, numbered from 0

    @property
    def signed(self) -> int:
        return int(self.section[-1]) + 1 if self.section.size else 0


def sign_query(raw: str, tables_dir: Path) -> SignedQuery:
    """The query's section count, and the signatures of its matchable sections' windows."""
    split = split_sections(raw)
    texts = [(section.text, normalize(section.text)) for section in split.sections]
    matchable = [(text, norm) for text, norm in texts if boilerplate_kind(norm) is None]
    cut = [normalized_windows(text, norm) for text, norm in matchable]
    kept = apply_stoplist(
        section_shingles([w for windows in cut for w in windows]), load_stoplist(tables_dir)
    )
    sets: list[NDArray[np.uint32]] = []
    owner: list[int] = []
    at = 0
    for windows in cut:
        signed = [s for s in kept[at : at + len(windows)] if s.size]
        at += len(windows)
        if signed:
            number = owner[-1] + 1 if owner else 0
            sets += signed
            owner += [number] * len(signed)
    if not sets:
        empty = np.empty((0, NUM_PERM), dtype=np.uint32)
        return SignedQuery(len(texts), empty, np.empty(0, dtype=np.int64))
    return SignedQuery(len(texts), signatures(sets, permutations()), np.array(owner, np.int64))


def bucket_sizes(keys: NDArray[np.uint64], index: BandIndex) -> NDArray[np.int64]:
    """The size of the bucket each key lands in, per query window and band."""
    sizes = np.empty(keys.shape, dtype=np.int64)
    for band in range(BANDS):
        column = index.keys[band]
        starts = np.searchsorted(column, keys[:, band], side="left")
        sizes[:, band] = np.searchsorted(column, keys[:, band], side="right") - starts
    return sizes


def _candidates(
    keys: NDArray[np.uint64], tables_dir: Path
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(query window, signature row) for every bucket a query window lands in."""
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


def require_index(tables_dir: Path) -> None:
    needed = [tables_dir / "bands" / "keys.npy", tables_dir / "families" / "families.parquet"]
    if not all(path.exists() for path in needed):
        raise IndexMissingError(f"no index in {tables_dir}; run `lineage index` first")


def query_text(raw: str, tables_dir: Path) -> QueryResult:
    """Every family that shares verified text with `raw`, best first."""
    require_index(tables_dir)
    query = sign_query(raw, tables_dir)
    if not query.signed:
        return QueryResult(query.sections, 0, [])
    query_rows, rows = _candidates(band_keys(query.values), tables_dir)
    files = open_signatures(tables_dir)
    counts = np.count_nonzero(signature_values(files, rows) == query.values[query_rows], axis=1)
    kept = counts >= min_agreements(THRESHOLD)
    query_rows, rows, counts = query.section[query_rows[kept]], rows[kept], counts[kept]

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
                share=len(covered) / query.signed,
                best_estimate=max(best[b] for b in bills) / NUM_PERM,
            )
        )
    results.sort(key=lambda m: (-m.share, -m.best_estimate, m.state, m.identifier, m.bill_id))
    return QueryResult(query.sections, query.signed, results)
