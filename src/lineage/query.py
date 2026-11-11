"""`lineage query`: find a bill's near-duplicates in the built index, offline.

The query text goes through the same steps as the corpus: cleaned, split, normalized, with
boilerplate-only sections set aside, then cut into windows, shingled, stoplisted and signed. A
query section matches a bill when any of its windows matches any of the bill's. Each band
keys are looked up in the memory-mapped band index by binary search, and a bucket over the
build's cap yields nothing here either. Every candidate is verified against its stored signature
at the build's threshold. Matches are grouped by family, so a state's copies of a bill count
once, and a family is scored from the query's side: the share of the query's words that fall in
the sections it matches, of the words of every section searched, so a provision carried inside
a larger bill still scores high, and a long matched section counts for more than a short one.

With the semantic stage's search (lineage.semantic), each matchable section's windows are also
embedded and searched by meaning, and a bill matches a query section when either stage finds
it. Every match keeps which stage found each of its sections, so the stages can be measured
alone and together. Nothing here needs a network connection or an API key. See
DESIGN.md#near-duplicate-stage and DESIGN.md#semantic-stage.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.bands import load_band_index
from lineage.boilerplate import boilerplate_kind
from lineage.lsh import BANDS, BUCKET_CAP, BandIndex, band_keys
from lineage.minhash import NUM_PERM, permutations, signatures
from lineage.sections import normalize, split_sections
from lineage.semantic import SemanticSearch, search_sections
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
    sections_matched: int  # of the query's sections either stage could match
    share: float  # of the searched sections' words, the share in sections the family matches
    best_estimate: float  # the highest estimated similarity of any matched section pair, or 0
    # Query sections each stage matched, by stage: "near_duplicate", "semantic".
    sections_by_stage: dict[str, int] = field(default_factory=dict[str, int])
    best_cosine: float | None = None  # the semantic stage's best, when it matched


@dataclass(frozen=True, slots=True)
class QueryResult:
    sections: int
    signed: int  # sections with at least one signed window
    matches: list[Match]
    # Sections either stage could match: the signed ones, and with the semantic stage every
    # matchable section with a window. Their words are the share's denominator.
    searched: int = 0
    words_searched: int = 0


@dataclass(frozen=True, slots=True)
class SignedQuery:
    sections: int
    values: NDArray[np.uint32]  # one signature per signed window
    section: NDArray[np.int64]  # each window's matchable section, numbered from 0

    @property
    def signed(self) -> int:
        return int(np.unique(self.section).size)


def matchable_sections(raw: str) -> tuple[int, list[tuple[str, str]]]:
    """The query's section count, and each matchable section's (display, normalized) text."""
    split = split_sections(raw)
    texts = [(section.text, normalize(section.text)) for section in split.sections]
    return len(texts), [(text, norm) for text, norm in texts if boilerplate_kind(norm) is None]


def sign_query(raw: str, tables_dir: Path) -> SignedQuery:
    """The query's section count, and the signatures of its matchable sections' windows."""
    sections, matchable = matchable_sections(raw)
    cut = [normalized_windows(text, norm) for text, norm in matchable]
    kept = apply_stoplist(
        section_shingles([w for windows in cut for w in windows]), load_stoplist(tables_dir)
    )
    sets: list[NDArray[np.uint32]] = []
    owner: list[int] = []
    at = 0
    for number, windows in enumerate(cut):
        signed = [s for s in kept[at : at + len(windows)] if s.size]
        at += len(windows)
        sets += signed
        owner += [number] * len(signed)
    if not sets:
        empty = np.empty((0, NUM_PERM), dtype=np.uint32)
        return SignedQuery(sections, empty, np.empty(0, dtype=np.int64))
    return SignedQuery(sections, signatures(sets, permutations()), np.array(owner, np.int64))


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


def _near_duplicate_hits(query: SignedQuery, tables_dir: Path) -> list[tuple[int, str, int]]:
    """(query section, bill id, agreements) for every verified window pair."""
    if not query.signed:
        return []
    query_rows, rows = _candidates(band_keys(query.values), tables_dir)
    files = open_signatures(tables_dir)
    counts = np.count_nonzero(signature_values(files, rows) == query.values[query_rows], axis=1)
    kept = counts >= min_agreements(THRESHOLD)
    query_rows, rows, counts = query.section[query_rows[kept]], rows[kept], counts[kept]
    bill_ids = signature_bill_ids(files, rows)
    return list(zip(query_rows.tolist(), bill_ids, counts.tolist(), strict=True))


def group_matches(
    near: Sequence[tuple[int, str, int]],
    by_meaning: Sequence[tuple[int, str, float]],
    words: Mapping[int, int],
    words_searched: int,
    tables_dir: Path,
) -> list[Match]:
    """Matches grouped by family, best first, scored from the query's side.

    `near` holds (query section, bill id, agreements) and `by_meaning` (query section, bill id,
    cosine); `words` gives each query section's words, and `words_searched` the share's
    denominator. A corpus bill's relatives are scored by the same function (lineage.relatives),
    so a bill and a query scored alike agree."""
    with queries.connect_quiet() as con:
        summaries = queries.bill_summaries(
            con,
            bills_glob=str(tables_dir / "bills" / "*.parquet"),
            families_path=tables_dir / "families" / "families.parquet",
            ids=sorted({b for _, b, _ in near} | {b for _, b, _ in by_meaning}),
        )
    by_id = {s.id: s for s in summaries}
    # bill id -> stage -> query sections it matches
    matched: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    best: dict[str, int] = defaultdict(int)
    cosine: dict[str, float] = {}
    for q, bill_id, count in near:
        matched[bill_id]["near_duplicate"].add(q)
        best[bill_id] = max(best[bill_id], count)
    for q, bill_id, c in by_meaning:
        matched[bill_id]["semantic"].add(q)
        cosine[bill_id] = max(cosine.get(bill_id, -1.0), c)

    def covered(bill_id: str) -> set[int]:
        return set[int]().union(*matched[bill_id].values())

    members: dict[str, list[str]] = defaultdict(list)
    for bill_id in matched:
        members[by_id[bill_id].family].append(bill_id)
    results: list[Match] = []
    for family, bills in members.items():
        # The family is shown through the bill that matches the most query sections.
        shown = by_id[min(bills, key=lambda b: (-len(covered(b)), -best[b], b))]
        stages = {stage for b in bills for stage in matched[b]}
        cosines = [cosine[b] for b in bills if b in cosine]
        family_covers = set[int]().union(*(covered(b) for b in bills))
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
                sections_matched=len(family_covers),
                share=sum(words[q] for q in family_covers) / max(words_searched, 1),
                best_estimate=max(best[b] for b in bills) / NUM_PERM,
                sections_by_stage={
                    stage: len(set[int]().union(*(matched[b][stage] for b in bills)))
                    for stage in sorted(stages)
                },
                best_cosine=round(max(cosines), 4) if cosines else None,
            )
        )
    results.sort(key=lambda m: (-m.share, -m.best_estimate, m.state, m.identifier, m.bill_id))
    return results


def query_text(
    raw: str, tables_dir: Path, *, semantic: SemanticSearch | None = None
) -> QueryResult:
    """Every family that shares text with `raw`, best first: verified near-duplicates, and with
    `semantic`, the semantic stage's matches too."""
    require_index(tables_dir)
    query = sign_query(raw, tables_dir)
    near = _near_duplicate_hits(query, tables_dir)
    searched: set[int] = set(query.section.tolist())
    _, matchable = matchable_sections(raw)
    words = [len(normalized.split()) for _, normalized in matchable]
    by_meaning: list[tuple[int, str, float]] = []
    if semantic is not None:
        texts = [text for text, _ in matchable]
        searched |= {i for i, text in enumerate(texts) if text.split()}
        by_meaning = [
            (h.query_section, h.bill_id, h.cosine) for h in search_sections(texts, semantic)
        ]
    if not searched:
        return QueryResult(query.sections, query.signed, [], 0)

    words_searched = sum(words[q] for q in searched)
    results = group_matches(near, by_meaning, dict(enumerate(words)), words_searched, tables_dir)
    return QueryResult(query.sections, query.signed, results, len(searched), words_searched)
