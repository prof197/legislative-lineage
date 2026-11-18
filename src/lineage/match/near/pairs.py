"""Verify the LSH candidates, store the cross-state pairs, and group same-state families.

Reads the signatures and band index that `lineage index` writes before it. Candidates are pairs
of windows; a section pair is verified when any pair of its windows is (DESIGN.md#windows).
Writes three tables:

- `<tables>/pairs/`: the verified cross-state section pairs as three aligned vectors
  (`left.npy` and `right.npy`, section numbers with left < right, and `agreements.npy`, the
  signature positions on which their best pair of windows agrees), sorted by left, then right.
  Sections are numbered in the signatures table's row order (`SignatureTable.sections`).
- `<tables>/same_state_pairs/`: the verified same-state section pairs, in the same layout. The
  state boilerplate flag (lineage.match.state_boilerplate) counts the families each section's
  near-duplicates reach in its own state.
- `<tables>/families/families.parquet`: every signed bill's family, named by the first bill id
  in the family in sort order. A bill with no near-duplicate in its state is its own family.

Same-state pairs form families and the state boilerplate flag; they never become matches. See
DESIGN.md#near-duplicate-stage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lineage.core import db
from lineage.core.dirs import replace_dir
from lineage.core.disk import require_free_space
from lineage.match.near import queries
from lineage.match.near.bands import load_band_index
from lineage.match.near.families import (
    FAMILY_SHARE,
    BillPairs,
    bill_pairs,
    components,
    linked,
)
from lineage.match.near.lsh import (
    BANDS,
    BUCKET_CAP,
    BandIndex,
    BandPairs,
    band_keys,
    band_pairs,
    bucket_sizes,
    candidates_by_band,
    merge_sorted,
    unseen,
)
from lineage.match.near.minhash import NUM_PERM
from lineage.match.near.signatures import RowLabels, SignatureFiles, signature_values
from lineage.match.near.verify import THRESHOLD, agreements_by, min_agreements

ESTIMATE_BINS = 16
SHARE_BINS = 20
FAMILY_SWEEP = (0.5, 2 / 3, 0.75, 0.8, 0.9, 1.0)
LARGEST_LISTED = 10
# Two uint32 rows and a uint8 count per pair; generous allowances for the family file.
_PAIR_BYTES = 4 + 4 + 1
_FAMILY_ROW_BYTES = 128
# Section pairs held, summed over bands, before repeats across bands are dropped: about 0.6 GB.
_HELD_SECTION_PAIRS = 1 << 26


@dataclass(frozen=True, slots=True)
class SectionPairs:
    left: NDArray[np.uint32]
    right: NDArray[np.uint32]
    agreements: NDArray[np.uint8]


def pair_codes(
    left: NDArray[np.integer[Any]], right: NDArray[np.integer[Any]]
) -> NDArray[np.uint64]:
    """Pairs as single sortable codes: left in the high 32 bits, right in the low."""
    return (np.asarray(left).astype(np.uint64) << np.uint64(32)) | np.asarray(right).astype(
        np.uint64
    )


def split_codes(codes: NDArray[np.uint64]) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Codes back into their left and right halves."""
    return (
        (codes >> np.uint64(32)).astype(np.uint32),
        (codes & np.uint64(0xFFFF_FFFF)).astype(np.uint32),
    )


def _best_per_code(codes: NDArray[np.uint64], best: NDArray[np.uint8]) -> SectionPairs:
    """Each code once, in code order, with the highest agreement count it was found with."""
    order = np.lexsort((-best.astype(np.int16), codes))
    codes, best = codes[order], best[order]
    first = np.r_[True, codes[1:] != codes[:-1]] if codes.size else np.empty(0, dtype=bool)
    return SectionPairs(*split_codes(codes[first]), best[first])


def section_pairs(
    section_of_row: NDArray[np.int64],
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
    counts: NDArray[np.uint8],
) -> SectionPairs:
    """Window pairs as section pairs: each once, with its best window pair's agreements.

    Sections are numbered in row order, so left < right rows give left <= right sections; a pair
    of windows of one section is not a section pair.
    """
    a, b = section_of_row[left], section_of_row[right]
    apart = a != b
    return _best_per_code(pair_codes(a[apart], b[apart]), counts[apart])


def merge_section_pairs(parts: list[SectionPairs]) -> SectionPairs:
    """Section pairs found in several parts, each once, with its best agreement count."""
    if not parts:
        empty = np.empty(0, dtype=np.uint32)
        return SectionPairs(empty, empty.copy(), np.empty(0, dtype=np.uint8))
    return _best_per_code(
        np.concatenate([pair_codes(p.left, p.right) for p in parts]),
        np.concatenate([p.agreements for p in parts]),
    )


def write_section_pairs(directory: Path, found: SectionPairs) -> None:
    """A pairs table's three vectors."""
    np.save(directory / "left.npy", np.asarray(found.left, dtype=np.uint32))
    np.save(directory / "right.npy", np.asarray(found.right, dtype=np.uint32))
    np.save(directory / "agreements.npy", np.asarray(found.agreements, dtype=np.uint8))


def _estimate_histogram(counts: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Pairs per 1/ESTIMATE_BINS of estimated similarity; the last bin includes 1."""
    bins = np.minimum(counts.astype(np.int64) * ESTIMATE_BINS // NUM_PERM, ESTIMATE_BINS - 1)
    return np.bincount(bins, minlength=ESTIMATE_BINS)


def _family_summary(
    labels: NDArray[np.int32], bills: NDArray[np.str_], state_of_bill: dict[str, str]
) -> dict[str, Any]:
    sizes = np.bincount(labels, minlength=labels.size)
    grouped = sizes >= 2
    order = np.argsort(-sizes, kind="stable")[: min(LARGEST_LISTED, int(grouped.sum()))]
    return {
        "families_of_two_or_more": int(grouped.sum()),
        "bills_in_them": int(sizes[grouped].sum()),
        "largest": [
            {"family": str(bills[f]), "state": state_of_bill[str(bills[f])], "bills": int(sizes[f])}
            for f in order
        ],
    }


def _histograms() -> dict[str, NDArray[np.int64]]:
    return {
        "cross_state": np.zeros(ESTIMATE_BINS, dtype=np.int64),
        "same_state": np.zeros(ESTIMATE_BINS, dtype=np.int64),
    }


@dataclass
class _Tally:
    """Counts over the distinct candidate pairs, added to band by band."""

    candidates: int = 0
    cross: int = 0
    kept_cross: int = 0
    kept_same: int = 0
    dropped: int = 0
    over: int = 0
    rows_capped: int = 0
    pairs_capped: int = 0
    largest: int = 0
    # Candidate window pairs before verification, by estimated similarity.
    histogram: dict[str, NDArray[np.int64]] = field(default_factory=_histograms)

    def add_band(self, found: BandPairs) -> None:
        """What a band's bucket cap skipped."""
        self.over += found.buckets_over_cap
        self.rows_capped += found.rows_in_capped_buckets
        self.pairs_capped += found.pairs_in_capped_buckets
        self.largest = max(self.largest, found.largest_bucket)

    def add_verified(
        self, counts: NDArray[np.uint8], cross: NDArray[np.bool_], kept: NDArray[np.bool_]
    ) -> None:
        """A band's new candidates: their agreement counts, which cross state lines, which
        were kept."""
        self.candidates += int(counts.size)
        self.cross += int(cross.sum())
        self.kept_cross += int((kept & cross).sum())
        self.kept_same += int((kept & ~cross).sum())
        self.dropped += int((~kept).sum())
        self.histogram["cross_state"] += _estimate_histogram(counts[cross])
        self.histogram["same_state"] += _estimate_histogram(counts[~cross])


@dataclass
class _Held:
    """The verified section pairs found so far, merged down to each pair once whenever they
    pass _HELD_SECTION_PAIRS."""

    cross: list[SectionPairs] = field(default_factory=list[SectionPairs])
    same: list[SectionPairs] = field(default_factory=list[SectionPairs])
    held: int = 0

    def add(self, cross: SectionPairs, same: SectionPairs) -> None:
        self.cross.append(cross)
        self.same.append(same)
        self.held += cross.left.size + same.left.size
        if self.held > _HELD_SECTION_PAIRS:
            self.cross = [merge_section_pairs(self.cross)]
            self.same = [merge_section_pairs(self.same)]
            self.held = self.cross[0].left.size + self.same[0].left.size


def _new_candidates(
    index: BandIndex, band: int, seen: NDArray[np.uint64], tally: _Tally, *, cap: int
) -> tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint64]]:
    """The window pairs band `band` forms that no earlier band did, and `seen` with them."""
    found = band_pairs(index, band, cap=cap)
    tally.add_band(found)
    new = found.codes[unseen(found.codes, seen)]
    del found
    left, right = split_codes(new)
    return left, right, merge_sorted(seen, new)


def _verify_bands(
    index: BandIndex,
    files: SignatureFiles,
    section_of_row: NDArray[np.int64],
    states: NDArray[np.int32],
    *,
    cap: int,
) -> tuple[SectionPairs, SectionPairs, _Tally]:
    """The verified cross-state and same-state section pairs, and the tally of candidates."""
    need = min_agreements(THRESHOLD)
    tally, held = _Tally(), _Held()
    seen = np.empty(0, dtype=np.uint64)

    def gather(rows: NDArray[np.int64]) -> NDArray[np.uint32]:
        return signature_values(files, rows)

    for band in range(BANDS):
        left, right, seen = _new_candidates(index, band, seen, tally, cap=cap)
        counts = agreements_by(gather, left, right)
        cross = states[left] != states[right]
        kept = counts >= need
        tally.add_verified(counts, cross, kept)
        cross_part, same_part = (
            section_pairs(section_of_row, left[mask], right[mask], counts[mask])
            for mask in (kept & cross, kept & ~cross)
        )
        held.add(cross_part, same_part)
    del seen
    return merge_section_pairs(held.cross), merge_section_pairs(held.same), tally


@dataclass(frozen=True, slots=True)
class _Families:
    shares: BillPairs
    labels: NDArray[np.int32]  # each bill's family at FAMILY_SHARE
    sweep: list[dict[str, Any]]


def _family_sweep(
    bills: NDArray[np.str_],
    bill_of_section: NDArray[np.int32],
    linking: SectionPairs,
    state_of_bill: dict[str, str],
) -> _Families:
    """The families at FAMILY_SHARE, and what each share of the sweep would form."""
    # Shares count sections, so a section matched through several windows counts once.
    shares = bill_pairs(bill_of_section, linking.left, linking.right)
    sweep: list[dict[str, Any]] = []
    labels = np.arange(bills.size, dtype=np.int32)
    for share in sorted({*FAMILY_SWEEP, FAMILY_SHARE}):
        links = linked(shares, share)
        found_labels = components(bills.size, shares.first[links], shares.second[links])
        if share == FAMILY_SHARE:
            labels = found_labels
        summary = _family_summary(found_labels, bills, state_of_bill)
        sweep.append(
            {
                "share": round(share, 3),
                "linked_bill_pairs": int(links.sum()),
                "families_of_two_or_more": summary["families_of_two_or_more"],
                "bills_in_them": summary["bills_in_them"],
                "largest_family": summary["largest"][0]["bills"] if summary["largest"] else 0,
            }
        )
    return _Families(shares, labels, sweep)


def _write_tables(
    tables_dir: Path,
    stored: SectionPairs,
    linking: SectionPairs,
    bills: NDArray[np.str_],
    labels: NDArray[np.int32],
) -> None:
    def write_families(directory: Path) -> None:
        with db.connect_quiet() as con:
            queries.write_families(
                con,
                parquet_path=directory / "families.parquet",
                bill_ids=bills.tolist(),
                families=bills[labels].tolist(),
            )

    replace_dir(tables_dir / "pairs", lambda d: write_section_pairs(d, stored))
    replace_dir(tables_dir / "same_state_pairs", lambda d: write_section_pairs(d, linking))
    replace_dir(tables_dir / "families", write_families)


def _found_report(tally: _Tally) -> dict[str, Any]:
    return {
        "largest_bucket": tally.largest,
        "capped": {
            "buckets": tally.over,
            "window_memberships": tally.rows_capped,
            "pairs": tally.pairs_capped,
        },
        "candidate_pairs": {
            "all": tally.candidates,
            "cross_state": tally.cross,
            "same_state": tally.candidates - tally.cross,
        },
    }


def _pairs_report(tally: _Tally, stored: SectionPairs, linking: SectionPairs) -> dict[str, Any]:
    return {
        "threshold": THRESHOLD,
        "min_agreements": min_agreements(THRESHOLD),
        # Pairs of windows.
        "verified": {
            "all": tally.kept_cross + tally.kept_same,
            "cross_state": tally.kept_cross,
            "same_state": tally.kept_same,
            "dropped": tally.dropped,
        },
        # The verified window pairs as pairs of distinct sections.
        "section_pairs": {
            "cross_state": int(stored.left.size),
            "same_state": int(linking.left.size),
        },
        # Candidate window pairs before verification, by estimated similarity in 1/16 steps.
        "estimate_histogram": {k: v.tolist() for k, v in tally.histogram.items()},
        "stored_bytes": int(stored.left.size + linking.left.size) * _PAIR_BYTES,
    }


def _families_report(
    families: _Families, bills: NDArray[np.str_], state_of_bill: dict[str, str]
) -> dict[str, Any]:
    shares = families.shares
    lowest = np.minimum(shares.first_share, shares.second_share)
    return {
        "share": FAMILY_SHARE,
        "same_state_bill_pairs": int(shares.first.size),
        "linked_bill_pairs": int(linked(shares).sum()),
        **_family_summary(families.labels, bills, state_of_bill),
        # Same-state bill pairs by the smaller of their two shares, in steps of 1/20.
        "min_share_histogram": np.histogram(lowest, bins=SHARE_BINS, range=(0.0, 1.0))[0].tolist(),
        "sweep": families.sweep,
    }


def build_pairs(
    tables_dir: Path,
    files: SignatureFiles,
    owners: RowLabels,
    states: NDArray[np.int32],
    *,
    cap: int,
    floor_bytes: int,
) -> dict[str, Any]:
    """Rebuild `<tables>/pairs/` and `<tables>/families/`; return their report sections, and
    the candidate counts for the `lsh` section under "found".

    Pairs are formed band by band. A pair an earlier band already formed is dropped, so each
    distinct candidate is verified once, against the memory-mapped signatures, as soon as its
    first band forms it. Only the codes of the candidates seen so far and the verified section
    pairs are held, never every candidate with its signatures (DESIGN.md#dr-3).
    """
    index = load_band_index(tables_dir)
    bills, inverse = np.unique(np.array(owners.section_bills), return_inverse=True)
    stored, linking, tally = _verify_bands(index, files, owners.section, states, cap=cap)
    require_free_space(
        tables_dir,
        needed=(stored.left.size + linking.left.size) * _PAIR_BYTES
        + bills.size * _FAMILY_ROW_BYTES
        + (1 << 20),
        floor=floor_bytes,
    )
    with db.connect_quiet() as con:
        state_of_bill = queries.bill_states(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    families = _family_sweep(bills, inverse.astype(np.int32), linking, state_of_bill)
    _write_tables(tables_dir, stored, linking, bills, families.labels)
    return {
        "found": _found_report(tally),
        "pairs": _pairs_report(tally, stored, linking),
        "families": _families_report(families, bills, state_of_bill),
    }


def _load_pairs(pairs: Path) -> SectionPairs:
    return SectionPairs(
        cast("NDArray[np.uint32]", np.load(pairs / "left.npy", mmap_mode="r")),
        cast("NDArray[np.uint32]", np.load(pairs / "right.npy", mmap_mode="r")),
        cast("NDArray[np.uint8]", np.load(pairs / "agreements.npy", mmap_mode="r")),
    )


def load_cross_state_pairs(tables_dir: Path) -> SectionPairs:
    """The verified cross-state pairs, memory-mapped."""
    return _load_pairs(tables_dir / "pairs")


def load_same_state_pairs(tables_dir: Path) -> SectionPairs:
    """The verified same-state pairs, memory-mapped; run `lineage index` if they are missing."""
    where = tables_dir / "same_state_pairs"
    if not (where / "left.npy").exists():
        raise FileNotFoundError(f"no same-state pairs in {where}; run `lineage index`")
    return _load_pairs(where)


def load_families(tables_dir: Path) -> dict[str, str]:
    """Every signed bill's family, by bill id."""
    with db.connect_quiet() as con:
        return queries.read_families(con, parquet_path=tables_dir / "families" / "families.parquet")


# Updates (DESIGN.md#updates, DR-7): the section pairs a day can change are recomputed exactly,
# from every pair of their windows, and the rest carried over under their new numbers.

# Window pairs compared per step when section pairs are recomputed.
_RECOMPUTE_CHUNK = 1 << 18


def section_bounds(section_of_row: NDArray[np.int64]) -> NDArray[np.int64]:
    """Each section's first signature row, and one past the last row: a section's windows are
    adjacent rows, so section s holds rows bounds[s] to bounds[s + 1]."""
    sections = int(section_of_row[-1]) + 1 if section_of_row.size else 0
    return np.searchsorted(section_of_row, np.arange(sections + 1)).astype(np.int64)


def recompute_section_pairs(
    files: SignatureFiles,
    index: BandIndex,
    bounds: NDArray[np.int64],
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
    *,
    cap: int = BUCKET_CAP,
    threshold: float = THRESHOLD,
) -> SectionPairs:
    """The verified pairs among the given section pairs, each with its best window pair's
    agreements, exactly as a build forms them: a window pair counts when some band puts both
    windows in one bucket within the cap and their signatures agree on enough positions."""
    need = min_agreements(threshold)
    lefts = np.asarray(left, dtype=np.int64)
    rights = np.asarray(right, dtype=np.int64)
    n_left = bounds[lefts + 1] - bounds[lefts]
    n_right = bounds[rights + 1] - bounds[rights]
    per_pair = n_left * n_right
    pair = np.repeat(np.arange(lefts.size), per_pair)
    within = np.arange(int(per_pair.sum()), dtype=np.int64) - np.repeat(
        np.cumsum(per_pair) - per_pair, per_pair
    )
    rows_left = bounds[lefts][pair] + within // n_right[pair]
    rows_right = bounds[rights][pair] + within % n_right[pair]
    involved = np.unique(np.concatenate([rows_left, rows_right]))
    values = signature_values(files, involved)
    keys = band_keys(values) if involved.size else np.empty((0, BANDS), dtype=np.uint64)
    sizes = np.empty(keys.shape, dtype=np.int64)
    for b in range(BANDS):
        sizes[:, b] = bucket_sizes(np.asarray(index.keys[b], dtype=np.uint64), keys[:, b])
    best = np.zeros(lefts.size, dtype=np.int16) - 1
    for start in range(0, pair.size, _RECOMPUTE_CHUNK):
        stop = start + _RECOMPUTE_CHUNK
        a = np.searchsorted(involved, rows_left[start:stop])
        b = np.searchsorted(involved, rows_right[start:stop])
        candidate = candidates_by_band(keys[a], keys[b], sizes[a], cap=cap)
        agree = np.count_nonzero(values[a] == values[b], axis=1).astype(np.int16)
        verified = candidate & (agree >= need)
        np.maximum.at(best, pair[start:stop][verified], agree[verified])
    kept = best >= 0
    return SectionPairs(
        lefts[kept].astype(np.uint32), rights[kept].astype(np.uint32), best[kept].astype(np.uint8)
    )


def renumbered(pairs: SectionPairs, section_map: NDArray[np.int64]) -> SectionPairs:
    """Stored pairs under new section numbers, those of a retired section (-1) dropped. Section
    numbers keep their order, so the pairs stay sorted."""
    a = section_map[np.asarray(pairs.left, dtype=np.int64)]
    b = section_map[np.asarray(pairs.right, dtype=np.int64)]
    keep = (a >= 0) & (b >= 0)
    return SectionPairs(
        a[keep].astype(np.uint32), b[keep].astype(np.uint32), np.asarray(pairs.agreements)[keep]
    )


def replaced_pairs(
    carried: SectionPairs, recomputed_codes: NDArray[np.uint64], found: SectionPairs
) -> SectionPairs:
    """`carried` without the pairs that were recomputed, with what the recomputation found,
    sorted by left, then right."""
    codes = pair_codes(carried.left, carried.right)
    keep = ~np.isin(codes, recomputed_codes)
    merged = np.concatenate([codes[keep], pair_codes(found.left, found.right)])
    agreements = np.concatenate([np.asarray(carried.agreements)[keep], found.agreements])
    order = np.argsort(merged, kind="stable")
    return SectionPairs(*split_codes(merged[order]), agreements[order].astype(np.uint8))


def sections_of_window_pairs(
    codes: NDArray[np.uint64], section_of_row: NDArray[np.int64]
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """The distinct section pairs, left < right, behind window pairs given as codes; a pair of
    windows of one section is not a section pair."""
    a = section_of_row[(codes >> np.uint64(32)).astype(np.int64)]
    b = section_of_row[(codes & np.uint64(0xFFFF_FFFF)).astype(np.int64)]
    apart = a != b
    return split_codes(np.unique(pair_codes(np.minimum(a, b)[apart], np.maximum(a, b)[apart])))
