"""Verify the LSH candidates, store the cross-state pairs, and group same-state families.

Reads the signatures and band index that `lineage index` writes before it. Writes two tables:

- `<tables>/pairs/`: the verified cross-state section pairs as three aligned vectors
  (`left.npy` and `right.npy`, signature rows with left < right, and `agreements.npy`, the
  signature positions on which the two agree), sorted by left row, then right.
- `<tables>/families/families.parquet`: every signed bill's family, named by the first bill id
  in the family in sort order. A bill with no near-duplicate in its state is its own family.

Same-state pairs serve only to form families and are not stored. See
DESIGN.md#near-duplicate-stage.
"""

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.bands import load_band_index, row_states
from lineage.disk import require_free_space
from lineage.families import FAMILY_SHARE, bill_pairs, components, linked
from lineage.lsh import BUCKET_CAP, candidate_pairs
from lineage.minhash import NUM_PERM
from lineage.signatures import load_signatures
from lineage.verify import THRESHOLD, agreements, min_agreements

ESTIMATE_BINS = 16
SHARE_BINS = 20
FAMILY_SWEEP = (0.5, 2 / 3, 0.75, 0.8, 0.9, 1.0)
LARGEST_LISTED = 10
# Two uint32 rows and a uint8 count per pair; generous allowances for the family file.
_PAIR_BYTES = 4 + 4 + 1
_FAMILY_ROW_BYTES = 128


@dataclass(frozen=True, slots=True)
class SectionPairs:
    left: NDArray[np.uint32]
    right: NDArray[np.uint32]
    agreements: NDArray[np.uint8]


def _estimate_histogram(counts: NDArray[np.uint8]) -> list[int]:
    """Pairs per 1/ESTIMATE_BINS of estimated similarity; the last bin includes 1."""
    bins = np.minimum(counts.astype(np.int64) * ESTIMATE_BINS // NUM_PERM, ESTIMATE_BINS - 1)
    return np.bincount(bins, minlength=ESTIMATE_BINS).tolist()


def _replace_dir(final: Path, write: Callable[[Path], None]) -> None:
    """Write a table into a fresh directory beside `final`, then swap it in."""
    building = Path(tempfile.mkdtemp(prefix=f"{final.name}.", dir=final.parent))
    try:
        write(building)
        if final.exists():
            shutil.rmtree(final)
        building.rename(final)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise


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


def build_pairs(tables_dir: Path, *, floor_bytes: int) -> dict[str, Any]:
    """Rebuild `<tables>/pairs/` and `<tables>/families/`; return their report sections."""
    table = load_signatures(tables_dir)
    found = candidate_pairs(load_band_index(tables_dir), cap=BUCKET_CAP)
    counts = agreements(table.values, found.left, found.right)
    states = row_states(tables_dir, table)
    cross = states[found.left] != states[found.right]
    kept = counts >= min_agreements(THRESHOLD)
    bills, inverse = np.unique(np.array(table.bill_ids), return_inverse=True)
    bill_of_row = inverse.astype(np.int32)
    keep_cross = kept & cross
    require_free_space(
        tables_dir,
        needed=int(keep_cross.sum()) * _PAIR_BYTES + bills.size * _FAMILY_ROW_BYTES + (1 << 20),
        floor=floor_bytes,
    )

    same = kept & ~cross
    shares = bill_pairs(bill_of_row, found.left[same], found.right[same])
    sweep: list[dict[str, Any]] = []
    labels = np.arange(bills.size, dtype=np.int32)
    with queries.connect_quiet() as con:
        state_of_bill = queries.bill_states(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
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

    def write_pairs(directory: Path) -> None:
        np.save(directory / "left.npy", found.left[keep_cross])
        np.save(directory / "right.npy", found.right[keep_cross])
        np.save(directory / "agreements.npy", counts[keep_cross])

    def write_families(directory: Path) -> None:
        with queries.connect_quiet() as con:
            queries.write_families(
                con,
                parquet_path=directory / "families.parquet",
                bill_ids=bills.tolist(),
                families=bills[labels].tolist(),
            )

    _replace_dir(tables_dir / "pairs", write_pairs)
    _replace_dir(tables_dir / "families", write_families)

    lowest = np.minimum(shares.first_share, shares.second_share)
    return {
        "pairs": {
            "threshold": THRESHOLD,
            "min_agreements": min_agreements(THRESHOLD),
            "verified": {
                "all": int(kept.sum()),
                "cross_state": int(keep_cross.sum()),
                "same_state": int(same.sum()),
                "dropped": int((~kept).sum()),
            },
            # Candidates before verification, by estimated similarity in steps of 1/16.
            "estimate_histogram": {
                "cross_state": _estimate_histogram(counts[cross]),
                "same_state": _estimate_histogram(counts[~cross]),
            },
            "stored_bytes": int(keep_cross.sum()) * _PAIR_BYTES,
        },
        "families": {
            "share": FAMILY_SHARE,
            "same_state_bill_pairs": int(shares.first.size),
            "linked_bill_pairs": int(linked(shares).sum()),
            **_family_summary(labels, bills, state_of_bill),
            # Same-state bill pairs by the smaller of their two shares, in steps of 1/20.
            "min_share_histogram": np.histogram(lowest, bins=SHARE_BINS, range=(0.0, 1.0))[
                0
            ].tolist(),
            "sweep": sweep,
        },
    }


def load_cross_state_pairs(tables_dir: Path) -> SectionPairs:
    """The verified cross-state pairs, memory-mapped."""
    pairs = tables_dir / "pairs"
    return SectionPairs(
        cast("NDArray[np.uint32]", np.load(pairs / "left.npy", mmap_mode="r")),
        cast("NDArray[np.uint32]", np.load(pairs / "right.npy", mmap_mode="r")),
        cast("NDArray[np.uint8]", np.load(pairs / "agreements.npy", mmap_mode="r")),
    )


def load_families(tables_dir: Path) -> dict[str, str]:
    """Every signed bill's family, by bill id."""
    with queries.connect_quiet() as con:
        return queries.read_families(con, parquet_path=tables_dir / "families" / "families.parquet")
