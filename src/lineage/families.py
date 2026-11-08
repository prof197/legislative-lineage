"""Group a state's near-duplicate bills into families, so each copy counts once.

A state files the same text many times: companion bills in each chamber, and a bill refiled
in a later session. Lineage counts the other states that carry a provision, so those copies
are grouped before cross-state scoring. Two bills of one state are linked when, in each
direction, at least `FAMILY_SHARE` of the bill's signed sections have a verified match among
the other's sections. Requiring a share keeps one shared clause, such as a state's own wording
of an appropriation, from chaining hundreds of bills together. Requiring it in both directions
keeps an omnibus bill from absorbing every small bill it contains. Families are the connected
components of the links. See DESIGN.md#near-duplicate-stage.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Proposed. Measured shares spike at the fractions of small bills (1/3, 1/2, 2/3, 3/4: a bill
# of three sections sharing one clause scores 1/3) and again at 1. Between 0.70 and 0.95 they
# are sparse, and 0.8 sits there, past the 3/4 spike (reports/index.json, families).
FAMILY_SHARE = 0.8

_LOW32 = np.uint64(0xFFFF_FFFF)
_U32 = np.uint64(32)


@dataclass(frozen=True, slots=True)
class BillPairs:
    """Pairs of bills that share at least one verified section pair; `first` < `second`."""

    first: NDArray[np.int32]
    second: NDArray[np.int32]
    first_share: NDArray[np.float64]  # share of first's sections matched in second
    second_share: NDArray[np.float64]


def bill_pairs(
    bill_of_section: NDArray[np.int32], left: NDArray[np.uint32], right: NDArray[np.uint32]
) -> BillPairs:
    """Match shares for every pair of bills joined by a section pair (left[i], right[i]).

    Sections are numbered as the signatures table numbers them (one number for all of a
    section's windows), and `bill_of_section` numbers their bills from 0. Pairs within one bill
    are ignored. A section counts once toward a share, however many sections of the other bill
    it matches.
    """
    sections = np.bincount(bill_of_section)
    a, b = bill_of_section[left], bill_of_section[right]
    apart = a != b
    owner = np.concatenate([a[apart], b[apart]]).astype(np.uint64)
    other = np.concatenate([b[apart], a[apart]]).astype(np.uint64)
    section = np.concatenate([left[apart], right[apart]])
    key = (owner << _U32) | other
    order = np.lexsort((section, key))
    key, section = key[order], section[order]
    first_of_run = np.ones(key.size, dtype=bool)
    first_of_run[1:] = (key[1:] != key[:-1]) | (section[1:] != section[:-1])
    keys, matched = np.unique(key[first_of_run], return_counts=True)
    owners = (keys >> _U32).astype(np.int32)
    others = (keys & _LOW32).astype(np.int32)
    shares = matched / sections[owners]
    # Every pair was entered in both orientations, so each key's reverse is present.
    forward = owners < others
    reverse = np.searchsorted(
        keys, (others[forward].astype(np.uint64) << _U32) | owners[forward].astype(np.uint64)
    )
    return BillPairs(owners[forward], others[forward], shares[forward], shares[reverse])


def components(n: int, first: NDArray[np.int32], second: NDArray[np.int32]) -> NDArray[np.int32]:
    """Label each of `n` nodes with the smallest node in its connected component."""
    labels = np.arange(n, dtype=np.int32)
    while True:
        lowest = np.minimum(labels[first], labels[second])
        merged = labels.copy()
        np.minimum.at(merged, first, lowest)
        np.minimum.at(merged, second, lowest)
        # Pointer jumping: follow labels to their own labels until each points at a root.
        while not np.array_equal(jumped := merged[merged], merged):
            merged = jumped
        if np.array_equal(merged, labels):
            return labels
        labels = merged


def linked(pairs: BillPairs, share: float = FAMILY_SHARE) -> NDArray[np.bool_]:
    """Which bill pairs reach `share` in both directions."""
    return np.minimum(pairs.first_share, pairs.second_share) >= share


def families(
    n_bills: int,
    bill_of_section: NDArray[np.int32],
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
    share: float = FAMILY_SHARE,
) -> NDArray[np.int32]:
    """Each bill's family, labelled by its smallest bill number, from same-state section pairs."""
    pairs = bill_pairs(bill_of_section, left, right)
    keep = linked(pairs, share)
    return components(n_bills, pairs.first[keep], pairs.second[keep])
