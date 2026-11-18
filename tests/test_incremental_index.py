"""The near-duplicate stage's incremental steps (DESIGN.md#updates, DR-7): a day's rows merged
into the band index, the section pairs it can change recomputed, and families regrouped, each
equal to a build from scratch over the same signatures, with a bucket cap small enough that
buckets cross it both ways."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from lineage.match.near.bands import load_band_index, update_bands
from lineage.match.near.families import families, update_families
from lineage.match.near.lsh import BANDS, ROWS, build_band_index, candidate_pairs
from lineage.match.near.minhash import NUM_PERM
from lineage.match.near.pairs import (
    SectionPairs,
    pair_codes,
    recompute_section_pairs,
    renumbered,
    replaced_pairs,
    section_bounds,
    section_pairs,
    sections_of_window_pairs,
)
from lineage.match.near.signatures import SignatureFiles, section_numbers
from lineage.match.near.verify import THRESHOLD, agreements, min_agreements

CAP = 6


def _bill_windows(rng: np.random.Generator, bases: NDArray[np.uint32]) -> list[NDArray[np.uint32]]:
    """A bill of one to three sections, each of one to three windows, each a copy of one of
    `bases` with a random quarter of its bands redrawn, so pairs share some bands and agree near
    the verification threshold."""
    sections: list[NDArray[np.uint32]] = []
    for _ in range(int(rng.integers(1, 4))):
        rows: list[NDArray[np.uint32]] = []
        for _ in range(int(rng.integers(1, 4))):
            row = bases[int(rng.integers(0, bases.shape[0]))].copy()
            for band in np.flatnonzero(rng.random(BANDS) < 0.25).tolist():
                row[band * ROWS : (band + 1) * ROWS] = rng.integers(0, 1 << 32, ROWS)
            rows.append(row)
        sections.append(np.stack(rows))
    return sections


Corpus = dict[str, tuple[str, list[NDArray[np.uint32]]]]  # bill -> (state, its sections' rows)


def _layout(
    corpus: Corpus, order: list[str]
) -> tuple[NDArray[np.uint32], NDArray[np.int64], list[str], NDArray[np.int32], list[str]]:
    """Signatures in `order`: values, each row's section, each section's bill, each row's state
    code, and each row's (bill, section, window) key."""
    values: list[NDArray[np.uint32]] = []
    bill_of_row: list[str] = []
    ordinal_of_row: list[int] = []
    keys: list[str] = []
    for bill in order:
        for ordinal, rows in enumerate(corpus[bill][1]):
            for window in range(rows.shape[0]):
                values.append(rows[window])
                bill_of_row.append(bill)
                ordinal_of_row.append(ordinal)
                keys.append(f"{bill}/{ordinal}/{window}")
    section = section_numbers(bill_of_row, np.array(ordinal_of_row, dtype=np.int64))
    section_bills = [bill_of_row[i] for i in np.flatnonzero(np.r_[True, np.diff(section) != 0])]
    states = sorted({s for s, _ in corpus.values()})
    state_of_row = np.array([states.index(corpus[b][0]) for b in bill_of_row], dtype=np.int32)
    return np.stack(values), section, section_bills, state_of_row, keys


def _reference(
    values: NDArray[np.uint32], section: NDArray[np.int64], state: NDArray[np.int32]
) -> tuple[SectionPairs, SectionPairs]:
    """Cross-state and same-state section pairs, as a build forms them."""
    found = candidate_pairs(build_band_index(values), cap=CAP)
    counts = agreements(values, found.left, found.right)
    kept = counts >= min_agreements(THRESHOLD)
    cross = state[found.left] != state[found.right]
    cross_pairs, same_pairs = (
        section_pairs(section, found.left[m], found.right[m], counts[m])
        for m in (kept & cross, kept & ~cross)
    )
    return cross_pairs, same_pairs


def _as_set(pairs: SectionPairs, bills: list[str]) -> set[tuple[str, str, int]]:
    return {
        (bills[a], bills[b], c)
        for a, b, c in zip(
            np.asarray(pairs.left).tolist(),
            np.asarray(pairs.right).tolist(),
            np.asarray(pairs.agreements).tolist(),
            strict=True,
        )
    }


@pytest.mark.parametrize("seed", range(6))
def test_an_update_of_the_index_gives_what_a_build_gives(seed: int, tmp_path: Path) -> None:
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 1 << 32, size=(30, NUM_PERM), dtype=np.uint32)
    corpus: Corpus = {
        f"b{i:03d}": (str(rng.choice(["AA", "BB", "CC"])), _bill_windows(rng, bases))
        for i in range(40)
    }
    before_order = sorted(corpus)
    retired = sorted(rng.choice(before_order, size=6, replace=False).tolist())
    added = [f"b{i:03d}x" for i in rng.choice(40, size=8, replace=False).tolist()]
    # A replaced bill keeps its id and takes new text, so it is retired and added at once.
    replaced = retired[:3]
    after_corpus: Corpus = {b: v for b, v in corpus.items() if b not in retired}
    for bill in [*added, *replaced]:
        after_corpus[bill] = (corpus.get(bill, ("AA", []))[0], _bill_windows(rng, bases))
    after_order = sorted(after_corpus)

    old_values, old_section, old_bills, old_state, old_keys = _layout(corpus, before_order)
    new_values, new_section, new_bills, new_state, new_keys = _layout(after_corpus, after_order)
    fresh_bills = set(added) | set(replaced)
    position = {k: i for i, k in enumerate(new_keys)}
    row_map = np.array(
        [-1 if k.split("/")[0] in retired else position[k] for k in old_keys], dtype=np.int64
    )
    fresh_rows = np.array(
        [i for i, k in enumerate(new_keys) if k.split("/")[0] in fresh_bills], dtype=np.int64
    )

    (tmp_path / "bands").mkdir()
    codes = update_bands(
        build_band_index(old_values),
        row_map,
        new_values[fresh_rows],
        fresh_rows,
        tmp_path / "bands",
        cap=CAP,
    )

    after_index = load_band_index(tmp_path)
    rebuilt = build_band_index(new_values)
    assert np.array_equal(after_index.keys, rebuilt.keys)
    assert np.array_equal(after_index.rows, rebuilt.rows)

    old_cross, old_same = _reference(old_values, old_section, old_state)
    first_rows = np.flatnonzero(np.r_[True, np.diff(old_section) != 0])
    section_map = np.where(row_map[first_rows] >= 0, new_section[row_map[first_rows]], -1)
    left, right = sections_of_window_pairs(codes, new_section)
    files = SignatureFiles([new_values], [], np.zeros(1, dtype=np.int64))
    found = recompute_section_pairs(
        files, after_index, section_bounds(new_section), left, right, cap=CAP
    )
    section_state = new_state[section_bounds(new_section)[:-1]]
    cross = section_state[found.left] != section_state[found.right]
    recomputed = pair_codes(left, right)
    want_cross, want_same = _reference(new_values, new_section, new_state)
    for carried, mask, want in (
        (renumbered(old_cross, section_map), cross, want_cross),
        (renumbered(old_same, section_map), ~cross, want_same),
    ):
        part = SectionPairs(found.left[mask], found.right[mask], found.agreements[mask])
        got = replaced_pairs(carried, recomputed, part)
        assert np.array_equal(got.left, want.left)
        assert np.array_equal(got.right, want.right)
        assert np.array_equal(got.agreements, want.agreements)
    # The case is not empty: both kinds of pair exist, and the update changed some of each.
    for old_pairs, want in ((old_cross, want_cross), (old_same, want_same)):
        assert _as_set(want, new_bills)
        assert _as_set(want, new_bills) != _as_set(old_pairs, old_bills)

    # Families: regroup only where the update can change them.
    bills = np.unique(np.array(new_bills))
    bill_of_section = np.searchsorted(bills, np.array(new_bills)).astype(np.int32)
    old_unique = np.unique(np.array(old_bills))
    old_labels = families(
        old_unique.size,
        np.searchsorted(old_unique, np.array(old_bills)).astype(np.int32),
        old_same.left,
        old_same.right,
    )
    old_families = {str(b): str(old_unique[f]) for b, f in zip(old_unique, old_labels, strict=True)}
    carried_same = renumbered(old_same, section_map)
    changed = np.setxor1d(
        pair_codes(carried_same.left, carried_same.right),
        pair_codes(want_same.left, want_same.right),
    )
    lost = np.isin(old_same.left, np.flatnonzero(section_map < 0)) | np.isin(
        old_same.right, np.flatnonzero(section_map < 0)
    )
    affected = (
        fresh_bills
        | {new_bills[int(s)] for c in changed.tolist() for s in (c >> 32, c & 0xFFFF_FFFF)}
        | {old_bills[int(s)] for s in np.concatenate([old_same.left[lost], old_same.right[lost]])}
    )
    got_families = update_families(
        old_families, bills, bill_of_section, want_same.left, want_same.right, affected
    )
    labels = families(bills.size, bill_of_section, want_same.left, want_same.right)
    assert got_families == {str(b): str(bills[f]) for b, f in zip(bills, labels, strict=True)}
