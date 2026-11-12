"""What a page lists: the matches at the stage's threshold, less a state's own boilerplate.

Both paths that list relatives read their matches through here: a corpus bill's page
(lineage.relatives) from the stored pairs, and a query of new text (lineage.query) from its own
hits, so the two list alike. A match by meaning counts at or above the stage's threshold
(lineage.semantic.THRESHOLD). A cross-state match one section of which is its state's
boilerplate (lineage.state_boilerplate) is dropped from both stages unless both stages found it:
of the dropped pairs the model judged, 30 of the 34 found by both stated the same provision,
against 8 and 10 of the 34 found by one stage (reports/results.json,
`precision.state_boilerplate_flag.dropped_pairs`). See DESIGN.md#boilerplate.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage.pairs import SectionPairs, load_cross_state_pairs
from lineage.semantic import MODEL, THRESHOLD
from lineage.semantic_pairs import SemanticPairs, load_semantic_pairs, section_keys
from lineage.state_boilerplate import MIN_FAMILIES, flagged, has_counts, load_families_reached

SectionKey = tuple[str, int]  # (bill id, ordinal)
NearHit = tuple[int, str, int, int]  # (query section, bill id, its section's ordinal, agreements)
MeaningHit = tuple[int, str, int, float]  # (query section, bill id, ordinal, cosine)


@dataclass(frozen=True, slots=True)
class ListedPairs:
    near: SectionPairs
    meaning: SemanticPairs
    # Pairs a flagged section kept because both stages found them, by stage.
    exempted: dict[str, int]


def shipped_flag(tables_dir: Path) -> NDArray[np.bool_] | None:
    """The flag as shipped, a mask over section numbers; None before its counts are built."""
    if not has_counts(tables_dir):
        return None
    return flagged(load_families_reached(tables_dir), MIN_FAMILIES)


def flagged_sections(tables_dir: Path) -> frozenset[SectionKey]:
    """The sections the shipped flag drops, by bill id and ordinal, for a query's hits."""
    flag = shipped_flag(tables_dir)
    if flag is None:
        return frozenset()
    bills, ordinals = section_keys(tables_dir)
    return frozenset((bills[s], int(ordinals[s])) for s in np.flatnonzero(flag).tolist())


def _keys(left: NDArray[np.uint32], right: NDArray[np.uint32]) -> NDArray[np.uint64]:
    return (np.asarray(left).astype(np.uint64) << np.uint64(32)) | np.asarray(right).astype(
        np.uint64
    )


def _kept(
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
    other: NDArray[np.uint64],
    flag: NDArray[np.bool_],
    exempt_both: bool,
) -> tuple[NDArray[np.bool_], int]:
    """Which pairs stand, and how many of them stand only because the other stage found them."""
    touched = flag[np.asarray(left)] | flag[np.asarray(right)]
    if not exempt_both:
        return ~touched, 0
    both = np.zeros(touched.shape, dtype=bool)
    at = np.flatnonzero(touched)
    both[at] = np.isin(_keys(left, right)[at], other)
    return ~touched | both, int(both.sum())


def listed_pairs(
    tables_dir: Path,
    *,
    model: str = MODEL,
    threshold: float = THRESHOLD,
    flag: NDArray[np.bool_] | None,
    exempt_both: bool = True,
) -> ListedPairs:
    """Both stages' stored cross-state pairs that a page lists: by meaning at or above
    `threshold`, and without the pairs of a section `flag` marks, but for those both stages
    found when `exempt_both`."""
    near = load_cross_state_pairs(tables_dir)
    meaning = load_semantic_pairs(tables_dir, model, threshold=threshold)
    if flag is None:
        return ListedPairs(near, meaning, {"near_duplicate": 0, "semantic": 0})
    near_keys = np.unique(_keys(near.left, near.right))
    meaning_keys = np.unique(_keys(meaning.left, meaning.right))
    keep_near, near_exempted = _kept(near.left, near.right, meaning_keys, flag, exempt_both)
    keep_meaning, meaning_exempted = _kept(
        meaning.left, meaning.right, near_keys, flag, exempt_both
    )
    return ListedPairs(
        SectionPairs(
            np.asarray(near.left)[keep_near],
            np.asarray(near.right)[keep_near],
            np.asarray(near.agreements)[keep_near],
        ),
        SemanticPairs(
            np.asarray(meaning.left)[keep_meaning],
            np.asarray(meaning.right)[keep_meaning],
            np.asarray(meaning.cosine)[keep_meaning],
            np.asarray(meaning.left_window)[keep_meaning],
            np.asarray(meaning.right_window)[keep_meaning],
        ),
        {"near_duplicate": near_exempted, "semantic": meaning_exempted},
    )


def listed_hits(
    near: Sequence[NearHit],
    by_meaning: Sequence[MeaningHit],
    *,
    threshold: float,
    flagged: Collection[SectionKey] = frozenset(),
    exempt_both: bool = True,
) -> tuple[list[tuple[int, str, int]], list[tuple[int, str, float]]]:
    """A query's hits a page lists, as (query section, bill id, score) for each stage: by
    meaning at or above `threshold`, and without a hit on a flagged section unless both stages
    hit that section from the same query section."""
    meaning = [h for h in by_meaning if h[3] >= threshold]
    near_at = {(q, b, o) for q, b, o, _ in near}
    meaning_at = {(q, b, o) for q, b, o, _ in meaning}

    def stands(q: int, b: str, o: int, other: set[tuple[int, str, int]]) -> bool:
        return (b, o) not in flagged or (exempt_both and (q, b, o) in other)

    return (
        [(q, b, a) for q, b, o, a in near if stands(q, b, o, meaning_at)],
        [(q, b, c) for q, b, o, c in meaning if stands(q, b, o, near_at)],
    )
