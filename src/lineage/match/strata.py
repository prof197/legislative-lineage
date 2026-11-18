"""Both stages' stored cross-state pairs by stratum: found by the near-duplicate stage only, by
meaning only, or by both. The precision samples draw from these strata and their estimates weigh
by them (lineage.precision); the state boilerplate flag counts them with its sections dropped.
See DESIGN.md#precision-sample.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage.match.near.pairs import load_cross_state_pairs
from lineage.match.semantic.pairs import load_semantic_pairs
from lineage.match.semantic.stage import BUILD_THRESHOLD
from lineage.match.semantic.stage import MODEL as MODEL_NAME

# Each stratum's name and the stages that found its pairs, in report order.
STRATA: dict[str, tuple[str, ...]] = {
    "near_duplicate_only": ("near_duplicate",),
    "semantic_only": ("semantic",),
    "both": ("near_duplicate", "semantic"),
}
STAGES = ("near_duplicate", "semantic")

PairKeys = NDArray[np.uint64]  # left << 32 | right, sorted and unique


class PrecisionError(Exception):
    pass


def pair_keys(left: NDArray[np.uint32], right: NDArray[np.uint32]) -> PairKeys:
    """One sortable key per section pair, from section numbers in the signatures' order."""
    return np.unique((left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64))


def strata(near_duplicate: PairKeys, semantic: PairKeys) -> dict[str, PairKeys]:
    """Every pair either stage found, in exactly one stratum."""
    return {
        "near_duplicate_only": np.setdiff1d(near_duplicate, semantic, assume_unique=True),
        "semantic_only": np.setdiff1d(semantic, near_duplicate, assume_unique=True),
        "both": np.intersect1d(near_duplicate, semantic, assume_unique=True),
    }


def stratum_of(stages: Sequence[str]) -> str:
    found = tuple(s for s in STAGES if s in stages)
    for name, covers in STRATA.items():
        if covers == found and len(found) == len(set(stages)):
            return name
    raise PrecisionError(f"no stratum is found by the stages {list(stages)}")


def stored_strata(
    tables_dir: Path,
    *,
    model: str = MODEL_NAME,
    dropped: NDArray[np.bool_] | None = None,
    threshold: float = BUILD_THRESHOLD,
) -> dict[str, PairKeys]:
    """Both stages' stored cross-state pairs by stratum, by meaning at or above `threshold`,
    by default all the build stores, which the samples were drawn from; with `dropped`, a mask
    over section numbers, without the pairs either side of which it drops."""
    near = load_cross_state_pairs(tables_dir)
    meaning = load_semantic_pairs(tables_dir, model, threshold=threshold)

    def keys(left: NDArray[np.uint32], right: NDArray[np.uint32]) -> PairKeys:
        left, right = np.asarray(left), np.asarray(right)
        if dropped is not None:
            kept = ~(dropped[left] | dropped[right])
            left, right = left[kept], right[kept]
        return pair_keys(left, right)

    return strata(keys(near.left, near.right), keys(meaning.left, meaning.right))
