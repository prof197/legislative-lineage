"""Verify LSH candidates against their full signatures.

A candidate shares at least one band, which a pair of quite different sections can do by
chance. Its similarity is re-estimated from all NUM_PERM positions, and pairs below the
threshold are dropped. The estimate is kept as a count of agreeing positions, so it stays exact
and takes one byte. See DESIGN.md#near-duplicate-stage.
"""

import numpy as np
from numpy.typing import NDArray

from lineage.minhash import NUM_PERM

# Proposed. At 0.5 the S-curve proposes 87% of pairs, and above it a pair's chance of being a
# candidate at all is high, so what is kept does not hinge on luck. The estimates' distribution
# has no valley to set it by (reports/index.json, pairs). The ablation sweeps it.
THRESHOLD = 0.5

# Pairs gathered per step: two (chunk, NUM_PERM) uint32 gathers of about 130 MB each.
_CHUNK = 1 << 18


def agreements(
    values: NDArray[np.uint32], left: NDArray[np.uint32], right: NDArray[np.uint32]
) -> NDArray[np.uint8]:
    """For each pair of rows, the number of signature positions at which the two agree."""
    if values.shape[1] != NUM_PERM:
        raise ValueError(f"signatures must have {NUM_PERM} columns, not {values.shape[1]}")
    out = np.empty(left.size, dtype=np.uint8)
    for start in range(0, left.size, _CHUNK):
        stop = start + _CHUNK
        same = values[left[start:stop]] == values[right[start:stop]]
        out[start:stop] = np.count_nonzero(same, axis=1)
    return out


def min_agreements(threshold: float) -> int:
    """The fewest agreeing positions whose estimate reaches `threshold`."""
    return int(np.ceil(threshold * NUM_PERM - 1e-9))
