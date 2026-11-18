"""The semantic stage's incremental neighbors (DESIGN.md#updates, DR-7): new texts searched
exactly, the stored lists they enter or that lose a retired text brought to what a build keeps,
the cap of K applied exactly, so a new text can push an existing neighbor out."""

import numpy as np
import pytest
from numpy.typing import NDArray

from lineage.match.semantic.neighbors import update_neighbors

K = 4
THRESHOLD = 0.8


def _clustered(
    rng: np.random.Generator, n: int, centers: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Unit vectors near one of `centers`, so lists fill past K and the cap binds."""
    picks = centers[rng.integers(0, centers.shape[0], n)]
    rows = picks + rng.normal(scale=0.12, size=picks.shape)
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def _from_scratch(vectors: NDArray[np.float16], live: NDArray[np.bool_]) -> list[set[int]]:
    """Each live text's K nearest live others at or above the threshold, ties to the lower id."""
    v = vectors.astype(np.float32)
    scores = v @ v.T
    out: list[set[int]] = []
    for x in range(v.shape[0]):
        if not live[x]:
            out.append(set())
            continue
        near: NDArray[np.bool_] = scores[x] >= THRESHOLD
        other = np.ones(v.shape[0], dtype=bool)
        other[x] = False
        others = np.flatnonzero(live & near & other)
        order = np.lexsort((others, -scores[x, others]))
        out.append(set(others[order][:K].tolist()))
    return out


def _lists(ids: NDArray[np.int32]) -> list[set[int]]:
    return [{int(i) for i in row if i >= 0} for row in ids]


@pytest.mark.parametrize("seed", range(5))
def test_an_update_of_the_neighbors_gives_what_a_build_keeps(seed: int) -> None:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(6, 16)).astype(np.float32)
    old = _clustered(rng, 60, centers).astype(np.float16)
    live_before = np.ones(60, dtype=bool)
    before = _from_scratch(old, live_before)
    ids = np.full((60, K), -1, dtype=np.int32)
    scores = np.zeros((60, K), dtype=np.float16)
    v = old.astype(np.float32)
    for x, found in enumerate(before):
        ordered = sorted(found, key=lambda y: (-float(v[x] @ v[y]), y))
        ids[x, : len(ordered)] = ordered
        scores[x, : len(ordered)] = [v[x] @ v[y] for y in ordered]
    assert (ids >= 0).all(axis=1).sum() > 10  # many lists are full, so the cap binds

    retired = np.sort(rng.choice(60, size=5, replace=False)).astype(np.int64)
    fresh = np.arange(60, 72, dtype=np.int64)
    after = np.concatenate([old, _clustered(rng, 12, centers).astype(np.float16)])
    after[retired] = 0
    live = np.ones(72, dtype=bool)
    live[retired] = False
    # Two shards, as the store keeps them, the second partly new.
    blocks = [after[:40], after[40:]]

    got_ids, got_scores, changed = update_neighbors(
        blocks, ids, scores, fresh=fresh, retired=retired, k=K, threshold=THRESHOLD
    )

    want = _from_scratch(after, live)
    assert _lists(got_ids) == want
    pushed_out = [
        x for x in range(60) if before[x] - want[x] and not before[x] & set(retired.tolist())
    ]
    assert pushed_out  # some stored list lost a neighbor to a new text
    w = after.astype(np.float32)
    for x, row in enumerate(got_ids):
        for slot, y in enumerate(row.tolist()):
            if y >= 0:
                assert abs(float(got_scores[x, slot]) - float(w[x] @ w[y])) < 2e-3
    assert set(changed.tolist()) >= {
        x for x in range(72) if _lists(got_ids)[x] != (before[x] if x < 60 else set())
    }
