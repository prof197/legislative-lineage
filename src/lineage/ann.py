"""Nearest neighbors over unit-length vectors: exact search, and faiss's approximate indexes.

faiss is a build-only optional dependency (`uv sync --extra build`): the build finds every
window's nearest neighbors with it, while a single query searches the memory-mapped float16
vectors exactly and needs no index. Its calls are kept here, typed, so the rest of the package
sees NumPy arrays only. Similarity is the inner product, which for unit vectors is the cosine.
See DESIGN.md#semantic-stage.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

# Corpus rows scored per matrix product in an exact search: 2,000 queries hold 262 MB of scores.
_EXACT_CHUNK = 1 << 15


@dataclass(frozen=True, slots=True)
class Neighbors:
    ids: NDArray[np.int64]  # (queries, k), best first; -1 where fewer than k were found
    scores: NDArray[np.float32]  # (queries, k), the cosine of each


def _faiss() -> Any:
    try:
        import faiss  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as e:  # pragma: no cover - the build extra is missing
        raise RuntimeError("faiss is not installed; run `uv sync --extra build`") from e
    return faiss


def _f32(x: NDArray[Any]) -> NDArray[np.float32]:
    return np.ascontiguousarray(x, dtype=np.float32)


def _merge(
    best: Neighbors, scores: NDArray[np.float32], ids: NDArray[np.int64], k: int
) -> Neighbors:
    all_scores = np.concatenate([best.scores, scores], axis=1)
    all_ids = np.concatenate([best.ids, ids], axis=1)
    # A stable sort on the negated scores keeps the lower id first among equal scores.
    order = np.argsort(-all_scores, axis=1, kind="stable")[:, :k]
    return Neighbors(
        np.take_along_axis(all_ids, order, axis=1), np.take_along_axis(all_scores, order, axis=1)
    )


def exact_search(corpus: Sequence[NDArray[Any]], queries: NDArray[Any], k: int) -> Neighbors:
    """The k most similar corpus rows to each query, by brute force. `corpus` is a list of
    blocks, memory-mapped float16 shards say, numbered on from one block to the next."""
    q = _f32(queries)
    best = Neighbors(
        np.full((q.shape[0], 0), -1, dtype=np.int64), np.empty((q.shape[0], 0), dtype=np.float32)
    )
    offset = 0
    for block in corpus:
        for start in range(0, block.shape[0], _EXACT_CHUNK):
            rows = _f32(block[start : start + _EXACT_CHUNK])
            scores = q @ rows.T
            take = min(k, rows.shape[0])
            top = np.argpartition(-scores, take - 1, axis=1)[:, :take]
            best = _merge(
                best,
                np.take_along_axis(scores, top, axis=1),
                top.astype(np.int64) + offset + start,
                k,
            )
        offset += block.shape[0]
    if best.ids.shape[1] < k:
        pad = k - best.ids.shape[1]
        best = Neighbors(
            np.pad(best.ids, ((0, 0), (0, pad)), constant_values=-1),
            np.pad(best.scores, ((0, 0), (0, pad)), constant_values=-np.inf),
        )
    return best


@dataclass(frozen=True, slots=True)
class Index:
    """A faiss index over unit vectors, and the factory string that made it."""

    spec: str
    handle: Any


def train_index(spec: str, dim: int, sample: NDArray[Any]) -> Index:
    """An empty index of the faiss factory `spec`, trained on `sample` if it needs training."""
    faiss = _faiss()
    handle = faiss.index_factory(dim, spec, faiss.METRIC_INNER_PRODUCT)
    if not handle.is_trained:
        handle.train(_f32(sample))
    return Index(spec, handle)


def add(index: Index, vectors: NDArray[Any]) -> None:
    """Add rows, numbered on from the rows already added."""
    index.handle.add(_f32(vectors))


def set_search(index: Index, *, nprobe: int | None = None, ef_search: int | None = None) -> None:
    """Search-time effort: IVF lists probed, or the HNSW candidate list's length."""
    faiss = _faiss()
    if nprobe is not None:
        faiss.extract_index_ivf(index.handle).nprobe = nprobe
    if ef_search is not None:
        faiss.ParameterSpace().set_index_parameter(index.handle, "efSearch", ef_search)


def search(index: Index, queries: NDArray[Any], k: int) -> Neighbors:
    found = cast(
        tuple[NDArray[np.float32], NDArray[np.int64]], index.handle.search(_f32(queries), k)
    )
    return Neighbors(found[1].astype(np.int64, copy=False), found[0])


def index_bytes(index: Index) -> int:
    faiss = _faiss()
    return int(cast(NDArray[np.uint8], faiss.serialize_index(index.handle)).size)


def set_threads(threads: int) -> None:
    _faiss().omp_set_num_threads(threads)


def rescore(
    candidates: NDArray[np.int64],
    queries: NDArray[Any],
    vector_of: Callable[[NDArray[np.int64]], NDArray[Any]],
    k: int,
) -> Neighbors:
    """Re-score each query's candidates against the stored vectors and keep the best k.

    `vector_of(ids)` returns the stored rows for an array of ids. An approximate index's
    scores come from compressed codes; the stored float16 vectors give each kept neighbor its
    true cosine, to within float16 rounding, and order the candidates by it."""
    q = _f32(queries)
    valid = candidates >= 0
    rows = _f32(vector_of(np.where(valid, candidates, 0).ravel())).reshape(
        candidates.shape[0], candidates.shape[1], -1
    )
    scores = np.einsum("qd,qcd->qc", q, rows)
    scores = np.where(valid, scores, -np.inf).astype(np.float32)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return Neighbors(
        np.where(
            np.take_along_axis(valid, order, axis=1),
            np.take_along_axis(candidates, order, axis=1),
            -1,
        ),
        np.take_along_axis(scores, order, axis=1),
    )


def recall_at(found: NDArray[np.int64], truth: NDArray[np.int64]) -> float:
    """The share of the true neighbors (ids >= 0) that were found."""
    hits = total = 0
    for f, t in zip(found.tolist(), truth.tolist(), strict=True):
        want = {i for i in t if i >= 0}
        hits += len(want & set(f))
        total += len(want)
    return hits / total if total else 1.0
