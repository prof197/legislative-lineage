"""`lineage neighbors`: every distinct window's nearest neighbors by meaning, for the build.

Each embedded text (`lineage embed`) is searched against all the others with an approximate
index: HNSW over the float16 vectors, which on a sample of the corpus kept 99.9% of the exact
neighbors at or above the threshold, where IVF-PQ re-scored kept 97.2%, and searched faster
(reports/ann_benchmark.json). HNSW's scores are cosines over the float16 vectors already; an
index of compressed codes, such as IVF-PQ, has its `candidates` best re-scored against the
vectors instead. The text itself is dropped, and the K best at or above the build's threshold
(lineage.semantic.BUILD_THRESHOLD) are kept. The result is two arrays beside the vectors, one
row per text id: `neighbors/ids.npy` (int32, -1 where fewer than K pass) and
`neighbors/scores.npy` (float16 cosines). The index is never saved: it exists only for the
build, and a single query needs neither it nor these arrays, since it searches the vectors
exactly (lineage.semantic). `check_at` measures the stored neighbors' recall at the stage's
stricter threshold on the same seeded texts.

The build holds to DR-3's memory bound. It refuses to start when its projected peak, the index
at its measured bytes per vector plus its buffers, is over the bound, and stops at the first
batch after which the process holds more, or has peaked higher inside a call. It searches
CHECKED seeded texts exactly at the end and reports the share of their true neighbors it
found, and the command stops below RECALL_FLOOR. See DESIGN.md#semantic-stage.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage.ann import (
    Neighbors,
    add,
    exact_search,
    index_bytes,
    recall_at,
    rescore,
    search,
    set_search,
    set_threads,
    train_index,
)
from lineage.core.disk import require_free_space
from lineage.core.memory import BOUND_BYTES, MemoryBoundError, peak_bytes, require_within
from lineage.semantic import BUILD_THRESHOLD as THRESHOLD
from lineage.semantic import MODEL, K
from lineage.vectors import EmbedError, load_plan, load_vectors, rows_of, vectors_dir

SEED = 20261111
SPEC = "HNSW32,SQfp16"
# 128, the widest benchmarked, fell under RECALL_FLOOR over the whole corpus, where each text
# has about twice the sample's true neighbors above the threshold.
EF_SEARCH = 256
# For an index that needs training (IVF): faiss's k-means wants at least 39 points per list,
# and this allows 40 per list for up to 16,384 lists. HNSW over float16 needs none.
TRAIN_ROWS = 40 * 16384
# The index's bytes per vector, measured on the benchmark's sample (reports/ann_benchmark.json:
# HNSW32,SQfp16 index_bytes over 1,000,000 vectors): the float16 vector and its links.
HNSW_BYTES_PER_VECTOR = 1041
# The build stops if this exact check finds less of the true neighbors above the threshold.
RECALL_FLOOR = 0.99
# Texts searched exactly once the build is done, to measure its recall over the whole corpus,
# where neighbors are denser than in the benchmark's sample.
CHECKED = 2000
THREADS = 10
_ADD_CHUNK = 1 << 17
_QUERY_CHUNK = 1 << 12  # queries per pass: their candidates' vectors are gathered at once
REPORT = Path("reports/neighbors.json")
COMMAND = "uv run lineage neighbors"


def projected_bytes(n: int, dim: int) -> int:
    """The build's peak, projected: the index, the training sample, a batch of queries with
    their float32 copies and results, and the exact check's largest block product."""
    train = min(TRAIN_ROWS, n) * dim * (2 + 4)  # float16 rows, and the float32 copy faiss takes
    batch = _QUERY_CHUNK * dim * (2 + 4) + _QUERY_CHUNK * (K + 1) * (8 + 4) * 3
    # A block's scores, negated, and partitioned, at exact search's 32,768 rows a block.
    check = min(CHECKED, n) * min(n, 1 << 15) * 4 * 3
    return n * HNSW_BYTES_PER_VECTOR + train + batch + check


def keep_neighbors(found: Neighbors, own: NDArray[np.int64], k: int, threshold: float) -> Neighbors:
    """Drop each query's own text, and neighbors below the threshold; keep the best k."""
    ok = (found.ids >= 0) & (found.ids != own[:, None]) & (found.scores >= threshold)
    scores = np.where(ok, found.scores, -np.inf).astype(np.float32)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    kept = np.take_along_axis(ok, order, axis=1)
    return Neighbors(
        np.where(kept, np.take_along_axis(found.ids, order, axis=1), -1),
        np.where(kept, np.take_along_axis(scores, order, axis=1), 0).astype(np.float32),
    )


def seeded_draws(n: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The build's training rows and the texts it checks exactly, drawn in that order from SEED,
    so a later check reads the same texts."""
    rng = np.random.default_rng(SEED)
    train = np.sort(rng.choice(n, size=min(TRAIN_ROWS, n), replace=False)).astype(np.int64)
    checked = np.sort(rng.choice(n, size=min(CHECKED, n), replace=False)).astype(np.int64)
    return train, checked


def check_at(tables_dir: Path, threshold: float, *, model: str = MODEL) -> dict[str, Any]:
    """The stored neighbors' recall at a threshold at or above the build's: of each checked
    text's K nearest at or above it by exact search, the share the stored neighbors at or above
    it hold. Exact, since the build keeps each text's K nearest before its threshold applies."""
    if threshold < THRESHOLD:
        raise EmbedError(f"the neighbors were kept down to {THRESHOLD}, not {threshold}")
    plan = load_plan(tables_dir)
    if plan is None:
        raise EmbedError(f"no semantic plan in {tables_dir}; run `lineage embed`")
    shards = load_vectors(tables_dir, model, plan)
    n = sum(s.shape[0] for s in shards)
    _, checked = seeded_draws(n)
    where = vectors_dir(tables_dir, model) / "neighbors"
    ids = np.asarray(np.load(where / "ids.npy", mmap_mode="r")[checked]).astype(np.int64)
    scores = np.asarray(np.load(where / "scores.npy", mmap_mode="r")[checked])
    stored = np.where(scores >= threshold, ids, -1)
    truth = keep_neighbors(
        exact_search(shards, rows_of(shards, checked), K + 1), checked, K, threshold
    )
    return {
        "threshold": threshold,
        "texts": int(checked.shape[0]),
        "true_neighbors": int((truth.ids >= 0).sum()),
        "recall": round(recall_at(stored, truth.ids), 4),
    }


def build_neighbors(
    tables_dir: Path,
    *,
    model: str = MODEL,
    spec: str = SPEC,
    ef_search: int | None = EF_SEARCH,  # for HNSW
    nprobe: int | None = None,  # for IVF
    candidates: int | None = None,  # re-scored per text, for an index of compressed codes
    floor_bytes: int,
    bound_bytes: int = BOUND_BYTES,
    progress: Any = print,
) -> dict[str, Any]:
    """Build the neighbors and report on them. The process's memory is checked against
    `bound_bytes` before the build, from a projection, and after every batch while it runs."""
    started = time.time()
    plan = load_plan(tables_dir)
    if plan is None:
        raise EmbedError(f"no semantic plan in {tables_dir}; run `lineage embed`")
    shards = load_vectors(tables_dir, model, plan)
    n = sum(s.shape[0] for s in shards)
    out = vectors_dir(tables_dir, model) / "neighbors"
    out.mkdir(parents=True, exist_ok=True)
    require_free_space(out, needed=n * K * (4 + 2), floor=floor_bytes)
    dim = shards[0].shape[1]
    projected = projected_bytes(n, dim)
    if projected > bound_bytes:
        raise MemoryBoundError(
            f"{spec} over {n:,} vectors is projected at {projected / 1024**3:.2f} GiB, over "
            f"the {bound_bytes / 1024**3:.0f} GiB bound; not started"
        )
    set_threads(THREADS)

    train_rows, checked = seeded_draws(n)
    train = rows_of(shards, train_rows)
    index = train_index(spec, train.shape[1], train)  # a no-op for an index needing no training
    del train
    require_within(bound_bytes, f"training {spec}")
    for block in shards:
        for i in range(0, block.shape[0], _ADD_CHUNK):
            add(index, block[i : i + _ADD_CHUNK])
            require_within(bound_bytes, f"indexing {spec}")
    set_search(index, nprobe=nprobe, ef_search=ef_search)
    built = time.time() - started
    progress(f"{spec}: {n:,} vectors indexed in {built:.0f}s")

    ids = np.lib.format.open_memmap(out / "ids.npy", mode="w+", dtype=np.int32, shape=(n, K))
    scores = np.lib.format.open_memmap(
        out / "scores.npy", mode="w+", dtype=np.float16, shape=(n, K)
    )
    at = 0
    for block in shards:
        for i in range(0, block.shape[0], _QUERY_CHUNK):
            queries = block[i : i + _QUERY_CHUNK]
            own = np.arange(at, at + queries.shape[0], dtype=np.int64)
            if candidates is None:  # one more than K, for the text itself
                found = search(index, queries, K + 1)
            else:
                approx = search(index, queries, candidates)
                found = rescore(approx.ids, queries, lambda r: rows_of(shards, r), candidates)
            kept = keep_neighbors(found, own, K, THRESHOLD)
            ids[at : at + queries.shape[0]] = kept.ids
            scores[at : at + queries.shape[0]] = kept.scores
            at += queries.shape[0]
            require_within(bound_bytes, f"searching {spec}")
        progress(f"{at:,} of {n:,} searched")
    ids.flush()
    scores.flush()
    found = int((np.asarray(ids) >= 0).sum())
    progress(f"checking {min(CHECKED, n):,} texts against exact search")
    truth = keep_neighbors(
        exact_search(shards, rows_of(shards, checked), K + 1), checked, K, THRESHOLD
    )
    return {
        "command": COMMAND,
        "model": model,
        "index": spec,
        "ef_search": ef_search,
        "nprobe": nprobe,
        "candidates_rescored": candidates,
        "k": K,
        "threshold": THRESHOLD,
        "texts": n,
        "neighbors_kept": found,
        "texts_with_a_neighbor": int((np.asarray(ids)[:, 0] >= 0).sum()),
        "exact_check": {
            "texts": int(checked.shape[0]),
            "true_neighbors": int((truth.ids >= 0).sum()),
            "recall": round(recall_at(np.asarray(ids)[checked].astype(np.int64), truth.ids), 4),
        },
        "index_bytes": index_bytes(index),
        "build_seconds": round(built),
        "seconds": round(time.time() - started),
        "peak_memory_bytes": peak_bytes(),
        "memory_bound_bytes": bound_bytes,
        "projected_memory_bytes": projected,
        "within_memory_bound": peak_bytes() <= bound_bytes,
    }


def write_report(doc: dict[str, Any], path: Path = REPORT) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")


# Updates (DESIGN.md#updates, DR-7): new texts searched exactly against every stored vector, and
# the stored lists they enter, or that lose a retired text, brought to what a build keeps. A
# list is each text's K nearest others at or above the threshold, best first, ties to the lower
# id; the stored float16 score of a neighbor already kept is kept as it was.

# Corpus rows scored per matrix product: 4,096 queries hold 512 MB of scores.
_SCAN_ROWS = 1 << 15


@dataclass(frozen=True, slots=True)
class Hits:
    """Every (query, text) pair at or above a threshold, the query's own text left out."""

    query: NDArray[np.int64]  # position among the queries
    text: NDArray[np.int64]
    score: NDArray[np.float32]


def scan(
    blocks: Sequence[NDArray[Any]], queries: NDArray[Any], own: NDArray[np.int64], threshold: float
) -> Hits:
    """Exact search: every stored text whose cosine with a query reaches `threshold`. `blocks`
    are the vectors in text id order, memory-mapped shards say; `own` is each query's own id."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    found: list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]] = []
    offset = 0
    for block in blocks:
        for start in range(0, block.shape[0], _SCAN_ROWS):
            rows = np.ascontiguousarray(block[start : start + _SCAN_ROWS], dtype=np.float32)
            scores = q @ rows.T
            qi, xi = np.nonzero(scores >= threshold)
            ids = xi.astype(np.int64) + offset + start
            mine = ids != own[qi]
            found.append((qi[mine].astype(np.int64), ids[mine], scores[qi[mine], xi[mine]]))
        offset += block.shape[0]
    if not found:
        empty = np.empty(0, dtype=np.int64)
        return Hits(empty, empty.copy(), np.empty(0, dtype=np.float32))
    return Hits(*(np.concatenate(parts) for parts in zip(*found, strict=True)))


def best_of(
    owner: NDArray[np.int64], ids: NDArray[np.int64], scores: NDArray[np.float32], k: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.intp]]:
    """Each owner's k best candidates, best first, ties to the lower id: (owner, id, the
    position of each kept candidate in the arrays given)."""
    order = np.lexsort((ids, -scores, owner))
    owner_sorted = owner[order]
    first = np.r_[True, owner_sorted[1:] != owner_sorted[:-1]] if owner.size else owner > 0
    starts = np.flatnonzero(first)
    rank = np.arange(owner.size) - np.repeat(starts, np.diff(np.r_[starts, owner.size]))
    kept = order[rank < k]
    return owner[kept], ids[kept], kept


def update_neighbors(
    blocks: Sequence[NDArray[Any]],
    ids: NDArray[np.int32],
    scores: NDArray[np.float16],
    *,
    fresh: NDArray[np.int64],
    retired: NDArray[np.int64],
    k: int = K,
    threshold: float = THRESHOLD,
) -> tuple[NDArray[np.int32], NDArray[np.float16], NDArray[np.int64]]:
    """The neighbor lists after an update, and the texts whose lists changed.

    `blocks` hold every vector after the update, `fresh` texts added and `retired` texts
    zeroed (no window carries them any more); `ids` and `scores` are the lists before, one row
    per text then stored. A fresh text's list comes from an exact search. A stored list keeps
    its entries and takes in each fresh text at or above the threshold, the best k kept. A full
    list that loses a retired text is searched again, since the text it had pushed out comes
    back; a list with room only loses it."""
    n = sum(int(b.shape[0]) for b in blocks)
    old_n = ids.shape[0]
    out_ids = np.full((n, k), -1, dtype=np.int32)
    out_scores = np.zeros((n, k), dtype=np.float16)
    out_ids[:old_n] = ids
    out_scores[:old_n] = scores
    gone = np.zeros(n, dtype=bool)
    gone[retired] = True
    out_ids[retired] = -1
    out_scores[retired] = 0
    holds_retired = (out_ids >= 0) & gone[np.maximum(out_ids, 0)]
    lost = holds_retired.any(axis=1)
    full = (np.asarray(ids) >= 0).all(axis=1) if old_n else np.zeros(0, dtype=bool)
    research = np.flatnonzero(lost[:old_n] & full)
    for row in np.flatnonzero(lost[:old_n] & ~full).tolist():
        keep = ~holds_retired[row] & (out_ids[row] >= 0)
        m = int(keep.sum())
        out_ids[row, :m], out_scores[row, :m] = out_ids[row, keep], out_scores[row, keep]
        out_ids[row, m:], out_scores[row, m:] = -1, 0

    searched = np.concatenate([fresh, research]).astype(np.int64)
    hits = scan(blocks, rows_of(blocks, searched), searched, threshold) if searched.size else None
    if hits is not None:
        hits = Hits(hits.query, hits.text, hits.score)
        live = ~gone[hits.text]
        owner, text, at = best_of(searched[hits.query[live]], hits.text[live], hits.score[live], k)
        out_ids[searched] = -1
        out_scores[searched] = 0
        slot = np.arange(owner.size) - np.searchsorted(owner, owner)
        out_ids[owner, slot] = text
        out_scores[owner, slot] = hits.score[live][at].astype(np.float16)

    changed = {
        *fresh.tolist(),
        *research.tolist(),
        *retired.tolist(),
        *np.flatnonzero(lost).tolist(),
    }
    if hits is not None and fresh.size:
        # The stored lists each fresh text enters: from the fresh queries' own hits, since a
        # cosine is the same both ways.
        from_fresh = hits.query < fresh.size
        target = hits.text[from_fresh]
        entering = fresh[hits.query[from_fresh]]
        score = hits.score[from_fresh]
        stored = ~np.isin(target, searched) & ~gone[target]
        target, entering, score = target[stored], entering[stored], score[stored]
        order = np.argsort(target, kind="stable")
        target, entering, score = target[order], entering[order], score[order]
        bounds = np.flatnonzero(np.r_[True, target[1:] != target[:-1], True])
        for lo, hi in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
            row = int(target[lo])
            have = out_ids[row][out_ids[row] >= 0].astype(np.int64)
            candidates = np.concatenate([have, entering[lo:hi]])
            vector = rows_of(blocks, np.array([row], dtype=np.int64))[0]
            # Rescored together, so the stored and the entering candidates are ordered by one
            # computation; the stored ones keep their stored scores.
            exact = rows_of(blocks, candidates) @ vector
            _, chosen, at = best_of(np.zeros(candidates.size, dtype=np.int64), candidates, exact, k)
            stored_score = np.concatenate(
                [out_scores[row][: have.size], score[lo:hi].astype(np.float16)]
            )
            out_ids[row] = -1
            out_scores[row] = 0
            out_ids[row, : chosen.size] = chosen
            out_scores[row, : chosen.size] = stored_score[at]
            changed.add(row)
    return out_ids, out_scores, np.array(sorted(changed), dtype=np.int64)
