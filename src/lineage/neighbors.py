"""`lineage neighbors`: every distinct window's nearest neighbors by meaning, for the build.

Each embedded text (`lineage embed`) is searched against all the others with an approximate
index: HNSW over the float16 vectors, which on a sample of the corpus kept 99.9% of the exact
neighbors at or above the threshold, where IVF-PQ re-scored kept 97.2%, and searched faster
(reports/ann_benchmark.json). HNSW's scores are cosines over the float16 vectors already; an
index of compressed codes, such as IVF-PQ, has its `candidates` best re-scored against the
vectors instead. The text itself is dropped, and the K best at or above the stage's threshold
are kept. The result is two arrays beside the vectors, one row per text id:
`neighbors/ids.npy` (int32, -1 where fewer than K pass) and `neighbors/scores.npy` (float16
cosines). The index is never saved: it exists only for the build, and a single query needs
neither it nor these arrays, since it searches the vectors exactly (lineage.semantic).

The build holds to DR-3's memory bound. It refuses to start when its projected peak, the index
at its measured bytes per vector plus its buffers, is over the bound, and stops at the first
batch after which the process holds more, or has peaked higher inside a call. It searches
CHECKED seeded texts exactly at the end and reports the share of their true neighbors it
found, and the command stops below RECALL_FLOOR. See DESIGN.md#semantic-stage.
"""

import json
import time
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
from lineage.disk import require_free_space
from lineage.memory import BOUND_BYTES, MemoryBoundError, peak_bytes, require_within
from lineage.semantic import MODEL, THRESHOLD, K
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

    rng = np.random.default_rng(SEED)
    train = rows_of(shards, np.sort(rng.choice(n, size=min(TRAIN_ROWS, n), replace=False)))
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
    checked = np.sort(rng.choice(n, size=min(CHECKED, n), replace=False)).astype(np.int64)
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
