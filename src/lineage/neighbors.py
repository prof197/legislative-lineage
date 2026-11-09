"""`lineage neighbors`: every distinct window's nearest neighbors by meaning, for the build.

Each embedded text (`lineage embed`) is searched against all the others with an approximate
index: HNSW over the float16 vectors, which on a sample of the corpus kept 99.9% of the exact
neighbors at or above the threshold, where IVF-PQ re-scored kept 97.2%, and searched faster
(reports/ann_benchmark.json). HNSW's scores are cosines over the float16 vectors already; an
index of compressed codes, such as IVF-PQ, has its `candidates` best re-scored against the
vectors instead. The text itself is dropped, and the K best at or above the stage's threshold
are kept. The result is two arrays beside the vectors, one row
per text id: `neighbors/ids.npy` (int32, -1 where fewer than K pass) and
`neighbors/scores.npy` (float16 cosines). A single query never needs them: it searches the
vectors exactly (lineage.semantic). See DESIGN.md#semantic-stage.
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
from lineage.memory import peak_bytes
from lineage.semantic import MODEL, THRESHOLD, K
from lineage.vectors import EmbedError, load_plan, load_vectors, rows_of, vectors_dir

SEED = 20261111
SPEC = "HNSW32,SQfp16"
EF_SEARCH = 128  # the widest benchmarked; its search over the corpus is projected at minutes
# For an index that needs training (IVF): faiss's k-means wants at least 39 points per list,
# and this allows 40 per list for up to 16,384 lists. HNSW over float16 needs none.
TRAIN_ROWS = 40 * 16384
# Texts searched exactly once the build is done, to measure its recall over the whole corpus,
# where neighbors are denser than in the benchmark's sample.
CHECKED = 2000
THREADS = 10
_ADD_CHUNK = 1 << 17
_QUERY_CHUNK = 1 << 12  # queries per pass: their candidates' vectors are gathered at once
REPORT = Path("reports/neighbors.json")
COMMAND = "uv run lineage neighbors"


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
    progress: Any = print,
) -> dict[str, Any]:
    started = time.time()
    plan = load_plan(tables_dir)
    if plan is None:
        raise EmbedError(f"no semantic plan in {tables_dir}; run `lineage embed`")
    shards = load_vectors(tables_dir, model, plan)
    n = sum(s.shape[0] for s in shards)
    out = vectors_dir(tables_dir, model) / "neighbors"
    out.mkdir(parents=True, exist_ok=True)
    require_free_space(out, needed=n * K * (4 + 2), floor=floor_bytes)
    set_threads(THREADS)

    rng = np.random.default_rng(SEED)
    train = rows_of(shards, np.sort(rng.choice(n, size=min(TRAIN_ROWS, n), replace=False)))
    index = train_index(spec, train.shape[1], train)  # a no-op for an index needing no training
    del train
    for block in shards:
        for i in range(0, block.shape[0], _ADD_CHUNK):
            add(index, block[i : i + _ADD_CHUNK])
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
    }


def write_report(doc: dict[str, Any], path: Path = REPORT) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")
