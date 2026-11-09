"""`lineage neighbors`: every distinct window's nearest neighbors by meaning, for the build.

Each embedded text (`lineage embed`) is searched against all the others with an approximate
index, IVF-PQ by default (reports/ann_benchmark.json). The index's CANDIDATES best for each text
are re-scored against the stored float16 vectors, the text itself is dropped, and the K best at
or above the stage's threshold are kept. The result is two arrays beside the vectors, one row
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
    index_bytes,
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
SPEC = "IVF16384,PQ48"
NPROBE = 32
CANDIDATES = 100
# faiss's k-means wants at least 39 training points per list; 40 per list of SPEC's 16,384.
TRAIN_ROWS = 40 * 16384
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
    nprobe: int | None = NPROBE,  # None for an index without lists
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
    index = train_index(spec, train.shape[1], train)
    del train
    for block in shards:
        for i in range(0, block.shape[0], _ADD_CHUNK):
            add(index, block[i : i + _ADD_CHUNK])
    if nprobe is not None:
        set_search(index, nprobe=nprobe)
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
            approx = search(index, queries, CANDIDATES)
            kept = keep_neighbors(
                rescore(approx.ids, queries, lambda r: rows_of(shards, r), CANDIDATES),
                own,
                K,
                THRESHOLD,
            )
            ids[at : at + queries.shape[0]] = kept.ids
            scores[at : at + queries.shape[0]] = kept.scores
            at += queries.shape[0]
        progress(f"{at:,} of {n:,} searched")
    ids.flush()
    scores.flush()
    found = int((np.asarray(ids) >= 0).sum())
    return {
        "command": COMMAND,
        "model": model,
        "index": spec,
        "nprobe": nprobe,
        "candidates_rescored": CANDIDATES,
        "k": K,
        "threshold": THRESHOLD,
        "texts": n,
        "neighbors_kept": found,
        "texts_with_a_neighbor": int((np.asarray(ids)[:, 0] >= 0).sum()),
        "index_bytes": index_bytes(index),
        "build_seconds": round(built),
        "seconds": round(time.time() - started),
        "peak_memory_bytes": peak_bytes(),
    }


def write_report(doc: dict[str, Any], path: Path = REPORT) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")
