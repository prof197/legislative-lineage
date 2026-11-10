"""The semantic stage's nearest-neighbor index: IVF-PQ and HNSW against exact search (dev only).

The build finds every distinct window's nearest neighbors. This compares the candidate indexes
on a seeded sample of the embedded corpus (`lineage embed`):

- exact search, by brute force over the sample's float16 vectors: the truth;
- HNSW over float16 vectors (faiss "HNSW32,SQfp16") at several search widths;
- IVF-PQ with 48-byte codes (faiss "IVF<n>,PQ48"), probing several numbers of lists, its
  CANDIDATES best re-scored against the float16 vectors and the best K kept.

The sample and the queries are drawn from the shards embedded so far, which a run takes in a
seeded random order (lineage.vectors), so the benchmark can run before the corpus is finished;
the report records how much was embedded. Queries are windows drawn apart from the sample, so no
query finds itself. For each index it
records build time, index bytes, queries a second, recall of the exact K nearest, and recall of
the exact neighbors at or above the stage's cosine threshold, the ones the stage keeps. The
build's cost on the whole corpus is projected from the sample: bytes and build time grow with
the rows, and the build makes one query per distinct window.

Run after `uv run lineage embed`, with the build extra (`uv sync --extra build`):

    uv run python -m benchmarks.ann

and it writes reports/ann_benchmark.json.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchmarks.minhash import machine
from lineage.ann import (
    Index,
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
from lineage.memory import peak_bytes
from lineage.semantic import MODEL, THRESHOLD
from lineage.vectors import embedded_shards, load_plan, rows_of

SEED = 20261110
TABLES = Path("data/tables")
REPORT = Path("reports/ann_benchmark.json")
SAMPLE = 1_000_000
QUERIES = 2000
K = 20
CANDIDATES = 100  # IVF-PQ candidates re-scored per query
THREADS = 10
HNSW = "HNSW32,SQfp16"
EF_SEARCH = (32, 64, 128)
IVF_LISTS = 4096
NPROBE = (8, 16, 32, 64)
TRAIN_ROWS = 200_000
ADD_CHUNK = 1 << 17


def _above(truth: Neighbors) -> NDArray[np.int64]:
    return np.where(truth.scores >= THRESHOLD, truth.ids, -1)


def _timed_search(index: Index, queries: NDArray[np.float16], k: int) -> tuple[Neighbors, float]:
    started = time.perf_counter()
    found = search(index, queries, k)
    return found, queries.shape[0] / (time.perf_counter() - started)


def _build(
    spec: str, sample: NDArray[np.float16], train: NDArray[np.float16]
) -> tuple[Index, float]:
    started = time.perf_counter()
    index = train_index(spec, sample.shape[1], train)
    for i in range(0, sample.shape[0], ADD_CHUNK):
        add(index, sample[i : i + ADD_CHUNK])
    return index, time.perf_counter() - started


def main() -> None:
    rng = np.random.default_rng(SEED)
    set_threads(THREADS)
    plan = load_plan(TABLES)
    if plan is None:
        raise SystemExit("no plan; run `uv run lineage embed`")
    shards = embedded_shards(TABLES, MODEL, plan)
    available = sum(s.shape[0] for s in shards)
    if available < SAMPLE + QUERIES:
        raise SystemExit(
            f"{available:,} vectors embedded; the benchmark needs {SAMPLE + QUERIES:,}"
        )
    # The corpus is every distinct text, embedded or not: the projection's scale.
    total = plan.distinct
    drawn = rng.choice(available, size=SAMPLE + QUERIES, replace=False)
    sample_ids = np.sort(drawn[:SAMPLE]).astype(np.int64)
    query_ids = np.sort(drawn[SAMPLE:]).astype(np.int64)
    sample = rows_of(shards, sample_ids).astype(np.float16)
    queries = rows_of(shards, query_ids).astype(np.float16)
    train = sample[rng.choice(SAMPLE, size=TRAIN_ROWS, replace=False)]
    scale = total / SAMPLE

    started = time.perf_counter()
    truth = exact_search([sample], queries, K)
    exact_rate = QUERIES / (time.perf_counter() - started)
    above = _above(truth)
    print("exact", round(exact_rate, 1), "queries/s", flush=True)

    def entry(found: Neighbors, rate: float, built: float, size: int) -> dict[str, Any]:
        return {
            "recall_at_k": round(recall_at(found.ids, truth.ids), 4),
            "recall_above_threshold": round(recall_at(found.ids, above), 4),
            "queries_per_second": round(rate, 1),
            "build_seconds": round(built, 1),
            "index_bytes": size,
            "corpus_projection": {
                "index_bytes": round(size * scale),
                "build_hours": round(built * scale / 3600, 2),
                "query_hours": round(total / rate / 3600, 2),
            },
        }

    results: dict[str, Any] = {}
    hnsw, built = _build(HNSW, sample, train)
    size = index_bytes(hnsw)
    for ef in EF_SEARCH:
        set_search(hnsw, ef_search=ef)
        found, rate = _timed_search(hnsw, queries, K)
        results[f"{HNSW} efSearch={ef}"] = entry(found, rate, built, size)
        print(HNSW, ef, results[f"{HNSW} efSearch={ef}"], flush=True)
    del hnsw

    spec = f"IVF{IVF_LISTS},PQ48"
    ivf, built = _build(spec, sample, train)
    size = index_bytes(ivf)
    for nprobe in NPROBE:
        set_search(ivf, nprobe=nprobe)
        started = time.perf_counter()
        approx = search(ivf, queries, CANDIDATES)
        found = rescore(approx.ids, queries, lambda ids: sample[ids], K)
        rate = QUERIES / (time.perf_counter() - started)
        key = f"{spec} nprobe={nprobe}, {CANDIDATES} re-scored"
        results[key] = entry(found, rate, built, size)
        print(key, results[key], flush=True)

    doc = {
        "command": "uv run python -m benchmarks.ann",
        "seed": SEED,
        "model": MODEL,
        "corpus_vectors": total,
        "embedded_when_run": {"shards": len(shards), "of": plan.shards, "vectors": available},
        "sample": SAMPLE,
        "queries": QUERIES,
        "k": K,
        "threshold": THRESHOLD,
        "threads": THREADS,
        "train_rows": TRAIN_ROWS,
        "exact_queries_per_second": round(exact_rate, 1),
        "true_neighbors_above_threshold": int((above >= 0).sum()),
        "indexes": results,
        "peak_memory_bytes": peak_bytes(),
        "machine": machine(),
    }
    REPORT.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
