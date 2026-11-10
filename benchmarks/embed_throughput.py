"""Embedding throughput on this machine: threads, batch order and worker processes (dev only).

Every configuration embeds the same seeded sample of distinct 94-word windows without
overlap, the semantic stage's unit, drawn from a seeded sample of matchable bills:

- one process, batches in the order drawn, as the semantic sizing measured its rates;
- one process, batches in order of token count (lineage.embed), at several thread counts;
- several worker processes, each with its own model and threads, each embedding an equal share
  of the windows in order of token count.

ONNX Runtime splits each batch's work across a process's threads. An Apple M1 Max has eight
performance and two efficiency cores, and a thread on an efficiency core can hold up the rest.
fastembed's own data-parallel mode gives each worker one thread, so it is not measured
separately: it is the eight-by-one row. The load average before each run is recorded, since
other programs on the machine lower every rate alike.

Run after `uv run lineage index` and `uv run lineage model fetch`:

    uv run python -m benchmarks.embed_throughput

and it writes reports/embed_throughput.json. The corpus hours divide the distinct windows
counted in reports/semantic_sizing.json by each rate.
"""

import json
import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.minhash import machine
from lineage import queries
from lineage.embed import BATCH_SIZE, embed, load_model
from lineage.models import MODELS
from lineage.windows import WINDOW_WORDS, display_windows

SEED = 20261109
TABLES = Path("data/tables")
SIZING = Path("reports/semantic_sizing.json")
REPORT = Path("reports/embed_throughput.json")
SAMPLE_BILLS = 4000
SAMPLE_WINDOWS = 3000
UNSORTED_BATCH = 64  # what the sizing measured with
THREADS = (4, 8, 10)
WORKERS = ((2, 4), (4, 2), (8, 1))  # (processes, threads each)
WARMUP = 32


def sample_windows(rng: np.random.Generator) -> list[str]:
    sections_glob = str(TABLES / "sections" / "*.parquet")
    eligible = TABLES / "eligible.parquet"
    with queries.connect_quiet() as con:
        ids = queries.matchable_bill_ids(con, sections_glob=sections_glob, eligible_path=eligible)
        picked: list[str] = sorted(rng.choice(ids, size=SAMPLE_BILLS, replace=False).tolist())
        rows = queries.matchable_section_texts(
            con, sections_glob=sections_glob, eligible_path=eligible, ids=picked
        )
    windows = sorted(
        {
            w
            for _, _, text in rows
            for w in display_windows(text, size=WINDOW_WORDS, stride=WINDOW_WORDS)
        }
    )
    chosen: list[int] = rng.choice(len(windows), size=SAMPLE_WINDOWS, replace=False).tolist()
    return [windows[i] for i in chosen]


def unsorted_rate(name: str, texts: Sequence[str], threads: int) -> float:
    model = load_model(name, threads=threads)
    list(model.embed(list(texts[:WARMUP]), batch_size=UNSORTED_BATCH))
    started = time.perf_counter()
    list(model.embed(list(texts), batch_size=UNSORTED_BATCH))
    return len(texts) / (time.perf_counter() - started)


def sorted_rate(name: str, texts: Sequence[str], threads: int) -> float:
    model = load_model(name, threads=threads)
    embed(model, texts[:WARMUP])
    started = time.perf_counter()
    embed(model, texts)
    return len(texts) / (time.perf_counter() - started)


_WORKER: dict[str, Any] = {}  # each worker process's own model


def _worker_setup(name: str, threads: int) -> None:
    _WORKER["model"] = load_model(name, threads=threads)
    embed(_WORKER["model"], ["warm up"])


def _worker_embed(texts: list[str]) -> int:
    return embed(_WORKER["model"], texts).shape[0]


def workers_rate(name: str, texts: Sequence[str], processes: int, threads: int) -> float:
    # Each worker takes every processes-th window, so each share spans the same lengths.
    shares = [list(texts[i::processes]) for i in range(processes)]
    context = mp.get_context("spawn")
    with context.Pool(processes, initializer=_worker_setup, initargs=(name, threads)) as pool:
        pool.map(_worker_embed, [s[:WARMUP] for s in shares])
        started = time.perf_counter()
        pool.map(_worker_embed, shares, chunksize=1)
        return len(texts) / (time.perf_counter() - started)


def model_runs(name: str, texts: Sequence[str], distinct: int) -> list[dict[str, Any]]:
    """Each configuration's rate, with the load average just before it ran."""
    plan: list[tuple[str, int, int, int]] = [
        ("drawn", 1, 8, UNSORTED_BATCH),
        *(("tokens", 1, t, BATCH_SIZE) for t in THREADS),
        *(("tokens", p, t, BATCH_SIZE) for p, t in WORKERS),
    ]
    runs: list[dict[str, Any]] = []
    for order, processes, threads, batch in plan:
        load = [round(v, 2) for v in os.getloadavg()]
        if order == "drawn":
            rate = unsorted_rate(name, texts, threads)
        elif processes == 1:
            rate = sorted_rate(name, texts, threads)
        else:
            rate = workers_rate(name, texts, processes, threads)
        runs.append(
            {
                "order": order,
                "processes": processes,
                "threads": threads,
                "batch": batch,
                "windows_per_second": round(rate, 1),
                "corpus_hours": round(distinct / rate / 3600, 1),
                "load_average_before": load,
            }
        )
        print(name, runs[-1], flush=True)
    return runs


def main() -> None:
    rng = np.random.default_rng(SEED)
    texts = sample_windows(rng)
    sizing = json.loads(SIZING.read_text())["schemes"][f"{WINDOW_WORDS}/{WINDOW_WORDS}"]
    distinct: int = sizing["distinct_units"]
    models: dict[str, Any] = {}
    for name in MODELS:
        runs = model_runs(name, texts, distinct)
        models[name] = {"runs": runs, "fastest": max(runs, key=lambda r: r["windows_per_second"])}
    doc = {
        "command": "uv run python -m benchmarks.embed_throughput",
        "seed": SEED,
        "sample": {"bills": SAMPLE_BILLS, "windows": SAMPLE_WINDOWS},
        "unit": f"{WINDOW_WORDS}-word windows, stride {WINDOW_WORDS}",
        "distinct_windows": distinct,
        "machine": machine(),
        "models": models,
    }
    REPORT.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
