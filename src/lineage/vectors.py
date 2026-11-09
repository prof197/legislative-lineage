"""`lineage embed`: a float16 vector for every distinct window of the semantic stage, resumable.

The semantic stage's unit is the 94-word window without overlap, cut from each matchable
section's display text. Identical windows, in companion bills, refiled bills and templates,
share one text, which is embedded once. The run has two passes:

- plan: every sections file's windows are cut and keyed by a 128-bit BLAKE2b digest of their
  text; a text seen before shares the first one's id. It writes
  `<tables>/semantic/units/<stem>.parquet`, one row per window with its text id, and
  `<tables>/semantic/texts/<shard>.parquet`, the distinct texts in id order, SHARD_TEXTS per
  shard. The plan does not depend on the model.
- embed: each text shard becomes `<tables>/semantic/<model>/<shard>.npy`, float16 rows of unit
  length. A shard is written under a temporary name and renamed, so a run that stops resumes
  at the shards not yet written. Worker processes each load the model with their own threads;
  four of two threads embed fastest on the reference laptop (reports/embed_throughput.json).

Each run logs its start and every shard it finishes, with the time, to `runs.jsonl` beside the
vectors, so the report counts every run's time up to its last shard, even a run that was
stopped. See DESIGN.md#semantic-stage.
"""

import hashlib
import json
import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.disk import require_free_space
from lineage.embed import embed, load_model
from lineage.memory import peak_bytes
from lineage.models import MODELS, MODELS_DIR, require_model
from lineage.windows import WINDOW_WORDS, display_windows

SEMANTIC_DIR = "semantic"
SHARD_TEXTS = 1 << 14
SHARD_SEED = 20261109
PROCESSES = 4
THREADS = 2
REPORT = Path("reports/embedding.json")
COMMAND = "uv run lineage embed"
_FETCH_SECTIONS = 4096
_SPAWN = multiprocessing.get_context("spawn")
# Bytes a distinct text takes in its shard, an upper bound for the disk check: 94 words of
# display text, compressed.
_TEXT_BYTES = 1024


class EmbedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Plan:
    # What the plan was cut from, to tell a stale plan: each sections file's size, and the
    # eligibility table's hash, since a change of document-type rules changes which sections
    # are matchable without touching a sections file.
    sections_files: dict[str, int]  # name -> size
    eligible_sha256: str
    windows: int
    distinct: int
    shards: int


def semantic_dir(tables_dir: Path) -> Path:
    return tables_dir / SEMANTIC_DIR


def _plan_path(tables_dir: Path) -> Path:
    return semantic_dir(tables_dir) / "plan.json"


def _sections_files(tables_dir: Path) -> list[Path]:
    return sorted((tables_dir / "sections").glob("*.parquet"))


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode(), digest_size=16).digest()


def shard_name(shard: int) -> str:
    return f"{shard:05d}"


def load_plan(tables_dir: Path) -> Plan | None:
    path = _plan_path(tables_dir)
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    return Plan(
        doc["sections_files"],
        doc["eligible_sha256"],
        doc["windows"],
        doc["distinct"],
        doc["shards"],
    )


def _inputs(tables_dir: Path) -> tuple[dict[str, int], str]:
    files = {p.name: p.stat().st_size for p in _sections_files(tables_dir)}
    eligible = tables_dir / "eligible.parquet"
    digest = hashlib.sha256(eligible.read_bytes()).hexdigest() if eligible.exists() else ""
    return files, digest


def _embedded_shards(tables_dir: Path) -> list[Path]:
    root = semantic_dir(tables_dir)
    return sorted(p for d in root.iterdir() if d.is_dir() for p in d.glob("[0-9]*.npy"))


def build_plan(tables_dir: Path, *, floor_bytes: int) -> Plan:
    """Cut every matchable section into windows and number the distinct texts."""
    files = _sections_files(tables_dir)
    if not files:
        raise EmbedError(f"no sections in {tables_dir / 'sections'}; run `lineage split`")
    eligible = tables_dir / "eligible.parquet"
    root = semantic_dir(tables_dir)
    if root.exists() and _embedded_shards(tables_dir):
        # A new plan numbers the texts afresh, so vectors embedded under another would be read
        # as the wrong texts.
        raise EmbedError(f"{root} holds vectors from an earlier plan; remove it and run again")
    units_dir, texts_dir = root / "units", root / "texts"
    for d in (units_dir, texts_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.parquet"):
            old.unlink()
    ids: dict[bytes, int] = {}
    pending: list[str] = []
    shards = windows = 0

    def flush(con: Any, *, final: bool) -> None:
        nonlocal shards, pending
        while len(pending) >= SHARD_TEXTS or (final and pending):
            batch, pending = pending[:SHARD_TEXTS], pending[SHARD_TEXTS:]
            require_free_space(texts_dir, needed=len(batch) * _TEXT_BYTES, floor=floor_bytes)
            queries.write_semantic_texts(
                con,
                parquet_path=texts_dir / f"{shard_name(shards)}.parquet",
                first_id=shards * SHARD_TEXTS,
                texts=batch,
            )
            shards += 1

    with queries.connect_quiet() as con:
        for part in files:
            bill_ids: list[str] = []
            ordinals: list[int] = []
            numbers: list[int] = []
            text_ids: list[int] = []
            scan = queries.matchable_sections(
                con, sections_path=part, eligible_path=eligible, display=True
            )
            while batch := scan.fetchmany(_FETCH_SECTIONS):
                for bill_id, ordinal, text, _ in batch:
                    cut = display_windows(text, size=WINDOW_WORDS, stride=WINDOW_WORDS)
                    for number, window in enumerate(cut):
                        key = digest(window)
                        text_id = ids.get(key)
                        if text_id is None:
                            text_id = ids[key] = len(ids)
                            pending.append(window)
                        bill_ids.append(bill_id)
                        ordinals.append(ordinal)
                        numbers.append(number)
                        text_ids.append(text_id)
            windows += len(text_ids)
            if text_ids:
                queries.write_semantic_units(
                    con,
                    parquet_path=units_dir / part.name,
                    bill_ids=bill_ids,
                    ordinals=ordinals,
                    windows=numbers,
                    text_ids=text_ids,
                )
            flush(con, final=False)
        flush(con, final=True)
    sections_files, eligible_sha256 = _inputs(tables_dir)
    plan = Plan(sections_files, eligible_sha256, windows, len(ids), shards)
    _plan_path(tables_dir).write_text(json.dumps(_plan_doc(plan), indent=1) + "\n")
    return plan


def _plan_doc(plan: Plan) -> dict[str, Any]:
    return {
        "sections_files": plan.sections_files,
        "eligible_sha256": plan.eligible_sha256,
        "windows": plan.windows,
        "distinct": plan.distinct,
        "shards": plan.shards,
    }


def current_plan(tables_dir: Path, *, floor_bytes: int) -> tuple[Plan, bool]:
    """The plan, built if missing; (plan, built now). A plan whose sections have changed since
    is refused rather than rebuilt, since vectors already embedded would no longer match it."""
    plan = load_plan(tables_dir)
    if plan is None:
        return build_plan(tables_dir, floor_bytes=floor_bytes), True
    if (plan.sections_files, plan.eligible_sha256) != _inputs(tables_dir):
        raise EmbedError(
            f"the sections or their eligibility changed since {_plan_path(tables_dir)} was "
            f"written; remove {semantic_dir(tables_dir)} and run again"
        )
    return plan, False


def vectors_dir(tables_dir: Path, model: str) -> Path:
    return semantic_dir(tables_dir) / model


_WORKER: dict[str, Any] = {}


def _worker_setup(model: str, models_dir: Path, threads: int) -> None:
    _WORKER["model"] = load_model(model, models_dir, threads=threads)


def _embed_shard(texts_path: Path, dest: Path) -> tuple[int, int]:
    """Embed one shard into `dest`; return its texts and the worker's peak memory. Runs in a
    worker process."""
    with queries.connect_quiet() as con:
        texts = queries.read_semantic_texts(con, parquet_path=texts_path)
    vectors = embed(_WORKER["model"], texts).astype(np.float16)
    partial = dest.with_name(dest.name + ".partial")
    with partial.open("wb") as f:
        np.save(f, vectors)
    partial.replace(dest)
    return len(texts), peak_bytes()


def missing_shards(tables_dir: Path, model: str, plan: Plan) -> list[int]:
    out = vectors_dir(tables_dir, model)
    return [s for s in range(plan.shards) if not (out / f"{shard_name(s)}.npy").exists()]


def shard_order(shards: list[int]) -> list[int]:
    """The shards in a fixed random order. Plan order runs state by state, so the shards done
    early in a run spread over every state instead of the first few alphabetically, and a
    sample of an unfinished run still stands for the corpus."""
    order = np.random.default_rng(SHARD_SEED).permutation(np.array(shards, dtype=np.int64))
    return [int(s) for s in order]


def embed_corpus(
    tables_dir: Path,
    model: str,
    *,
    models_dir: Path = MODELS_DIR,
    processes: int = PROCESSES,
    threads: int = THREADS,
    floor_bytes: int,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Plan if needed, embed every shard not yet embedded, and return the report."""
    if model not in MODELS:
        raise EmbedError(f"unknown model {model}; known: {', '.join(MODELS)}")
    require_model(model, models_dir)
    started = time.time()
    log = vectors_dir(tables_dir, model) / "runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    def record(event: dict[str, Any]) -> None:
        with log.open("a") as f:
            f.write(json.dumps({"run": round(started, 3), "at": round(time.time(), 3)} | event))
            f.write("\n")

    record({"event": "start", "processes": processes, "threads": threads})
    plan, planned = current_plan(tables_dir, floor_bytes=floor_bytes)
    if planned:
        progress(
            f"planned {plan.windows:,} windows, {plan.distinct:,} distinct, {plan.shards} shards"
        )
    out = vectors_dir(tables_dir, model)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.partial"):
        stale.unlink()
    todo = missing_shards(tables_dir, model, plan)
    dim = MODELS[model].dim
    require_free_space(out, needed=len(todo) * SHARD_TEXTS * dim * 2, floor=floor_bytes)
    texts_dir = semantic_dir(tables_dir) / "texts"
    done_texts = 0
    if todo:
        progress(f"{len(todo)} of {plan.shards} shards to embed with {model}")
        with ProcessPoolExecutor(
            max_workers=processes,
            mp_context=_SPAWN,
            initializer=_worker_setup,
            initargs=(model, models_dir, threads),
        ) as pool:
            queue = shard_order(todo)
            running: dict[Future[tuple[int, int]], int] = {}
            while queue or running:
                while queue and len(running) < processes:
                    s = queue.pop(0)
                    name = shard_name(s)
                    running[
                        pool.submit(
                            _embed_shard, texts_dir / f"{name}.parquet", out / f"{name}.npy"
                        )
                    ] = s
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    shard = running.pop(future)
                    count, worker_peak = future.result()
                    record(
                        {
                            "event": "shard",
                            "shard": shard,
                            "texts": count,
                            "worker_peak_bytes": worker_peak,
                            "main_peak_bytes": peak_bytes(),
                        }
                    )
                    done_texts += count
                    elapsed = time.time() - started
                    left = len(queue) + len(running)
                    rate = done_texts / elapsed if elapsed else 0.0
                    eta = left * SHARD_TEXTS / rate / 3600 if rate else float("nan")
                    progress(
                        f"{plan.shards - left} of {plan.shards} shards, {rate:.1f} texts/s "
                        f"this run, about {eta:.1f} h left"
                    )
    record({"event": "end"})
    return report(tables_dir, model, plan)


def report(tables_dir: Path, model: str, plan: Plan) -> dict[str, Any]:
    out = vectors_dir(tables_dir, model)
    events = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines() if line]
    runs: dict[float, dict[str, Any]] = {}
    for e in events:
        run = runs.setdefault(e["run"], {"start": e["run"], "last": e["run"], "texts": 0})
        run["last"] = max(run["last"], e["at"])
        if e["event"] == "start":
            run["processes_threads"] = (e["processes"], e["threads"])
        elif e["event"] == "shard":
            run["texts"] += e["texts"]
    seconds = round(sum(r["last"] - r["start"] for r in runs.values()))
    peaks = {
        key: max((e[key] for e in events if key in e), default=None)
        for key in ("worker_peak_bytes", "main_peak_bytes")
    }
    embedded = sum(r["texts"] for r in runs.values())
    left = missing_shards(tables_dir, model, plan)
    return {
        "command": COMMAND,
        "model": model,
        "unit": f"{WINDOW_WORDS}-word windows, stride {WINDOW_WORDS}",
        "windows": plan.windows,
        "distinct_texts": plan.distinct,
        "shards": plan.shards,
        "shards_left": len(left),
        "vector_bytes": sum(
            (out / f"{shard_name(s)}.npy").stat().st_size
            for s in range(plan.shards)
            if (out / f"{shard_name(s)}.npy").exists()
        ),
        "runs": len(runs),
        "seconds": seconds,
        "texts_per_second": round(embedded / seconds, 1) if seconds else None,
        "texts_embedded_in_runs": embedded,
        "processes_threads": sorted({r["processes_threads"] for r in runs.values()}),
        # The highest any logged shard saw: each worker, and the process that ran the workers.
        "peak_memory_bytes": peaks,
    }


def write_report(doc: dict[str, Any], path: Path = REPORT) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")


def load_vectors(tables_dir: Path, model: str, plan: Plan) -> list[NDArray[np.float16]]:
    """Every shard's vectors, memory-mapped, in text id order."""
    out = vectors_dir(tables_dir, model)
    if missing_shards(tables_dir, model, plan):
        raise EmbedError(f"{out} is incomplete; run `lineage embed`")
    return [np.load(out / f"{shard_name(s)}.npy", mmap_mode="r") for s in range(plan.shards)]


def rows_of(blocks: Sequence[NDArray[Any]], ids: NDArray[np.int64]) -> NDArray[np.float32]:
    """The rows of the given global ids, in the order given, from blocks numbered on from
    one to the next (memory-mapped shards, say)."""
    sizes = np.array([b.shape[0] for b in blocks], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    out = np.empty((ids.size, blocks[0].shape[1]), dtype=np.float32)
    which = np.searchsorted(starts, ids, side="right") - 1
    for block in np.unique(which).tolist():
        at = np.flatnonzero(which == block)
        out[at] = blocks[block][ids[at] - starts[block]]
    return out
