"""Time and peak memory of the two MinHash engines doing the same work, for DESIGN.md#dr-3.

    uv run python -m benchmarks.minhash

Signing. Each engine runs in a fresh process. It first reads every matchable section of the
corpus and keeps its stoplisted shingle set, then signs all of them in batches, keeping every
signature in memory as an index build would. Reading is the same work for both engines and is
reported separately; only signing is timed. datasketch runs through its batch path
(`MinHash.bulk`) with its default permutation scheme. Its input is converted to Python ints
inside the timed region, since the library cannot take a NumPy array without that. A separate
pass signs a fixed sample under `tracemalloc`, which counts exactly the bytes the signatures
still hold afterwards, per section; it runs apart from the timed pass because tracing slows
allocation.

LSH index. Each engine then builds its index, 32 bands of 4 rows, in another fresh process
from the same input: the signatures table that `lineage index` wrote, loaded into memory.
datasketch inserts every row through `MinHashLSH.insertion_session`. Its index is checked
against the memory bound as it grows, and the build stops if it crosses the bound or the free
disk falls to the floor, so that a run that does not fit is measured stopping there, not
extrapolated, and never pushes the machine into swap.

Peak memory is the process's physical footprint (see benchmarks/memory.py). Both engines run
on one core. The comparison is per process, and both parallelize the same way.
"""

import argparse
import gc
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from benchmarks import memory
from lineage import queries
from lineage.disk import DEFAULT_FLOOR_BYTES, GiB
from lineage.lsh import BANDS, ROWS, build_band_index
from lineage.minhash import NUM_PERM, permutations, signatures
from lineage.shingles import section_shingles
from lineage.signatures import load_signatures
from lineage.stoplist import apply_stoplist, load_stoplist

ENGINES = ("numpy", "datasketch")
COMMAND = "uv run python -m benchmarks.minhash"
_BATCH = 5_000
# Sections signed under tracemalloc to measure what each signature holds.
_MEMORY_SAMPLE = 20_000
# The design's bounds on the reference laptop (DR-3): signing the full corpus overnight, and
# a full index build (signatures plus LSH index) within half of the machine's 32 GiB.
OVERNIGHT_SECONDS = 8 * 3600
MEMORY_BOUND_BYTES = 16 * GiB
# Rows datasketch inserts between checks of memory and disk.
_WATCH_EVERY = 25_000


@dataclass(frozen=True, slots=True)
class EngineRun:
    engine: str
    sections: int
    shingles: int
    read_seconds: float
    sign_seconds: float
    retained_bytes_per_section: float
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class LshRun:
    engine: str
    sections: int
    inserted: int
    seconds: float
    index_bytes: int  # the footprint the index added
    peak_memory_bytes: int  # signatures, index and anything transient
    completed: bool
    stopped_because: str | None


def _identity(value: int) -> int:
    return value


def _datasketch_minhash() -> Any:
    from datasketch import MinHash  # pyright: ignore[reportMissingTypeStubs]

    return cast("Any", MinHash)


def _read_sets(tables_dir: Path, parts: int | None) -> list[NDArray[np.uint32]]:
    files = sorted((tables_dir / "sections").glob("*.parquet"))[:parts]
    eligible = tables_dir / "eligible.parquet"
    stoplist = load_stoplist(tables_dir)
    sets: list[NDArray[np.uint32]] = []
    with queries.connect_quiet() as con:
        for part in files:
            scan = queries.matchable_sections(con, sections_path=part, eligible_path=eligible)
            while batch := scan.fetchmany(_BATCH):
                kept = apply_stoplist(section_shingles([t for *_, t in batch]), stoplist)
                sets += (s for s in kept if s.size)
    return sets


def _sign(minhash: Any, sets: list[NDArray[np.uint32]]) -> list[Any]:
    perms = permutations()
    kept: list[Any] = []
    for i in range(0, len(sets), _BATCH):
        batch = sets[i : i + _BATCH]
        if minhash is None:
            kept.append(signatures(batch, perms))
        else:
            kept += minhash.bulk(
                [s.tolist() for s in batch], num_perm=NUM_PERM, seed=1, hashfunc=_identity
            )
    return kept


def run_engine(engine: str, tables_dir: Path, parts: int | None) -> EngineRun:
    """Sign the matchable sections of the first `parts` sections files (all when None)."""
    started = time.perf_counter()
    sets = _read_sets(tables_dir, parts)
    read_seconds = time.perf_counter() - started
    minhash = _datasketch_minhash() if engine == "datasketch" else None

    sample = sets[:_MEMORY_SAMPLE]
    gc.collect()
    tracemalloc.start()
    held = _sign(minhash, sample)
    retained, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del held

    started = time.perf_counter()
    kept = _sign(minhash, sets)
    sign_seconds = time.perf_counter() - started
    peak = memory.peak_bytes()
    del kept
    return EngineRun(
        engine=engine,
        sections=len(sets),
        shingles=sum(s.size for s in sets),
        read_seconds=round(read_seconds, 1),
        sign_seconds=round(sign_seconds, 1),
        retained_bytes_per_section=round(retained / max(len(sample), 1), 1),
        peak_memory_bytes=peak,
    )


def _datasketch_lsh() -> tuple[Any, Any]:
    from datasketch import LeanMinHash, MinHashLSH  # pyright: ignore[reportMissingTypeStubs]

    return cast("Any", MinHashLSH), cast("Any", LeanMinHash)


def _insert_watched(
    values: NDArray[np.uint32], tables_dir: Path, bound: int, floor: int
) -> tuple[int, str | None]:
    """Insert every row into a datasketch index, stopping at the memory bound or disk floor."""
    lsh_class, lean = _datasketch_lsh()
    index = lsh_class(num_perm=NUM_PERM, params=(BANDS, ROWS))
    with index.insertion_session() as session:
        for i, row in enumerate(values):
            if i % _WATCH_EVERY == 0:
                if memory.current_bytes() > bound:
                    return i, "memory bound"
                if shutil.disk_usage(tables_dir).free < floor:
                    return i, "disk floor"
            # The values are this project's signatures; the scheme only labels them.
            session.insert(i, lean(seed=1, hashvalues=row, scheme="affine32"))
    return len(values), None


def run_lsh(
    engine: str,
    tables_dir: Path,
    *,
    bound: int = MEMORY_BOUND_BYTES,
    floor: int = DEFAULT_FLOOR_BYTES,
) -> LshRun:
    """Build one engine's LSH index over the stored signatures, in this process."""
    values = load_signatures(tables_dir).values
    before = memory.current_bytes()
    started = time.perf_counter()
    if engine == "numpy":
        held = build_band_index(values)
        inserted, stopped = len(values), None
    else:
        held = None
        inserted, stopped = _insert_watched(values, tables_dir, bound, floor)
    seconds = time.perf_counter() - started
    index_bytes = memory.current_bytes() - before
    del held
    return LshRun(
        engine=engine,
        sections=len(values),
        inserted=inserted,
        seconds=round(seconds, 1),
        index_bytes=index_bytes,
        peak_memory_bytes=memory.peak_bytes(),
        completed=stopped is None,
        stopped_because=stopped,
    )


def _fresh_process[T](fn: Callable[..., T], *args: Any) -> T:
    # A fresh process per run, so none inherits another's heap or peak.
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(fn, *args).result()


def machine() -> dict[str, Any]:
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "datasketch": version("datasketch"),
    }


def _lsh_summary(run: LshRun, signing: EngineRun) -> dict[str, Any]:
    return asdict(run) | {
        "index_bytes_per_section": round(run.index_bytes / max(run.inserted, 1), 1),
        "within_memory_bound": run.completed and run.peak_memory_bytes <= MEMORY_BOUND_BYTES,
        "signing_matches_index_build": signing.sections == run.sections,
    }


def benchmark(tables_dir: Path, *, parts: int | None = None) -> dict[str, Any]:
    runs = {engine: _fresh_process(run_engine, engine, tables_dir, parts) for engine in ENGINES}
    ours, theirs = runs["numpy"], runs["datasketch"]
    if (ours.sections, ours.shingles) != (theirs.sections, theirs.shingles):
        raise RuntimeError("the engines were given different work")
    lsh = {engine: _fresh_process(run_lsh, engine, tables_dir) for engine in ENGINES}
    return {
        "command": COMMAND,
        "machine": machine(),
        "scope": "full corpus" if parts is None else f"first {parts} sections files",
        "num_perm": NUM_PERM,
        "memory_sample_sections": _MEMORY_SAMPLE,
        "memory_bound_bytes": MEMORY_BOUND_BYTES,
        "engines": {
            name: asdict(run)
            | {
                "seconds_per_million_shingles": round(run.sign_seconds / run.shingles * 1e6, 3)
                if run.shingles
                else None,
                "signature_memory_bytes": round(run.retained_bytes_per_section * run.sections),
                "within_overnight_bound": run.sign_seconds <= OVERNIGHT_SECONDS,
            }
            for name, run in runs.items()
        },
        "datasketch_over_numpy_sign_time": round(theirs.sign_seconds / ours.sign_seconds, 2)
        if ours.sign_seconds
        else None,
        "lsh": {
            "bands": BANDS,
            "rows": ROWS,
            "input": "the signatures table, loaded into memory",
            "engines": {name: _lsh_summary(run, runs[name]) for name, run in lsh.items()},
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MinHash engine benchmark (DESIGN.md#dr-3).")
    parser.add_argument("--tables", type=Path, default=Path("data/tables"))
    parser.add_argument("--report", type=Path, default=Path("reports/minhash_benchmark.json"))
    parser.add_argument("--parts", type=int, help="Only the first N sections files (a trial).")
    args = parser.parse_args(argv)
    report = cast(Path, args.report)
    doc = benchmark(cast(Path, args.tables), parts=cast("int | None", args.parts))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"signing": doc["engines"], "lsh": doc["lsh"]["engines"]}, indent=1))


if __name__ == "__main__":
    main()
