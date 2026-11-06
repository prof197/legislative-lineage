from pathlib import Path

import duckdb
import numpy as np
import pytest

from benchmarks import memory
from benchmarks.minhash import ENGINES, run_engine, run_lsh
from lineage import queries
from lineage.minhash import NUM_PERM
from lineage.signatures import build_signatures

TEXTS = [
    "the department of revenue shall publish the rates each year",
    "a person who violates this section commits a misdemeanor",
    "too short",
]


@pytest.fixture
def tables(tmp_path: Path) -> Path:
    tables = tmp_path / "tables"
    (tables / "sections").mkdir(parents=True)
    ids = [f"b{i}" for i in range(len(TEXTS))]
    n = len(TEXTS)
    with duckdb.connect() as con:
        queries.write_sections(
            con,
            parquet_path=tables / "sections" / "s.parquet",
            columns=queries.SectionColumns(
                ids, [1] * n, [None] * n, TEXTS, TEXTS, [True] * n, [None] * n
            ),
        )
        queries.write_eligibility(
            con,
            parquet_path=tables / "eligible.parquet",
            ids=ids,
            eligible=[True] * n,
            rules=[1] * n,
        )
        queries.write_stoplist(
            con, parquet_path=tables / "stoplist.parquet", hashes=[], sections=[]
        )
    return tables


# Speed is never asserted: a timing assertion would fail on a busy machine, not on a bug. The
# benchmark's numbers come from its committed report.
@pytest.mark.parametrize("engine", ENGINES)
def test_each_engine_signs_the_same_work(tables: Path, engine: str) -> None:
    run = run_engine(engine, tables, parts=None)

    assert run.engine == engine
    assert (run.sections, run.shingles) == (2, 11)
    assert run.retained_bytes_per_section >= NUM_PERM * 4
    assert run.peak_memory_bytes > 0


@pytest.fixture
def signed(tables: Path) -> Path:
    build_signatures(tables, workers=1, floor_bytes=0)
    return tables


@pytest.mark.parametrize("engine", ENGINES)
def test_each_engine_indexes_every_signature(signed: Path, engine: str) -> None:
    run = run_lsh(engine, signed)

    assert (run.sections, run.inserted) == (2, 2)
    assert run.completed
    assert run.stopped_because is None


def test_datasketch_stops_at_the_memory_bound(signed: Path) -> None:
    run = run_lsh("datasketch", signed, bound=0)

    assert not run.completed
    assert (run.inserted, run.stopped_because) == (0, "memory bound")


def test_datasketch_stops_at_the_disk_floor(signed: Path) -> None:
    run = run_lsh("datasketch", signed, floor=1 << 62)

    assert not run.completed
    assert (run.inserted, run.stopped_because) == (0, "disk floor")


def test_memory_counts_an_allocation_and_keeps_its_peak() -> None:
    before = memory.current_bytes()
    block = np.ones(64 << 20, dtype=np.uint8)

    assert memory.current_bytes() - before >= 60 << 20
    del block
    assert memory.peak_bytes() >= before + (60 << 20)
