from pathlib import Path

import duckdb
import pytest

from benchmarks.minhash import ENGINES, run_engine
from lineage import queries
from lineage.minhash import NUM_PERM

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
    assert run.rss_peak_bytes > 0
