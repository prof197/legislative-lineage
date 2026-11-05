import shutil
from pathlib import Path

import duckdb
import pytest

from lineage import disk, queries
from lineage.disk import InsufficientDiskSpaceError, check_free_space

GiB = 1 << 30


def test_allows_a_write_that_leaves_the_floor_intact() -> None:
    check_free_space(free=40 * GiB, needed=7 * GiB, floor=20 * GiB)


def test_allows_a_write_that_lands_exactly_on_the_floor() -> None:
    check_free_space(free=27 * GiB, needed=7 * GiB, floor=20 * GiB)


def test_refuses_a_write_that_would_cross_the_floor() -> None:
    with pytest.raises(InsufficientDiskSpaceError, match=r"would leave 13\.0 GiB"):
        check_free_space(free=20 * GiB, needed=7 * GiB, floor=20 * GiB)


def _with_free_space(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    usage = shutil.disk_usage(".")._replace(free=free)

    def disk_usage(_: object) -> object:
        return usage

    monkeypatch.setattr(disk.shutil, "disk_usage", disk_usage)


def _spill_limit(con: duckdb.DuckDBPyConnection) -> str:
    (row,) = con.execute("SELECT current_setting('max_temp_directory_size')").fetchall()
    return str(row[0])


def test_a_query_may_spill_only_down_to_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_free_space(monkeypatch, disk.DEFAULT_FLOOR_BYTES + 6 * GiB)

    with queries.connect_quiet() as con:
        assert _spill_limit(con) == "6.0 GiB"


def test_a_query_may_not_spill_at_all_below_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_free_space(monkeypatch, disk.DEFAULT_FLOOR_BYTES - GiB)

    with queries.connect_quiet() as con:
        assert _spill_limit(con) == "0 bytes"
        con.execute("SET memory_limit = '32MB'")
        with pytest.raises(duckdb.OutOfMemoryException, match="max_temp_directory_size"):
            con.execute("SELECT md5(range::VARCHAR) AS h FROM range(5000000) ORDER BY h").fetchone()


def test_the_spill_directory_is_held_to_the_same_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _with_free_space(monkeypatch, disk.DEFAULT_FLOOR_BYTES + 2 * GiB)

    with queries.connect(memory_limit="1GB", temp_directory=tmp_path) as con:
        assert _spill_limit(con) == "2.0 GiB"
