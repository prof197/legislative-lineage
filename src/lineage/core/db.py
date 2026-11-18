"""DuckDB connections, every one capped to spill no lower than the free-space floor, and the
literal helpers the queries modules build SQL with. Opening DuckDB any other way is refused by a
test. Each package's SQL lives in its own queries module. See DESIGN.md#dr-4.
"""

from pathlib import Path

import duckdb

from lineage.core.disk import spill_budget


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def struct_literal(columns: dict[str, str]) -> str:
    return "{" + ", ".join(f"{sql_string(k)}: {sql_string(v)}" for k, v in columns.items()) + "}"


def _cap_spill(con: duckdb.DuckDBPyConnection, directory: Path) -> None:
    # DuckDB lets a query that outgrows memory spill up to 90% of the free disk. Capping it at
    # the floor makes such a query fail with an out-of-memory error instead of filling the disk.
    # The cap is per connection; the worker processes scan one file each and do not spill.
    con.execute(f"SET max_temp_directory_size = '{spill_budget(directory)}B'")


def connect_quiet() -> duckdb.DuckDBPyConnection:
    """A connection that prints no progress bar, which would interleave across processes."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    # Without a temp_directory setting, DuckDB spills to .tmp under the working directory.
    _cap_spill(con, Path.cwd())
    return con


def connect(*, memory_limit: str, temp_directory: Path) -> duckdb.DuckDBPyConnection:
    con = connect_quiet()
    con.execute(f"SET memory_limit = {sql_string(memory_limit)}")
    # Spill to the data disk, which the caller has checked for space, not the system temp dir.
    con.execute(f"SET temp_directory = {sql_string(str(temp_directory))}")
    _cap_spill(con, temp_directory)
    return con
