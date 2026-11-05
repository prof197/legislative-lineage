"""All SQL the pipeline runs, kept in one module so it can be reviewed and tested as a unit.

pyright cannot see inside SQL strings, so each query here is exercised against committed
fixtures in the tests. See DESIGN.md#dr-4.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

# The fields of Open States' JSON export (format 1.0) that the pipeline reads. Declaring the
# schema instead of letting DuckDB infer it per file keeps every archive's Parquet identical in
# shape, turns absent keys (some records have no `raw_text`) into NULL, and ignores the rest.
BILL_JSON_COLUMNS: dict[str, str] = {
    "id": "VARCHAR",
    "legislative_session": "VARCHAR",
    "identifier": "VARCHAR",
    "title": "VARCHAR",
    "chamber": "VARCHAR",
    "classification": "VARCHAR[]",
    "actions": (
        "STRUCT(organization__name VARCHAR, description VARCHAR, date VARCHAR, "
        'classification VARCHAR[], "order" INTEGER)[]'
    ),
    "versions": (
        "STRUCT(note VARCHAR, date VARCHAR, links STRUCT(url VARCHAR, media_type VARCHAR)[])[]"
    ),
    "raw_text": "VARCHAR",
    "raw_text_url": "VARCHAR",
}


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _struct_literal(columns: dict[str, str]) -> str:
    return "{" + ", ".join(f"{_sql_string(k)}: {_sql_string(v)}" for k, v in columns.items()) + "}"


def connect_quiet() -> duckdb.DuckDBPyConnection:
    """A connection that prints no progress bar, which would interleave across processes."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con


def connect(*, memory_limit: str, temp_directory: Path) -> duckdb.DuckDBPyConnection:
    con = connect_quiet()
    con.execute(f"SET memory_limit = {_sql_string(memory_limit)}")
    # Spill to the data disk, which the caller has checked for space, not the system temp dir.
    con.execute(f"SET temp_directory = {_sql_string(str(temp_directory))}")
    return con


def outcome_actions(
    con: duckdb.DuckDBPyConnection, *, json_path: Path, kinds: list[str]
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Each bill's id with only the actions classified as one of `kinds`.

    Status depends on a handful of classifications, so filtering in SQL keeps the rows that
    reach Python small.
    """
    query = f"""
        SELECT id, list_filter(actions, lambda a: list_has_any(a.classification, $kinds))
        FROM read_json($json_path, format = 'array',
                       columns = {_struct_literal(BILL_JSON_COLUMNS)})
    """
    rows: list[tuple[str, list[dict[str, Any]]]] = con.execute(
        query, {"json_path": str(json_path), "kinds": kinds}
    ).fetchall()
    return rows


# Columns built from Python lists are cast explicitly: a list that is all None would otherwise
# be typed NULL, and Parquet files with different column types cannot be read as one table.


@dataclass(frozen=True, slots=True)
class StatusColumns:
    """Derived status per bill id, as parallel lists, to be joined onto the bills table."""

    id: list[str]
    status: list[str]
    reason: list[str | None]
    on: list[date | None]


def copy_bills_to_parquet(
    con: duckdb.DuckDBPyConnection,
    *,
    json_path: Path,
    parquet_path: Path,
    state: str,
    generated_at: str,
    archive: str,
    status: StatusColumns,
) -> int:
    """Write one archive's bills file, with status joined on, and return the rows written."""
    # COPY's target cannot be a bound parameter, so it is quoted; every value is bound.
    query = f"""
        COPY (
            WITH s AS (
                SELECT
                    CAST(unnest($status_id) AS VARCHAR) AS id,
                    CAST(unnest($status) AS VARCHAR) AS status,
                    CAST(unnest($status_reason) AS VARCHAR) AS status_reason,
                    CAST(unnest($status_date) AS DATE) AS status_date
            )
            SELECT
                b.id,
                $state AS state,
                b.legislative_session AS session,
                b.identifier,
                b.title,
                b.chamber,
                b.classification,
                s.status,
                s.status_reason,
                s.status_date,
                b.actions,
                b.versions,
                b.raw_text,
                b.raw_text_url,
                CAST($generated_at AS TIMESTAMP) AS generated_at,
                $archive AS archive
            FROM read_json($json_path, format = 'array',
                           columns = {_struct_literal(BILL_JSON_COLUMNS)}) AS b
            LEFT JOIN s USING (id)
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    params = {
        "state": state,
        "generated_at": generated_at,
        "archive": archive,
        "json_path": str(json_path),
        "status_id": status.id,
        "status": status.status,
        "status_reason": status.reason,
        "status_date": status.on,
    }
    row = con.execute(query, params).fetchone()
    if row is None:
        raise RuntimeError("COPY returned no row count")
    count: int = row[0]
    return count


def text_bytes(con: duckdb.DuckDBPyConnection, *, bills_glob: str) -> int:
    """Total UTF-8 bytes of extracted text in the bills table, to bound what splitting writes."""
    row = con.execute(
        "SELECT coalesce(sum(strlen(raw_text)), 0) FROM read_parquet($glob)", {"glob": bills_glob}
    ).fetchone()
    return int(row[0]) if row else 0


def open_bill_texts(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path
) -> duckdb.DuckDBPyConnection:
    """Start a scan of one bills file's ids and texts; read it with `fetchmany`.

    No ORDER BY: sorting would hold the whole file's text in memory before the first row.
    DuckDB keeps a single file's rows in stored order, so the output is still deterministic.
    """
    return con.execute("SELECT id, raw_text FROM read_parquet($path)", {"path": str(parquet_path)})


@dataclass(frozen=True, slots=True)
class SectionColumns:
    """Sections as parallel lists, one entry per section."""

    bill_id: list[str]
    ordinal: list[int]
    heading: list[str | None]
    text: list[str]
    normalized: list[str]
    by_heading: list[bool]
    boilerplate: list[str | None]


def write_sections(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, columns: SectionColumns
) -> None:
    query = f"""
        COPY (
            SELECT
                CAST(unnest($bill_id) AS VARCHAR) AS bill_id,
                CAST(unnest($ordinal) AS INTEGER) AS ordinal,
                CAST(unnest($heading) AS VARCHAR) AS heading,
                CAST(unnest($text) AS VARCHAR) AS text,
                CAST(unnest($normalized) AS VARCHAR) AS normalized,
                CAST(unnest($by_heading) AS BOOLEAN) AS by_heading,
                CAST(unnest($boilerplate) AS VARCHAR) AS boilerplate
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(
        query,
        {
            "bill_id": columns.bill_id,
            "ordinal": columns.ordinal,
            "heading": columns.heading,
            "text": columns.text,
            "normalized": columns.normalized,
            "by_heading": columns.by_heading,
            "boilerplate": columns.boilerplate,
        },
    )


def write_stoplist(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, hashes: list[int], sections: list[int]
) -> None:
    query = f"""
        COPY (
            SELECT
                CAST(unnest($hashes) AS UINTEGER) AS hash,
                CAST(unnest($sections) AS BIGINT) AS sections
            ORDER BY hash
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"hashes": hashes, "sections": sections})


def boilerplate_counts(con: duckdb.DuckDBPyConnection, *, sections_glob: str) -> dict[str, int]:
    """Sections flagged by each boilerplate pattern."""
    rows: list[tuple[str, int]] = con.execute(
        "SELECT boilerplate, count(*) FROM read_parquet($glob) WHERE boilerplate IS NOT NULL "
        "GROUP BY boilerplate ORDER BY boilerplate",
        {"glob": sections_glob},
    ).fetchall()
    return dict(rows)
