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
) -> list[tuple[str, list[str], list[dict[str, Any]]]]:
    """Each bill's id and classification, with only the actions classified as one of `kinds`.

    Status depends on a handful of classifications, so filtering in SQL keeps the rows that
    reach Python small.
    """
    query = f"""
        SELECT id, coalesce(classification, []),
               list_filter(actions, lambda a: list_has_any(a.classification, $kinds))
        FROM read_json($json_path, format = 'array',
                       columns = {_struct_literal(BILL_JSON_COLUMNS)})
    """
    rows: list[tuple[str, list[str], list[dict[str, Any]]]] = con.execute(
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


def matchable_sections(
    con: duckdb.DuckDBPyConnection, *, sections_path: Path, eligible_path: Path
) -> duckdb.DuckDBPyConnection:
    """Start a scan of one sections file's matchable sections: (bill id, ordinal, normalized).

    Matchable means the bill's document type is eligible (DR-5) and the section is not flagged
    as boilerplate. Rows come in key order, so anything built from them is reproducible. Read
    it with `fetchmany`.
    """
    return con.execute(
        """
        SELECT s.bill_id, s.ordinal, s.normalized
        FROM read_parquet($sections) s
        SEMI JOIN (SELECT id FROM read_parquet($eligible) WHERE eligible) e ON s.bill_id = e.id
        WHERE s.boilerplate IS NULL
        ORDER BY s.bill_id, s.ordinal
        """,
        {"sections": str(sections_path), "eligible": str(eligible_path)},
    )


def count_sections(con: duckdb.DuckDBPyConnection, *, sections_glob: str) -> int:
    row: tuple[int] | None = con.execute(
        "SELECT count(*) FROM read_parquet($glob)", {"glob": sections_glob}
    ).fetchone()
    return row[0] if row else 0


def write_signature_keys(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, bill_ids: list[str], ordinals: list[int]
) -> None:
    """The keys of a signature matrix, row for row."""
    query = f"""
        COPY (
            SELECT
                CAST(unnest($bill_ids) AS VARCHAR) AS bill_id,
                CAST(unnest($ordinals) AS INTEGER) AS ordinal
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"bill_ids": bill_ids, "ordinals": ordinals})


def read_signature_keys(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path
) -> tuple[list[str], list[int]]:
    rows: list[tuple[str, int]] = con.execute(
        "SELECT bill_id, ordinal FROM read_parquet($p)", {"p": str(parquet_path)}
    ).fetchall()
    return [b for b, _ in rows], [o for _, o in rows]


def read_stoplist(con: duckdb.DuckDBPyConnection, *, parquet_path: Path) -> list[int]:
    rows: list[tuple[int]] = con.execute(
        "SELECT hash FROM read_parquet($p) ORDER BY hash", {"p": str(parquet_path)}
    ).fetchall()
    return [h for (h,) in rows]


def boilerplate_counts(con: duckdb.DuckDBPyConnection, *, sections_glob: str) -> dict[str, int]:
    """Sections flagged by each boilerplate pattern."""
    rows: list[tuple[str, int]] = con.execute(
        "SELECT boilerplate, count(*) FROM read_parquet($glob) WHERE boilerplate IS NOT NULL "
        "GROUP BY boilerplate ORDER BY boilerplate",
        {"glob": sections_glob},
    ).fetchall()
    return dict(rows)


# ---- Document-type eligibility (`lineage eligible`, DR-5) -----------------------------------


@dataclass(frozen=True, slots=True)
class RecordType:
    id: str
    state: str
    type: str  # classifications joined by "+"
    title: str


def record_types(con: duckdb.DuckDBPyConnection, *, bills_glob: str) -> list[RecordType]:
    rows: list[tuple[str, str, str, str]] = con.execute(
        "SELECT id, state, array_to_string(classification, '+'), coalesce(title, '') "
        "FROM read_parquet($bills) ORDER BY id",
        {"bills": bills_glob},
    ).fetchall()
    return [RecordType(*row) for row in rows]


def write_eligibility(
    con: duckdb.DuckDBPyConnection,
    *,
    parquet_path: Path,
    ids: list[str],
    eligible: list[bool],
    rules: list[int | None],
) -> None:
    query = f"""
        COPY (
            SELECT
                CAST(unnest($ids) AS VARCHAR) AS id,
                CAST(unnest($eligible) AS BOOLEAN) AS eligible,
                CAST(unnest($rules) AS INTEGER) AS rule
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"ids": ids, "eligible": eligible, "rules": rules})


def eligible_text_by_state(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, sections_glob: str, eligible_path: str
) -> dict[str, tuple[int, int]]:
    """Per state: eligible records that yielded sections, and eligible records."""
    rows: list[tuple[str, int, int]] = con.execute(
        """
        WITH sectioned AS (SELECT DISTINCT bill_id AS id FROM read_parquet($sections))
        SELECT b.state,
               count(*) FILTER (WHERE b.id IN (SELECT id FROM sectioned)),
               count(*)
        FROM read_parquet($bills) b JOIN read_parquet($eligible) e USING (id)
        WHERE e.eligible
        GROUP BY b.state ORDER BY b.state
        """,
        {"bills": bills_glob, "sections": sections_glob, "eligible": eligible_path},
    ).fetchall()
    return {state: (sectioned, total) for state, sectioned, total in rows}


# ---- Coverage statistics (`lineage stats`) -------------------------------------------------

_TYPE = "array_to_string(classification, '+')"
_ACTION_CLASSES = "flatten(list_transform(actions, lambda a: a.classification))"


def coverage_by_session(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, sections_glob: str, blank_pattern: str
) -> list[dict[str, Any]]:
    query = """
        WITH per_bill AS (
            SELECT bill_id, count(*) AS n, bool_or(by_heading) AS by_heading
            FROM read_parquet($sections)
            GROUP BY bill_id
        ),
        b AS (
            SELECT
                *,
                raw_text IS NOT NULL AND regexp_full_match(raw_text, $blank) AS blank
            FROM read_parquet($bills)
        )
        SELECT
            b.state,
            b.session,
            CAST(any_value(b.generated_at) AS VARCHAR) AS generated_at,
            count(*) AS records,
            count(*) FILTER (WHERE b.raw_text IS NULL) AS missing_text,
            count(*) FILTER (WHERE b.blank) AS blank_text,
            count(*) FILTER (
                WHERE b.raw_text IS NOT NULL AND NOT b.blank AND p.bill_id IS NULL
            ) AS text_without_sections,
            count(p.bill_id) AS with_sections,
            count(*) FILTER (WHERE p.by_heading) AS split_at_headings,
            coalesce(sum(p.n), 0) AS sections,
            quantile_cont(p.n, 0.5) AS sections_median,
            quantile_disc(p.n, 0.9) AS sections_p90,
            max(p.n) AS sections_max,
            count(*) FILTER (WHERE b.status = 'introduced') AS introduced
        FROM b LEFT JOIN per_bill AS p ON p.bill_id = b.id
        GROUP BY b.state, b.session
        ORDER BY b.state, b.session
    """
    cursor = con.execute(
        query, {"bills": bills_glob, "sections": sections_glob, "blank": blank_pattern}
    )
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def type_counts(con: duckdb.DuckDBPyConnection, *, bills_glob: str) -> list[tuple[str, str, int]]:
    """Records per state and document type (the classification list joined with '+')."""
    rows: list[tuple[str, str, int]] = con.execute(
        f"SELECT state, {_TYPE} AS type, count(*) FROM read_parquet($bills) "
        "GROUP BY ALL ORDER BY state, type",
        {"bills": bills_glob},
    ).fetchall()
    return rows


def title_matches(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, pattern: str
) -> list[tuple[str, str, int]]:
    """Records per state and type whose title matches `pattern`."""
    rows: list[tuple[str, str, int]] = con.execute(
        f"SELECT state, {_TYPE} AS type, count(*) FROM read_parquet($bills) "
        "WHERE regexp_matches(title, $pattern) GROUP BY ALL ORDER BY state, type",
        {"bills": bills_glob, "pattern": pattern},
    ).fetchall()
    return rows


def status_by_type(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str
) -> dict[tuple[str, str], dict[str, int]]:
    rows: list[tuple[str, str, str, int]] = con.execute(
        f"SELECT state, {_TYPE} AS type, status, count(*) FROM read_parquet($bills) "
        "GROUP BY ALL ORDER BY state, type, status",
        {"bills": bills_glob},
    ).fetchall()
    out: dict[tuple[str, str], dict[str, int]] = {}
    for state, kind, status, n in rows:
        out.setdefault((state, kind), {})[status] = n
    return out


def resolutions_adopted_by_one_chamber(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str
) -> list[tuple[str, str, int]]:
    """Resolutions with a passage action that still died when the session ended.

    Adoption passes a resolution, so these are joint and concurrent measures that only one
    chamber adopted.
    """
    rows: list[tuple[str, str, int]] = con.execute(
        f"SELECT state, {_TYPE} AS type, count(*) FROM read_parquet($bills) "
        f"WHERE {_TYPE} LIKE '%resolution%' AND status_reason = 'session-ended' "
        f"AND list_contains({_ACTION_CLASSES}, 'passage') GROUP BY ALL ORDER BY state, type",
        {"bills": bills_glob},
    ).fetchall()
    return rows


def reached_governor_shown_died(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str
) -> list[tuple[str, str, int]]:
    """Records sent to the governor, with no recorded outcome, that died at session end."""
    rows: list[tuple[str, str, int]] = con.execute(
        f"SELECT state, {_TYPE} AS type, count(*) FROM read_parquet($bills) "
        "WHERE status_reason = 'session-ended' "
        f"AND list_has_any({_ACTION_CLASSES}, ['executive-receipt', 'enrolled']) "
        "GROUP BY ALL ORDER BY state, type",
        {"bills": bills_glob},
    ).fetchall()
    return rows


def carryover_overlap(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, state: str, first: str, second: str
) -> dict[str, Any]:
    """How many first-session records reappear in the second under the same number and title."""
    query = """
        WITH b AS (
            SELECT session, identifier, lower(trim(title)) AS title
            FROM read_parquet($bills) WHERE state = $state
        )
        SELECT
            count(*) FILTER (WHERE session = $first) AS first_records,
            count(*) FILTER (
                WHERE session = $first AND (identifier, title) IN (
                    SELECT (identifier, title) FROM b WHERE session = $second
                )
            ) AS reappearing
        FROM b
    """
    row = con.execute(
        query, {"bills": bills_glob, "state": state, "first": first, "second": second}
    ).fetchall()[0]
    first_records, reappearing = int(row[0]), int(row[1])
    share = round(reappearing / first_records, 4) if first_records else None
    return {"first_records": first_records, "reappearing": reappearing, "share": share}
