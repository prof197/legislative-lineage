"""The SQL lineage.data's modules run: SQL lives only in a package's own queries module, and a
package above may call these. pyright cannot see inside SQL strings, so each query is exercised
against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from lineage.core.db import sql_string, struct_literal

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


def outcome_actions(
    con: duckdb.DuckDBPyConnection, *, json_path: Path, kinds: list[str], wording: str
) -> list[tuple[str, list[str], list[dict[str, Any]]]]:
    """Each bill's id and classification, with only the actions classified as one of `kinds` or
    described in words matching the `wording` pattern.

    Status depends on a handful of actions, so filtering in SQL keeps the rows that reach Python
    small.
    """
    query = f"""
        SELECT id, coalesce(classification, []),
               list_filter(actions, lambda a: list_has_any(a.classification, $kinds)
                                              OR regexp_matches(coalesce(a.description, ''),
                                                                $wording))
        FROM read_json($json_path, format = 'array',
                       columns = {struct_literal(BILL_JSON_COLUMNS)})
    """
    rows: list[tuple[str, list[str], list[dict[str, Any]]]] = con.execute(
        query, {"json_path": str(json_path), "kinds": kinds, "wording": wording}
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
                -- The project's names from here on: later stages never see the export's.
                list_transform(b.actions, lambda a: {{
                    'body': a.organization__name, 'description': a.description,
                    'date': a.date, 'classification': a.classification, 'order': a."order"
                }}) AS actions,
                b.versions,
                b.raw_text AS text,
                b.raw_text_url AS text_url,
                CAST($generated_at AS TIMESTAMP) AS generated_at,
                $archive AS archive
            FROM read_json($json_path, format = 'array',
                           columns = {struct_literal(BILL_JSON_COLUMNS)}) AS b
            LEFT JOIN s USING (id)
        ) TO {sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
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
        ) TO {sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
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
    con: duckdb.DuckDBPyConnection,
    *,
    bills_glob: str,
    texts_glob: str,
    sections_glob: str,
    blank_pattern: str,
) -> list[dict[str, Any]]:
    """Per state and session: the merged text's coverage, and where the text came from."""
    query = """
        WITH per_bill AS (
            SELECT bill_id, count(*) AS n, bool_or(by_heading) AS by_heading
            FROM read_parquet($sections)
            GROUP BY bill_id
        ),
        b AS (
            SELECT
                b.*,
                t.text AS merged,
                t.text_source,
                t.text IS NOT NULL AND regexp_full_match(t.text, $blank) AS blank
            FROM read_parquet($bills) AS b
            JOIN read_parquet($texts) AS t USING (id)
        )
        SELECT
            b.state,
            b.session,
            CAST(any_value(b.generated_at) AS VARCHAR) AS generated_at,
            count(*) AS records,
            count(*) FILTER (WHERE b.merged IS NULL) AS missing_text,
            count(*) FILTER (WHERE b.blank) AS blank_text,
            count(*) FILTER (
                WHERE b.merged IS NOT NULL AND NOT b.blank AND p.bill_id IS NULL
            ) AS text_without_sections,
            count(p.bill_id) AS with_sections,
            count(*) FILTER (WHERE p.by_heading) AS split_at_headings,
            coalesce(sum(p.n), 0) AS sections,
            quantile_cont(p.n, 0.5) AS sections_median,
            quantile_disc(p.n, 0.9) AS sections_p90,
            max(p.n) AS sections_max,
            count(*) FILTER (WHERE b.text_source = 'legislature') AS text_fetched,
            count(*) FILTER (WHERE b.text_source = 'ca_bulk') AS text_from_bulk,
            count(*) FILTER (WHERE b.status = 'introduced') AS introduced
        FROM b LEFT JOIN per_bill AS p ON p.bill_id = b.id
        GROUP BY b.state, b.session
        ORDER BY b.state, b.session
    """
    cursor = con.execute(
        query,
        {
            "bills": bills_glob,
            "texts": texts_glob,
            "sections": sections_glob,
            "blank": blank_pattern,
        },
    )
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def text_layer_coverage(
    con: duckdb.DuckDBPyConnection,
    *,
    eligible_path: Path,
    texts_glob: str,
    sections_glob: str,
    plan_path: Path,
) -> list[tuple[str, int, int, int, int, int]]:
    """Per state, over eligible records: how many there are, how many have usable exported
    text (the fetch plan's kind), how many yield sections from the merged text, and how many
    of those took fetched or bulk text."""
    rows: list[tuple[str, int, int, int, int, int]] = con.execute(
        """
        WITH split AS (SELECT DISTINCT bill_id FROM read_parquet($sections))
        SELECT p.state, count(*),
            count(*) FILTER (WHERE p.kind = 'usable'),
            count(s.bill_id),
            count(s.bill_id) FILTER (WHERE t.text_source = 'legislature'),
            count(s.bill_id) FILTER (WHERE t.text_source = 'ca_bulk')
        FROM read_parquet($plan) AS p
        JOIN read_parquet($eligible) AS e ON e.id = p.id AND e.eligible
        JOIN read_parquet($texts) AS t ON t.id = p.id
        LEFT JOIN split AS s ON s.bill_id = p.id
        GROUP BY p.state ORDER BY p.state
        """,
        {
            "plan": str(plan_path), "eligible": str(eligible_path), "texts": texts_glob,
            "sections": sections_glob,
        },
    ).fetchall()  # fmt: skip
    return rows


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


def notice_only_by_session(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, state: str, notice: str, enacting: str
) -> list[tuple[str, int]]:
    """Records per session of `state` whose text matches `notice` but not `enacting`."""
    rows: list[tuple[str, int]] = con.execute(
        "SELECT session, count(*) FROM read_parquet($bills) "
        "WHERE state = $state AND regexp_matches(text, $notice) "
        "AND NOT regexp_matches(text, $enacting) GROUP BY ALL ORDER BY session",
        {"bills": bills_glob, "state": state, "notice": notice, "enacting": enacting},
    ).fetchall()
    return rows


def amendment_only_by_session(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, opening: str, chars: int, substitute: str
) -> list[tuple[str, str, int, int]]:
    """Per state and session: records whose text opens like an amendment, and how many of those
    replace the whole bill (match `substitute` anywhere)."""
    rows: list[tuple[str, str, int, int]] = con.execute(
        "SELECT state, session, count(*), count(*) FILTER (regexp_matches(text, $substitute)) "
        "FROM read_parquet($bills) WHERE regexp_matches(left(text, $chars), $opening) "
        "GROUP BY ALL ORDER BY state, session",
        {"bills": bills_glob, "opening": opening, "chars": chars, "substitute": substitute},
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


def status_counts(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, status: str
) -> list[tuple[str, str, int]]:
    """Records per state and type with `status`."""
    rows: list[tuple[str, str, int]] = con.execute(
        f"SELECT state, {_TYPE} AS type, count(*) FROM read_parquet($bills) "
        "WHERE status = $status GROUP BY ALL ORDER BY state, type",
        {"bills": bills_glob, "status": status},
    ).fetchall()
    return rows


def status_reason_counts(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, reason: str
) -> list[tuple[str, str, int]]:
    """Records per state and type whose status was decided by `reason`."""
    rows: list[tuple[str, str, int]] = con.execute(
        f"SELECT state, {_TYPE} AS type, count(*) FROM read_parquet($bills) "
        "WHERE status_reason = $reason GROUP BY ALL ORDER BY state, type",
        {"bills": bills_glob, "reason": reason},
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
