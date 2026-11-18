"""The SQL not yet moved to its package's own queries module (M13; DR-4).

pyright cannot see inside SQL strings, so each query here is exercised against committed
fixtures in the tests. See DESIGN.md#dr-4.
"""

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from lineage.core.db import sql_string, struct_literal
from lineage.data.queries import BILL_JSON_COLUMNS, RecordType, StatusColumns
from lineage.match.queries import MATCHABLE


def bill_statuses(con: duckdb.DuckDBPyConnection, *, bills_glob: str) -> dict[str, str | None]:
    """Every bill's status, by bill id."""
    rows: list[tuple[str, str | None]] = con.execute(
        "SELECT id, status FROM read_parquet($bills)", {"bills": bills_glob}
    ).fetchall()
    return dict(rows)


def bill_id_of(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, state: str, session: str, identifier: str
) -> str | None:
    """The id of the bill a state, session and identifier name, if the table holds it."""
    row: tuple[str] | None = con.execute(
        "SELECT id FROM read_parquet($bills) "
        "WHERE state = $state AND session = $session AND identifier = $identifier",
        {"bills": bills_glob, "state": state, "session": session, "identifier": identifier},
    ).fetchone()
    return None if row is None else row[0]


def bill_details(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, ids: list[str]
) -> dict[str, tuple[str, str, str, str, str | None, str | None]]:
    """(state, session, identifier, title, status, first action's day) for each id."""
    rows: list[tuple[str, str, str, str, str, str | None, str | None]] = con.execute(
        "SELECT id, state, session, identifier, title, status, "
        "list_min(list_transform(actions, lambda a: left(a.date, 10))) "
        "FROM read_parquet($bills) WHERE list_contains($ids, id)",
        {"bills": bills_glob, "ids": ids},
    ).fetchall()
    return {r[0]: (r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows}


def matchable_bill_ids(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path
) -> list[str]:
    """Every bill with at least one matchable section, sorted."""
    rows: list[tuple[str]] = con.execute(
        f"SELECT DISTINCT s.bill_id {MATCHABLE} ORDER BY 1",
        {"sections": sections_glob, "eligible": str(eligible_path)},
    ).fetchall()
    return [bill for (bill,) in rows]


def section_headings(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, bill_id: str
) -> dict[int, str | None]:
    """Each section's heading of one bill, by ordinal."""
    rows: list[tuple[int, str | None]] = con.execute(
        "SELECT ordinal, heading FROM read_parquet($sections) WHERE bill_id = $id",
        {"sections": sections_glob, "id": bill_id},
    ).fetchall()
    return dict(rows)


def matchable_section_texts(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path, ids: list[str]
) -> list[tuple[str, int, str]]:
    """(bill id, ordinal, display text) of the wanted bills' matchable sections, in key order."""
    rows: list[tuple[str, int, str]] = con.execute(
        f"SELECT s.bill_id, s.ordinal, s.text {MATCHABLE} "
        "AND s.bill_id IN (SELECT unnest($ids)) ORDER BY s.bill_id, s.ordinal",
        {"sections": sections_glob, "eligible": str(eligible_path), "ids": ids},
    ).fetchall()
    return rows


def hosted_bill_ids(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, eligible_path: Path, state: str
) -> list[str]:
    """The ids of `state`'s bills the document-type rules match, sorted: the bills a hosted
    site has a page for."""
    rows: list[tuple[str]] = con.execute(
        "SELECT b.id FROM read_parquet($bills) b "
        "SEMI JOIN (SELECT id FROM read_parquet($eligible) WHERE eligible) e ON b.id = e.id "
        "WHERE b.state = $state ORDER BY 1",
        {"bills": bills_glob, "eligible": str(eligible_path), "state": state},
    ).fetchall()
    return [bill for (bill,) in rows]


ExportBillRow = tuple[
    str,  # id
    str,  # state
    str,  # session
    str,  # identifier
    str,  # title
    str | None,  # status
    str | None,  # status_reason
    date | None,  # status_date
    str | None,  # first action's day
    str | None,  # text_source
    str | None,  # text_version
    str | None,  # text_url
    bool,  # the text is a Georgia local notice alone
    bool,  # the text opens like an amendment and does not replace the bill
]


def export_bills(
    con: duckdb.DuckDBPyConnection,
    *,
    bills_glob: str,
    texts_glob: str,
    ids: list[str],
    notice: str,
    enacting: str,
    amendment_opening: str,
    amendment_chars: int,
    substitute: str,
) -> list[ExportBillRow]:
    """What the export says of each wanted bill: its record, and the text matching read with
    what it is, tested on that text (the texts table), not the export's."""
    rows: list[ExportBillRow] = con.execute(
        """
        SELECT b.id, b.state, b.session, b.identifier, b.title, b.status, b.status_reason,
            b.status_date, list_min(list_transform(b.actions, lambda a: left(a.date, 10))),
            t.text_source, t.text_version, t.text_url,
            coalesce(regexp_matches(t.text, $notice) AND NOT regexp_matches(t.text, $enacting),
                false),
            coalesce(regexp_matches(left(t.text, $chars), $opening)
                AND NOT regexp_matches(t.text, $substitute), false)
        FROM read_parquet($bills) b
        LEFT JOIN read_parquet($texts) t ON t.id = b.id
        WHERE list_contains($ids, b.id)
        ORDER BY b.id
        """,
        {
            "bills": bills_glob,
            "texts": texts_glob,
            "ids": ids,
            "notice": notice,
            "enacting": enacting,
            "opening": amendment_opening,
            "chars": amendment_chars,
            "substitute": substitute,
        },
    ).fetchall()
    return rows


def section_infos(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, ids: list[str]
) -> list[tuple[str, int, str | None, int, str | None]]:
    """(bill id, ordinal, heading, display words, boilerplate kind) of every section of the
    wanted bills, in order."""
    rows: list[tuple[str, int, str | None, int, str | None]] = con.execute(
        "SELECT bill_id, ordinal, heading, len(string_split_regex(trim(text), '\\s+')), "
        "boilerplate FROM read_parquet($sections) WHERE list_contains($ids, bill_id) "
        "ORDER BY bill_id, ordinal",
        {"sections": sections_glob, "ids": ids},
    ).fetchall()
    return rows


def archive_stamps(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, state: str
) -> list[tuple[str, datetime]]:
    """Each of `state`'s sessions with its archive's Generated At."""
    rows: list[tuple[str, datetime]] = con.execute(
        "SELECT DISTINCT session, generated_at FROM read_parquet($bills) "
        "WHERE state = $state ORDER BY 1",
        {"bills": bills_glob, "state": state},
    ).fetchall()
    return rows


def states_indexed(con: duckdb.DuckDBPyConnection, *, bills_glob: str) -> int:
    row: tuple[int] | None = con.execute(
        "SELECT count(DISTINCT state) FROM read_parquet($bills)", {"bills": bills_glob}
    ).fetchone()
    return 0 if row is None else row[0]


# v2 lanes add their SQL in their own region below, never editing a function above or another
# lane's region, so lanes merged in any order do not conflict (DESIGN.md#lanes).

# --- lane B: collection ---

# --- lane C: matching ---


def matchable_sections_of(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path, ids: list[str]
) -> list[tuple[str, int, str, str]]:
    """(bill id, ordinal, display text, normalized text) of the wanted bills' matchable
    sections, in key order: what an origin match's spans and bill share are read from."""
    rows: list[tuple[str, int, str, str]] = con.execute(
        f"SELECT s.bill_id, s.ordinal, s.text, s.normalized {MATCHABLE} "
        "AND s.bill_id IN (SELECT unnest($ids)) ORDER BY s.bill_id, s.ordinal",
        {"sections": sections_glob, "eligible": str(eligible_path), "ids": ids},
    ).fetchall()
    return rows


def split_by_heading_of(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, ids: list[str]
) -> dict[str, bool]:
    """Whether each wanted bill's sections were split at headings or are the window fallback's
    windows of one text (lineage.text.sections.Split.by_heading)."""
    rows: list[tuple[str, bool]] = con.execute(
        "SELECT bill_id, bool_and(by_heading) FROM read_parquet($sections) "
        "WHERE bill_id IN (SELECT unnest($ids)) GROUP BY bill_id",
        {"sections": sections_glob, "ids": ids},
    ).fetchall()
    return dict(rows)


# --- lane D: evals ---

# --- lane L1: incremental core ---

# A bills file's rows in Open States' shape, the inverse of `copy_bills_to_parquet`'s renaming,
# so a day's records and the records they join are read by one reader, as ingest reads them.
_OPEN_STATES_SHAPE = """
    id, session AS legislative_session, identifier, title, chamber, classification,
    list_transform(actions, lambda a: {
        'organization__name': a.body, 'description': a.description, 'date': a.date,
        'classification': a.classification, 'order': a."order"
    }) AS actions,
    versions, text AS raw_text, text_url AS raw_text_url
"""


def bills_files(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str
) -> list[tuple[str, str, str, str]]:
    """(file name, state, session, archive) of every bills file; each holds one archive."""
    rows: list[tuple[str, str, str, str]] = con.execute(
        "SELECT parse_filename(filename), any_value(state), any_value(session), "
        "any_value(archive) FROM read_parquet($bills, filename = true) GROUP BY 1 ORDER BY 1",
        {"bills": bills_glob},
    ).fetchall()
    return rows


def stored_records(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, ids: list[str]
) -> list[tuple[str, str, str, str | None]]:
    """(id, state, session, exported text) of each wanted record already in the corpus."""
    rows: list[tuple[str, str, str, str | None]] = con.execute(
        "SELECT id, state, session, text FROM read_parquet($bills) "
        "WHERE id IN (SELECT unnest($ids)) ORDER BY id",
        {"bills": bills_glob, "ids": ids},
    ).fetchall()
    return rows


def latest_stamp(con: duckdb.DuckDBPyConnection, *, bills_glob: str) -> datetime | None:
    """The latest Generated At any archive in the corpus was stamped with."""
    row: tuple[datetime | None] | None = con.execute(
        "SELECT max(generated_at) FROM read_parquet($bills)", {"bills": bills_glob}
    ).fetchone()
    return row[0] if row else None


def merged_records_json(
    con: duckdb.DuckDBPyConnection,
    *,
    bills_path: Path | None,
    day_path: Path,
    out_path: Path,
    replaced: list[str],
) -> None:
    """One archive's records as a JSON array in Open States' shape, sorted by id: the bills
    file's rows but the `replaced` ones, and the day's records from `day_path` (a JSON array)."""
    stored = (
        f"SELECT {_OPEN_STATES_SHAPE} FROM read_parquet($bills) "
        "WHERE id NOT IN (SELECT unnest($replaced)) UNION ALL BY NAME "
        if bills_path is not None
        else ""
    )
    query = f"""
        COPY (
            {stored}
            SELECT id, legislative_session, identifier, title, chamber, classification, actions,
                   versions, raw_text, raw_text_url
            FROM read_json($day, format = 'array', columns = {struct_literal(BILL_JSON_COLUMNS)})
            ORDER BY id
        ) TO {sql_string(str(out_path))} (FORMAT json, ARRAY true)
    """
    params: dict[str, Any] = {"day": str(day_path)}
    if bills_path is not None:
        params |= {"bills": str(bills_path), "replaced": replaced}
    con.execute(query, params)


def stored_outcome_actions(
    con: duckdb.DuckDBPyConnection, *, bills_path: Path, kinds: list[str], wording: str
) -> list[tuple[str, list[str], list[dict[str, Any]]]]:
    """`outcome_actions` over a bills file rather than an archive: each record's id and
    classification, with the actions its status can turn on, under the export's names."""
    query = f"""
        WITH b AS (SELECT {_OPEN_STATES_SHAPE} FROM read_parquet($bills))
        SELECT id, coalesce(classification, []),
               list_filter(actions, lambda a: list_has_any(a.classification, $kinds)
                                              OR regexp_matches(coalesce(a.description, ''),
                                                                $wording))
        FROM b
    """
    rows: list[tuple[str, list[str], list[dict[str, Any]]]] = con.execute(
        query, {"bills": str(bills_path), "kinds": kinds, "wording": wording}
    ).fetchall()
    return rows


def restamp_bills(
    con: duckdb.DuckDBPyConnection,
    *,
    bills_path: Path,
    out_path: Path,
    generated_at: str,
    status: StatusColumns,
) -> None:
    """A bills file as of a later Generated At: its rows unchanged but their status."""
    query = f"""
        COPY (
            WITH s AS (
                SELECT
                    CAST(unnest($status_id) AS VARCHAR) AS id,
                    CAST(unnest($status) AS VARCHAR) AS status,
                    CAST(unnest($status_reason) AS VARCHAR) AS status_reason,
                    CAST(unnest($status_date) AS DATE) AS status_date
            )
            SELECT b.* REPLACE (
                s.status AS status, s.status_reason AS status_reason,
                s.status_date AS status_date,
                CAST($generated_at AS TIMESTAMP) AS generated_at
            )
            FROM read_parquet($bills) b LEFT JOIN s USING (id)
        ) TO {sql_string(str(out_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(
        query,
        {
            "bills": str(bills_path),
            "generated_at": generated_at,
            "status_id": status.id,
            "status": status.status,
            "status_reason": status.reason,
            "status_date": status.on,
        },
    )


def merge_texts_file(
    con: duckdb.DuckDBPyConnection,
    *,
    texts_path: Path | None,
    bills_path: Path,
    fresh: list[str],
    out_path: Path,
) -> None:
    """A texts file after a day: each record's text as before, but the `fresh` records' (those
    a day added or whose text it replaced), which take the text their record exports, as
    `write_texts` gives it with nothing fetched."""
    stored = (
        "SELECT * FROM read_parquet($texts) WHERE id NOT IN (SELECT unnest($fresh)) "
        "AND id IN (SELECT id FROM read_parquet($bills)) UNION ALL BY NAME "
        if texts_path is not None
        else ""
    )
    query = f"""
        COPY (
            {stored}
            SELECT
                b.id,
                b.text,
                CASE WHEN b.text IS NOT NULL THEN 'openstates' END AS text_source,
                b.text_url,
                list_filter(coalesce(b.versions, []), lambda v: list_contains(
                    list_transform(coalesce(v.links, []), lambda l: l.url), b.text_url
                ))[1].note AS text_version
            FROM read_parquet($bills) AS b
            WHERE b.id IN (SELECT unnest($fresh))
            ORDER BY id
        ) TO {sql_string(str(out_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    params: dict[str, Any] = {"bills": str(bills_path), "fresh": fresh}
    if texts_path is not None:
        params["texts"] = str(texts_path)
    con.execute(query, params)


def texts_of(
    con: duckdb.DuckDBPyConnection, *, texts_path: Path, ids: list[str]
) -> list[tuple[str, str | None]]:
    """(id, text) of the wanted records of one texts file, in the file's order."""
    rows: list[tuple[str, str | None]] = con.execute(
        "SELECT id, text FROM read_parquet($p) WHERE id IN (SELECT unnest($ids))",
        {"p": str(texts_path), "ids": ids},
    ).fetchall()
    return rows


def copy_without(
    con: duckdb.DuckDBPyConnection, *, path: Path, out_path: Path, column: str, ids: list[str]
) -> int:
    """A Parquet file without the rows whose `column` is one of `ids`, in the order stored;
    returns the rows kept."""
    if not re.fullmatch(r"[a-z_]+", column):
        raise ValueError(f"not a column name: {column}")
    row = con.execute(
        f"COPY (SELECT * FROM read_parquet($p) WHERE {column} NOT IN (SELECT unnest($ids))) "
        f"TO {sql_string(str(out_path))} (FORMAT parquet, COMPRESSION zstd)",
        {"p": str(path), "ids": ids},
    ).fetchone()
    return int(row[0]) if row else 0


def files_holding(
    con: duckdb.DuckDBPyConnection, *, glob: str, column: str, ids: list[str]
) -> list[str]:
    """The names of the Parquet files under `glob` with a row whose `column` is one of `ids`."""
    if not re.fullmatch(r"[a-z_]+", column):
        raise ValueError(f"not a column name: {column}")
    rows: list[tuple[str]] = con.execute(
        f"SELECT DISTINCT parse_filename(filename) FROM read_parquet($g, filename = true) "
        f"WHERE {column} IN (SELECT unnest($ids)) ORDER BY 1",
        {"g": glob, "ids": ids},
    ).fetchall()
    return [name for (name,) in rows]


def eligibility_rows(
    con: duckdb.DuckDBPyConnection, *, path: Path
) -> list[tuple[str, bool, int | None]]:
    rows: list[tuple[str, bool, int | None]] = con.execute(
        "SELECT id, eligible, rule FROM read_parquet($p) ORDER BY id", {"p": str(path)}
    ).fetchall()
    return rows


def record_types_of(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, ids: list[str]
) -> list[RecordType]:
    """`record_types` of the wanted records."""
    rows: list[tuple[str, str, str, str]] = con.execute(
        "SELECT id, state, array_to_string(classification, '+'), coalesce(title, '') "
        "FROM read_parquet($bills) WHERE id IN (SELECT unnest($ids)) ORDER BY id",
        {"bills": bills_glob, "ids": ids},
    ).fetchall()
    return [RecordType(*row) for row in rows]


def semantic_text_ids(
    con: duckdb.DuckDBPyConnection, *, texts_glob: str, texts: list[str]
) -> dict[str, int]:
    """The id of each of `texts` the plan already holds."""
    rows: list[tuple[int, str]] = con.execute(
        "SELECT text_id, text FROM read_parquet($t) WHERE text IN (SELECT unnest($texts))",
        {"t": texts_glob, "texts": texts},
    ).fetchall()
    return {text: text_id for text_id, text in rows}


def carried_text_ids(
    con: duckdb.DuckDBPyConnection, *, units_glob: str, text_ids: list[int]
) -> list[int]:
    """Which of `text_ids` some window still carries."""
    rows: list[tuple[int]] = con.execute(
        "SELECT DISTINCT text_id FROM read_parquet($u) WHERE text_id IN (SELECT unnest($ids)) "
        "ORDER BY 1",
        {"u": units_glob, "ids": text_ids},
    ).fetchall()
    return [t for (t,) in rows]


def row_count(con: duckdb.DuckDBPyConnection, *, glob: str) -> int:
    row = con.execute("SELECT count(*) FROM read_parquet($g)", {"g": glob}).fetchone()
    return int(row[0]) if row else 0


def unit_text_ids(con: duckdb.DuckDBPyConnection, *, units_path: Path, ids: list[str]) -> list[int]:
    """The text ids the wanted bills' windows carry in one windows file."""
    rows: list[tuple[int]] = con.execute(
        "SELECT DISTINCT text_id FROM read_parquet($u) WHERE bill_id IN (SELECT unnest($ids))",
        {"u": str(units_path), "ids": ids},
    ).fetchall()
    return [t for (t,) in rows]


# --- lane L2: intake ---


def bills_acted_per_day(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, month: str
) -> list[tuple[str, str, int]]:
    """(day, state, bills) for each day of `month` (YYYY-MM): the distinct bills with an action
    that day, the intake's proxy for the bills the API reports updated."""
    rows: list[tuple[str, str, int]] = con.execute(
        """
        WITH acted AS (
            SELECT DISTINCT state, id, left(a.date, 10) AS day
            FROM (SELECT state, id, unnest(actions) AS a FROM read_parquet($bills))
            WHERE left(a.date, 7) = $month
        )
        SELECT day, state, count(*)::INTEGER FROM acted GROUP BY day, state ORDER BY day, state
        """,
        {"bills": bills_glob, "month": month},
    ).fetchall()
    return rows


def bills_first_acted(con: duckdb.DuckDBPyConnection, *, bills_glob: str, month: str) -> int:
    """Bills whose first action falls in `month` (YYYY-MM): the month's new bills."""
    row: tuple[int] | None = con.execute(
        """
        SELECT count(*)::INTEGER FROM (
            SELECT min(left(a.date, 10)) AS first
            FROM (SELECT state, id, unnest(actions) AS a FROM read_parquet($bills))
            GROUP BY state, id
        ) WHERE left(first, 7) = $month
        """,
        {"bills": bills_glob, "month": month},
    ).fetchone()
    return 0 if row is None else row[0]
