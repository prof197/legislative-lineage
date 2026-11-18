"""The SQL not yet moved to its package's own queries module (M13; DR-4).

pyright cannot see inside SQL strings, so each query here is exercised against committed
fixtures in the tests. See DESIGN.md#dr-4.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from lineage.core.db import sql_string, struct_literal
from lineage.data.queries import BILL_JSON_COLUMNS, RecordType, StatusColumns


def write_semantic_units(
    con: duckdb.DuckDBPyConnection,
    *,
    parquet_path: Path,
    bill_ids: list[str],
    ordinals: list[int],
    windows: list[int],
    text_ids: list[int],
) -> None:
    """The semantic stage's windows, one row each: bill id, section ordinal, window number, and
    the id of its text, shared by every window with the same text."""
    query = f"""
        COPY (
            SELECT
                CAST(unnest($bill_ids) AS VARCHAR) AS bill_id,
                CAST(unnest($ordinals) AS INTEGER) AS ordinal,
                CAST(unnest($windows) AS INTEGER) AS window_number,
                CAST(unnest($text_ids) AS BIGINT) AS text_id
        ) TO {sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(
        query,
        {"bill_ids": bill_ids, "ordinals": ordinals, "windows": windows, "text_ids": text_ids},
    )


def write_semantic_texts(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, first_id: int, texts: list[str]
) -> None:
    """A shard of distinct window texts, ids counting up from `first_id`."""
    query = f"""
        COPY (
            SELECT CAST(unnest($ids) AS BIGINT) AS text_id, CAST(unnest($texts) AS VARCHAR) AS text
        ) TO {sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"ids": list(range(first_id, first_id + len(texts))), "texts": texts})


def read_semantic_texts(con: duckdb.DuckDBPyConnection, *, parquet_path: Path) -> list[str]:
    """A shard's texts, in id order."""
    rows: list[tuple[str]] = con.execute(
        "SELECT text FROM read_parquet($p) ORDER BY text_id", {"p": str(parquet_path)}
    ).fetchall()
    return [t for (t,) in rows]


def semantic_units_of(
    con: duckdb.DuckDBPyConnection, *, units_glob: str, text_ids: list[int]
) -> list[tuple[int, str, int]]:
    """(text id, bill id, section ordinal) of every window whose text is one of `text_ids`,
    each section once per text."""
    rows: list[tuple[int, str, int]] = con.execute(
        """
        SELECT DISTINCT text_id, bill_id, ordinal FROM read_parquet($units)
        WHERE text_id IN (SELECT unnest($ids))
        ORDER BY text_id, bill_id, ordinal
        """,
        {"units": units_glob, "ids": text_ids},
    ).fetchall()
    return rows


def semantic_text_ids_of(
    con: duckdb.DuckDBPyConnection, *, units_glob: str, bill_ids: list[str]
) -> list[tuple[str, int]]:
    """(bill id, text id) of every window of the given bills, each text once per bill."""
    rows: list[tuple[str, int]] = con.execute(
        """
        SELECT DISTINCT bill_id, text_id FROM read_parquet($units)
        WHERE bill_id IN (SELECT unnest($ids))
        ORDER BY bill_id, text_id
        """,
        {"units": units_glob, "ids": bill_ids},
    ).fetchall()
    return rows


def semantic_section_pairs(
    con: duckdb.DuckDBPyConnection,
    *,
    units_glob: str,
    bills_glob: str,
    sections: dict[str, Any],
    neighbors: dict[str, Any],
) -> dict[str, Any]:
    """Cross-state section pairs whose windows carry the same text or neighboring texts, each
    once, at its best cosine and with the window pair that gave it.

    `sections` numbers the signed sections (`section`, `bill_id`, `ordinal`); a window of any
    other section is left out. `neighbors` holds each kept neighbor (`a`, `b`, `cosine`), in
    either direction. A text shared by two windows pairs them at a cosine of 1. Pairs come out
    with left < right, sorted, and ties at the best cosine go to the lowest window pair."""
    con.register("signed_sections", sections)
    con.register("kept_neighbors", neighbors)
    try:
        result = con.execute(
            """
            WITH units AS (
                SELECT u.text_id, s.section, u.window_number, b.state
                FROM read_parquet($units) u
                JOIN signed_sections s ON s.bill_id = u.bill_id AND s.ordinal = u.ordinal
                JOIN (SELECT id, state FROM read_parquet($bills)) b ON b.id = u.bill_id
            ), linked AS (
                SELECT a, b, CAST(cosine AS FLOAT) AS cosine FROM kept_neighbors
                UNION ALL
                SELECT DISTINCT text_id, text_id, CAST(1.0 AS FLOAT) FROM units
            ), expanded AS (
                SELECT
                    least(x.section, y.section) AS l,
                    greatest(x.section, y.section) AS r,
                    linked.cosine,
                    CASE WHEN x.section < y.section THEN x.window_number
                         ELSE y.window_number END AS lw,
                    CASE WHEN x.section < y.section THEN y.window_number
                         ELSE x.window_number END AS rw
                FROM linked
                JOIN units x ON x.text_id = linked.a
                JOIN units y ON y.text_id = linked.b
                WHERE x.state <> y.state
            )
            SELECT
                CAST(l AS UINTEGER) AS l, CAST(r AS UINTEGER) AS r, cosine,
                CAST(lw AS UINTEGER) AS lw, CAST(rw AS UINTEGER) AS rw
            FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY l, r ORDER BY cosine DESC, lw, rw
                ) AS n
                FROM expanded
            )
            WHERE n = 1
            ORDER BY l, r
            """,
            {"units": units_glob, "bills": bills_glob},
        )
        found: dict[str, Any] = result.fetchnumpy()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    finally:
        con.unregister("signed_sections")
        con.unregister("kept_neighbors")
    return found


def semantic_units_left_out(
    con: duckdb.DuckDBPyConnection, *, units_glob: str, sections: dict[str, Any]
) -> tuple[int, int]:
    """Windows, and sections, of the semantic stage that no signed section holds."""
    con.register("signed_sections", sections)
    try:
        rows: list[tuple[int, int]] = con.execute(
            """
            SELECT count(*), count(DISTINCT (u.bill_id, u.ordinal))
            FROM read_parquet($units) u
            ANTI JOIN signed_sections s ON s.bill_id = u.bill_id AND s.ordinal = u.ordinal
            """,
            {"units": units_glob},
        ).fetchall()
    finally:
        con.unregister("signed_sections")
    return int(rows[0][0]), int(rows[0][1])


@dataclass(frozen=True, slots=True)
class BillSummary:
    id: str
    state: str
    session: str
    identifier: str
    title: str
    status: str | None
    family: str
    family_bills: int


def bill_summaries(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, families_path: Path, ids: list[str]
) -> list[BillSummary]:
    """What a query result shows for each wanted bill, with its family and the family's size."""
    rows: list[tuple[str, str, str, str, str, str | None, str, int]] = con.execute(
        """
        WITH families AS (SELECT bill_id, family FROM read_parquet($families)),
        wanted AS (SELECT bill_id, family FROM families WHERE bill_id IN (SELECT unnest($ids))),
        sizes AS (
            SELECT family, count(*) AS bills FROM families
            WHERE family IN (SELECT family FROM wanted) GROUP BY family
        )
        SELECT b.id, b.state, b.session, b.identifier, b.title, b.status, w.family, s.bills
        FROM wanted w
        JOIN read_parquet($bills) b ON b.id = w.bill_id
        JOIN sizes s USING (family)
        """,
        {"families": str(families_path), "bills": bills_glob, "ids": ids},
    ).fetchall()
    return [BillSummary(*row) for row in rows]


@dataclass(frozen=True, slots=True)
class EnactmentBill:
    """A bill an enactment may name, with what the recall eval checks about it."""

    id: str
    state: str
    session: str
    identifier: str
    status: str | None
    status_date: date | None
    passage_chambers: int
    text_chars: int  # the matched text (texts table), ignoring surrounding whitespace
    amendment_only: bool  # opens like an amendment and does not replace the whole bill
    eligible: bool
    family: str | None  # None when the bill has no signed sections


def enactment_bills(
    con: duckdb.DuckDBPyConnection,
    *,
    bills_glob: str,
    texts_glob: str,
    eligible_path: Path,
    families_path: Path,
    states: list[str],
    opening: str,
    chars: int,
    substitute: str,
) -> list[EnactmentBill]:
    """Every bill of the given states, with its passage, text, eligibility and family. The text is
    the one the index matched (the texts table), fetched or exported."""
    passages = "list_filter(actions, lambda a: list_contains(a.classification, 'passage'))"
    rows: list[tuple[Any, ...]] = con.execute(
        f"""
        SELECT
            b.id, b.state, b.session, b.identifier, b.status, b.status_date,
            len(list_distinct(list_transform({passages}, lambda a: a.body))),
            length(trim(coalesce(t.text, ''))),
            coalesce(regexp_matches(left(t.text, $chars), $opening)
                     AND NOT regexp_matches(t.text, $substitute), false),
            coalesce(e.eligible, false),
            f.family
        FROM read_parquet($bills) b
        LEFT JOIN read_parquet($texts) t USING (id)
        LEFT JOIN read_parquet($eligible) e USING (id)
        LEFT JOIN read_parquet($families) f ON f.bill_id = b.id
        WHERE b.state IN (SELECT unnest($states))
        ORDER BY b.id
        """,
        {
            "bills": bills_glob,
            "texts": texts_glob,
            "eligible": str(eligible_path),
            "families": str(families_path),
            "states": states,
            "opening": opening,
            "chars": chars,
            "substitute": substitute,
        },
    ).fetchall()
    return [EnactmentBill(*row) for row in rows]


def listed_bills(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, states: list[str]
) -> list[tuple[Any, ...]]:
    """Every bill of the given states: what matching needs, its title and last passage date."""
    passages = "list_filter(actions, lambda a: list_contains(a.classification, 'passage'))"
    return con.execute(
        f"""
        SELECT
            id, state, session, identifier, status, status_date,
            len(list_distinct(list_transform({passages}, lambda a: a.body))),
            coalesce(title, ''),
            TRY_CAST(list_max(list_transform({passages}, lambda a: left(a.date, 10))) AS DATE)
        FROM read_parquet($bills)
        WHERE state IN (SELECT unnest($states))
        ORDER BY id
        """,
        {"bills": bills_glob, "states": states},
    ).fetchall()


def signature_rows_of(
    con: duckdb.DuckDBPyConnection, *, keys_paths: list[Path], bill_ids: list[str]
) -> list[tuple[int, int, str, int]]:
    """(file index, row within the file, bill id, section ordinal) of every signature row of the
    wanted bills."""
    if not keys_paths or not bill_ids:
        return []
    found: list[tuple[str, int, str, int]] = con.execute(
        """
        SELECT filename, file_row_number, bill_id, ordinal
        FROM read_parquet($paths, filename = true, file_row_number = true)
        WHERE bill_id IN (SELECT unnest($ids))
        ORDER BY filename, file_row_number
        """,
        {"paths": [str(p) for p in keys_paths], "ids": bill_ids},
    ).fetchall()
    index = {str(p): i for i, p in enumerate(keys_paths)}
    return [(index[name], row, bill_id, ordinal) for name, row, bill_id, ordinal in found]


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


_MATCHABLE = """
    FROM read_parquet($sections) s
    SEMI JOIN (SELECT id FROM read_parquet($eligible) WHERE eligible) e ON s.bill_id = e.id
    WHERE s.boilerplate IS NULL
"""


def section_word_counts(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path
) -> list[int]:
    """Display words of every matchable section, in key order."""
    rows: list[tuple[int]] = con.execute(
        f"SELECT len(string_split_regex(trim(s.text), '\\s+')) {_MATCHABLE} "
        "ORDER BY s.bill_id, s.ordinal",
        {"sections": sections_glob, "eligible": str(eligible_path)},
    ).fetchall()
    return [n for (n,) in rows]


def matchable_bill_ids(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path
) -> list[str]:
    """Every bill with at least one matchable section, sorted."""
    rows: list[tuple[str]] = con.execute(
        f"SELECT DISTINCT s.bill_id {_MATCHABLE} ORDER BY 1",
        {"sections": sections_glob, "eligible": str(eligible_path)},
    ).fetchall()
    return [bill for (bill,) in rows]


def sections_at(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, keys: list[tuple[str, int]]
) -> dict[tuple[str, int], tuple[str | None, str, str]]:
    """The heading, display text and normalized text of each wanted (bill id, ordinal) section."""
    rows: list[tuple[str, int, str | None, str, str]] = con.execute(
        """
        SELECT s.bill_id, s.ordinal, s.heading, s.text, s.normalized
        FROM read_parquet($sections) s
        JOIN (SELECT unnest($ids) AS bill_id, unnest($ordinals) AS ordinal) k
            ON s.bill_id = k.bill_id AND s.ordinal = k.ordinal
        """,
        {
            "sections": sections_glob,
            "ids": [b for b, _ in keys],
            "ordinals": [o for _, o in keys],
        },
    ).fetchall()
    return {(b, o): (h, t, n) for b, o, h, t, n in rows}


def section_headings(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, bill_id: str
) -> dict[int, str | None]:
    """Each section's heading of one bill, by ordinal."""
    rows: list[tuple[int, str | None]] = con.execute(
        "SELECT ordinal, heading FROM read_parquet($sections) WHERE bill_id = $id",
        {"sections": sections_glob, "id": bill_id},
    ).fetchall()
    return dict(rows)


def normalized_texts_of(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, bill_ids: list[str]
) -> list[tuple[str, str]]:
    """(bill id, normalized text) of every section of the given bills."""
    rows: list[tuple[str, str]] = con.execute(
        """
        SELECT bill_id, normalized FROM read_parquet($sections)
        WHERE bill_id IN (SELECT unnest($ids))
        """,
        {"sections": sections_glob, "ids": bill_ids},
    ).fetchall()
    return rows


def matchable_section_texts(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path, ids: list[str]
) -> list[tuple[str, int, str]]:
    """(bill id, ordinal, display text) of the wanted bills' matchable sections, in key order."""
    rows: list[tuple[str, int, str]] = con.execute(
        f"SELECT s.bill_id, s.ordinal, s.text {_MATCHABLE} "
        "AND s.bill_id IN (SELECT unnest($ids)) ORDER BY s.bill_id, s.ordinal",
        {"sections": sections_glob, "eligible": str(eligible_path), "ids": ids},
    ).fetchall()
    return rows


def matchable_section_normalized(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path, ids: list[str]
) -> list[tuple[str, int, str]]:
    """(bill id, ordinal, normalized text) of the wanted bills' matchable sections."""
    rows: list[tuple[str, int, str]] = con.execute(
        f"SELECT s.bill_id, s.ordinal, s.normalized {_MATCHABLE} "
        "AND s.bill_id IN (SELECT unnest($ids)) ORDER BY s.bill_id, s.ordinal",
        {"sections": sections_glob, "eligible": str(eligible_path), "ids": ids},
    ).fetchall()
    return rows


def sections_naming(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, phrases: list[str]
) -> list[tuple[str, int]]:
    """(bill id, ordinal) of every section whose normalized text holds one of `phrases`."""
    rows: list[tuple[str, int]] = con.execute(
        "SELECT bill_id, ordinal FROM read_parquet($sections) "
        "WHERE regexp_matches(normalized, $pattern) ORDER BY bill_id, ordinal",
        {"sections": sections_glob, "pattern": "|".join(re.escape(p) for p in phrases)},
    ).fetchall()
    return rows


def related_bill_ids(
    con: duckdb.DuckDBPyConnection,
    *,
    bills_glob: str,
    families_path: Path,
    listed: list[tuple[str, str, str]],
    title_regex: str,
) -> list[str]:
    """Bills named by (state, session, identifier) or whose title matches `title_regex`, ignoring
    case, with every bill in the same family as one of them, sorted."""
    rows: list[tuple[str]] = con.execute(
        """
        WITH listed AS (
            SELECT unnest($states) AS state, unnest($sessions) AS session,
                unnest($identifiers) AS identifier
        ),
        named AS (
            SELECT b.id FROM read_parquet($bills) b
            LEFT JOIN listed l USING (state, session, identifier)
            WHERE l.state IS NOT NULL OR regexp_matches(coalesce(b.title, ''), $regex, 'i')
        ),
        families AS (SELECT bill_id, family FROM read_parquet($families))
        SELECT id FROM named
        UNION
        SELECT f.bill_id FROM families f
        WHERE f.family IN (SELECT family FROM families WHERE bill_id IN (SELECT id FROM named))
        ORDER BY 1
        """,
        {
            "bills": bills_glob,
            "families": str(families_path),
            "states": [s for s, _, _ in listed],
            "sessions": [s for _, s, _ in listed],
            "identifiers": [i for _, _, i in listed],
            "regex": title_regex,
        },
    ).fetchall()
    return [bill for (bill,) in rows]


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
        f"SELECT s.bill_id, s.ordinal, s.text, s.normalized {_MATCHABLE} "
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
