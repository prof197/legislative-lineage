"""The SQL lineage.export's modules run: SQL lives only in a package's own queries
module, and a package above may call these. pyright cannot see inside SQL strings, so
each query is exercised against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

from datetime import date, datetime
from pathlib import Path

import duckdb


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


def section_headings(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, bill_id: str
) -> dict[int, str | None]:
    """Each section's heading of one bill, by ordinal."""
    rows: list[tuple[int, str | None]] = con.execute(
        "SELECT ordinal, heading FROM read_parquet($sections) WHERE bill_id = $id",
        {"sections": sections_glob, "id": bill_id},
    ).fetchall()
    return dict(rows)


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
