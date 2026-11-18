"""The SQL lineage.match's modules run: SQL lives only in a package's own queries
module, and a package above may call these. pyright cannot see inside SQL strings, so
each query is exercised against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb


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


MATCHABLE = """
    FROM read_parquet($sections) s
    SEMI JOIN (SELECT id FROM read_parquet($eligible) WHERE eligible) e ON s.bill_id = e.id
    WHERE s.boilerplate IS NULL
"""


def section_word_counts(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path
) -> list[int]:
    """Display words of every matchable section, in key order."""
    rows: list[tuple[int]] = con.execute(
        f"SELECT len(string_split_regex(trim(s.text), '\\s+')) {MATCHABLE} "
        "ORDER BY s.bill_id, s.ordinal",
        {"sections": sections_glob, "eligible": str(eligible_path)},
    ).fetchall()
    return [n for (n,) in rows]


def matchable_section_normalized(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path, ids: list[str]
) -> list[tuple[str, int, str]]:
    """(bill id, ordinal, normalized text) of the wanted bills' matchable sections."""
    rows: list[tuple[str, int, str]] = con.execute(
        f"SELECT s.bill_id, s.ordinal, s.normalized {MATCHABLE} "
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
