"""The SQL lineage.origins's modules run: SQL lives only in a package's own queries
module, and a package above may call these. pyright cannot see inside SQL strings, so
each query is exercised against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

from pathlib import Path

import duckdb

from lineage.match.queries import MATCHABLE


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
