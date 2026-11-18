"""The SQL lineage.match.semantic's modules run: SQL lives only in a package's own queries
module, and a package above may call these. pyright cannot see inside SQL strings, so
each query is exercised against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

from pathlib import Path
from typing import Any

import duckdb

from lineage.core.db import sql_string


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
