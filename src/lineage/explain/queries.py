"""The SQL lineage.explain's modules run: SQL lives only in a package's own queries
module, and a package above may call these. pyright cannot see inside SQL strings, so
each query is exercised against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

import duckdb


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
