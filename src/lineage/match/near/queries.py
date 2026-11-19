"""The SQL lineage.match.near's modules run: SQL lives only in a package's own queries
module, and a package above may call these. pyright cannot see inside SQL strings, so
each query is exercised against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

from pathlib import Path

import duckdb

from lineage.core.db import sql_string


def write_stoplist(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, hashes: list[int], sections: list[int]
) -> None:
    query = f"""
        COPY (
            SELECT
                CAST(unnest($hashes) AS UINTEGER) AS hash,
                CAST(unnest($sections) AS BIGINT) AS sections
            ORDER BY hash
        ) TO {sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"hashes": hashes, "sections": sections})


def matchable_sections(
    con: duckdb.DuckDBPyConnection,
    *,
    sections_path: Path,
    eligible_path: Path,
    display: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Start a scan of one sections file's matchable sections: (bill id, ordinal, normalized),
    or (bill id, ordinal, display text, normalized) with `display`.

    Matchable means the bill's document type is eligible (DR-5) and the section is not flagged
    as boilerplate. Rows come in key order, so anything built from them is reproducible. Read
    it with `fetchmany`.
    """
    text = "s.text, " if display else ""
    return con.execute(
        f"""
        SELECT s.bill_id, s.ordinal, {text}s.normalized
        FROM read_parquet($sections) s
        SEMI JOIN (SELECT id FROM read_parquet($eligible) WHERE eligible) e ON s.bill_id = e.id
        WHERE s.boilerplate IS NULL
        ORDER BY s.bill_id, s.ordinal
        """,
        {"sections": str(sections_path), "eligible": str(eligible_path)},
    )


def window_bound(con: duckdb.DuckDBPyConnection, *, sections_glob: str, stride: int) -> int:
    """At most how many windows the sections yield: one per `stride` of a section's display
    words, plus one."""
    row: tuple[int | None] | None = con.execute(
        "SELECT sum(1 + len(string_split_regex(trim(text), '\\s+')) // $stride) "
        "FROM read_parquet($glob)",
        {"glob": sections_glob, "stride": stride},
    ).fetchone()
    return int(row[0] or 0) if row else 0


def section_word_quantiles(
    con: duckdb.DuckDBPyConnection, *, sections_glob: str, eligible_path: Path
) -> dict[str, dict[str, int]]:
    """Display words per matchable section, over all of them and over those split at headings:
    the median, 90th and 99th percentiles."""
    rows: list[tuple[str, int | None, int | None, int | None]] = con.execute(
        """
        WITH w AS (
            SELECT s.by_heading, len(string_split_regex(trim(s.text), '\\s+')) AS words
            FROM read_parquet($sections) s
            SEMI JOIN (SELECT id FROM read_parquet($eligible) WHERE eligible) e ON s.bill_id = e.id
            WHERE s.boilerplate IS NULL
        )
        SELECT 'all', quantile_disc(words, 0.5), quantile_disc(words, 0.9),
            quantile_disc(words, 0.99) FROM w
        UNION ALL
        SELECT 'split_at_headings', quantile_disc(words, 0.5), quantile_disc(words, 0.9),
            quantile_disc(words, 0.99) FROM w WHERE by_heading
        """,
        {"sections": sections_glob, "eligible": str(eligible_path)},
    ).fetchall()
    return {
        name: {"median": median, "p90": p90 or 0, "p99": p99 or 0}
        for name, median, p90, p99 in sorted(rows)
        if median is not None
    }


def write_signature_keys(
    con: duckdb.DuckDBPyConnection,
    *,
    parquet_path: Path,
    bill_ids: list[str],
    ordinals: list[int],
    windows: list[int],
) -> None:
    """The keys of a signature matrix, row for row: bill id, section ordinal, window number."""
    query = f"""
        COPY (
            SELECT
                CAST(unnest($bill_ids) AS VARCHAR) AS bill_id,
                CAST(unnest($ordinals) AS INTEGER) AS ordinal,
                CAST(unnest($windows) AS INTEGER) AS window_number
        ) TO {sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"bill_ids": bill_ids, "ordinals": ordinals, "windows": windows})


def read_signature_keys(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path
) -> tuple[list[str], list[int], list[int]]:
    rows: list[tuple[str, int, int]] = con.execute(
        "SELECT bill_id, ordinal, window_number FROM read_parquet($p)", {"p": str(parquet_path)}
    ).fetchall()
    return [b for b, _, _ in rows], [o for _, o, _ in rows], [w for _, _, w in rows]


def signature_sections_at(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, rows: list[int]
) -> dict[int, tuple[str, int]]:
    """The bill id and section ordinal of each wanted row of one signature keys file."""
    found: list[tuple[int, str, int]] = con.execute(
        """
        SELECT file_row_number, bill_id, ordinal FROM read_parquet($p, file_row_number = true)
        WHERE file_row_number IN (SELECT unnest($rows))
        """,
        {"p": str(parquet_path), "rows": rows},
    ).fetchall()
    return {row: (bill_id, ordinal) for row, bill_id, ordinal in found}


def bill_states(con: duckdb.DuckDBPyConnection, *, bills_glob: str) -> dict[str, str]:
    """Every bill's state, by bill id."""
    rows: list[tuple[str, str]] = con.execute(
        "SELECT id, state FROM read_parquet($bills)", {"bills": bills_glob}
    ).fetchall()
    return dict(rows)


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


def write_families(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, bill_ids: list[str], families: list[str]
) -> None:
    query = f"""
        COPY (
            SELECT
                CAST(unnest($ids) AS VARCHAR) AS bill_id,
                CAST(unnest($families) AS VARCHAR) AS family
        ) TO {sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"ids": bill_ids, "families": families})


def read_families(con: duckdb.DuckDBPyConnection, *, parquet_path: Path) -> dict[str, str]:
    rows: list[tuple[str, str]] = con.execute(
        "SELECT bill_id, family FROM read_parquet($p)", {"p": str(parquet_path)}
    ).fetchall()
    return dict(rows)
