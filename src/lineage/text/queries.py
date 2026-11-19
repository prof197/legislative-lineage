"""The SQL lineage.text's modules run: SQL lives only in a package's own queries
module, and a package above may call these. pyright cannot see inside SQL strings, so
each query is exercised against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from lineage.core.db import sql_string


def write_texts(
    con: duckdb.DuckDBPyConnection, *, bills_path: Path, fetched_path: Path, out_path: Path
) -> None:
    """One bills file's texts: the fetched text where `fetched_path` (JSON lines) has one,
    else the exported text, each with its source, URL and version label."""
    # The exported text's version: the one whose links include the URL it was extracted from.
    query = f"""
        COPY (
            WITH f AS (
                SELECT * FROM read_json($fetched, format = 'newline_delimited', columns = {{
                    'id': 'VARCHAR', 'text': 'VARCHAR', 'source': 'VARCHAR', 'url': 'VARCHAR',
                    'version': 'VARCHAR', 'date': 'VARCHAR'}})
            )
            SELECT
                b.id,
                coalesce(f.text, b.text) AS text,
                CASE WHEN f.id IS NOT NULL THEN f.source
                     WHEN b.text IS NOT NULL THEN 'openstates' END AS text_source,
                CASE WHEN f.id IS NOT NULL THEN f.url ELSE b.text_url END AS text_url,
                CASE WHEN f.id IS NOT NULL THEN f.version
                     ELSE list_filter(coalesce(b.versions, []), lambda v: list_contains(
                         list_transform(coalesce(v.links, []), lambda l: l.url), b.text_url
                     ))[1].note END AS text_version
            FROM read_parquet($bills) AS b
            LEFT JOIN f USING (id)
        ) TO {sql_string(str(out_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"bills": str(bills_path), "fetched": str(fetched_path)})


def text_sources(con: duckdb.DuckDBPyConnection, *, texts_glob: str) -> dict[str, int]:
    """Records per text source; records with no text count under "none"."""
    rows: list[tuple[str | None, int]] = con.execute(
        "SELECT text_source, count(*) FROM read_parquet($t) GROUP BY ALL", {"t": texts_glob}
    ).fetchall()
    return {source or "none": n for source, n in sorted(rows, key=lambda r: r[0] or "")}


def text_bytes(con: duckdb.DuckDBPyConnection, *, bills_glob: str) -> int:
    """Total UTF-8 bytes of text in a table, to bound what a stage writes."""
    row = con.execute(
        "SELECT coalesce(sum(strlen(text)), 0) FROM read_parquet($glob)", {"glob": bills_glob}
    ).fetchone()
    return int(row[0]) if row else 0


def file_state(con: duckdb.DuckDBPyConnection, *, bills_path: Path) -> str | None:
    """The one state of a bills file, which holds one archive; None if it holds no rows."""
    states: list[tuple[str]] = con.execute(
        "SELECT DISTINCT state FROM read_parquet($path)", {"path": str(bills_path)}
    ).fetchall()
    if len(states) > 1:
        raise ValueError(f"{bills_path} holds {len(states)} states")
    return states[0][0] if states else None


def open_bill_texts(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path
) -> duckdb.DuckDBPyConnection:
    """Start a scan of one texts file's ids and texts; read it with `fetchmany`.

    No ORDER BY: sorting would hold the whole file's text in memory before the first row.
    DuckDB keeps a single file's rows in stored order, so the output is still deterministic.
    """
    return con.execute("SELECT id, text FROM read_parquet($path)", {"path": str(parquet_path)})


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
        ) TO {sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
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


@dataclass(frozen=True, slots=True)
class TextPatterns:
    """What makes a record's stored text unusable (see lineage.data.stats for each pattern)."""

    blank: str
    notice: str
    enacting: str
    opening: str
    chars: int
    substitute: str


def text_population(
    con: duckdb.DuckDBPyConnection,
    *,
    bills_glob: str,
    eligible_path: Path,
    sections_glob: str,
    patterns: TextPatterns,
    texts_glob: str | None = None,
) -> list[tuple[str, str, str, str, str, list[dict[str, Any]]]]:
    """Each eligible record's id, state, session, status, why its stored text is unusable
    ('usable' if it is not), and its versions.

    The checks run in this order, so a record counts once: no text, blank text, text that
    yields no section, a Georgia local notice, an amendment stored in place of the bill.
    Whether the stored text yields a section is read from the sections table, which holds the
    split of each record's text in `texts_glob`. Where that text was fetched, not exported, the
    sections say nothing about the stored text, and the kind is 'recheck:<later kind>': the
    caller splits the stored text and keeps the later kind if it yields a section.
    """
    exported = (
        "JOIN read_parquet($texts) AS t ON t.id = x.bill_id AND t.text_source = 'openstates'"
        if texts_glob
        else ""
    )
    fetched = "LEFT JOIN read_parquet($texts) AS t2 ON t2.id = b.id" if texts_glob else ""
    recheck = (
        "WHEN s.bill_id IS NULL AND t2.text_source IS DISTINCT FROM 'openstates' THEN "
        "'recheck:' || CASE WHEN b.state = 'GA' AND regexp_matches(b.text, $notice) "
        "AND NOT regexp_matches(b.text, $enacting) THEN 'ga_notice' "
        "WHEN regexp_matches(left(b.text, $chars), $opening) "
        "AND NOT regexp_matches(b.text, $substitute) THEN 'amendment_only' ELSE 'usable' END"
        if texts_glob
        else ""
    )
    query = f"""
        WITH split AS (
            SELECT DISTINCT x.bill_id FROM read_parquet($sections) AS x {exported}
        )
        SELECT b.id, b.state, b.session, b.status,
            CASE
                WHEN b.text IS NULL THEN 'missing'
                WHEN regexp_full_match(b.text, $blank) THEN 'blank'
                {recheck}
                WHEN s.bill_id IS NULL THEN 'no_sections'
                WHEN b.state = 'GA' AND regexp_matches(b.text, $notice)
                     AND NOT regexp_matches(b.text, $enacting) THEN 'ga_notice'
                WHEN regexp_matches(left(b.text, $chars), $opening)
                     AND NOT regexp_matches(b.text, $substitute) THEN 'amendment_only'
                ELSE 'usable'
            END AS kind,
            coalesce(b.versions, [])
        FROM read_parquet($bills) AS b
        JOIN read_parquet($eligible) AS e ON e.id = b.id AND e.eligible
        LEFT JOIN split AS s ON s.bill_id = b.id
        {fetched}
        ORDER BY b.id
    """
    rows: list[tuple[str, str, str, str, str, list[dict[str, Any]]]] = con.execute(
        query,
        {
            "bills": bills_glob,
            "eligible": str(eligible_path),
            "sections": sections_glob,
            "blank": patterns.blank,
            "notice": patterns.notice,
            "enacting": patterns.enacting,
            "chars": patterns.chars,
            "opening": patterns.opening,
            "substitute": patterns.substitute,
        }
        | ({"texts": texts_glob} if texts_glob else {}),
    ).fetchall()
    return rows


def stored_texts(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, ids: list[str]
) -> dict[str, str]:
    """The exported text of each id."""
    rows: list[tuple[str, str]] = con.execute(
        "SELECT id, text FROM read_parquet($bills) WHERE list_contains($ids, id)",
        {"bills": bills_glob, "ids": ids},
    ).fetchall()
    return dict(rows)


def write_fetch_plan(
    con: duckdb.DuckDBPyConnection,
    *,
    columns: dict[str, list[Any]],
    path: Path,
    fingerprint: str,
) -> None:
    """Write the text layer's fetch plan, one row per record, from parallel lists.

    The fingerprint of what built it goes in the file's key-value metadata, so it cannot be
    separated from the rows.
    """
    names = list(columns)
    select = ", ".join(f"unnest(${n})::VARCHAR AS {n}" for n in names)
    options = f"FORMAT parquet, KV_METADATA {{fingerprint: {sql_string(fingerprint)}}}"
    con.execute(f"COPY (SELECT {select}) TO {sql_string(str(path))} ({options})", columns)


def fetch_plan_fingerprint(con: duckdb.DuckDBPyConnection, *, path: Path) -> str | None:
    row: tuple[bytes] | None = con.execute(
        "SELECT value FROM parquet_kv_metadata($path) WHERE key = 'fingerprint'",
        {"path": str(path)},
    ).fetchone()
    return row[0].decode() if row else None


def fetch_plan(con: duckdb.DuckDBPyConnection, *, path: Path) -> list[dict[str, Any]]:
    cursor = con.execute("SELECT * FROM read_parquet($path) ORDER BY id", {"path": str(path)})
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def identifiers(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, ids: list[str]
) -> list[tuple[str, str, str]]:
    """(id, session, identifier) for each id."""
    rows: list[tuple[str, str, str]] = con.execute(
        "SELECT id, session, identifier FROM read_parquet($bills) "
        "WHERE list_contains($ids, id) ORDER BY id",
        {"bills": bills_glob, "ids": ids},
    ).fetchall()
    return rows


def stored_text(
    con: duckdb.DuckDBPyConnection, *, bills_glob: str, ids: list[str]
) -> dict[str, tuple[str | None, str | None, list[dict[str, Any]]]]:
    """Open States' stored text, the URL it was extracted from, and the versions, per id."""
    rows: list[tuple[str, str | None, str | None, list[dict[str, Any]]]] = con.execute(
        "SELECT id, text, text_url, coalesce(versions, []) FROM read_parquet($bills) "
        "WHERE list_contains($ids, id)",
        {"bills": bills_glob, "ids": ids},
    ).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}
