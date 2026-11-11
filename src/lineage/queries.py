"""All SQL the pipeline runs, kept in one module so it can be reviewed and tested as a unit.

pyright cannot see inside SQL strings, so each query here is exercised against committed
fixtures in the tests. See DESIGN.md#dr-4.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from lineage.disk import spill_budget

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


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _struct_literal(columns: dict[str, str]) -> str:
    return "{" + ", ".join(f"{_sql_string(k)}: {_sql_string(v)}" for k, v in columns.items()) + "}"


def _cap_spill(con: duckdb.DuckDBPyConnection, directory: Path) -> None:
    # DuckDB lets a query that outgrows memory spill up to 90% of the free disk. Capping it at
    # the floor makes such a query fail with an out-of-memory error instead of filling the disk.
    # The cap is per connection; the worker processes scan one file each and do not spill.
    con.execute(f"SET max_temp_directory_size = '{spill_budget(directory)}B'")


def connect_quiet() -> duckdb.DuckDBPyConnection:
    """A connection that prints no progress bar, which would interleave across processes."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    # Without a temp_directory setting, DuckDB spills to .tmp under the working directory.
    _cap_spill(con, Path.cwd())
    return con


def connect(*, memory_limit: str, temp_directory: Path) -> duckdb.DuckDBPyConnection:
    con = connect_quiet()
    con.execute(f"SET memory_limit = {_sql_string(memory_limit)}")
    # Spill to the data disk, which the caller has checked for space, not the system temp dir.
    con.execute(f"SET temp_directory = {_sql_string(str(temp_directory))}")
    _cap_spill(con, temp_directory)
    return con


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
                       columns = {_struct_literal(BILL_JSON_COLUMNS)})
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
                           columns = {_struct_literal(BILL_JSON_COLUMNS)}) AS b
            LEFT JOIN s USING (id)
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
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
        ) TO {_sql_string(str(out_path))} (FORMAT parquet, COMPRESSION zstd)
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
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
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


def write_stoplist(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, hashes: list[int], sections: list[int]
) -> None:
    query = f"""
        COPY (
            SELECT
                CAST(unnest($hashes) AS UINTEGER) AS hash,
                CAST(unnest($sections) AS BIGINT) AS sections
            ORDER BY hash
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
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
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"bill_ids": bill_ids, "ordinals": ordinals, "windows": windows})


def read_signature_keys(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path
) -> tuple[list[str], list[int], list[int]]:
    rows: list[tuple[str, int, int]] = con.execute(
        "SELECT bill_id, ordinal, window_number FROM read_parquet($p)", {"p": str(parquet_path)}
    ).fetchall()
    return [b for b, _, _ in rows], [o for _, o, _ in rows], [w for _, _, w in rows]


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
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
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
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
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


def signature_bill_ids_at(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, rows: list[int]
) -> dict[int, str]:
    """The bill id of each wanted row of one signature keys file, by row number."""
    found: list[tuple[int, str]] = con.execute(
        """
        SELECT file_row_number, bill_id FROM read_parquet($p, file_row_number = true)
        WHERE file_row_number IN (SELECT unnest($rows))
        """,
        {"p": str(parquet_path), "rows": rows},
    ).fetchall()
    return dict(found)


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
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"ids": ids, "eligible": eligible, "rules": rules})


def write_families(
    con: duckdb.DuckDBPyConnection, *, parquet_path: Path, bill_ids: list[str], families: list[str]
) -> None:
    query = f"""
        COPY (
            SELECT
                CAST(unnest($ids) AS VARCHAR) AS bill_id,
                CAST(unnest($families) AS VARCHAR) AS family
        ) TO {_sql_string(str(parquet_path))} (FORMAT parquet, COMPRESSION zstd)
    """
    con.execute(query, {"ids": bill_ids, "families": families})


def read_families(con: duckdb.DuckDBPyConnection, *, parquet_path: Path) -> dict[str, str]:
    rows: list[tuple[str, str]] = con.execute(
        "SELECT bill_id, family FROM read_parquet($p)", {"p": str(parquet_path)}
    ).fetchall()
    return dict(rows)


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


@dataclass(frozen=True, slots=True)
class TextPatterns:
    """What makes a record's stored text unusable (see lineage.stats for each pattern)."""

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
    options = f"FORMAT parquet, KV_METADATA {{fingerprint: {_sql_string(fingerprint)}}}"
    con.execute(f"COPY (SELECT {select}) TO {_sql_string(str(path))} ({options})", columns)


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
