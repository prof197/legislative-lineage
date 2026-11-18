"""The SQL lineage.evals's modules run: SQL lives only in a package's own queries
module, and a package above may call these. pyright cannot see inside SQL strings, so
each query is exercised against committed fixtures in the tests. See DESIGN.md#dr-4.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb


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
