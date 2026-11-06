"""The resolved answer key: every commission list row and the corpus bills it resolved to.

Recall is only as good as the mapping from a list row to the bills it names, so the mapping is
committed in `reports/answer_key.csv` for checking by eye. Each named bill gets its Open States
id, title, chambers passed and last passage date, the rule that found it, and the listed number
it matched. Rows that are not counted are kept, with the reason. Resolution is
`lineage.ulc.find_bills`, the function the recall eval uses, so the two cannot disagree. See
DESIGN.md#commission-recall.
"""

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from lineage import queries
from lineage.commission import FIRST_YEAR, excluded_reason, session_years
from lineage.sessions import load_sessions
from lineage.ulc import (
    ENACTMENTS,
    BillRecord,
    Snapshot,
    canonical_number,
    find_bills,
    load_enactments,
)

ANSWER_KEY = Path("reports/answer_key.csv")
COMMAND = "uv run lineage eval answer-key"
COLUMNS: tuple[str, ...] = (
    "act",
    "jurisdiction",
    "year",
    "listed",
    "listed_status",
    "counted",
    "rule",
    "matched_number",
    "openstates_id",
    "session",
    "identifier",
    "title",
    "status",
    "chambers_passed",
    "last_passage_on",
)


@dataclass(frozen=True, slots=True)
class KeyBill:
    title: str
    last_passage_on: date | None  # the last recorded passage action, in either chamber


def answer_key_rows(
    snapshot: Snapshot,
    bills: list[BillRecord],
    details: dict[str, KeyBill],
    session_years: dict[tuple[str, str], tuple[int, int]],
    *,
    first_year: int = FIRST_YEAR,
) -> list[dict[str, str]]:
    """One row per named bill of each list row; one row with no bill when none resolved."""
    out: list[dict[str, str]] = []
    for row in snapshot.rows:
        base = {
            "act": row.act,
            "jurisdiction": row.jurisdiction,
            "year": str(row.year),
            "listed": row.bill_cell,
            "listed_status": row.status,
        }
        why = "listed as introduced" if row.stratum is None else excluded_reason(row, first_year)
        if why is not None:
            out.append(_row(base, counted=f"no: {why}"))
            continue
        found = find_bills(row, bills, session_years)
        if not found.bills:
            out.append(_row(base, counted="yes"))
            continue
        listed = {canonical_number(n): n for n in row.bill_numbers}
        for bill in found.bills:
            detail = details[bill.id]
            out.append(
                _row(
                    base,
                    counted="yes",
                    rule=found.rule,
                    matched_number=listed[canonical_number(bill.identifier)],
                    openstates_id=bill.id,
                    session=bill.session,
                    identifier=bill.identifier,
                    title=detail.title,
                    status=bill.status or "",
                    chambers_passed=str(bill.passage_chambers),
                    last_passage_on=(
                        detail.last_passage_on.isoformat() if detail.last_passage_on else ""
                    ),
                )
            )
    return out


def _row(base: dict[str, str], **fields: str) -> dict[str, str]:
    return {c: base.get(c, fields.get(c, "")) for c in COLUMNS}


def build_answer_key(
    tables_dir: Path, *, enactments_path: Path = ENACTMENTS, sessions_path: Path
) -> tuple[str, list[dict[str, str]]]:
    """The answer key from the bills table: needs ingest only, not the index."""
    snapshot = load_enactments(enactments_path)
    states = sorted({r.state for r in snapshot.rows if r.state is not None})
    with queries.connect_quiet() as con:
        found = queries.listed_bills(
            con, bills_glob=str(tables_dir / "bills" / "*.parquet"), states=states
        )
    bills = [BillRecord(*row[:7]) for row in found]
    details = {row[0]: KeyBill(row[7], row[8]) for row in found}
    rows = answer_key_rows(snapshot, bills, details, session_years(load_sessions(sessions_path)))
    return f"{COMMAND}; {snapshot.read_note}", rows


def render_answer_key(rows: list[dict[str, str]], note: str) -> str:
    buffer = io.StringIO()
    buffer.write(f"# {note}\n")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def read_answer_key(path: Path) -> tuple[str, list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    note = lines[0].removeprefix("#").strip() if lines and lines[0].startswith("#") else ""
    return note, list(csv.DictReader(line for line in lines if not line.startswith("#")))
