"""`lineage stats`: what the snapshot actually contains, per state and session.

The design's estimates (text coverage, how often headings split a bill, sections per bill) are
replaced by these measurements, and the document-type counts are the evidence for DR-5's type
rules. Every figure is computed from the tables, never typed in. See DESIGN.md#dr-5.
"""

import json
from pathlib import Path
from typing import Any

from lineage import queries
from lineage.sessions import CARRYOVER, load_sessions, status_end_date
from lineage.stoplist import file_sha256

COMMAND = "uv run lineage stats"
# Blank means no character that Python's str.isspace() would not call whitespace. RE2's \s is
# ASCII-only, so the Unicode separators and the remaining control whitespace are listed.
BLANK_PATTERN = r"[\s\p{Z}\x{0b}\x{1c}-\x{1f}\x{85}]*"
# Titles that announce a ceremonial measure. These are counted as candidates, as evidence for
# the per-state type rules, and never used as a filter by themselves.
COMMEMORATIVE_TITLE = (
    r"(?i)\b(commend|congratulat|condolence|mourn|in memory of|honou?ring|recogniz(e|es|ing))"
)


def _nest(rows: list[tuple[str, str, int]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    for outer, inner, n in rows:
        nested.setdefault(outer, {})[inner] = n
    return nested


def coverage(tables_dir: Path, *, sessions_path: Path, manifest_path: Path) -> dict[str, Any]:
    bills = str(tables_dir / "bills" / "*.parquet")
    sections = str(tables_dir / "sections" / "*.parquet")
    session_dates = load_sessions(sessions_path)
    with queries.connect_quiet() as con:
        per_session = queries.coverage_by_session(
            con, bills_glob=bills, sections_glob=sections, blank_pattern=BLANK_PATTERN
        )
        types = queries.type_counts(con, bills_glob=bills)
        commemorative = queries.title_matches(con, bills_glob=bills, pattern=COMMEMORATIVE_TITLE)
        status = queries.status_by_type(con, bills_glob=bills)
        adopted = queries.adopted_resolutions_shown_died(con, bills_glob=bills)
        governor = queries.reached_governor_shown_died(con, bills_glob=bills)
        carryover = [
            {"state": state, "from": first, "to": second}
            | queries.carryover_overlap(
                con, bills_glob=bills, state=state, first=first, second=second
            )
            for (state, first), second in sorted(CARRYOVER.items())
        ]

    states: dict[str, dict[str, Any]] = {}
    unknown_end = {"sessions": 0, "bills_left_introduced": 0}
    totals = dict.fromkeys(
        (
            "records",
            "missing_text",
            "blank_text",
            "text_without_sections",
            "with_sections",
            "split_at_headings",
            "sections",
        ),
        0,
    )
    for row in per_session:
        state, session = row["state"], row["session"]
        known = session_dates.get((state, session))
        end = known.end_date if known else None
        biennium_end = status_end_date(session_dates, state, session)
        if end is None:
            unknown_end["sessions"] += 1
            unknown_end["bills_left_introduced"] += row["introduced"]
        entry = states.setdefault(state, {"sessions": []})
        entry["sessions"].append(
            {
                "session": session,
                "generated_at": row["generated_at"],
                "end_date": end.isoformat() if end else None,
                "status_end_date": biennium_end.isoformat() if biennium_end else None,
                "records": row["records"],
                "missing_text": row["missing_text"],
                "blank_text": row["blank_text"],
                "text_without_sections": row["text_without_sections"],
                "with_sections": row["with_sections"],
                "split_at_headings": row["split_at_headings"],
                "sections": {
                    "total": row["sections"],
                    "median": row["sections_median"],
                    "p90": row["sections_p90"],
                    "max": row["sections_max"],
                },
            }
        )
        for key in totals:
            totals[key] += int(row[key])

    for state, entry in states.items():
        dates = sorted(s["generated_at"][:10] for s in entry["sessions"])
        entry["generated_at"] = {"first": dates[0], "last": dates[-1]}
        entry["types"] = _nest(types).get(state, {})
        entry["commemorative_title_candidates"] = _nest(commemorative).get(state, {})
        entry["status"] = {
            kind: by_status for (s, kind), by_status in sorted(status.items()) if s == state
        }

    return {
        "command": COMMAND,
        "manifest_sha256": file_sha256(manifest_path),
        "sessions_csv_sha256": file_sha256(sessions_path),
        "definitions": {
            "missing_text": "the record has no raw_text",
            "blank_text": "raw_text has no character Python's str.isspace() rejects",
            "text_without_sections": "non-blank text that cleaning reduces to nothing",
            "split_at_headings": "bills split by the heading rule, not the window fallback",
            "commemorative_title_candidates": COMMEMORATIVE_TITLE,
        },
        "totals": totals,
        "sessions_without_end_date": unknown_end,
        "status_gaps": {
            "adopted_resolutions_shown_died": _nest(adopted),
            "reached_governor_shown_died": _nest(governor),
        },
        "carryover": carryover,
        "states": dict(sorted(states.items())),
    }


def write_coverage(
    tables_dir: Path, *, sessions_path: Path, manifest_path: Path, report_path: Path
) -> dict[str, Any]:
    doc = coverage(tables_dir, sessions_path=sessions_path, manifest_path=manifest_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc
