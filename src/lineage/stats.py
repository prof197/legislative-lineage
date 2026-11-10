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
from lineage.status import ENACTED_IN_WORDS, Status
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
# Georgia publishes a notice before a local bill is introduced (O.C.G.A. § 28-1-14), and the
# export often stores that notice and its affidavit in place of the bill. Every Georgia bill
# carries the enacting clause, so text with the notice but no clause holds no bill text.
GEORGIA_NOTICE = r"(?i)notice\s+of\s+intention\s+to\s+introduce\s+local\s+legislation"
GEORGIA_ENACTING = r"(?i)be\s+it\s+enacted\s+by\s+the\s+general\s+assembly"
# Some exports store an amendment in place of the bill. An amendment document opens by naming
# itself or the bill it changes, in each chamber's own form: "AMENDMENT NO. 1 On page 1" (LA),
# "AMENDMENT TO SENATE BILL" (IL), "Amend Bill, page 2" (PA), "AMEND House Bill No. 117" (DE),
# "Floor Amend 2" (GA), "FLOOR AMENDMENT No." (OK), "Strike the original sections" (NE), "as
# inserted by section 1 of the bill" (NH), or an amendment's synopsis stored without the
# amendment, "SYNOPSIS This Amendment" (DE). Only the opening is read, since bills quote these
# phrases in their bodies, and a bill's own title ("A bill to amend ...") never matches; nor
# does Alabama's bill synopsis, which describes the bill.
AMENDMENT_CHARS = 400
AMENDMENT_OPENING = (
    r"(?i)\bamendment\s+(?:no\.?|number)\s*[\d_]+\.?\s+"
    r"(?:on\s+page|proposed|amend|delete|strike|offered|by\b|in\s+the\b)"
    r"|\bamendment\s+to\s+(?:house|senate|assembly)\s+(?:bill|joint|concurrent|resolution)"
    r"|\bamendment\s+in\s+the\s+nature\s+of\s+a\s+substitute\b"
    r"|\b(?:floor|committee)\s+amend(?:ment)?s?\s+"
    r"(?:no\.?\s|number\b|\d|to\s+(?:lb|ab|hb|sb|house|senate|assembly)\b)"
    r"|\bamend\s+(?:the\s+(?:bill|title|substitute|amendments?)\b|bill\b|amendments?\b)"
    r"|\bamend\s+(?:house|senate|assembly)\s+(?:bill|substitute|joint|concurrent|resolution)"
    r"|\bamend\s+(?:h|s|a|l)\.?\s?(?:b|r|j\.?\s?r|c\.?\s?r)\.?\s*\d+"
    r"|\bas\s+inserted\s+by\s+section\s+\d+\s+of\s+the\s+bill\b"
    r"|\bstrike\s+(?:the\s+original\s+sections|amendment\s+\d+)"
    r"|\bon\s+page\s+\d+,?\s+(?:line|at|after|delete|strike|by)"
    r"|^\s*synopsis\s+this\s+amendment\b"
)
# An amendment that replaces the whole bill still carries a full text to match.
AMENDMENT_SUBSTITUTE = (
    r"(?i)(?:strik\w*|replac\w*|delet\w*)\s+(?:out\s+)?(?:all|everything)\s+after\s+the\s+"
    r"enacting\s+clause|in\s+the\s+nature\s+of\s+a\s+substitute"
)


def _nest(rows: list[tuple[str, str, int]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = {}
    for outer, inner, n in rows:
        nested.setdefault(outer, {})[inner] = n
    return nested


def georgia_local_notices(bills_glob: str) -> dict[str, int]:
    """Georgia records per session whose stored text is a local bill's notice alone."""
    with queries.connect_quiet() as con:
        rows = queries.notice_only_by_session(
            con,
            bills_glob=bills_glob,
            state="GA",
            notice=GEORGIA_NOTICE,
            enacting=GEORGIA_ENACTING,
        )
    return dict(rows)


def amendment_only(bills_glob: str) -> dict[tuple[str, str], tuple[int, int]]:
    """Per (state, session): records stored as an amendment, and how many replace the bill."""
    with queries.connect_quiet() as con:
        rows = queries.amendment_only_by_session(
            con,
            bills_glob=bills_glob,
            opening=AMENDMENT_OPENING,
            chars=AMENDMENT_CHARS,
            substitute=AMENDMENT_SUBSTITUTE,
        )
    return {(state, session): (n, substitutes) for state, session, n, substitutes in rows}


def _text_layer(rows: list[tuple[str, int, int, int, int, int]]) -> dict[str, Any]:
    states = {
        state: {
            "eligible": eligible,
            "usable_exported": exported,
            "with_sections": merged,
            "with_sections_fetched": fetched,
            "with_sections_bulk": bulk,
        }
        for state, eligible, exported, merged, fetched, bulk in rows
    }
    keys = ("eligible", "usable_exported", "with_sections", "with_sections_fetched",
            "with_sections_bulk")  # fmt: skip
    return {"totals": {k: sum(v[k] for v in states.values()) for k in keys}, "states": states}


def coverage(
    tables_dir: Path,
    *,
    sessions_path: Path,
    manifest_path: Path,
    plan_path: Path = Path("data/text/plan.parquet"),
) -> dict[str, Any]:
    bills = str(tables_dir / "bills" / "*.parquet")
    texts = str(tables_dir / "texts" / "*.parquet")
    sections = str(tables_dir / "sections" / "*.parquet")
    session_dates = load_sessions(sessions_path)
    with queries.connect_quiet() as con:
        per_session = queries.coverage_by_session(
            con,
            bills_glob=bills,
            texts_glob=texts,
            sections_glob=sections,
            blank_pattern=BLANK_PATTERN,
        )
        text_layer = (
            queries.text_layer_coverage(
                con,
                eligible_path=tables_dir / "eligible.parquet",
                texts_glob=texts,
                sections_glob=sections,
                plan_path=plan_path,
            )
            if plan_path.exists() and (tables_dir / "eligible.parquet").exists()
            else []
        )
        types = queries.type_counts(con, bills_glob=bills)
        commemorative = queries.title_matches(con, bills_glob=bills, pattern=COMMEMORATIVE_TITLE)
        status = queries.status_by_type(con, bills_glob=bills)
        one_chamber = queries.resolutions_adopted_by_one_chamber(con, bills_glob=bills)
        governor = queries.reached_governor_shown_died(con, bills_glob=bills)
        in_words = queries.status_reason_counts(con, bills_glob=bills, reason=ENACTED_IN_WORDS)
        finished = queries.status_counts(
            con, bills_glob=bills, status=Status.PASSED_LEGISLATURE.value
        )
        carryover = [
            {"state": state, "from": first, "to": second}
            | queries.carryover_overlap(
                con, bills_glob=bills, state=state, first=first, second=second
            )
            for (state, first), second in sorted(CARRYOVER.items())
        ]

    notices = georgia_local_notices(bills)
    amendments = amendment_only(bills)
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
            "text_fetched",
            "text_from_bulk",
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
                "text_fetched": row["text_fetched"],
                "text_from_bulk": row["text_from_bulk"],
                "amendment_only": dict(
                    zip(
                        ("all", "substitutes"),
                        amendments.get((state, session), (0, 0)),
                        strict=True,
                    )
                ),
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
            "missing_text": "the record has no stored text",
            "blank_text": "the stored text has no character Python's str.isspace() rejects",
            "text_without_sections": (
                "non-blank text that cleaning reduces to nothing or to a notice that the text "
                "is not available"
            ),
            "split_at_headings": "bills split by the heading rule, not the window fallback",
            "text_fetched": "records whose text the text layer fetched from the legislature",
            "text_from_bulk": "records whose text came from California's bulk download",
            "text_layer": (
                "per state, over eligible records: usable exported text (the fetch plan's "
                "kind is usable) against text that yields sections once fetched text is merged"
            ),
            "commemorative_title_candidates": COMMEMORATIVE_TITLE,
            "amendment_only": (
                f"text whose first {AMENDMENT_CHARS} characters open like an amendment "
                f"({AMENDMENT_OPENING}); substitutes also replace the whole bill "
                f"({AMENDMENT_SUBSTITUTE})"
            ),
            "georgia_local_notices": (
                "Georgia records whose text has the notice of intention to introduce local "
                f"legislation ({GEORGIA_NOTICE}) and no enacting clause ({GEORGIA_ENACTING})"
            ),
        },
        "totals": totals
        | {
            "amendment_only": sum(n for n, _ in amendments.values()),
            "amendment_substitutes": sum(m for _, m in amendments.values()),
        },
        "text_layer": _text_layer(text_layer),
        "sessions_without_end_date": unknown_end,
        "status_gaps": {
            "resolutions_adopted_by_one_chamber": _nest(one_chamber),
            "reached_governor_shown_died": _nest(governor),
            "enacted_in_words": _nest(in_words),
            "passed_legislature": _nest(finished),
        },
        "carryover": carryover,
        "georgia_local_notices": {"total": sum(notices.values()), "by_session": notices},
        "states": dict(sorted(states.items())),
    }


def write_coverage(
    tables_dir: Path, *, sessions_path: Path, manifest_path: Path, report_path: Path
) -> dict[str, Any]:
    doc = coverage(tables_dir, sessions_path=sessions_path, manifest_path=manifest_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc
