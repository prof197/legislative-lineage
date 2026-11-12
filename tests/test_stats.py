import json
import sys
from pathlib import Path
from typing import Any

import pytest

from lineage import queries
from lineage.ingest import ingest
from lineage.sectioning import build_sections
from lineage.snapshot import scan, write_manifest
from lineage.stats import (
    BLANK_PATTERN,
    amendment_only,
    coverage,
    georgia_local_notices,
    write_coverage,
)
from lineage.text_merge import merge_texts
from tests.conftest import SESSIONS


def test_blank_means_no_character_python_does_not_call_whitespace() -> None:
    whitespace = "".join(chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace())
    con = queries.connect_quiet()

    def blank(text: str) -> bool:
        row = con.execute("SELECT regexp_full_match($t, $p)", {"t": text, "p": BLANK_PATTERN})
        result: bool = row.fetchall()[0][0]
        return result

    assert blank("")
    assert blank(whitespace)
    # A zero-width space is invisible but is not whitespace, so text holding one is not blank;
    # the splitter's cleaning reports it separately, as text that yields no sections.
    assert not blank(" \u200b ")
    assert not blank("\u00a0x\u3000")


@pytest.fixture
def tables(raw_dir: Path, tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.json"
    write_manifest(scan(raw_dir), manifest)
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)
    merge_texts(out, exported_only=True, floor_bytes=0)
    build_sections(out, workers=1, floor_bytes=0)
    return out, manifest


def _session(doc: dict[str, Any], state: str) -> dict[str, Any]:
    (session,) = doc["states"][state]["sessions"]
    return session


def test_reports_text_coverage_and_sections_per_session(tables: tuple[Path, Path]) -> None:
    out, manifest = tables

    doc = coverage(out, sessions_path=SESSIONS, manifest_path=manifest)

    assert doc["command"] == "uv run lineage stats"
    assert {k: _session(doc, "AK")[k] for k in ("records", "missing_text", "blank_text")} == {
        "records": 1,
        "missing_text": 0,
        "blank_text": 1,
    }
    assert _session(doc, "HI")["missing_text"] == 1
    ga = _session(doc, "GA")
    assert ga["generated_at"] == "2026-07-25 06:17:51.59996"
    assert ga["end_date"] == "2026-04-02"
    assert (ga["records"], ga["with_sections"], ga["split_at_headings"]) == (2, 2, 1)
    assert ga["sections"] == {"total": 4, "median": 2, "p90": 3, "max": 3}
    assert doc["totals"]["records"] == 5
    assert doc["totals"]["with_sections"] == 3


def test_reports_types_status_and_freshness_per_state(tables: tuple[Path, Path]) -> None:
    out, manifest = tables

    ga = coverage(out, sessions_path=SESSIONS, manifest_path=manifest)["states"]["GA"]

    assert ga["generated_at"] == {"first": "2026-07-25", "last": "2026-07-25"}
    assert ga["types"] == {"bill": 1, "resolution": 1}
    # SR 1065 recognizes a week; its title says so.
    assert ga["commemorative_title_candidates"] == {"resolution": 1}
    # SB 238 was vetoed the day Georgia records as "Date Signed by Governor" (test_ingest).
    assert ga["status"] == {
        "bill": {"died": 1},
        "resolution": {"passed": 1},
    }


def test_reports_the_known_status_gaps(tables: tuple[Path, Path]) -> None:
    out, manifest = tables

    doc = coverage(out, sessions_path=SESSIONS, manifest_path=manifest)

    # SR 1065 was adopted by the Senate, which passes a simple resolution.
    assert doc["status_gaps"]["resolutions_adopted_by_one_chamber"] == {}
    assert doc["status_gaps"]["reached_governor_shown_died"] == {}
    assert doc["status_gaps"]["enacted_in_words"] == {}
    assert doc["status_gaps"]["passed_legislature"] == {}
    assert doc["sessions_without_end_date"] == {"sessions": 0, "bills_left_introduced": 0}


def test_counts_bills_left_introduced_by_a_missing_end_date(
    tables: tuple[Path, Path], tmp_path: Path
) -> None:
    out, manifest = tables
    without_ak = tmp_path / "sessions.csv"
    lines = SESSIONS.read_text().splitlines()
    without_ak.write_text("\n".join(line for line in lines if not line.startswith("AK,")) + "\n")
    ingest(out.parent / "raw", manifest, without_ak, out, floor_bytes=0)

    doc = coverage(out, sessions_path=without_ak, manifest_path=manifest)

    assert doc["sessions_without_end_date"] == {"sessions": 1, "bills_left_introduced": 1}


def test_writes_the_report(tables: tuple[Path, Path], tmp_path: Path) -> None:
    out, manifest = tables
    report = tmp_path / "reports" / "coverage.json"

    write_coverage(out, sessions_path=SESSIONS, manifest_path=manifest, report_path=report)

    assert json.loads(report.read_text())["totals"]["records"] == 5


NOTICE = (
    "NOTICE OF INTENTION TO INTRODUCE LOCAL LEGISLATION Notice is given that there will be "
    "introduced at the 2025 regular session of the General Assembly of Georgia a bill to "
    "reconstitute the board of education of a county; and for other purposes.\n\nAFFIDAVIT\n"
)
ENACTED = "A BILL TO BE ENTITLED AN ACT\nBE IT ENACTED BY THE GENERAL ASSEMBLY OF GEORGIA:\n"


def test_counts_georgia_local_bills_stored_as_their_notice_alone(tmp_path: Path) -> None:
    bills = tmp_path / "bills.parquet"
    rows = [
        ("GA", "2025_26", NOTICE),  # the notice and affidavit, no bill
        ("GA", "2025_26", " NOTICE   OF\nINTENTION TO INTRODUCE\nLOCAL LEGISLATION\n"),
        ("GA", "2025_26", NOTICE + ENACTED + "SECTION 1.\nThe board shall have five members."),
        ("GA", "2023_24", ENACTED + "SECTION 1.\nA general bill."),
        ("GA", "2023_24", NOTICE),
        ("AL", "2025rs", NOTICE),  # another state's text is not counted
    ]
    queries.connect_quiet().execute(
        "COPY (SELECT * FROM (VALUES "
        + ", ".join(f"('{s}', '{n}', $t{i})" for i, (s, n, _) in enumerate(rows))
        + f") t(state, session, text)) TO '{bills}'",
        {f"t{i}": text for i, (_, _, text) in enumerate(rows)},
    )

    assert georgia_local_notices(str(bills)) == {"2023_24": 1, "2025_26": 2}


def test_reports_georgia_local_notices_per_session(tables: tuple[Path, Path]) -> None:
    out, manifest = tables

    doc = coverage(out, sessions_path=SESSIONS, manifest_path=manifest)

    # SB 238 is a Georgia local bill whose stored text is the bill itself.
    assert doc["georgia_local_notices"] == {"total": 0, "by_session": {}}
    assert "enacting clause" in doc["definitions"]["georgia_local_notices"]


# Openings in each state's own format; only the first six are amendments.
AMENDMENTS = [
    ("LA", 'AMENDMENT NO. 1\nOn page 1, line 8, after "sewerage" delete the remainder of the line'),
    (
        "IL",
        "AMENDMENT TO SENATE BILL 1212\nAMENDMENT NO. ______. Amend Senate Bill 1212 by replacing "
        'everything after the enacting clause with the following:\n"Section 5. The Act ..."',
    ),
    ("PA", 'Amend Bill, page 2, line 7, by striking out the period after "DEFINITIONS)"'),
    ("DE", 'AMEND House Bill No. 117 on line 9 by deleting " When " as it appears therein'),
    (
        "NH",
        "Amend RSA 359-S:1, II as inserted by section 1 of the bill by replacing it with the "
        "following:",
    ),
    # Delaware's amendment synopsis, stored without the amendment it explains.
    ("DE", "\n      \n        SYNOPSIS\n      \n        This Amendment clarifies that the Act"),
    # Alabama's bills open with their own synopsis.
    ("AL", "SYNOPSIS:\nThis bill would provide that an amendment to a county plan"),
    ("MI", 'HOUSE BILL NO. 4748\nA bill to amend 1974 PA 258, entitled "Mental health code,"'),
    ("CA", "california legislature\nAssembly Constitutional Amendment No. 2\nIntroduced by"),
    ("UT", "State Weed Committee Amendments\nGENERAL SESSION\nLONG TITLE"),
    ("NY", "Section 1. Under the authority of state plan amendment #17-0058, the department"),
    ("GA", "LC 36 5429\nTo amend Title 34 of the Official Code of Georgia Annotated"),
    # A marker deep inside a bill is not the document's opening.
    ("LA", "AN ACT\nSection 1. " + "The board shall meet. " * 30 + "AMENDMENT NO. 1 On page 2"),
]


def test_counts_records_whose_stored_text_is_an_amendment(tmp_path: Path) -> None:
    bills = tmp_path / "bills.parquet"
    queries.connect_quiet().execute(
        "COPY (SELECT * FROM (VALUES "
        + ", ".join(f"('{s}', '2025', $t{i})" for i, (s, _) in enumerate(AMENDMENTS))
        + f") t(state, session, text)) TO '{bills}'",
        {f"t{i}": text for i, (_, text) in enumerate(AMENDMENTS)},
    )

    found = amendment_only(str(bills))

    assert found == {
        ("DE", "2025"): (2, 0),
        ("IL", "2025"): (1, 1),  # replaces everything after the enacting clause: a substitute
        ("LA", "2025"): (1, 0),
        ("NH", "2025"): (1, 0),
        ("PA", "2025"): (1, 0),
    }


def test_reports_amendment_only_text_per_session(tables: tuple[Path, Path]) -> None:
    out, manifest = tables

    doc = coverage(out, sessions_path=SESSIONS, manifest_path=manifest)

    assert _session(doc, "GA")["amendment_only"] == {"all": 0, "substitutes": 0}
    assert doc["totals"]["amendment_only"] == 0
    assert "first 400 characters" in doc["definitions"]["amendment_only"]
