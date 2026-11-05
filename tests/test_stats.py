import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pytest

from lineage.ingest import ingest
from lineage.sectioning import build_sections
from lineage.snapshot import scan, write_manifest
from lineage.stats import BLANK_PATTERN, coverage, write_coverage
from tests.conftest import SESSIONS


def test_blank_means_no_character_python_does_not_call_whitespace() -> None:
    whitespace = "".join(chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace())
    con = duckdb.connect()

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
    assert ga["status"] == {
        "bill": {"passed": 1},
        "resolution": {"passed": 1},
    }


def test_reports_the_known_status_gaps(tables: tuple[Path, Path]) -> None:
    out, manifest = tables

    doc = coverage(out, sessions_path=SESSIONS, manifest_path=manifest)

    # SR 1065 was adopted by the Senate, which passes a simple resolution.
    assert doc["status_gaps"]["resolutions_adopted_by_one_chamber"] == {}
    assert doc["status_gaps"]["reached_governor_shown_died"] == {}
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
