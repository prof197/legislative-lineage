from datetime import date
from pathlib import Path

import pytest

from lineage.sessions import (
    CARRYOVER,
    Session,
    SessionsError,
    load_sessions,
    parse_legislative_sessions,
    session_rows,
    status_end_date,
    write_sessions,
)
from lineage.snapshot import load_manifest

# Shaped like an openstates-scrapers state module: imports and a class around a literal list.
SCRAPER_SOURCE = """
from openstates.scrape import State
from .bills import XXBillScraper


class Exampleland(State):
    scrapers = {"bills": XXBillScraper}
    legislative_sessions = [
        {
            "_scraped_name": "2025 Regular Session",
            "identifier": "2025",
            "name": "2025 Regular Session",
            "start_date": "2025-01-13",
            "end_date": "2025-05-30",
        },
        {
            "identifier": "2026",
            "name": "2026 Regular Session",
            "start_date": "2026-01-12",
            "end_date": "2026-05-29",
            "active": True,
        },
        {"identifier": "2026S1", "name": "2026 First Special Session", "start_date": "2026-09-01"},
    ]

    def get_session_list(self):
        return [s["_scraped_name"] for s in self.legislative_sessions]
"""


def test_reads_the_session_list_without_running_the_module() -> None:
    sessions = parse_legislative_sessions(SCRAPER_SOURCE)

    assert [s["identifier"] for s in sessions] == ["2025", "2026", "2026S1"]
    assert sessions[0]["end_date"] == "2025-05-30"


def test_rejects_a_session_list_that_is_not_a_literal() -> None:
    source = "class X:\n    legislative_sessions = load_sessions()\n"

    with pytest.raises(SessionsError, match="not a literal"):
        parse_legislative_sessions(source)


def test_keeps_only_snapshot_sessions_and_leaves_unknown_dates_empty() -> None:
    rows = session_rows(
        {"XX": parse_legislative_sessions(SCRAPER_SOURCE)},
        [("XX", "2026"), ("XX", "2026S1"), ("XX", "1999")],
        carryover={},
    )

    assert rows == [
        Session("XX", "2026", "2026 Regular Session", date(2026, 1, 12), date(2026, 5, 29), None),
        Session("XX", "2026S1", "2026 First Special Session", date(2026, 9, 1), None, None),
        Session("XX", "1999", None, None, None, None),
    ]


def test_csv_round_trips_and_names_its_source(tmp_path: Path) -> None:
    path = tmp_path / "sessions.csv"
    rows = [
        Session("XX", "2025", "2025 Regular Session", date(2025, 1, 13), date(2025, 5, 30), "2026"),
        Session("XX", "2026", "2026 Regular Session", date(2026, 1, 12), date(2026, 5, 29), None),
        Session("XX", "1999", None, None, None, None),
    ]

    write_sessions(rows, path, source="https://github.com/example/scrapers", commit="abc123")

    assert load_sessions(path) == {(s.state, s.session): s for s in rows}
    header = path.read_text().splitlines()[:2]
    assert any("https://github.com/example/scrapers" in line for line in header)
    assert any("abc123" in line for line in header)


def test_a_session_that_carries_over_ends_with_its_biennium() -> None:
    sessions = {
        (s.state, s.session): s
        for s in [
            Session("XX", "2025", "2025", date(2025, 1, 13), date(2025, 5, 30), "2026"),
            Session("XX", "2026", "2026", date(2026, 1, 12), date(2026, 5, 29), None),
            Session("XX", "2026S1", "Special", date(2026, 9, 1), date(2026, 9, 3), None),
        ]
    }

    assert status_end_date(sessions, "XX", "2025") == date(2026, 5, 29)
    assert status_end_date(sessions, "XX", "2026") == date(2026, 5, 29)
    # Special sessions keep their own dates.
    assert status_end_date(sessions, "XX", "2026S1") == date(2026, 9, 3)


def test_an_unknown_end_date_stays_unknown() -> None:
    sessions = {("XX", "2026S1"): Session("XX", "2026S1", "Special", date(2026, 9, 1), None, None)}

    assert status_end_date(sessions, "XX", "2026S1") is None
    assert status_end_date(sessions, "XX", "not-in-the-csv") is None


SNAPSHOT = Path(__file__).parents[1] / "snapshot"


def test_the_committed_csv_covers_every_archive_and_carryover_target() -> None:
    sessions = load_sessions(SNAPSHOT / "sessions.csv")

    assert {(a.state, a.session) for a in load_manifest(SNAPSHOT / "manifest.json")} <= set(
        sessions
    )
    assert all((state, target) in sessions for (state, _), target in CARRYOVER.items())
