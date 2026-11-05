"""Session start and end dates for the snapshot, pinned as a committed CSV.

The JSON export has no session dates, but bill status needs to know whether a session has
ended. Open States keeps each state's sessions, with dates, as a literal list in its scraper
repository. This module reads that list at a pinned commit, parsing the source rather than
importing it, and keeps only dates and names. See DESIGN.md#data-snapshot.
"""

import ast
import csv
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from lineage.fetch import open_url

SCRAPERS_REPO = "https://github.com/openstates/openstates-scrapers"
_RAW_URL = "https://raw.githubusercontent.com/openstates/openstates-scrapers/{commit}/scrapers/{state}/__init__.py"
# A scraper module is tens of kilobytes; anything far larger is not the file we expect.
_MAX_SOURCE_BYTES = 1 << 20
_COLUMNS = ["state", "session", "name", "start_date", "end_date", "carries_over_to"]

# Legislatures with two-year terms that Open States publishes one year at a time, although
# bills carry over from the first year to the second. A first-year bill is not dead until the
# second year ends. Each pair is confirmed by the share of first-year bills that reappear in
# the second-year archive, which `lineage stats` reports.
CARRYOVER: dict[tuple[str, str], str] = {
    ("HI", "2023 Regular Session"): "2024",
    ("HI", "2025"): "2026",
    ("NH", "2023"): "2024",
    ("NH", "2025"): "2026",
    ("OK", "2023"): "2024",
    ("OK", "2025"): "2026",
    ("VA", "2024"): "2025",
}


class SessionsError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Session:
    state: str
    session: str
    name: str | None
    start_date: date | None
    end_date: date | None
    carries_over_to: str | None


def parse_legislative_sessions(source: str) -> list[dict[str, Any]]:
    """The `legislative_sessions` list from a scraper module, evaluated as a literal only."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "legislative_sessions" for t in node.targets
        ):
            try:
                sessions: list[dict[str, Any]] = ast.literal_eval(node.value)
            except ValueError:
                raise SessionsError("legislative_sessions is not a literal list") from None
            return sessions
    raise SessionsError("no legislative_sessions assignment")


def fetch_scraper_sessions(
    states: list[str], *, commit: str, pause: float = 0.5
) -> dict[str, list[dict[str, Any]]]:
    """Each state's session list from openstates-scrapers at `commit`, one request at a time."""
    found: dict[str, list[dict[str, Any]]] = {}
    for i, state in enumerate(states):
        if i:
            time.sleep(pause)
        with open_url(_RAW_URL.format(commit=commit, state=state.lower()), pause) as response:
            body = response.read(_MAX_SOURCE_BYTES + 1)
        if len(body) > _MAX_SOURCE_BYTES:
            raise SessionsError(f"{state}: scraper module larger than {_MAX_SOURCE_BYTES} bytes")
        try:
            found[state] = parse_legislative_sessions(body.decode())
        except SessionsError as e:
            raise SessionsError(f"{state}: {e}") from None
    return found


def _date(value: object) -> date | None:
    return date.fromisoformat(value) if isinstance(value, str) and value else None


def session_rows(
    scraped: dict[str, list[dict[str, Any]]],
    wanted: list[tuple[str, str]],
    *,
    carryover: dict[tuple[str, str], str],
) -> list[Session]:
    """One row per wanted (state, session); a session the scrapers lack gets empty fields."""
    rows: list[Session] = []
    for state, session in wanted:
        meta = {s["identifier"]: s for s in scraped.get(state, [])}.get(session, {})
        rows.append(
            Session(
                state=state,
                session=session,
                name=meta.get("name"),
                start_date=_date(meta.get("start_date")),
                end_date=_date(meta.get("end_date")),
                carries_over_to=carryover.get((state, session)),
            )
        )
    return rows


def write_sessions(rows: list[Session], path: Path, *, source: str, commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(f"# Session dates from {source} at commit {commit}\n")
        f.write(f"# Regenerate with: uv run lineage sessions --commit {commit}\n")
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(_COLUMNS)
        for r in rows:
            writer.writerow(
                [
                    r.state,
                    r.session,
                    r.name or "",
                    r.start_date.isoformat() if r.start_date else "",
                    r.end_date.isoformat() if r.end_date else "",
                    r.carries_over_to or "",
                ]
            )


def load_sessions(path: Path) -> dict[tuple[str, str], Session]:
    with path.open(newline="") as f:
        lines = (line for line in f if not line.startswith("#"))
        return {
            (r["state"], r["session"]): Session(
                state=r["state"],
                session=r["session"],
                name=r["name"] or None,
                start_date=_date(r["start_date"]),
                end_date=_date(r["end_date"]),
                carries_over_to=r["carries_over_to"] or None,
            )
            for r in csv.DictReader(lines)
        }


def status_end_date(
    sessions: dict[tuple[str, str], Session], state: str, session: str
) -> date | None:
    """When a bill from this session stops being alive: the session's end, or its biennium's."""
    row = sessions.get((state, session))
    if row is None:
        return None
    if row.carries_over_to is not None:
        return status_end_date(sessions, state, row.carries_over_to)
    return row.end_date
