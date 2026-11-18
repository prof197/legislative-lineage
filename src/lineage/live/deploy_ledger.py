"""The deploy's ledger: one record per day the daily job deployed or tried to, under
`<data>/deploy/days/` (lineage.live.deploy).

See DESIGN.md#updates, DR-13.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from lineage.core.paths import DATA

DEPLOY_DIR = DATA / "deploy"


class DeployError(Exception):
    pass


# The ledger: one record per day the daily job deployed or tried to.

DayState = Literal["deployed", "held", "released", "rejected", "failed", "superseded"]


class DayRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: date
    state: DayState
    tag: str | None = None
    asset: str | None = None
    # Held on its own counts, as against held behind an earlier held day.
    held_on_counts: bool = False
    held_behind: date | None = None
    reasons: list[str] = []
    counts: dict[str, int] = {}
    baseline: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    done: list[str] = []  # the publishing steps done, so a resumed run skips them
    error: str | None = None
    updated: str = ""


@dataclass(frozen=True, slots=True)
class DeployPaths:
    root: Path = DEPLOY_DIR
    # The job's own clone of the repository on main: it builds the site, commits the pin and
    # pushes from here, never from a checkout someone works in.
    checkout: Path = DEPLOY_DIR / "checkout"

    @property
    def days(self) -> Path:
        return self.root / "days"

    @property
    def candidates(self) -> Path:
        return self.root / "candidates"

    @property
    def reviews(self) -> Path:
        return self.root / "reviews"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def read_record(paths: DeployPaths, day: date) -> DayRecord | None:
    path = paths.days / f"{day.isoformat()}.json"
    return DayRecord.model_validate_json(path.read_text()) if path.exists() else None


def write_record(paths: DeployPaths, record: DayRecord) -> Path:
    paths.days.mkdir(parents=True, exist_ok=True)
    record.updated = _now()
    path = paths.days / f"{record.day.isoformat()}.json"
    partial = path.with_name(path.name + ".partial")
    partial.write_text(record.model_dump_json(indent=1) + "\n")
    partial.replace(path)
    return path


def records(paths: DeployPaths) -> list[DayRecord]:
    if not paths.days.exists():
        return []
    return sorted(
        (DayRecord.model_validate_json(p.read_text()) for p in paths.days.glob("*.json")),
        key=lambda r: r.day,
    )


def blocking_day(paths: DeployPaths, day: date) -> DayRecord | None:
    """The earliest day before `day` held on its own counts and not yet released."""
    return next(
        (r for r in records(paths) if r.day < day and r.state == "held" and r.held_on_counts),
        None,
    )
