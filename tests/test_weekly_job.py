"""The weekly job end to end at fixture scale (DESIGN.md#updates, DR-7): `lineage rebuild`
twice over the recorded days, pinned and refreshed, then the agreement check, each step run as
its `lineage` command, and the job's status and report."""

import json
import shutil
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.schedule import (
    STATUS_FILE,
    Outcome,
    Step,
    WeeklyPaths,
    run_job,
    weekly_job_steps,
)
from tests.conftest import FAKE_MODEL
from tests.days import DAYS, sessions_csv
from tests.live import REFRESHED_SHARE, Recorded, recorded

D3 = date.fromisoformat(DAYS[2])


@pytest.fixture(scope="module")
def rec(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    return recorded(tmp_path_factory)


class Alerts:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> None:
        self.sent.append((title, message))


def _cli_runner(root: Path) -> Callable[[Step], Outcome]:
    """Each step as its `lineage` command, in this process, with what the fixture builds with
    added: the days as the whole corpus, its sessions, the stand-in model, exact search."""
    sessions, no_plan = sessions_csv(root), root / "no-plan.parquet"
    fixture = {
        "rebuild": ["--days-only", "--exact", "--model", FAKE_MODEL, "--sessions", str(sessions),
                    "--text-plan", str(no_plan), "--max-share", REFRESHED_SHARE, "--workers", "1",
                    "--min-free-gib", "0"],
        "publish": ["--model", FAKE_MODEL],
    }  # fmt: skip

    def run(step: Step) -> Outcome:
        result = CliRunner().invoke(app, [*step.argv, *fixture[step.argv[0]]])
        return Outcome(result.exit_code, result.output)

    return run


def test_the_weekly_job_rebuilds_twice_and_the_updated_tables_agree(
    rec: Recorded, tmp_path: Path
) -> None:
    paths = WeeklyPaths(
        days=rec.days,
        tables=tmp_path / "tables",
        pinned=tmp_path / "rebuild" / "pinned",
        refreshed=tmp_path / "rebuild" / "refreshed",
        report=tmp_path / "weekly.json",
    )
    shutil.copytree(rec.tables[3], paths.tables, symlinks=False)
    alerts = Alerts()
    steps = weekly_job_steps(paths)
    assert [(s.name, s.day) for s in steps] == [
        ("rebuild-pinned", D3),
        ("rebuild-refreshed", D3),
        ("check", D3),
    ]

    status = run_job(
        "weekly", steps, run=_cli_runner(tmp_path), notify=alerts, status_dir=tmp_path / "s"
    )

    assert status.ok, status.steps
    assert alerts.sent == []
    doc = json.loads(paths.report.read_text())
    assert doc["as_of"] == DAYS[2]
    assert doc["agreement"]["share"] == 1.0
    assert doc["agreement"]["relatives_agreed"] > 0
    assert doc["drift"]["relatives_changed"] + doc["drift"]["kept_but_changed"] > 0
    pinned = json.loads((tmp_path / "rebuild" / "pinned.build" / "rebuild.json").read_text())
    assert pinned["statistics"] == "pinned"
    assert pinned["vectors"]["shards_embedded"] == 0  # every vector reused from the tables
    refreshed = json.loads((tmp_path / "rebuild" / "refreshed.build" / "rebuild.json").read_text())
    assert refreshed["statistics"] == "refreshed"
    assert refreshed["vectors"]["shards_embedded"] == 0
    saved = json.loads((tmp_path / "s" / STATUS_FILE).read_text())
    assert saved["ok"] is True
    assert saved["steps"][0]["argv"][:2] == ["lineage", "rebuild"]


def test_the_weekly_job_has_nothing_to_rebuild_before_any_day_is_applied(
    rec: Recorded, tmp_path: Path
) -> None:
    paths = WeeklyPaths(days=rec.days, tables=rec.tables[1])  # the base build: no day applied
    assert weekly_job_steps(paths) == []
