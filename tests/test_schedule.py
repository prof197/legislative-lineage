"""The scheduled jobs (DESIGN.md#updates, Publishing; DR-7): the daily job's steps in order, a
failed step that leaves the last good export in place with its status and an alert, a dry run
over the recorded days with the network stubbed, the weekly rebuild's agreement, which fails
below 98%, and its drift, and the launchd templates."""

import fcntl
import json
import plistlib
import shutil
import socket
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.core import http
from lineage.live.feed import check, write_feed
from lineage.live.incremental import update
from lineage.live.publish import export_changes, promote, staging_dir
from lineage.live.schedule import (
    AGREEMENT_FLOOR,
    LOCK_FILE,
    SCHEDULE,
    STATUS_FILE,
    TEMPLATES,
    DailyPaths,
    DisagreementError,
    Outcome,
    ScheduleError,
    Step,
    WeeklyPaths,
    When,
    agreement,
    calendar_intervals,
    daily_steps,
    placeholders,
    render_template,
    run_job,
    weekly_check,
    weekly_report,
    weekly_steps,
)
from lineage.live.updates import Snapshot, load_day, snapshot
from tests.conftest import FAKE_MODEL
from tests.days import DAYS
from tests.live import (
    ExportInputs,
    Recorded,
    export_inputs,
    files_seen,
    full_export,
    normalized,
    recorded,
)
from tools.leak_check import LOCAL_PATH

SITE = "https://example.org/lineage"
D1, D2, D3 = (date.fromisoformat(d) for d in DAYS)


class Alerts:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> None:
        self.sent.append((title, message))


@pytest.fixture(scope="module")
def rec(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    return recorded(tmp_path_factory)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> Any:
        raise AssertionError("a dry run sends no request")

    monkeypatch.setattr(http, "open_url", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def _runner(rec: Recorded, paths: DailyPaths, inputs: ExportInputs, fail: str | None = None):
    """The job's steps run in this process on the library, the network stubbed: the intake
    "fetches" a recorded day by copying it where `lineage intake` writes days."""
    export = paths.export("GA")

    def run(step: Step) -> Outcome:
        if step.name == fail:
            return Outcome(1, f"error: {step.name} failed (test)")
        if step.name == "intake":
            assert step.day is not None
            target = paths.days / step.day.isoformat()
            if not target.exists():
                shutil.copytree(rec.days / step.day.isoformat(), target)
        elif step.name == "update":
            assert step.day is not None
            day = load_day(paths.days / step.day.isoformat())
            update(paths.tables, day, model=FAKE_MODEL, changes_dir=paths.changes, floor_bytes=0)
        elif step.name == "pages":
            export_changes(
                paths.tables,
                state="GA",
                export=export,
                changes_dir=paths.changes,
                models_dir=None,
                floor=0,
                **inputs,
            )
        elif step.name == "feed":
            write_feed(
                paths.tables,
                state="GA",
                export=staging_dir(export),
                site_url=SITE,
                changes_dir=paths.changes,
                modeldocs_root=inputs["modeldocs_root"],
                collector_package=inputs["collector_package"],
            )
        elif step.name == "promote":
            promote(export)
        else:
            raise AssertionError(f"no step {step.name}")
        return Outcome(0, f"{step.name} done")

    return run


def _setup(rec: Recorded, root: Path) -> tuple[DailyPaths, ExportInputs]:
    """A data directory as it stands before the live track: the base build's tables, a full
    export of them, and no days yet."""
    paths = DailyPaths(
        days=root / "days", tables=root / "tables", changes=root / "changes", export_root=root
    )
    paths.days.mkdir(parents=True)
    shutil.copytree(rec.tables[1], paths.tables, symlinks=False)
    inputs = export_inputs(root / "inputs")
    full_export(paths.tables, paths.export("GA"), inputs)
    return paths, inputs


def test_a_dry_run_over_the_recorded_days_publishes_each_day_with_no_request(
    rec: Recorded, tmp_path: Path, no_network: None
) -> None:
    paths, inputs = _setup(rec, tmp_path / "data")
    status_dir = tmp_path / "data" / "schedule"
    alerts = Alerts()

    for day in (D2, D3):
        steps = daily_steps(day, state="GA", site_url=SITE, paths=paths)
        status = run_job(
            "daily", steps, run=_runner(rec, paths, inputs), notify=alerts, status_dir=status_dir
        )

        assert status.ok, status.steps
        assert [s["name"] for s in status.steps] == ["intake", "update", "pages", "feed", "promote"]
        live = paths.export("GA")
        assert normalized(live) == normalized(
            full_export(paths.tables, tmp_path / f"full-{day}", inputs)
        )
        check((live / "feed.xml").read_bytes())
        assert json.loads((live / "meta.json").read_text())["as_of"] == day.isoformat()
        assert not staging_dir(live).exists()
        saved = json.loads((status_dir / STATUS_FILE).read_text())
        assert saved["ok"] is True
        assert saved["steps"][1]["argv"][:2] == ["lineage", "update"]
    assert alerts.sent == []
    assert len((status_dir / "runs.jsonl").read_text().splitlines()) == 2


def test_a_failed_step_stops_the_job_and_leaves_the_last_good_export(
    rec: Recorded, tmp_path: Path, no_network: None
) -> None:
    paths, inputs = _setup(rec, tmp_path / "data")
    status_dir = tmp_path / "data" / "schedule"
    alerts = Alerts()
    good = _runner(rec, paths, inputs)
    run_job(
        "daily",
        daily_steps(D2, state="GA", site_url=SITE, paths=paths),
        run=good,
        notify=alerts,
        status_dir=status_dir,
    )
    live = paths.export("GA")
    before = files_seen(live)

    status = run_job(
        "daily",
        daily_steps(D3, state="GA", site_url=SITE, paths=paths),
        run=_runner(rec, paths, inputs, fail="feed"),
        notify=alerts,
        status_dir=status_dir,
    )

    assert status.ok is False
    assert status.failed_step == "feed"
    assert [s["name"] for s in status.steps] == ["intake", "update", "pages", "feed"]
    assert files_seen(live) == before  # the same files, untouched: day 2's export
    saved = json.loads((status_dir / STATUS_FILE).read_text())
    assert saved["failed_step"] == "feed"
    assert saved["steps"][-1]["output"] == "error: feed failed (test)"
    [(title, message)] = alerts.sent
    assert title == "Legislative Lineage"
    assert "stopped at feed" in message
    assert "unchanged" in message

    # The next run finds day 3 applied to the tables but not to the export, and publishes it.
    again = run_job(
        "daily",
        daily_steps(D3, state="GA", site_url=SITE, paths=paths),
        run=good,
        notify=alerts,
        status_dir=status_dir,
    )
    assert again.ok
    assert [s["name"] for s in again.steps] == ["intake", "pages", "feed", "promote"]
    assert normalized(live) == normalized(full_export(paths.tables, tmp_path / "full", inputs))


def test_a_runner_that_raises_is_a_failed_step_and_a_held_lock_starts_nothing(
    tmp_path: Path,
) -> None:
    alerts = Alerts()
    ran: list[str] = []

    def run(step: Step) -> Outcome:
        ran.append(step.name)
        raise OSError("no such command")

    steps = [Step("a", ("a",)), Step("b", ("b",))]
    status = run_job("daily", steps, run=run, notify=alerts, status_dir=tmp_path)
    assert (status.failed_step, ran) == ("a", ["a"])
    assert status.steps[0]["output"] == "OSError: no such command"

    with (tmp_path / LOCK_FILE).open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        status = run_job("weekly", steps, run=run, notify=alerts, status_dir=tmp_path)
    assert status.failed_step == "lock"
    assert ran == ["a"]
    assert "another job is running" in alerts.sent[-1][1]


def test_the_daily_steps_are_the_cli_commands_in_order(tmp_path: Path) -> None:
    paths = DailyPaths(
        days=tmp_path / "days",
        tables=tmp_path / "tables",
        changes=tmp_path / "changes",
        export_root=tmp_path / "export",
    )
    (paths.tables / "updates").mkdir(parents=True)
    (paths.tables / "updates" / "applied.jsonl").write_text(
        json.dumps({"as_of": DAYS[0], "sha256": "x"}) + "\n"
    )
    steps = daily_steps(D3, state="GA", site_url=SITE, paths=paths)

    intake = next(steps)
    assert intake.argv == (
        "intake",
        "--day",
        DAYS[2],
        "--days",
        str(paths.days),
        "--tier",
        "default",
    )
    for day in DAYS:  # the intake leaves day 3, and an earlier run left day 2 unapplied
        (paths.days / day).mkdir(parents=True)
        (paths.days / day / "day.json").write_text(json.dumps({"as_of": day, "sources": []}))
    rest = list(steps)
    assert [(s.name, s.day) for s in rest] == [
        ("update", D2),
        ("update", D3),
        ("pages", None),
        ("feed", None),
        ("promote", None),
    ]
    assert rest[0].argv == (
        "update",
        "--day",
        str(paths.days / DAYS[1]),
        "--tables",
        str(paths.tables),
        "--changes",
        str(paths.changes),
    )
    export = paths.export("GA")
    assert rest[2].argv == (
        "publish",
        "pages",
        "--state",
        "GA",
        "--tables",
        str(paths.tables),
        "--export",
        str(export),
        "--changes",
        str(paths.changes),
        "--paraphrase",
    )
    assert rest[3].argv[:6] == ("publish", "feed", "--state", "GA", "--site-url", SITE)
    assert str(staging_dir(export)) in rest[3].argv
    assert rest[4].argv == ("publish", "promote", "--export", str(export))


# The weekly rebuild.


def test_the_updated_tables_agree_with_a_rebuild_and_the_refreshed_statistics_drift(
    rec: Recorded, tmp_path: Path
) -> None:
    report = tmp_path / "weekly.json"
    doc = weekly_report(rec.tables[3], rec.rebuilt, rec.refreshed, model=FAKE_MODEL, report=report)

    assert json.loads(report.read_text()) == doc
    assert doc["as_of"] == DAYS[2]
    a = doc["agreement"]
    assert a["ok"] is True
    assert a["share"] == 1.0  # exact search at fixture scale: an update gives a rebuild
    assert a["relatives_exact"] == a["relatives_agreed"] == a["relatives_updated"] > 0
    assert (a["statuses_differ"], a["families_differ"]) == (0, 0)
    d = doc["drift"]
    assert d["relatives_pinned"] == a["relatives_rebuilt"]
    assert d["relatives_changed"] == d["gained"] + d["lost"]
    assert d["relatives_changed"] + d["kept_but_changed"] > 0  # the refreshed stoplist moves some
    assert d["bills_affected"] > 0


def _drop(s: Snapshot, n: int) -> Snapshot:
    """`s` with its first `n` relatives, a bill and one it lists, taken out."""
    out: dict[str, tuple[Any, ...]] = {}
    left = n
    for bill, rs in sorted(s.relatives.items()):
        take = min(left, len(rs))
        out[bill] = rs[take:]
        left -= take
    return replace(s, relatives=out)


def test_a_recorded_disagreement_below_98_percent_fails_the_weekly_check(
    rec: Recorded, tmp_path: Path
) -> None:
    updated = snapshot(rec.tables[3], model=FAKE_MODEL)
    total = sum(len(rs) for rs in updated.relatives.values())
    below = int(total * (1 - AGREEMENT_FLOOR)) + 1  # just under the floor
    report = tmp_path / "weekly.json"

    with pytest.raises(DisagreementError, match="below 98%"):
        weekly_check(updated, _drop(updated, below), None, as_of=D3, report=report)

    doc = json.loads(report.read_text())  # on record with its numbers
    assert doc["agreement"]["ok"] is False
    assert doc["agreement"]["share"] < AGREEMENT_FLOOR
    assert doc["agreement"]["relatives_agreed"] == total - below


def test_the_floor_is_98_percent_of_the_relatives_either_lists(tmp_path: Path) -> None:
    def rel(i: int) -> tuple[str, tuple[str, ...], int, float]:
        return (f"r{i}", ("near_duplicate",), 1, 0.5)

    def snap(n: int, extra: int = 0) -> Snapshot:
        return Snapshot(
            statuses={},
            families=frozenset(),
            relatives={"b": tuple(rel(i) for i in range(n + extra))},
        )

    assert agreement(snap(98), snap(98, 2)).share == 0.98  # 98 of the 100 either lists
    assert agreement(snap(98), snap(98, 2)).ok
    assert not agreement(snap(97), snap(97, 3)).ok
    assert agreement(snap(0), snap(0)).share == 1.0
    with pytest.raises(DisagreementError):
        weekly_check(snap(97), snap(97, 3), None, as_of=D3, report=tmp_path / "weekly.json")


def test_the_weekly_steps_check_before_anything_is_adopted(tmp_path: Path) -> None:
    paths = WeeklyPaths(
        days=tmp_path / "days",
        tables=tmp_path / "t",
        pinned=tmp_path / "p",
        refreshed=tmp_path / "f",
        report=tmp_path / "w.json",
    )
    steps = weekly_steps(D3, paths=paths)
    assert [s.name for s in steps] == ["rebuild-pinned", "rebuild-refreshed", "check"]
    assert "--pinned-from" in steps[0].argv
    assert "--pinned-from" not in steps[1].argv
    assert steps[2].argv[:2] == ("publish", "weekly")
    alerts = Alerts()
    status = run_job(
        "weekly",
        steps,
        run=lambda s: Outcome(1 if s.name == "check" else 0, "error: below 98%"),
        notify=alerts,
        status_dir=tmp_path / "s",
    )
    assert status.failed_step == "check"
    assert "stopped at check" in alerts.sent[0][1]


# The launchd templates.


def test_the_jobs_run_when_ruled_daily_monday_to_saturday_and_weekly_saturday_evening() -> None:
    # launchd counts weekdays from 0, Sunday: no daily run on Sunday.
    assert SCHEDULE["daily"] == When(hour=5, minute=30, weekdays=(1, 2, 3, 4, 5, 6))
    assert SCHEDULE["weekly"] == When(hour=20, minute=0, weekdays=(6,))


def _times(plist: dict[str, Any]) -> list[tuple[int, int, int]]:
    return [(w["Weekday"], w["Hour"], w["Minute"]) for w in plist["StartCalendarInterval"]]


def test_the_plist_command_fills_the_ruled_times(tmp_path: Path) -> None:
    daily = [(d, 5, 30) for d in range(1, 7)]
    for job, expected in (("daily", daily), ("weekly", [(6, 20, 0)])):
        out = tmp_path / f"{job}.plist"
        result = CliRunner().invoke(
            app, ["schedule", "plist", job, "--out", str(out), "--site-url", SITE]
        )
        assert result.exit_code == 0, result.output
        assert _times(plistlib.loads(out.read_bytes())) == expected
    out = tmp_path / "custom.plist"
    custom = ["--hour", "6", "--minute", "5", "--weekdays", "0,3"]
    result = CliRunner().invoke(
        app, ["schedule", "plist", "weekly", "--out", str(out), "--site-url", SITE, *custom]
    )
    assert result.exit_code == 0, result.output
    assert _times(plistlib.loads(out.read_bytes())) == [(0, 6, 5), (3, 6, 5)]


@pytest.mark.parametrize("job", ["daily", "weekly"])
def test_a_launchd_template_names_no_real_path_and_renders_to_a_property_list(job: str) -> None:
    template = (TEMPLATES / f"lineage-{job}.plist.template").read_text()
    assert not LOCAL_PATH.search(template)
    assert placeholders(template) == [
        "DATA", "INTERVALS", "LABEL", "PATH", "REPO", "SITE_URL", "STATE",
    ]  # fmt: skip
    values = {
        "DATA": "/srv/lineage-data",
        "INTERVALS": calendar_intervals(When(hour=6, minute=30, weekdays=(2, 6))),
        "LABEL": "org.example.lineage",
        "PATH": "/usr/bin:/bin",
        "REPO": "/srv/legislative-lineage",
        "SITE_URL": SITE,
        "STATE": "GA",
    }
    plist = plistlib.loads(render_template(template, values).encode())

    assert plist["Label"] == f"org.example.lineage.{job}"
    assert _times(plist) == [(2, 6, 30), (6, 6, 30)]
    assert plist["ProgramArguments"] == [
        "/bin/bash",
        "/srv/legislative-lineage/scheduling/run.sh",
        job,
    ]
    assert plist["EnvironmentVariables"]["LINEAGE_STATE"] == "GA"
    assert "OPENSTATES_API_KEY" not in plist["EnvironmentVariables"]  # read from the keychain
    with pytest.raises(ScheduleError, match="@STATE@"):
        render_template(template, {k: v for k, v in values.items() if k != "STATE"})


def test_the_runner_script_calls_the_cli_and_names_no_real_path() -> None:
    script = (TEMPLATES / "run.sh").read_text()
    assert not LOCAL_PATH.search(script)
    assert "uv run --frozen lineage schedule daily --state" in script
    assert "uv run --frozen lineage schedule weekly" in script
    assert "launchctl" not in script
