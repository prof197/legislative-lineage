"""The daily job (DESIGN.md#updates, Publishing): its steps in order, a failed step that leaves
the last good export in place with its status and an alert, a dry run over the recorded days
with the network stubbed, and the launchd template."""

import fcntl
import json
import plistlib
import shutil
import socket
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from lineage import fetch
from lineage.feed import check, write_feed
from lineage.publish import export_changes, promote, staging_dir
from lineage.schedule import (
    LOCK_FILE,
    STATUS_FILE,
    TEMPLATES,
    DailyPaths,
    Outcome,
    ScheduleError,
    Step,
    daily_steps,
    placeholders,
    render_template,
    run_job,
)
from lineage.updates import load_day, update
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

    monkeypatch.setattr(fetch, "open_url", refuse)
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


# The launchd templates.


@pytest.mark.parametrize("job", ["daily"])
def test_a_launchd_template_names_no_real_path_and_renders_to_a_property_list(job: str) -> None:
    template = (TEMPLATES / f"lineage-{job}.plist.template").read_text()
    assert not LOCAL_PATH.search(template)
    assert placeholders(template) == ["DATA", "HOUR", "LABEL", "PATH", "REPO", "SITE_URL", "STATE"]
    values = {
        "DATA": "/srv/lineage-data",
        "HOUR": "6",
        "LABEL": "org.example.lineage",
        "PATH": "/usr/bin:/bin",
        "REPO": "/srv/legislative-lineage",
        "SITE_URL": SITE,
        "STATE": "GA",
    }
    plist = plistlib.loads(render_template(template, values).encode())

    assert plist["Label"] == f"org.example.lineage.{job}"
    assert plist["ProgramArguments"] == [
        "/bin/bash",
        "/srv/legislative-lineage/scheduling/run.sh",
        job,
    ]
    assert plist["EnvironmentVariables"]["LINEAGE_STATE"] == "GA"
    assert "OPENSTATES_API_KEY" not in plist["EnvironmentVariables"]  # read from the keychain
    assert plist["StartCalendarInterval"]["Hour"] == 6
    with pytest.raises(ScheduleError, match="@STATE@"):
        render_template(template, {k: v for k, v in values.items() if k != "STATE"})


def test_the_runner_script_calls_the_cli_and_names_no_real_path() -> None:
    script = (TEMPLATES / "run.sh").read_text()
    assert not LOCAL_PATH.search(script)
    assert "uv run --frozen lineage schedule daily --state" in script
    assert "launchctl" not in script
