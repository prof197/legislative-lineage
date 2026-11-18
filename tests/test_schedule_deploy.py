"""The scheduled jobs with the deploy in them (DESIGN.md#updates, DR-13): before the first day of
session the daily job stops at the local export and the weekly job prepares a candidate; from
it the daily job deploys, and a held day stops the job with status "held", its counts and
baseline, and an alert. The recorded days run in this process, every command that would leave
the machine faked."""

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from lineage import deploy
from lineage.deploy import DeployPaths, DeployRules, Limits, load_rules
from lineage.feed import write_feed
from lineage.incremental import update
from lineage.publish import export_changes, promote, staging_dir
from lineage.schedule import (
    STATUS_FILE,
    TEMPLATES,
    DailyPaths,
    Outcome,
    Step,
    WeeklyPaths,
    daily_steps,
    run_job,
    weekly_job_steps,
)
from lineage.updates import load_day
from tests.conftest import FAKE_MODEL
from tests.days import DAYS
from tests.live import ExportInputs, Recorded, export_inputs, full_export, recorded
from tests.test_deploy import Fake, job_checkout


@pytest.fixture(autouse=True)
def deny_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deny list the deploy reads from the environment, as the daily job sets it."""
    monkeypatch.setenv("LEAK_CHECK_TERMS", "zzdenyterm")


SITE = "https://example.org/lineage"
D2, D3 = (date.fromisoformat(d) for d in DAYS[1:])


class Alerts:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def __call__(self, title: str, message: str) -> None:
        self.sent.append((title, message))


@pytest.fixture(scope="module")
def rec(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    return recorded(tmp_path_factory)


def _rules(first_day: date, *, ceiling: int = 2500) -> DeployRules:
    rules = load_rules()
    limits = {"origin_claims": Limits(30, 50), "changed_files": Limits(1000, ceiling)}
    return replace(
        rules,
        first_day={"GA": first_day},
        confirmed={"GA": True},
        breaker=replace(rules.breaker, limits=limits),
    )


def _setup(rec: Recorded, root: Path) -> tuple[DailyPaths, ExportInputs]:
    paths = DailyPaths(
        days=root / "days",
        tables=root / "tables",
        changes=root / "changes",
        export_root=root,
        deploy=root / "deploy",
    )
    paths.days.mkdir(parents=True)
    shutil.copytree(rec.tables[1], paths.tables, symlinks=False)
    inputs = export_inputs(root / "inputs")
    full_export(paths.tables, paths.export("GA"), inputs)
    return paths, inputs


def _runner(
    rec: Recorded, paths: DailyPaths, inputs: ExportInputs, rules: DeployRules, fake: Fake
) -> Callable[[Step], Outcome]:
    """Each step on the library, in this process; the deploy's commands go to the fake. A
    held deploy answers as `lineage publish deploy` does: exit code 3, the record last."""
    export = paths.export("GA")
    dpaths = DeployPaths(root=paths.deploy, checkout=fake.checkout)

    def run(step: Step) -> Outcome:
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
                paths.tables, state="GA", export=export, changes_dir=paths.changes,
                models_dir=None, floor=0, **inputs,
            )  # fmt: skip
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
        elif step.name == "deploy":
            assert step.argv[:2] == ("publish", "deploy")
            record = deploy.deploy_day(
                state="GA", export=export, changes_dir=paths.changes, paths=dpaths,
                rules=rules, shell=fake, floor=0,
            )  # fmt: skip
            if record.state == "held":
                detail = {"day": str(record.day), "reasons": record.reasons,
                          "counts": record.counts, "baseline": record.baseline}  # fmt: skip
                return Outcome(deploy.HELD, json.dumps(detail))
        else:
            raise AssertionError(f"no step {step.name}")
        return Outcome(0, f"{step.name} done")

    return run


def test_before_the_first_day_the_daily_job_stops_at_the_local_export(
    rec: Recorded, tmp_path: Path
) -> None:
    paths, _ = _setup(rec, tmp_path / "data")
    steps = list(daily_steps(D2, state="GA", site_url=SITE, paths=paths, rules=_rules(D3)))
    assert [s.name for s in steps][-1] == "promote"
    assert "deploy" not in [s.name for s in steps]


def test_from_the_first_day_the_daily_job_deploys_after_the_promote(
    rec: Recorded, tmp_path: Path
) -> None:
    paths, inputs = _setup(rec, tmp_path / "data")
    fake = Fake(job_checkout(tmp_path))
    rules = _rules(D2)
    alerts = Alerts()
    status_dir = tmp_path / "data" / "schedule"

    steps = daily_steps(D2, state="GA", site_url=SITE, paths=paths, rules=rules)
    status = run_job(
        "daily", steps, run=_runner(rec, paths, inputs, rules, fake), notify=alerts,
        status_dir=status_dir,
    )  # fmt: skip

    assert status.ok, status.steps
    assert status.state == "ok"
    assert [s["name"] for s in status.steps][-2:] == ["promote", "deploy"]
    deploy_step = status.steps[-1]["argv"]
    assert deploy_step[:6] == ["lineage", "publish", "deploy", "--state", "GA", "--export"]
    assert str(paths.deploy) in deploy_step
    assert "gh workflow run pages.yml --ref main" in fake.lines()
    assert alerts.sent == []


def test_a_held_day_stops_the_job_with_its_counts_and_an_alert(
    rec: Recorded, tmp_path: Path
) -> None:
    paths, inputs = _setup(rec, tmp_path / "data")
    fake = Fake(job_checkout(tmp_path))
    rules = _rules(D2, ceiling=1)  # too little history, and day 2 changes more than one file
    alerts = Alerts()
    status_dir = tmp_path / "data" / "schedule"

    status = run_job(
        "daily",
        daily_steps(D2, state="GA", site_url=SITE, paths=paths, rules=rules),
        run=_runner(rec, paths, inputs, rules, fake),
        notify=alerts,
        status_dir=status_dir,
    )

    assert status.ok is False
    assert status.state == "held"
    assert status.failed_step is None
    assert status.held is not None
    assert status.held["day"] == DAYS[1]
    assert status.held["counts"]["changed_files"] > 1
    assert status.held["baseline"]["changed_files"]["rule"] == "ceiling"
    saved = json.loads((status_dir / STATUS_FILE).read_text())
    assert saved["state"] == "held"
    assert saved["held"] == status.held
    assert fake.published() == []
    [(title, message)] = alerts.sent
    assert title == "Legislative Lineage"
    assert "held 2026-01-13 for review" in message
    assert "nothing was published" in message
    assert "lineage publish release --day 2026-01-13" in message


def test_a_failed_step_records_its_state(tmp_path: Path) -> None:
    alerts = Alerts()
    status = run_job(
        "daily", [Step("a", ("a",))], run=lambda _: Outcome(1, "no"), notify=alerts,
        status_dir=tmp_path,
    )  # fmt: skip
    assert (status.state, status.failed_step, status.held) == ("failed", "a", None)


def test_the_weekly_job_prepares_a_candidate_before_session_and_a_review_during_it(
    rec: Recorded, tmp_path: Path
) -> None:
    paths = WeeklyPaths(
        days=rec.days,
        tables=rec.tables[3],
        pinned=tmp_path / "p",
        refreshed=tmp_path / "f",
        report=tmp_path / "w.json",
        changes=tmp_path / "changes",
        export_root=tmp_path / "export",
        deploy=tmp_path / "deploy",
    )
    before = weekly_job_steps(
        paths, state="GA", site_url=SITE, day=D3, rules=_rules(date(2027, 1, 11))
    )
    assert [s.name for s in before] == ["rebuild-pinned", "rebuild-refreshed", "check", "candidate"]
    candidate = before[-1].argv
    assert candidate[:2] == ("publish", "candidate")
    assert ("--export", str(tmp_path / "export" / "GA")) == candidate[
        candidate.index("--export") : candidate.index("--export") + 2
    ]
    assert "--deploy-dir" in candidate
    assert "--site-url" in candidate

    during = weekly_job_steps(paths, state="GA", site_url=SITE, day=D3, rules=_rules(D2))
    assert [s.name for s in during][-1] == "review"
    review = during[-1].argv
    assert review[:2] == ("publish", "review")
    assert review[review.index("--through") + 1] == DAYS[2]
    # Without a state the job is the rebuilds and the check alone, as before.
    assert [s.name for s in weekly_job_steps(paths)] == [
        "rebuild-pinned",
        "rebuild-refreshed",
        "check",
    ]


def test_the_runner_script_reads_the_leak_terms_from_the_keychain_and_knows_a_held_day() -> None:
    script = (TEMPLATES / "run.sh").read_text()
    assert "security find-generic-password -w -s legislative-lineage-leak-terms" in script
    assert "export LEAK_CHECK_TERMS" in script
    assert "uv run --frozen lineage schedule weekly --state" in script
    assert "-ne 3" in script  # a held day is already alerted, like a failed step
    # The GitHub token is the deploy's to read before each call (lineage.deploy.github_shell).
    assert "export GH_TOKEN" not in script
    # Held awake for the length of the job, which the Mac is woken for.
    assert "caffeinate -i uv run --frozen lineage schedule daily" in script
