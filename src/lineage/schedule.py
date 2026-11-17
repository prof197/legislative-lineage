"""The live track's scheduled jobs on the Mac, under launchd (DESIGN.md#updates, Publishing).

**The daily job** runs `lineage` commands in order: `intake` of the day; `update` of every day
taken in that the tables have not had, oldest first, so a day an earlier run could not apply
is applied now; `publish pages`, the incremental export into a staging copy of the live
export; `publish feed` into that copy; and `publish promote`, which swaps the copy in. A step
that fails stops the job, so the live export stays the last good one: nothing before the swap
touches it. Each run writes its outcome to a status file under the data directory, and a failed
run raises a macOS notification, since the job has no outside service to post to. Before the
hosted state's first day of session (rules/deploy.toml) the job stops there, at a local export
and feed; from that day it ends with `publish deploy` (lineage.deploy), which publishes the day
behind its gates, or holds it: a held day stops the job with status "held", its counts and
baseline, and an alert, as a failure does.

**The weekly job** compares the updated tables with a full rebuild over the same days and the
same pinned corpus statistics (`lineage.updates.snapshot`): the share of relatives the two agree
on must be at least 98%, or the job fails loudly (DR-7). It then reports the drift: how many
relatives changed when the stoplist and the state boilerplate flag were counted afresh. Before
session it then prepares a release candidate for review (`publish candidate`), and during it
lists the week's new origin claims for the maintainer (`publish review`).

The steps are `Step`s, each a `lineage` command line, run by an injectable runner (the CLI in
a subprocess, or the library in-process in the tests) and alerted by an injectable notifier
(osascript, or a fake). Two jobs never run at once: the second finds the lock held and stops.
`scheduling/` holds the launchd templates and the shell runner launchd starts.

Owned by lane L3 (M12), the deploy and the weekly job's rebuilds lane L4's. See
DESIGN.md#updates, DR-7.

See DESIGN.md#dr-13.
"""

import fcntl
import json
import plistlib
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from lineage.deploy import DEPLOY_DIR, HELD, DeployRules, in_session, load_rules
from lineage.incremental import applied_days
from lineage.paths import DATA
from lineage.publish import staging_dir
from lineage.semantic import MODEL
from lineage.updates import CHANGES_DIR, DAY_FILE, DayInfo, Relative, Snapshot, snapshot

SCHEDULE_DIR = DATA / "schedule"
STATUS_FILE = "status.json"
RUNS_FILE = "runs.jsonl"
LOCK_FILE = "lock"
WEEKLY_REPORT = SCHEDULE_DIR / "weekly.json"
AGREEMENT_FLOOR = 0.98  # DR-7: the weekly rebuild fails below it
OUTPUT_TAIL = 4000  # characters of a step's output kept in the status file
NOTIFY_TITLE = "Legislative Lineage"
TEMPLATES = Path("scheduling")
_PLACEHOLDER = re.compile(r"@[A-Z_]+@")


class ScheduleError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    argv: tuple[str, ...]  # a `lineage` command line, without `lineage`
    day: date | None = None  # the day an intake or update step is for


@dataclass(frozen=True, slots=True)
class Outcome:
    code: int
    output: str = ""


Runner = Callable[[Step], Outcome]
Notifier = Callable[[str, str], None]


# The daily job.


def pending_days(days_dir: Path, tables: Path) -> list[date]:
    """The days taken in that the tables have not had applied, oldest first: every day after
    the last applied one, since an update refuses a day before it."""
    applied = applied_days(tables)
    last = max(applied, default=None)
    found = (
        sorted(
            DayInfo.model_validate_json((p / DAY_FILE).read_text()).as_of
            for p in days_dir.iterdir()
            if (p / DAY_FILE).exists()
        )
        if days_dir.exists()
        else []
    )
    return [d for d in found if last is None or d > last]


@dataclass(frozen=True, slots=True)
class DailyPaths:
    days: Path = DATA / "days"
    tables: Path = DATA / "tables"
    changes: Path = CHANGES_DIR
    export_root: Path = DATA / "export"
    deploy: Path = DEPLOY_DIR

    def export(self, state: str) -> Path:
        return self.export_root / state


def daily_steps(
    day: date,
    *,
    state: str,
    site_url: str,
    tier: str = "default",
    paths: DailyPaths | None = None,
    paraphrase: bool = True,
    rules: DeployRules | None = None,
) -> Iterator[Step]:
    """The daily job's steps, in order. The update steps are listed once the intake has run,
    since they are the days it leaves to apply. From the state's first day of session the
    deploy follows the promote."""
    paths = paths if paths is not None else DailyPaths()
    rules = rules if rules is not None else load_rules()
    export = paths.export(state)
    yield Step(
        "intake",
        ("intake", "--day", day.isoformat(), "--days", str(paths.days), "--tier", tier),
        day,
    )
    for pending in pending_days(paths.days, paths.tables):
        yield Step(
            "update",
            (
                "update",
                "--day",
                str(paths.days / pending.isoformat()),
                "--tables",
                str(paths.tables),
                "--changes",
                str(paths.changes),
            ),
            pending,
        )
    yield Step(
        "pages",
        (
            "publish",
            "pages",
            "--state",
            state,
            "--tables",
            str(paths.tables),
            "--export",
            str(export),
            "--changes",
            str(paths.changes),
            "--paraphrase" if paraphrase else "--no-paraphrase",
        ),
    )
    yield Step(
        "feed",
        (
            "publish",
            "feed",
            "--state",
            state,
            "--site-url",
            site_url,
            "--tables",
            str(paths.tables),
            "--export",
            str(staging_dir(export)),
            "--changes",
            str(paths.changes),
        ),
    )
    yield Step("promote", ("publish", "promote", "--export", str(export)))
    if in_session(rules, state, day):
        yield Step(
            "deploy",
            (
                "publish",
                "deploy",
                "--state",
                state,
                "--export",
                str(export),
                "--changes",
                str(paths.changes),
                "--deploy-dir",
                str(paths.deploy),
            ),
            day,
        )


# Running a job.


@dataclass
class JobStatus:
    job: str
    started: str
    ok: bool | None = None
    finished: str | None = None
    failed_step: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    state: str = "running"  # then ok, failed, held (a day held for review) or locked
    held: dict[str, Any] | None = None  # a held day's counts, baseline and reasons


def _held_detail(output: str) -> dict[str, Any]:
    """The held day's record, the last line `publish deploy` prints when it holds a day."""
    for line in reversed(output.strip().splitlines()):
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict):
            return dict(doc)  # pyright: ignore[reportUnknownArgumentType]
    return {"output": output[-OUTPUT_TAIL:]}


def _now() -> datetime:
    return datetime.now(UTC)


def _write_status(status: JobStatus, status_dir: Path) -> Path:
    status_dir.mkdir(parents=True, exist_ok=True)
    doc = json.dumps(status.__dict__, indent=1) + "\n"
    path = status_dir / STATUS_FILE
    partial = path.with_name(path.name + ".partial")
    partial.write_text(doc)
    partial.replace(path)
    with (status_dir / RUNS_FILE).open("a") as runs:
        runs.write(json.dumps(status.__dict__) + "\n")
    return path


def run_job(
    job: str,
    steps: Iterable[Step],
    *,
    run: Runner,
    notify: Notifier,
    status_dir: Path = SCHEDULE_DIR,
    now: Callable[[], datetime] = _now,
) -> JobStatus:
    """Run each step in order and stop at the first that fails or holds a day. The outcome
    goes to the status file, and a failure, a held day, or a job already running, to the
    notifier."""
    status = JobStatus(job=job, started=now().isoformat())
    status_dir.mkdir(parents=True, exist_ok=True)
    with (status_dir / LOCK_FILE).open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            status.ok, status.failed_step, status.state = False, "lock", "locked"
            status.finished = now().isoformat()
            notify(NOTIFY_TITLE, f"The {job} job did not start: another job is running.")
            return status
        for step in steps:
            started = time.monotonic()
            try:
                outcome = run(step)
            except Exception as e:  # a runner that raises is a failed step, not a crashed job
                outcome = Outcome(code=-1, output=f"{type(e).__name__}: {e}")
            status.steps.append(
                {
                    "name": step.name,
                    "argv": ["lineage", *step.argv],
                    "day": None if step.day is None else step.day.isoformat(),
                    "code": outcome.code,
                    "seconds": round(time.monotonic() - started, 1),
                    "output": outcome.output[-OUTPUT_TAIL:],
                }
            )
            if outcome.code == HELD and step.name == "deploy":
                status.ok, status.state = False, "held"
                status.held = _held_detail(outcome.output)
                status.finished = now().isoformat()
                path = _write_status(status, status_dir)
                held_day = status.held.get("day", step.day.isoformat() if step.day else "the day")
                reasons = "; ".join(status.held.get("reasons", []))[:200]
                notify(
                    NOTIFY_TITLE,
                    f"The {job} job held {held_day} for review; nothing was published."
                    + (f" {reasons}." if reasons else "")
                    + f" After review: lineage publish release --day {held_day}, or reject it with"
                    + f" lineage publish reject --day {held_day} (see {path})",
                )
                return status
            if outcome.code != 0:
                status.ok, status.failed_step, status.state = False, step.name, "failed"
                status.finished = now().isoformat()
                path = _write_status(status, status_dir)
                last = outcome.output.strip().splitlines()[-1:] or [f"exit code {outcome.code}"]
                notify(
                    NOTIFY_TITLE,
                    f"The {job} job stopped at {step.name}; the published export is unchanged. "
                    f"{last[0][:200]} (see {path})",
                )
                return status
        status.ok, status.state = True, "ok"
        status.finished = now().isoformat()
        _write_status(status, status_dir)
        return status


def cli_runner(step: Step) -> Outcome:
    """Run a step as the `lineage` command of the environment this process runs in."""
    command = [str(Path(sys.executable).with_name("lineage")), *step.argv]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    return Outcome(code=done.returncode, output=done.stdout + done.stderr)


def _applescript(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def osascript_notifier(title: str, message: str) -> None:
    """A macOS notification. A notification that cannot be shown is no reason to fail."""
    script = f"display notification {_applescript(message)} with title {_applescript(title)}"
    subprocess.run(["osascript", "-e", script], check=False, timeout=30, capture_output=True)


# The weekly rebuild's check.


def _directed(s: Snapshot) -> dict[tuple[str, str], Relative]:
    return {(bill, r[0]): r for bill, rs in s.relatives.items() for r in rs}


@dataclass(frozen=True, slots=True)
class Agreement:
    """How far the updated tables' relatives agree with a full rebuild's over the same days."""

    updated: int  # relatives, a bill and one it lists, in the updated tables
    rebuilt: int
    agreed: int  # listed by both
    share: float  # agreed, of those either lists
    exact: int  # of those agreed, with the same stages, sections matched and share
    statuses_differ: int
    families_differ: int  # same-state families either has that the other does not
    origins_differ: int  # bills whose origin matches differ

    @property
    def ok(self) -> bool:
        return self.share >= AGREEMENT_FLOOR


def agreement(updated: Snapshot, rebuilt: Snapshot) -> Agreement:
    u, r = _directed(updated), _directed(rebuilt)
    both = set(u) & set(r)
    either = set(u) | set(r)
    bills = set(updated.statuses) | set(rebuilt.statuses)
    origin_bills = set(updated.origins) | set(rebuilt.origins)
    return Agreement(
        updated=len(u),
        rebuilt=len(r),
        agreed=len(both),
        share=len(both) / len(either) if either else 1.0,
        exact=sum(1 for k in both if u[k] == r[k]),
        statuses_differ=sum(1 for b in bills if updated.statuses.get(b) != rebuilt.statuses.get(b)),
        families_differ=len(updated.families ^ rebuilt.families),
        origins_differ=sum(
            1 for b in origin_bills if updated.origins.get(b) != rebuilt.origins.get(b)
        ),
    )


@dataclass(frozen=True, slots=True)
class Drift:
    """How many relatives changed when the corpus statistics were counted afresh."""

    pinned: int  # relatives with the base build's stoplist and flag
    refreshed: int  # with the stoplist and flag counted over the corpus as it stands
    gained: int
    lost: int
    changed: int  # kept, with other stages, sections matched or share
    bills: int  # bills with any of those

    @property
    def relatives_changed(self) -> int:
        return self.gained + self.lost


def drift(pinned: Snapshot, refreshed: Snapshot) -> Drift:
    p, f = _directed(pinned), _directed(refreshed)
    gained, lost = set(f) - set(p), set(p) - set(f)
    changed = {k for k in set(p) & set(f) if p[k] != f[k]}
    return Drift(
        pinned=len(p),
        refreshed=len(f),
        gained=len(gained),
        lost=len(lost),
        changed=len(changed),
        bills=len({bill for bill, _ in gained | lost | changed}),
    )


class DisagreementError(ScheduleError):
    """The updated tables and the rebuild agree on less than the floor: the live track has
    drifted from what a build gives, and must not go on as if it had not."""


def weekly_check(
    updated: Snapshot,
    pinned: Snapshot,
    refreshed: Snapshot | None,
    *,
    as_of: date,
    report: Path = WEEKLY_REPORT,
) -> dict[str, Any]:
    """The weekly report: the agreement between the updated tables and the pinned rebuild, and
    the drift to the refreshed one. Written first, then raised on when the agreement is below
    the floor, so a failure is on record with its numbers."""
    a = agreement(updated, pinned)
    doc: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "agreement": {
            "floor": AGREEMENT_FLOOR,
            "ok": a.ok,
            "share": round(a.share, 6),
            "relatives_updated": a.updated,
            "relatives_rebuilt": a.rebuilt,
            "relatives_agreed": a.agreed,
            "relatives_exact": a.exact,
            "statuses_differ": a.statuses_differ,
            "families_differ": a.families_differ,
            "origins_differ": a.origins_differ,
        },
        "drift": None,
    }
    if refreshed is not None:
        d = drift(pinned, refreshed)
        doc["drift"] = {
            "relatives_changed": d.relatives_changed,
            "relatives_pinned": d.pinned,
            "relatives_refreshed": d.refreshed,
            "gained": d.gained,
            "lost": d.lost,
            "kept_but_changed": d.changed,
            "bills_affected": d.bills,
        }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(doc, indent=1) + "\n")
    if not a.ok:
        raise DisagreementError(
            f"the updated tables agree with the rebuild on {a.share:.2%} of relatives, below "
            f"{AGREEMENT_FLOOR:.0%} ({a.agreed} of {a.updated} updated and {a.rebuilt} rebuilt); "
            f"see {report}"
        )
    return doc


def weekly_report(
    updated: Path,
    pinned: Path,
    refreshed: Path | None,
    *,
    model: str = MODEL,
    report: Path = WEEKLY_REPORT,
) -> dict[str, Any]:
    """`weekly_check` on three tables directories: the updated tables, the rebuild over the
    same days with their pinned statistics, and the rebuild with the statistics refreshed."""
    applied = applied_days(updated)
    if not applied:
        raise ScheduleError(f"{updated} has had no day applied; there is nothing to check")
    return weekly_check(
        snapshot(updated, model=model),
        snapshot(pinned, model=model),
        None if refreshed is None else snapshot(refreshed, model=model),
        as_of=max(applied),
        report=report,
    )


@dataclass(frozen=True, slots=True)
class WeeklyPaths:
    days: Path = DATA / "days"
    tables: Path = DATA / "tables"
    pinned: Path = DATA / "rebuild" / "pinned"
    refreshed: Path = DATA / "rebuild" / "refreshed"
    report: Path = WEEKLY_REPORT
    changes: Path = CHANGES_DIR
    export_root: Path = DATA / "export"
    deploy: Path = DEPLOY_DIR


def weekly_steps(as_of: date, *, paths: WeeklyPaths | None = None) -> list[Step]:
    """The weekly job's steps: the two rebuilds over the days so far, then the check, which
    fails the job below the floor before anything adopts the refreshed rebuild."""
    paths = paths if paths is not None else WeeklyPaths()
    days, tables = str(paths.days), str(paths.tables)
    return [
        Step(
            "rebuild-pinned",
            (
                "rebuild",
                "--days",
                days,
                "--through",
                as_of.isoformat(),
                "--pinned-from",
                tables,
                "--out",
                str(paths.pinned),
            ),
            as_of,
        ),
        Step(
            "rebuild-refreshed",
            (
                "rebuild",
                "--days",
                days,
                "--through",
                as_of.isoformat(),
                # The same texts' vectors, not the statistics: a window's vector does not
                # depend on the stoplist or the flag.
                "--vectors-from",
                tables,
                "--out",
                str(paths.refreshed),
            ),
            as_of,
        ),
        Step(
            "check",
            (
                "publish",
                "weekly",
                "--updated",
                tables,
                "--pinned",
                str(paths.pinned),
                "--refreshed",
                str(paths.refreshed),
                "--report",
                str(paths.report),
            ),
            as_of,
        ),
    ]


def weekly_job_steps(
    paths: WeeklyPaths | None = None,
    *,
    state: str | None = None,
    site_url: str | None = None,
    day: date | None = None,
    rules: DeployRules | None = None,
) -> list[Step]:
    """The weekly job's rebuilds and check, over the days the updated tables have had: through
    the last one applied, since pinned statistics come from tables at the same day and a day
    taken in but not applied is not in the tables to agree with; none before any day is.

    For a hosted `state`, then: before its first day of session, a release candidate packed and
    gated for review; from it, the week's new origin claims for the maintainer's review. Both
    come after the check, so a live track that has drifted prepares nothing."""
    paths = paths if paths is not None else WeeklyPaths()
    applied = applied_days(paths.tables)
    steps = weekly_steps(max(applied), paths=paths) if applied else []
    if state is None or site_url is None or day is None:
        return steps
    rules = rules if rules is not None else load_rules()
    common = (
        "--state",
        state,
        "--site-url",
        site_url,
        "--tables",
        str(paths.tables),
        "--changes",
        str(paths.changes),
        "--deploy-dir",
        str(paths.deploy),
    )
    if in_session(rules, state, day):
        steps.append(
            Step("review", ("publish", "review", *common, "--through", day.isoformat()), day)
        )
    else:
        steps.append(
            Step(
                "candidate",
                (
                    "publish",
                    "candidate",
                    *common,
                    "--export",
                    str(paths.export_root / state),
                    "--day",
                    day.isoformat(),
                ),
                day,
            )
        )
    return steps


# The launchd templates.


def render_template(template: str, values: Mapping[str, str]) -> str:
    """A launchd template with its @NAME@ placeholders filled; one left unfilled is an error,
    as is a result that is not a property list."""
    out = template
    for name, value in values.items():
        out = out.replace(f"@{name}@", value)
    left = sorted(set(_PLACEHOLDER.findall(out)))
    if left:
        raise ScheduleError(f"placeholders left unfilled: {', '.join(left)}")
    plistlib.loads(out.encode())
    return out


def placeholders(template: str) -> Sequence[str]:
    return sorted({p.strip("@") for p in _PLACEHOLDER.findall(template)})
