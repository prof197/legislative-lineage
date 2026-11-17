"""The live track's scheduled jobs on the Mac, under launchd (DESIGN.md#updates, Publishing).

**The daily job** runs `lineage` commands in order: `intake` of the day; `update` of every day
taken in that the tables have not had, oldest first, so a day an earlier run could not apply
is applied now; `publish pages`, the incremental export into a staging copy of the live
export; `publish feed` into that copy; and `publish promote`, which swaps the copy in. A step
that fails stops the job, so the live export stays the last good one: nothing before the swap
touches it. Each run writes its outcome to a status file under the data directory, and a failed
run raises a macOS notification, since the job has no outside service to post to. The job
builds a local export and feed ready to publish; it deploys and publishes nothing.

The steps are `Step`s, each a `lineage` command line, run by an injectable runner (the CLI in
a subprocess, or the library in-process in the tests) and alerted by an injectable notifier
(osascript, or a fake). Two jobs never run at once: the second finds the lock held and stops.
`scheduling/` holds the launchd templates and the shell runner launchd starts.

Owned by lane L3 (M12). See DESIGN.md#updates.
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

from lineage.incremental import applied_days
from lineage.paths import DATA
from lineage.publish import staging_dir
from lineage.updates import CHANGES_DIR, DAY_FILE, DayInfo

SCHEDULE_DIR = DATA / "schedule"
STATUS_FILE = "status.json"
RUNS_FILE = "runs.jsonl"
LOCK_FILE = "lock"
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
) -> Iterator[Step]:
    """The daily job's steps, in order. The update steps are listed once the intake has run,
    since they are the days it leaves to apply."""
    paths = paths if paths is not None else DailyPaths()
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


# Running a job.


@dataclass
class JobStatus:
    job: str
    started: str
    ok: bool | None = None
    finished: str | None = None
    failed_step: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


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
    """Run each step in order and stop at the first that fails. The outcome goes to the
    status file, and a failure, or a job already running, to the notifier."""
    status = JobStatus(job=job, started=now().isoformat())
    status_dir.mkdir(parents=True, exist_ok=True)
    with (status_dir / LOCK_FILE).open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            status.ok, status.failed_step = False, "lock"
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
            if outcome.code != 0:
                status.ok, status.failed_step = False, step.name
                status.finished = now().isoformat()
                path = _write_status(status, status_dir)
                last = outcome.output.strip().splitlines()[-1:] or [f"exit code {outcome.code}"]
                notify(
                    NOTIFY_TITLE,
                    f"The {job} job stopped at {step.name}; the published export is unchanged. "
                    f"{last[0][:200]} (see {path})",
                )
                return status
        status.ok = True
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
