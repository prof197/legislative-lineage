"""The live track's deploy: each day's export packed as a release, gated, held when it jumps
past a normal day, and published (DESIGN.md#updates, DR-13).

Before a hosted state's first day of session (rules/deploy.toml) the site deploys only from a
reviewed release: the daily job stops at the local export, and the weekly job prepares a release
candidate, packed and gated with a short review summary, and publishes nothing
(`prepare_candidate`). From that day the daily job deploys after promoting the export
(`deploy_day`):

1. **Pack** the export as release `data-<st>-<as of>` (lineage.release), which refuses any file
   or field the export contract does not name, and check every file for a local path.
2. **Build the site** from the unpacked release in the job's own checkout of the repository, as
   the Pages workflow builds it, and run the copy check; then measure it with
   `site/scripts/size.mjs` against the Pages limit.
3. **The circuit breaker.** The day's new origin claims (the change log's, as a page or the feed
   lists them) and its changed files (against the release before it) are set against the days
   deployed without a hold; a count past the limit holds the day (`breaker`).
4. **Commit the pin** to `site/data-release.json` and run the leak check on that commit, the
   check CI runs, with the deny list from the environment.
5. **Publish:** create the GitHub release with the asset, push the commit, and dispatch the
   Pages workflow, which verifies the asset against the pin before it builds.

A held day publishes nothing and is recorded with its counts and baseline; the daily job alerts
on it as on a failure. Only `release_day`, which the maintainer runs after review, publishes it.
A later day is held behind it rather than published past it, since its release would carry the
held day's changes; once the held day is released, the next day's run publishes as before.
Every step that leaves this machine (git, gh, npm, the leak check) goes through an injectable
`Shell`, so the tests run the whole flow with fakes. Each day's record, under `<data>/deploy/`,
lists the steps done, so a run stopped part way resumes rather than publishing twice.

See DESIGN.md#updates, DR-13.
"""

import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from lineage.export_schema import Meta, Release
from lineage.origins import FLOOR
from lineage.paths import DATA
from lineage.release import (
    RELEASE_MANIFEST,
    Checksums,
    ReleaseError,
    changed_files,
    package,
    read_checksums,
    write_release_manifest,
)
from lineage.updates import Change, OriginGained, OriginLost, read_changes

RULES = Path("rules/deploy.toml")
DEPLOY_DIR = DATA / "deploy"
HELD = 3  # the exit code of a held day: neither a success nor a failed step
KINDS = ("origin_claims", "changed_files")
PAGES_LIMIT = 1_000_000_000  # GitHub Pages' published site limit, as site/scripts/size.mjs has it
SITE_REPORT = Path("reports/site.json")  # where size.mjs writes, committed, so restored after
WORKFLOW = "pages.yml"
BRANCH = "main"
REVIEW_DAYS = 7
# A local path as tools/leak_check.py matches it; a test holds the two equal, since the job's
# `lineage` runs where the dev tools are not importable.
LOCAL_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+|(?:/private)?/(?:tmp|var/folders)/")
# A path inside a web address is the site's, not this machine's: bill data links to
# legislatures' pages, some of them under a path named home.
_URL = re.compile(r"[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_SHOWN = 10


class DeployError(Exception):
    pass


# The rules.


@dataclass(frozen=True, slots=True)
class Limits:
    floor: int
    ceiling: int


@dataclass(frozen=True, slots=True)
class Breaker:
    window: int
    min_history: int
    multiple: float
    limits: Mapping[str, Limits]


@dataclass(frozen=True, slots=True)
class DeployRules:
    first_day: Mapping[str, date]
    confirmed: Mapping[str, bool]
    breaker: Breaker
    max_share: float  # of the Pages limit, for the size gate


def load_rules(path: Path = RULES) -> DeployRules:
    doc = tomllib.loads(path.read_text())
    sessions: dict[str, Any] = doc.get("session", {})
    b = doc["breaker"]
    return DeployRules(
        first_day={st: s["first_day"] for st, s in sessions.items()},
        confirmed={st: bool(s.get("confirmed", False)) for st, s in sessions.items()},
        breaker=Breaker(
            window=int(b["window"]),
            min_history=int(b["min_history"]),
            multiple=float(b["multiple"]),
            limits={k: Limits(int(b[k]["floor"]), int(b[k]["ceiling"])) for k in KINDS},
        ),
        max_share=float(doc["size"]["max_share_of_pages_limit"]),
    )


def in_session(rules: DeployRules, state: str, day: date) -> bool:
    """Whether `day` is on or after the state's first day of session; a state with none
    recorded never deploys by itself."""
    first = rules.first_day.get(state)
    return first is not None and day >= first


# What leaves this machine.


@dataclass(frozen=True, slots=True)
class Call:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict[str, str])  # added to the job's own


@dataclass(frozen=True, slots=True)
class Result:
    code: int
    output: str = ""


Shell = Callable[[Call], Result]


def subprocess_shell(call: Call) -> Result:
    done = subprocess.run(
        list(call.argv),
        cwd=call.cwd,
        env={**os.environ, **call.env},
        capture_output=True,
        text=True,
        check=False,
    )
    return Result(done.returncode, done.stdout + done.stderr)


def _run(shell: Shell, argv: Sequence[str], cwd: Path, what: str) -> str:
    result = shell(Call(tuple(argv), cwd))
    if result.code != 0:
        tail = result.output.strip().splitlines()[-1:] or [f"exit code {result.code}"]
        raise DeployError(f"{what} failed: {tail[0][:300]}")
    return result.output


# The ledger: one record per day the daily job deployed or tried to.

DayState = Literal["deployed", "held", "released", "failed", "superseded"]


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


# Packing and the data's own checks.


def tag_for(state: str, day: date) -> str:
    return f"data-{state.lower()}-{day.isoformat()}"


def export_day(export: Path) -> date | None:
    meta = Meta.model_validate_json((export / "meta.json").read_text())
    return None if meta.as_of is None else date.fromisoformat(meta.as_of)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Candidate:
    tag: str
    asset: Path
    checksums: Checksums

    @property
    def pin(self) -> Release:
        return Release(tag=self.tag, asset=self.asset.name, sha256=_sha256(self.asset))


def local_paths(export: Path) -> list[str]:
    """Every file of the export that names a local path, with the line: a release names none.
    Web addresses are set aside first. Deny-listed terms are not scanned for here: public
    records name many people, and whether they should be is the maintainer's to rule
    (DESIGN.md#updates)."""
    found: list[str] = []
    for p in sorted(export.rglob("*")):
        if not p.is_file():
            continue
        text = p.read_bytes().decode("utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), start=1):
            if LOCAL_PATH.search(_URL.sub(" ", line)):
                found.append(f"{p.relative_to(export).as_posix()}:{n}")
    return found


def pack(export: Path, *, state: str, tag: str, paths: DeployPaths, floor: int) -> Candidate:
    """The export as release `tag`, under `<deploy>/candidates/<tag>/`: the contract's check
    (lineage.release.check_export, inside `package`) and the local-path check."""
    found = local_paths(export)
    if found:
        shown = ", ".join(found[:_SHOWN])
        raise DeployError(f"the export names a local path: {shown}")
    out = paths.candidates / tag
    asset, _ = package(export, state=state, tag=tag, out_dir=out, floor=floor)
    checksums = read_checksums(asset)
    (out / "checksums.json").write_text(checksums.model_dump_json(indent=1) + "\n")
    return Candidate(tag=tag, asset=asset, checksums=checksums)


def _candidate_of(paths: DeployPaths, tag: str) -> Candidate | None:
    out = paths.candidates / tag
    sums = out / "checksums.json"
    assets = sorted(out.glob("export-*.tar.gz")) if out.exists() else []
    if not sums.exists() or len(assets) != 1:
        return None
    return Candidate(tag, assets[0], Checksums.model_validate_json(sums.read_text()))


def _tag_day(tag: str) -> date:
    return date.fromisoformat(tag[-10:])


def previous_candidate(paths: DeployPaths, state: str, before: date) -> Candidate | None:
    """The latest release packed here for a day before `before`, deployed or not: the
    reference a day's changed files are counted against."""
    if not paths.candidates.exists():
        return None
    prefix = f"data-{state.lower()}-"
    tags = sorted(
        p.name
        for p in paths.candidates.iterdir()
        if p.name.startswith(prefix) and _tag_day(p.name) < before
    )
    for tag in reversed(tags):
        if (found := _candidate_of(paths, tag)) is not None:
            return found
    return None


def pinned_checksums(paths: DeployPaths, shell: Shell) -> tuple[Release, Checksums]:
    """The release the checkout's `site/data-release.json` pins, and its checksum manifest:
    from the candidate packed here when there is one, else downloaded with gh and checked
    against the pin's digest."""
    pin = Release.model_validate_json((paths.checkout / RELEASE_MANIFEST).read_text())
    local = _candidate_of(paths, pin.tag)
    if local is not None and _sha256(local.asset) == pin.sha256:
        return pin, local.checksums
    paths.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.root) as tmp:
        _run(
            shell,
            ["gh", "release", "download", pin.tag, "--pattern", pin.asset, "--dir", tmp],
            paths.checkout,
            f"downloading {pin.tag}",
        )
        asset = Path(tmp) / pin.asset
        if not asset.exists() or _sha256(asset) != pin.sha256:
            raise DeployError(f"{pin.asset} does not match the digest the pin commits")
        try:
            return pin, read_checksums(asset)
        except ReleaseError as e:  # a release packed before the manifest existed
            raise DeployError(f"{pin.tag} has no checksum manifest to compare with: {e}") from e


# The site's gates, in the job's checkout.


def site_gates(
    candidate: Candidate, *, paths: DeployPaths, shell: Shell, max_share: float
) -> dict[str, Any]:
    """Build the site from the unpacked release as the Pages workflow does, run the copy check,
    and measure the site against the Pages limit; returns both gates' results, raising on the
    first that fails. The size report the measure writes is committed, so it is restored."""
    site = paths.checkout / "site"
    data = site / "public" / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    with tarfile.open(candidate.asset) as tar:
        tar.extractall(data, filter="data")
    (data / "release.json").write_text(json.dumps(candidate.pin.model_dump(), indent=1) + "\n")
    _run(shell, ["npm", "ci"], site, "npm ci")
    _run(shell, ["npm", "run", "build"], site, "the site's build")
    _run(shell, ["npm", "run", "test:copy"], site, "the copy check")
    gates: dict[str, Any] = {"copy": {"ok": True}}
    try:
        _run(shell, ["npm", "run", "size"], site, "the size measure")
        report = json.loads((paths.checkout / SITE_REPORT).read_text())
    finally:
        _run(shell, ["git", "checkout", "--", str(SITE_REPORT)], paths.checkout, "restoring")
    share = float(report["total_bytes"]) / PAGES_LIMIT
    gates["size"] = {
        "ok": share <= max_share,
        "total_bytes": int(report["total_bytes"]),
        "share_of_pages_limit": round(share, 4),
        "max_share": max_share,
    }
    if share > max_share:
        raise DeployError(
            f"the site is {share:.1%} of the Pages limit, past the {max_share:.0%} the size gate "
            "allows (DR-2)"
        )
    return gates


# The circuit breaker.


def listed_origin_claims(changes: Iterable[Change], days: Iterable[date]) -> list[OriginGained]:
    """The origin matches the given days gained that a page or the feed lists: by the larger
    of the two shares, at the listing floor, as lineage.feed lists them."""
    wanted = set(days)
    return [
        c
        for c in changes
        if isinstance(c, OriginGained)
        and c.as_of in wanted
        and max(c.bill_share, c.model_share) >= FLOOR
    ]


def day_counts(
    candidate: Candidate,
    reference: Checksums | None,
    changes: Sequence[Change],
    days: Iterable[date],
) -> dict[str, int]:
    """A day's counts: its listed new origin claims, and the files its release adds, changes or
    removes against the one before (every file, without one)."""
    if reference is None:
        files = len(candidate.checksums.files)
    else:
        diff = changed_files(reference, candidate.checksums)
        files = sum(len(v) for v in diff.values())
    return {
        "origin_claims": len(listed_origin_claims(changes, days)),
        "changed_files": files,
    }


@dataclass(frozen=True, slots=True)
class Decision:
    hold: bool
    reasons: list[str]
    baseline: dict[str, Any]


def breaker(
    counts: Mapping[str, int], history: Sequence[Mapping[str, int]], b: Breaker
) -> Decision:
    """Hold when a count passes its limit: with at least `min_history` normal days, both
    `multiple` times the median of the last `window` and the floor; with fewer, the ceiling."""
    recent = list(history)[-b.window :]
    reasons: list[str] = []
    baseline: dict[str, Any] = {"days": len(recent)}
    for kind in KINDS:
        limits = b.limits[kind]
        if len(recent) < b.min_history:
            limit = float(limits.ceiling)
            baseline[kind] = {"median": None, "limit": limit, "rule": "ceiling"}
            why = (
                f"its ceiling of {limits.ceiling} while fewer than {b.min_history} days are "
                "deployed"
            )
        else:
            median = float(statistics.median(h[kind] for h in recent))
            limit = max(b.multiple * median, float(limits.floor))
            baseline[kind] = {"median": median, "limit": limit, "rule": "median"}
            why = (
                f"{b.multiple:g} times the median of {median:g} over {len(recent)} days and the "
                f"floor of {limits.floor}"
            )
        if counts[kind] > limit:
            reasons.append(f"{counts[kind]} {kind.replace('_', ' ')}, past {why}")
    return Decision(hold=bool(reasons), reasons=reasons, baseline=baseline)


def normal_history(paths: DeployPaths, before: date) -> list[dict[str, int]]:
    """The counts of the days deployed without a hold before `before`, oldest first. A day held
    and then released was a jump by definition, so it does not set what normal is."""
    return [r.counts for r in records(paths) if r.day < before and r.state == "deployed"]


# Publishing.


def _commit_message(tag: str, counts: Mapping[str, int], *, reviewed: bool) -> tuple[str, str]:
    subject = f"build(data): {tag} from the live track"
    how = (
        "released after the maintainer's review of a held day"
        if reviewed
        else ("deployed by the daily job behind its gates and circuit breaker")
    )
    body = (
        f"Pins {tag}, {how} (DESIGN.md#updates). "
        f"{counts.get('changed_files', 0)} files changed against the release before it; "
        f"{counts.get('origin_claims', 0)} new origin claims listed."
    )
    return subject, body


def publish(
    record: DayRecord,
    candidate: Candidate,
    *,
    paths: DeployPaths,
    shell: Shell,
    reviewed: bool = False,
) -> DayRecord:
    """Commit the pin, check the commit for leaks, then create the release, push and dispatch
    the Pages workflow; each step is recorded as done, so a stopped run resumes after it."""
    checkout = paths.checkout

    def step(name: str, action: Callable[[], object]) -> None:
        if name in record.done:
            return
        action()
        record.done.append(name)
        write_record(paths, record)

    def commit() -> None:
        write_release_manifest(candidate.asset, tag=candidate.tag, path=checkout / RELEASE_MANIFEST)
        subject, body = _commit_message(candidate.tag, record.counts, reviewed=reviewed)
        _run(shell, ["git", "add", str(RELEASE_MANIFEST)], checkout, "git add")
        _run(shell, ["git", "commit", "-m", subject, "-m", body], checkout, "git commit")

    def leak() -> None:
        result = shell(
            Call(
                ("uv", "run", "--frozen", "python", "-m", "tools.leak_check", "--history",
                 "--require-terms"),
                checkout,
            )
        )  # fmt: skip
        if result.code != 0:
            # The commit never leaves this machine: it is undone, and the day fails.
            _run(shell, ["git", "reset", "--hard", "HEAD~1"], checkout, "undoing the commit")
            record.done.remove("commit")
            tail = result.output.strip().splitlines()[-1:] or [f"exit code {result.code}"]
            raise DeployError(f"the leak check failed on the commit: {tail[0][:300]}")

    notes = (
        "The live track's data for the site, packed by `lineage publish deploy` "
        "(DESIGN.md#updates). checksums.json inside lists every file's SHA-256."
    )
    step("commit", commit)
    step("leak", leak)
    step(
        "release",
        lambda: _run(
            shell,
            ["gh", "release", "create", candidate.tag, str(candidate.asset), "--title",
             candidate.tag, "--notes", notes],
            checkout,
            "gh release create",
        ),
    )  # fmt: skip
    step(
        "push",
        lambda: _run(shell, ["git", "push", "origin", f"HEAD:{BRANCH}"], checkout, "git push"),
    )
    step(
        "dispatch",
        lambda: _run(
            shell, ["gh", "workflow", "run", WORKFLOW, "--ref", BRANCH], checkout, "dispatch"
        ),
    )
    return record


def _ready_checkout(paths: DeployPaths, shell: Shell, *, resuming: bool) -> None:
    """The checkout brought up to main, and clean but for a commit a stopped run left."""
    if not (paths.checkout / ".git").exists():
        raise DeployError(
            f"no checkout of the repository at {paths.checkout}; clone it there once, on {BRANCH}"
        )
    if not resuming:
        _run(shell, ["git", "pull", "--ff-only", "origin", BRANCH], paths.checkout, "git pull")
    dirty = _run(shell, ["git", "status", "--porcelain"], paths.checkout, "git status")
    if dirty.strip():
        raise DeployError(f"the checkout at {paths.checkout} has uncommitted changes")


def deploy_day(
    *,
    state: str,
    export: Path,
    changes_dir: Path,
    paths: DeployPaths | None = None,
    rules: DeployRules | None = None,
    shell: Shell = subprocess_shell,
    floor: int,
) -> DayRecord:
    """Deploy the live export, as of the last day applied to it: pack, gate, check the breaker,
    publish. Returns the day's record: deployed (or already), held, publishing nothing, or
    failed, raised after it is recorded."""
    paths = paths if paths is not None else DeployPaths()
    rules = rules if rules is not None else load_rules()
    day = export_day(export)
    if day is None:
        raise DeployError(f"{export} is a full export; the live track deploys a day applied to it")
    record = read_record(paths, day)
    if record is not None and record.state in ("deployed", "released", "superseded"):
        return record
    if record is not None and record.state == "held":
        # Held on its counts, it waits for `release_day`; held behind a day since released, it
        # goes out as it would have, its gates run again on the checkout as it stands.
        if record.held_on_counts or blocking_day(paths, day) is not None:
            return record
        try:
            return _publish_held(record, paths=paths, rules=rules, shell=shell, reviewed=False)
        except (DeployError, ReleaseError, OSError) as e:
            record.state, record.error = "failed", str(e)
            write_record(paths, record)
            raise DeployError(str(e)) from e
    resuming = record is not None and bool(record.done)
    record = record if resuming and record is not None else DayRecord(day=day, state="failed")
    try:
        tag = tag_for(state, day)
        candidate = _candidate_of(paths, tag) if resuming else None
        if candidate is None:
            candidate = pack(export, state=state, tag=tag, paths=paths, floor=floor)
            record.gates["contract"] = {"ok": True, "files": len(candidate.checksums.files)}
            record.gates["local_paths"] = {"ok": True}
        record.tag, record.asset = tag, candidate.asset.name
        if not resuming:
            _ready_checkout(paths, shell, resuming=False)
            record.gates |= site_gates(
                candidate, paths=paths, shell=shell, max_share=rules.max_share
            )
            reference = previous_candidate(paths, state, day)
            since = _tag_day(reference.tag) if reference is not None else None
            logged = sorted({c.as_of for c in read_changes(changes_dir)})
            days = [d for d in logged if d <= day and (since is None or d > since)]
            record.counts = day_counts(
                candidate,
                None if reference is None else reference.checksums,
                read_changes(changes_dir),
                days,
            )
            decision = breaker(record.counts, normal_history(paths, day), rules.breaker)
            record.baseline = decision.baseline
            if decision.hold:
                record.state, record.held_on_counts, record.reasons = "held", True, decision.reasons
                write_record(paths, record)
                return record
            blocker = blocking_day(paths, day)
            if blocker is not None:
                record.state, record.held_behind = "held", blocker.day
                record.reasons = [f"held behind {blocker.day}, which awaits review"]
                write_record(paths, record)
                return record
        else:
            _ready_checkout(paths, shell, resuming=True)
        record.state = "failed"
        record.error = None
        write_record(paths, record)
        publish(record, candidate, paths=paths, shell=shell)
        record.state = "deployed"
        write_record(paths, record)
        _supersede(paths, day)
        return record
    except (DeployError, ReleaseError, OSError) as e:
        record.state, record.error = "failed", str(e)
        write_record(paths, record)
        raise DeployError(str(e)) from e


def _supersede(paths: DeployPaths, day: date) -> None:
    """Days held behind another and published now as part of `day`'s release."""
    for r in records(paths):
        if r.day < day and r.state == "held" and not r.held_on_counts:
            r.state = "superseded"
            write_record(paths, r)


def release_day(
    day: date,
    *,
    paths: DeployPaths | None = None,
    rules: DeployRules | None = None,
    shell: Shell = subprocess_shell,
) -> DayRecord:
    """Publish a held day after the maintainer's review: its own release, packed and gated when
    it was held, with the site's gates run again on the checkout as it stands. Refused while an
    earlier day is held on its counts, so no day is published past another."""
    paths = paths if paths is not None else DeployPaths()
    rules = rules if rules is not None else load_rules()
    record = read_record(paths, day)
    if record is None or record.state != "held":
        state = "no record" if record is None else record.state
        raise DeployError(f"{day} is not held ({state}); only a held day is released")
    blocker = blocking_day(paths, day)
    if blocker is not None:
        raise DeployError(f"{blocker.day} is held before it; release that day first")
    return _publish_held(record, paths=paths, rules=rules, shell=shell, reviewed=True)


def _publish_held(
    record: DayRecord, *, paths: DeployPaths, rules: DeployRules, shell: Shell, reviewed: bool
) -> DayRecord:
    """A held day's own release published: released after review, or deployed once the day it
    waited behind is out."""
    candidate = _candidate_of(paths, record.tag or "")
    if candidate is None:
        raise DeployError(f"{record.day}'s release is not under {paths.candidates}")
    resuming = bool(record.done)
    _ready_checkout(paths, shell, resuming=resuming)
    if not resuming:
        record.gates |= site_gates(candidate, paths=paths, shell=shell, max_share=rules.max_share)
    publish(record, candidate, paths=paths, shell=shell, reviewed=reviewed)
    record.state = "released" if reviewed else "deployed"
    record.error = None
    write_record(paths, record)
    _supersede(paths, record.day)
    return record


# Before session: a candidate for review. During session: the week's origin claims.


def origin_claims_review(
    changes: Sequence[Change],
    *,
    through: date,
    days: int = REVIEW_DAYS,
    describe: Callable[[list[str]], Mapping[str, Mapping[str, Any]]] = lambda _: {},
    documents: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The listed origin claims first listed in the `days` days ending `through`, each with its
    bill, document, both shares, stages, the day first listed, and whether it still stands."""
    start = through - timedelta(days=days - 1)
    window = [start + timedelta(days=n) for n in range(days)]
    claims = listed_origin_claims(changes, window)
    lost = {(c.bill_id, c.doc_id): c.as_of for c in changes if isinstance(c, OriginLost)}
    bills = describe(sorted({c.bill_id for c in claims}))
    docs = documents or {}
    out: list[dict[str, Any]] = []
    for c in sorted(claims, key=lambda c: (c.as_of, c.bill_id, c.doc_id)):
        gone = lost.get((c.bill_id, c.doc_id))
        out.append(
            {
                "bill_id": c.bill_id,
                "bill": dict(bills.get(c.bill_id, {})),
                "doc_id": c.doc_id,
                "document": dict(docs.get(c.doc_id, {})),
                "bill_share": c.bill_share,
                "model_share": c.model_share,
                "stages": list(c.stages),
                "first_listed": c.as_of.isoformat(),
                "still_listed": gone is None or gone < c.as_of,
            }
        )
    return out


def _claim_line(c: Mapping[str, Any]) -> str:
    bill = c["bill"]
    name = f"{bill['state']} {bill['identifier']}" if bill else c["bill_id"]
    doc = c["document"]
    title = f"“{doc['title']}” ({doc['organization']})" if doc else c["doc_id"]
    standing = "" if c["still_listed"] else "; no longer listed"
    return (
        f"- {c['first_listed']}: {name} and {title}: {c['bill_share']:.0%} of the bill's wording, "
        f"{c['model_share']:.0%} of the model's; {' and '.join(c['stages'])}{standing}"
    )


def write_review(
    claims: Sequence[Mapping[str, Any]], *, through: date, out: Path, extra: Sequence[str] = ()
) -> Path:
    """The review as JSON and as text beside it, for the maintainer to read."""
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"through": through.isoformat(), "days": REVIEW_DAYS, "origin_claims": list(claims)}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    lines = [
        *extra,
        f"New origin claims listed in the {REVIEW_DAYS} days through {through}: {len(claims)}",
        *(_claim_line(c) for c in claims),
    ]
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    return out


def weekly_review(
    through: date,
    *,
    changes_dir: Path,
    paths: DeployPaths | None = None,
    describe: Callable[[list[str]], Mapping[str, Mapping[str, Any]]] = lambda _: {},
    documents: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """The week's new origin claims, for the maintainer's weekly review during session ->
    `<deploy>/reviews/<through>.json` and `.txt`."""
    paths = paths if paths is not None else DeployPaths()
    claims = origin_claims_review(
        read_changes(changes_dir), through=through, describe=describe, documents=documents
    )
    return write_review(claims, through=through, out=paths.reviews / f"{through.isoformat()}.json")


def prepare_candidate(
    day: date,
    *,
    state: str,
    export: Path,
    changes_dir: Path,
    paths: DeployPaths | None = None,
    rules: DeployRules | None = None,
    shell: Shell = subprocess_shell,
    describe: Callable[[list[str]], Mapping[str, Mapping[str, Any]]] = lambda _: {},
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    floor: int,
) -> dict[str, Any]:
    """Before session: the live export packed and gated as a release candidate, with a review
    summary of the files it changes against the pinned release and the week's new origin
    claims -> `<deploy>/candidates/<tag>/review.json` and `.txt`. Publishes nothing."""
    paths = paths if paths is not None else DeployPaths()
    rules = rules if rules is not None else load_rules()
    as_of = export_day(export) or day
    tag = tag_for(state, as_of)
    candidate = pack(export, state=state, tag=tag, paths=paths, floor=floor)
    _ready_checkout(paths, shell, resuming=False)
    gates: dict[str, Any] = {
        "contract": {"ok": True, "files": len(candidate.checksums.files)},
        "local_paths": {"ok": True},
    }
    gates |= site_gates(candidate, paths=paths, shell=shell, max_share=rules.max_share)
    pin, pinned = pinned_checksums(paths, shell)
    diff = changed_files(pinned, candidate.checksums)
    claims = origin_claims_review(
        read_changes(changes_dir), through=as_of, describe=describe, documents=documents
    )
    by_part: dict[str, int] = {}
    for paths_changed in diff.values():
        for p in paths_changed:
            part = p.split("/")[0] if "/" in p else p
            by_part[part] = by_part.get(part, 0) + 1
    summary = [
        f"Release candidate {tag} for review, packed from the live export as of {as_of}; "
        "nothing is published.",
        f"Against the pinned release {pin.tag}: {len(diff['added'])} files added, "
        f"{len(diff['changed'])} changed, {len(diff['removed'])} removed "
        f"({', '.join(f'{k} {v}' for k, v in sorted(by_part.items())) or 'none'}).",
        f"Gates passed: contract ({gates['contract']['files']} files), local paths, copy, size "
        f"({gates['size']['share_of_pages_limit']:.1%} of the Pages limit). The leak check runs "
        "on the commit that pins it.",
        f"To publish it after review: gh release create {tag} {candidate.asset}, then commit its "
        f"pin to {RELEASE_MANIFEST} and run the Pages workflow (DESIGN.md#export).",
        "",
    ]
    out = paths.candidates / tag / "review.json"
    write_review(claims, through=as_of, out=out, extra=summary)
    doc = json.loads(out.read_text()) | {
        "tag": tag,
        "asset": candidate.asset.name,
        "pinned": pin.tag,
        "files": diff,
        "gates": gates,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def describe_bills(
    tables: Path, *, state: str, site_url: str
) -> Callable[[list[str]], dict[str, dict[str, Any]]]:
    """Each bill a review names, as the feed names it: state, identifier, title and link."""
    from lineage.feed import bill_infos  # the feed's naming pulls in the export's stack

    def describe(ids: list[str]) -> dict[str, dict[str, Any]]:
        infos = bill_infos(tables, state=state, site_url=site_url.rstrip("/"), ids=ids)
        return {
            b: {"state": i.state, "identifier": i.identifier, "title": i.title, "link": i.link}
            for b, i in infos.items()
        }

    return describe


def document_names(
    modeldocs_root: Path | None = None, collector_package: str | None = None
) -> dict[str, dict[str, Any]]:
    """Each collected model document's title and the organization whose library holds it."""
    from lineage.collect import PACKAGE, collectors
    from lineage.modeldocs import MODELDOCS_DIR, read_docs

    root = modeldocs_root if modeldocs_root is not None else MODELDOCS_DIR
    return {
        d.id: {"title": d.title, "organization": c.source.organization}
        for c in collectors(collector_package or PACKAGE).values()
        for d in read_docs(c.source, root)
    }
