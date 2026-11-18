"""The live track's deploy: each day's export packed as a release, gated, held when it jumps
past a normal day, and published (DESIGN.md#updates, DR-13).

Before a hosted state's first day of session (rules/deploy.toml) the site deploys only from a
reviewed release: the daily job stops at the local export, and the weekly job prepares a release
candidate, packed and gated with a short review summary, and publishes nothing
(`prepare_candidate`). From that day the daily job deploys after promoting the export
(`deploy_day`):

1. **Pack** the export as release `data-<st>-<as of>` (lineage.export.release), which refuses any
file
   or field the export contract does not name, and check every file for a local path.
2. **Build the site** from the unpacked release in the job's own checkout of the repository, as
   the Pages workflow builds it, and run the copy check; then measure it with
   `site/scripts/size.mjs` against the Pages limit.
3. **The circuit breaker.** The day's new origin claims (the change log's, as a page or the feed
   lists them) and its changed files (against the release before it) are set against the days
   deployed without a hold; a count past the limit holds the day (`breaker`). The files the day
   adds or changes are read for the leak check's deny list, as whole words; a file naming a
   term holds the day too (`deny_listed`), since public records name many people and whether
   one should be shown is the maintainer's to rule.
4. **Commit the pin** to `site/data-release.json` and run the leak check on that commit, the
   check CI runs, with the deny list from the environment.
5. **Publish:** create the GitHub release with the asset, push the commit, and dispatch the
   Pages workflow, which verifies the asset against the pin before it builds.

A held day publishes nothing and is recorded with its counts and baseline; the daily job alerts
on it as on a failure. Only `release_day`, which the maintainer runs after review, publishes it.
A later day is held behind it rather than published past it, since its release would carry the
held day's changes; once the held day is released, the next day's run publishes as before. Or
the maintainer rejects it (`reject_day`): its release is never published, the live track keeps
its data, and the next day's run publishes as usual, counted against the rejected day's release
so its jump is not counted twice.
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

from lineage.core.paths import DATA
from lineage.export.release import (
    RELEASE_MANIFEST,
    Checksums,
    ReleaseError,
    changed_files,
    package,
    read_checksums,
    write_release_manifest,
)
from lineage.export.schema import Meta, Release
from lineage.live.feed import listed_origin
from lineage.live.updates import Change, OriginGained, OriginLost, read_changes
from lineage.origins.withheld import REPORT as WITHHELD_REPORT
from lineage.origins.withheld import load_withheld

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
    """Whether `day` is on or after the state's first day of session, once the maintainer has
    confirmed that date; a state with none recorded, or with its date unconfirmed, never
    deploys by itself, since deploying without review is what the date turns on."""
    first = rules.first_day.get(state)
    return first is not None and rules.confirmed.get(state, False) and day >= first


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


# The job's GitHub access (ruled at the deploy review): a fine-grained token scoped to this
# repository, read from the keychain before every git or gh call, never the maintainer's own login.
TOKEN_SERVICE = "legislative-lineage-gh-token"
TokenSource = Callable[[], str | None]


def keychain_token() -> str | None:
    """The job's token from the login keychain, or None when it holds none."""
    found = subprocess.run(
        ["security", "find-generic-password", "-w", "-s", TOKEN_SERVICE],
        capture_output=True,
        text=True,
        check=False,
    )
    token = found.stdout.strip()
    return token if found.returncode == 0 and token else None


def github_shell(shell: Shell, token: TokenSource, config_dir: Path) -> Shell:
    """`shell`, with every git and gh call given the job's token, read afresh, and stopped when
    there is none. gh gets GH_TOKEN and a config directory of the job's own, empty, so it cannot
    read a stored login; git clears every credential helper configured anywhere and uses only
    gh's, which answers with that token. Neither may prompt."""

    def run(call: Call) -> Result:
        if call.argv[:1] not in (("git",), ("gh",)):
            return shell(call)
        value = token()
        if not value:
            raise DeployError(
                f"no GitHub token in the keychain ({TOKEN_SERVICE}); the job uses no other login"
            )
        argv = call.argv
        if argv[0] == "git":
            argv = ("git", "-c", "credential.helper=",
                    "-c", "credential.helper=!gh auth git-credential", *argv[1:])  # fmt: skip
        config_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **call.env,
            "GH_TOKEN": value,
            "GH_CONFIG_DIR": str(config_dir),
            "GH_PROMPT_DISABLED": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        return shell(Call(argv, call.cwd, env))

    return run


JOB_SHELL = github_shell(subprocess_shell, keychain_token, DEPLOY_DIR / "gh-config")


def _run(shell: Shell, argv: Sequence[str], cwd: Path, what: str) -> str:
    result = shell(Call(tuple(argv), cwd))
    if result.code != 0:
        tail = result.output.strip().splitlines()[-1:] or [f"exit code {result.code}"]
        raise DeployError(f"{what} failed: {tail[0][:300]}")
    return result.output


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
    (lineage.export.release.check_export, inside `package`) and the local-path check."""
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


# The deny list over the release's files.

DENY_ENV = "LEAK_CHECK_TERMS"


def deny_pattern(terms: str | None = None) -> re.Pattern[str]:
    """The leak check's deny list (tools.leak_check, from the environment), as whole words: a
    term inside a longer public word, such as a surname that contains it, is not the term."""
    raw = os.environ.get(DENY_ENV, "") if terms is None else terms
    words = [t.strip() for t in raw.split(",") if t.strip()]
    if not words:
        raise DeployError(
            f"{DENY_ENV} is not set, so the release's files cannot be read for the deny list"
        )
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(w) for w in words) + r")(?!\w)", re.IGNORECASE
    )


def changed_paths(candidate: Candidate, reference: Checksums | None) -> list[str]:
    """The files a release adds or changes against the one before it; every file, without one."""
    if reference is None:
        return sorted(f.path for f in candidate.checksums.files)
    diff = changed_files(reference, candidate.checksums)
    return diff["added"] + diff["changed"]


def deny_listed(asset: Path, files: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    """Of `files` in the release, those that name a deny-listed term, by path only: the match
    itself is never repeated, as the leak check never repeats it."""
    wanted = set(files)
    found: list[str] = []
    with tarfile.open(asset) as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name not in wanted:
                continue
            handle = tar.extractfile(member)
            if handle is not None and pattern.search(handle.read().decode("utf-8", "replace")):
                found.append(member.name)
    return sorted(found)


def deny_gate(found: Sequence[str], checked: int) -> dict[str, Any]:
    return {"ok": not found, "files_checked": checked, "files": list(found[:_SHOWN]),
            "count": len(found)}  # fmt: skip


def deny_reason(found: Sequence[str]) -> str:
    more = f" and {len(found) - _SHOWN} more" if len(found) > _SHOWN else ""
    return (
        f"{len(found)} files added or changed name a deny-listed term "
        f"({', '.join(found[:_SHOWN])}{more}); read them before releasing"
    )


# The circuit breaker.


def listed_origin_claims(
    changes: Iterable[Change], days: Iterable[date], withheld: frozenset[str] = frozenset()
) -> list[OriginGained]:
    """The origin matches the given days gained that a page or the feed lists: by the larger
    of the two shares, at the listing floor, their document not withheld, as lineage.live.feed lists
    them."""
    wanted = set(days)
    return [c for c in changes if isinstance(c, OriginGained) and c.as_of in wanted and
            listed_origin(c, withheld)]  # fmt: skip


def day_counts(
    candidate: Candidate,
    reference: Checksums | None,
    changes: Sequence[Change],
    days: Iterable[date],
    withheld: frozenset[str] = frozenset(),
) -> dict[str, int]:
    """A day's counts: its listed new origin claims, and the files its release adds, changes or
    removes against the one before (every file, without one)."""
    if reference is None:
        files = len(candidate.checksums.files)
    else:
        diff = changed_files(reference, candidate.checksums)
        files = sum(len(v) for v in diff.values())
    return {
        "origin_claims": len(listed_origin_claims(changes, days, withheld)),
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
    shell: Shell = JOB_SHELL,
    floor: int,
    withheld_report: Path = WITHHELD_REPORT,
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
        pattern = deny_pattern()
        tag = tag_for(state, day)
        candidate = _candidate_of(paths, tag) if resuming else None
        if candidate is None:
            candidate = pack(export, state=state, tag=tag, paths=paths, floor=floor)
            record.gates["contract"] = {"ok": True, "files": len(candidate.checksums.files)}
            record.gates["local_paths"] = {"ok": True}
            record.gates["withheld"] = {"report": withheld_digest(withheld_report)}
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
                load_withheld(withheld_report).ids,
            )
            checked = changed_paths(candidate, None if reference is None else reference.checksums)
            named = deny_listed(candidate.asset, checked, pattern)
            record.gates["deny_list"] = deny_gate(named, len(checked))
            decision = breaker(record.counts, normal_history(paths, day), rules.breaker)
            record.baseline = decision.baseline
            reasons = decision.reasons + ([deny_reason(named)] if named else [])
            if reasons:
                record.state, record.held_on_counts, record.reasons = "held", True, reasons
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


def withheld_digest(path: Path = WITHHELD_REPORT) -> str:
    """The withhold report as it stood when a release was packed, so a release made after the
    maintainer withholds documents can tell it is out of date."""
    return _sha256(path) if path.exists() else "none"


def exported_withheld(export: Path) -> frozenset[str]:
    """The documents the export's sources page lists as withheld: what its pages were written
    under."""
    doc = json.loads((export / "sources.json").read_text())
    return frozenset(w["id"] for s in doc["sources"] for w in s.get("withheld", []))


def release_day(
    day: date,
    *,
    paths: DeployPaths | None = None,
    rules: DeployRules | None = None,
    shell: Shell = JOB_SHELL,
    repack: bool = False,
    export: Path | None = None,
    state: str = "GA",
    floor: int = 0,
    withheld_report: Path = WITHHELD_REPORT,
) -> DayRecord:
    """Approve a held day after the maintainer's review and publish it, its site gates run
    again. Refused while an earlier day is held on its counts, so no day is published past
    another.

    As packed, its own release goes out, unless the withhold report has changed since it was
    packed: its pages would still list the documents just withheld. With `repack`, after
    withholding (`lineage withhold --doc`, then `publish pages`, `feed` and `promote`), the live
    `export` as it now stands is packed instead, as of its own day, and refused unless its
    sources page lists exactly the report's withheld documents; days held behind the reviewed
    one up to then go out with it."""
    paths = paths if paths is not None else DeployPaths()
    rules = rules if rules is not None else load_rules()
    record = read_record(paths, day)
    if record is None or record.state != "held":
        state_now = "no record" if record is None else record.state
        raise DeployError(f"{day} is not held ({state_now}); only a held day is released")
    blocker = blocking_day(paths, day)
    if blocker is not None:
        raise DeployError(f"{blocker.day} is held before it; release that day first")
    digest = withheld_digest(withheld_report)
    if not repack:
        packed = record.gates.get("withheld", {}).get("report")
        if packed is not None and packed != digest:
            raise DeployError(
                f"the withhold report has changed since {day} was packed; publish pages, feed "
                "and promote, then release it with --repack"
            )
        return _publish_held(record, paths=paths, rules=rules, shell=shell, reviewed=True)
    if export is None:
        raise DeployError("--repack packs the live export; name it")
    through = export_day(export)
    if through is None or through < day:
        raise DeployError(f"the live export is not as of {day} or later")
    later = [r for r in records(paths) if day < r.day <= through and r.held_on_counts
             and r.state == "held"]  # fmt: skip
    if later:
        raise DeployError(f"{later[0].day} is held on its own counts too; review it first")
    if exported_withheld(export) != load_withheld(withheld_report).ids:
        raise DeployError(
            "the live export was not written under the withhold report as it stands; run "
            "`lineage publish pages`, `publish feed` and `publish promote` first"
        )
    candidate = pack(export, state=state, tag=tag_for(state, through), paths=paths, floor=floor)
    record.tag, record.asset = candidate.tag, candidate.asset.name
    record.gates["contract"] = {"ok": True, "files": len(candidate.checksums.files)}
    record.gates["local_paths"] = {"ok": True}
    record.gates["withheld"] = {"report": digest}
    record.reasons = [*record.reasons, f"approved after review; repacked as of {through}"]
    write_record(paths, record)
    out = _publish_held(record, paths=paths, rules=rules, shell=shell, reviewed=True)
    for r in records(paths):
        if day < r.day <= through and r.state == "held":
            r.state = "superseded"
            write_record(paths, r)
    return out


def reject_day(day: date, *, paths: DeployPaths | None = None) -> DayRecord:
    """Reject a held day after the maintainer's review: its release is never published and
    nothing leaves this machine. The live track keeps its data, so the next day's release
    carries it, counted against the rejected day's release, which stays packed as the
    reference (`previous_candidate`); a day held behind it waits no longer, and is published
    with the next day's release. A rejected day does not set what a normal day is."""
    paths = paths if paths is not None else DeployPaths()
    record = read_record(paths, day)
    if record is None or record.state != "held":
        state = "no record" if record is None else record.state
        raise DeployError(f"{day} is not held ({state}); only a held day is rejected")
    record.state = "rejected"
    record.reasons = [*record.reasons, f"rejected after review on {datetime.now(UTC).date()}"]
    write_record(paths, record)
    return record


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
    withheld: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """The listed origin claims first listed in the `days` days ending `through`, each with its
    bill, document, both shares, stages, the day first listed, and whether it still stands; a
    withheld document's are not listed, so not reviewed."""
    start = through - timedelta(days=days - 1)
    window = [start + timedelta(days=n) for n in range(days)]
    claims = listed_origin_claims(changes, window, withheld)
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
    withheld_report: Path = WITHHELD_REPORT,
) -> Path:
    """The week's new origin claims, for the maintainer's weekly review during session ->
    `<deploy>/reviews/<through>.json` and `.txt`."""
    paths = paths if paths is not None else DeployPaths()
    claims = origin_claims_review(
        read_changes(changes_dir),
        through=through,
        describe=describe,
        documents=documents,
        withheld=load_withheld(withheld_report).ids,
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
    shell: Shell = JOB_SHELL,
    describe: Callable[[list[str]], Mapping[str, Mapping[str, Any]]] = lambda _: {},
    documents: Mapping[str, Mapping[str, Any]] | None = None,
    floor: int,
    withheld_report: Path = WITHHELD_REPORT,
) -> dict[str, Any]:
    """Before session: the live export packed and gated as a release candidate, with a review
    summary of the files it changes against the pinned release and the week's new origin
    claims -> `<deploy>/candidates/<tag>/review.json` and `.txt`. Publishes nothing."""
    paths = paths if paths is not None else DeployPaths()
    rules = rules if rules is not None else load_rules()
    pattern = deny_pattern()
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
    checked = changed_paths(candidate, pinned)
    named = deny_listed(candidate.asset, checked, pattern)
    gates["deny_list"] = deny_gate(named, len(checked))
    claims = origin_claims_review(
        read_changes(changes_dir),
        through=as_of,
        describe=describe,
        documents=documents,
        withheld=load_withheld(withheld_report).ids,
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
        (
            f"Deny list: {deny_reason(named)}."
            if named
            else f"Deny list: none of the {len(checked)} files added or changed names a term."
        ),
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
    from lineage.live.feed import bill_infos  # the feed's naming pulls in the export's stack

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
    from lineage.origins.collect import PACKAGE, collectors
    from lineage.origins.modeldocs import MODELDOCS_DIR, read_docs

    root = modeldocs_root if modeldocs_root is not None else MODELDOCS_DIR
    return {
        d.id: {"title": d.title, "organization": c.source.organization}
        for c in collectors(collector_package or PACKAGE).values()
        for d in read_docs(c.source, root)
    }
