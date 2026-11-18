"""A day's release packed from the export, and the gates it passes before anything is
published: the contract and local paths, the site's build, copy check and size, and the
deny list over its files (lineage.live.deploy).

See DESIGN.md#updates, DR-13.
"""

import json
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from lineage.core.hashing import file_sha256
from lineage.export.release import (
    RELEASE_MANIFEST,
    Checksums,
    ReleaseError,
    changed_files,
    package,
    read_checksums,
)
from lineage.export.schema import Meta, Release
from lineage.live.deploy_ledger import DeployError, DeployPaths
from lineage.live.deploy_shell import Shell, run_checked

PAGES_LIMIT = 1_000_000_000  # GitHub Pages' published site limit, as site/scripts/size.mjs has it
SITE_REPORT = Path("reports/site.json")  # where size.mjs writes, committed, so restored after
# A local path as tools/leak_check.py matches it; a test holds the two equal, since the job's
# `lineage` runs where the dev tools are not importable.
LOCAL_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+|(?:/private)?/(?:tmp|var/folders)/")
# A path inside a web address is the site's, not this machine's: bill data links to
# legislatures' pages, some of them under a path named home.
_URL = re.compile(r"[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
SHOWN = 10


# Packing and the data's own checks.


def tag_for(state: str, day: date) -> str:
    return f"data-{state.lower()}-{day.isoformat()}"


def export_day(export: Path) -> date | None:
    meta = Meta.model_validate_json((export / "meta.json").read_text())
    return None if meta.as_of is None else date.fromisoformat(meta.as_of)


@dataclass(frozen=True, slots=True)
class Candidate:
    tag: str
    asset: Path
    checksums: Checksums

    @property
    def pin(self) -> Release:
        return Release(tag=self.tag, asset=self.asset.name, sha256=file_sha256(self.asset))


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
        shown = ", ".join(found[:SHOWN])
        raise DeployError(f"the export names a local path: {shown}")
    out = paths.candidates / tag
    asset, _ = package(export, state=state, tag=tag, out_dir=out, floor=floor)
    checksums = read_checksums(asset)
    (out / "checksums.json").write_text(checksums.model_dump_json(indent=1) + "\n")
    return Candidate(tag=tag, asset=asset, checksums=checksums)


def candidate_of(paths: DeployPaths, tag: str) -> Candidate | None:
    out = paths.candidates / tag
    sums = out / "checksums.json"
    assets = sorted(out.glob("export-*.tar.gz")) if out.exists() else []
    if not sums.exists() or len(assets) != 1:
        return None
    return Candidate(tag, assets[0], Checksums.model_validate_json(sums.read_text()))


def tag_day(tag: str) -> date:
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
        if p.name.startswith(prefix) and tag_day(p.name) < before
    )
    for tag in reversed(tags):
        if (found := candidate_of(paths, tag)) is not None:
            return found
    return None


def pinned_checksums(paths: DeployPaths, shell: Shell) -> tuple[Release, Checksums]:
    """The release the checkout's `site/data-release.json` pins, and its checksum manifest:
    from the candidate packed here when there is one, else downloaded with gh and checked
    against the pin's digest."""
    pin = Release.model_validate_json((paths.checkout / RELEASE_MANIFEST).read_text())
    local = candidate_of(paths, pin.tag)
    if local is not None and file_sha256(local.asset) == pin.sha256:
        return pin, local.checksums
    paths.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.root) as tmp:
        run_checked(
            shell,
            ["gh", "release", "download", pin.tag, "--pattern", pin.asset, "--dir", tmp],
            paths.checkout,
            f"downloading {pin.tag}",
        )
        asset = Path(tmp) / pin.asset
        if not asset.exists() or file_sha256(asset) != pin.sha256:
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
    run_checked(shell, ["npm", "ci"], site, "npm ci")
    run_checked(shell, ["npm", "run", "build"], site, "the site's build")
    run_checked(shell, ["npm", "run", "test:copy"], site, "the copy check")
    gates: dict[str, Any] = {"copy": {"ok": True}}
    try:
        run_checked(shell, ["npm", "run", "size"], site, "the size measure")
        report = json.loads((paths.checkout / SITE_REPORT).read_text())
    finally:
        run_checked(shell, ["git", "checkout", "--", str(SITE_REPORT)], paths.checkout, "restoring")
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
    return {"ok": not found, "files_checked": checked, "files": list(found[:SHOWN]),
            "count": len(found)}  # fmt: skip


def deny_reason(found: Sequence[str]) -> str:
    more = f" and {len(found) - SHOWN} more" if len(found) > SHOWN else ""
    return (
        f"{len(found)} files added or changed name a deny-listed term "
        f"({', '.join(found[:SHOWN])}{more}); read them before releasing"
    )
