"""site/scripts/fetch-data.sh, the deploy's download of the pinned data release
(DESIGN.md#export): the asset is checked against the digest committed in data-release.json,
then every unpacked file against the checksum manifest inside it, with none missing and none
extra, before public/data is replaced. Run here with a fake `gh` that serves a local asset."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from lineage.core.paths import DATA
from lineage.export.release import CHECKSUMS, package
from tests.fixture_export import COMMITTED

REPO = Path(__file__).parent.parent
SCRIPT = REPO / "site" / "scripts" / "fetch-data.sh"
PINNED = json.loads((REPO / "site" / "data-release.json").read_text())
TAG = "data-ut-2026-11-16"
ASSET = f"export-UT-{TAG}.tar.gz"
# Serves `gh release download TAG --pattern ASSET --dir DIR` from $FAKE_RELEASES/TAG/ASSET.
FAKE_GH = """#!/usr/bin/env bash
set -euo pipefail
[ "$1 $2" = "release download" ] || { echo "fake gh: unexpected $*" >&2; exit 64; }
tag="$3"; shift 3
while [ $# -gt 0 ]; do
  case "$1" in
    --pattern) pattern="$2"; shift 2 ;;
    --dir) dir="$2"; shift 2 ;;
    *) echo "fake gh: unexpected $1" >&2; exit 64 ;;
  esac
done
cp "$FAKE_RELEASES/$tag/$pattern" "$dir/"
"""

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("node") is None or shutil.which("tar") is None,
    reason="fetch-data.sh needs bash, node and tar",
)


def _site(tmp_path: Path, asset: Path, *, tag: str = TAG) -> Path:
    """A site directory with the script and a data-release.json pinning `asset`, and `gh`
    serving it."""
    site = tmp_path / "site"
    (site / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, site / "scripts" / "fetch-data.sh")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    pin = {"tag": tag, "asset": asset.name, "sha256": digest}
    (site / "data-release.json").write_text(json.dumps(pin, indent=1) + "\n")
    served = tmp_path / "releases" / tag
    served.mkdir(parents=True)
    shutil.copy(asset, served / asset.name)
    return site


def _run(site: Path) -> subprocess.CompletedProcess[str]:
    bin_dir = site.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_RELEASES": str(site.parent / "releases"),
    }
    return subprocess.run(
        ["bash", str(site / "scripts" / "fetch-data.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def asset(tmp_path: Path) -> Path:
    export = tmp_path / "export" / "UT"
    shutil.copytree(COMMITTED, export)
    built, _ = package(export, state="UT", tag=TAG, out_dir=tmp_path / "built", floor=0)
    return built


def _members(asset: Path) -> dict[str, bytes]:
    with tarfile.open(asset) as tar:
        out: dict[str, bytes] = {}
        for m in tar.getmembers():
            f = tar.extractfile(m)
            assert f is not None
            out[m.name] = f.read()
        return out


def _repack(members: dict[str, bytes], out: Path) -> Path:
    """A tarball of `members`, as a tampered release would be packed."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return out


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_release_verified_against_its_manifest_is_unpacked(tmp_path: Path, asset: Path):
    site = _site(tmp_path, asset)

    done = _run(site)

    assert done.returncode == 0, done.stderr
    unpacked = _tree(site / "public" / "data")
    release = json.loads(unpacked.pop("release.json"))
    assert release == json.loads((site / "data-release.json").read_text())
    assert unpacked == _members(asset)
    files = len(_members(asset)) - 1
    assert f"{files} files verified against {CHECKSUMS}" in done.stdout


def test_an_asset_that_does_not_match_the_committed_digest_is_refused(
    tmp_path: Path, asset: Path
) -> None:
    site = _site(tmp_path, asset)
    served = tmp_path / "releases" / TAG / ASSET
    members = _members(asset)
    _repack(members, served)  # the same files, packed again: another digest

    done = _run(site)

    assert done.returncode == 1
    assert f"{ASSET} does not match the digest committed in data-release.json" in done.stderr
    assert not (site / "public" / "data").exists()


def _tampered(tmp_path: Path, asset: Path, change: str) -> tuple[Path, str]:
    members = _members(asset)
    target = "bills/ut-2023-sb-18.json"
    if change == "tampered":  # the same size, so only its digest tells
        members[target] = members[target].replace(b'"passed"', b'"PASSED"', 1)
    elif change == "missing":
        del members[target]
    elif change == "extra":
        target = "bills/ut-2099-hb-1.json"
        members[target] = members["bills/ut-2023-sb-18.json"]
    elif change == "malformed manifest":
        target = CHECKSUMS
        members[CHECKSUMS] = members[CHECKSUMS].replace(b'"path"', b'"name"', 1)
    # The digest is pinned to the changed asset, as a reviewed commit of a bad release would be.
    return _repack(members, tmp_path / ASSET), target


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ("tampered", "does not match its SHA-256 in"),
        ("missing", "missing, though listed in"),
        ("extra", "not listed in"),
        ("malformed manifest", "is not a checksum manifest"),
    ],
)
def test_a_file_that_disagrees_with_the_manifest_is_refused(
    tmp_path: Path, asset: Path, change: str, says: str
) -> None:
    bad, target = _tampered(tmp_path / "bad", asset, change)
    site = _site(tmp_path, bad)
    existing = site / "public" / "data"
    existing.mkdir(parents=True)
    (existing / "meta.json").write_text("{}")

    done = _run(site)

    assert done.returncode == 1, done.stdout
    assert target in done.stderr
    assert says in done.stderr
    # A release that fails is never unpacked over the data already there.
    assert _tree(existing) == {"meta.json": b"{}"}


def test_a_changed_size_is_refused(tmp_path: Path, asset: Path) -> None:
    members = _members(asset)
    members["meta.json"] += b" "
    site = _site(tmp_path, _repack(members, tmp_path / ASSET))

    done = _run(site)

    assert done.returncode == 1
    assert f"meta.json: {len(members['meta.json'])} bytes, not" in done.stderr


def _without_manifest(asset: Path, out: Path) -> Path:
    members = _members(asset)
    del members[CHECKSUMS]
    return _repack(members, out)


def test_the_release_packed_before_the_manifest_is_accepted_on_its_digest(
    tmp_path: Path, asset: Path
) -> None:
    # The one release packed before the manifest existed: its committed digest verifies it.
    tag = "data-ga-2026-11-12"
    old = _without_manifest(asset, tmp_path / "old" / f"export-GA-{tag}.tar.gz")
    site = _site(tmp_path, old, tag=tag)

    done = _run(site)

    assert done.returncode == 0, done.stderr
    assert f"{tag} predates the checksum manifest" in done.stdout
    assert _tree(site / "public" / "data").keys() == {*_members(old), "release.json"}


def test_any_other_release_without_a_manifest_is_refused(tmp_path: Path, asset: Path) -> None:
    site = _site(tmp_path, _without_manifest(asset, tmp_path / "new" / ASSET))

    done = _run(site)

    assert done.returncode == 1
    assert f"{ASSET} holds no {CHECKSUMS}" in done.stderr
    assert not (site / "public" / "data").exists()


def test_the_pinned_release_is_accepted(tmp_path: Path) -> None:
    """The release site/data-release.json pins today, served from the local copy packed with
    it; skipped where that copy is absent, as in CI, whose data workflow downloads it."""
    local = DATA / "export" / PINNED["asset"]
    if not local.exists():
        pytest.skip(f"no local copy of {PINNED['asset']}")
    site = _site(tmp_path, local, tag=PINNED["tag"])
    assert json.loads((site / "data-release.json").read_text()) == PINNED

    done = _run(site)

    assert done.returncode == 0, done.stderr
    assert f"{PINNED['asset']} verified against data-release.json" in done.stdout
