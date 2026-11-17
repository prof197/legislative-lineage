"""The data release (DESIGN.md#export): one reproducible tarball of the export with its checksum
manifest inside, packed only from files the export contract names, each valid against its
model; pinned by its digest in site/data-release.json."""

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage import release
from lineage.cli import app
from lineage.export_schema import TOP_LEVEL, Release
from lineage.release import (
    CHECKSUMS,
    CONTRACT,
    Checksums,
    ReleaseError,
    changed_files,
    check_export,
    checksums_bytes,
    package,
    read_checksums,
    write_release_manifest,
)
from tests.fixture_export import COMMITTED

TAG = "data-ut-2026-11-16"


@pytest.fixture
def export(tmp_path: Path) -> Path:
    out = tmp_path / "UT"
    shutil.copytree(COMMITTED, out)
    return out


def _pack(export: Path, out_dir: Path, *, tag: str = TAG, state: str = "UT") -> tuple[Path, Path]:
    return package(export, state=state, tag=tag, out_dir=out_dir, floor=0)


def _members(asset: Path) -> dict[str, bytes]:
    with tarfile.open(asset) as tar:
        out: dict[str, bytes] = {}
        for m in tar.getmembers():
            f = tar.extractfile(m)
            assert f is not None
            out[m.name] = f.read()
        return out


def test_the_contract_names_every_file_the_export_writes() -> None:
    # Release is release.json, which the site build writes beside the export; it never ships.
    assert {model for _, model in CONTRACT} == set(TOP_LEVEL) - {Release}


def test_a_valid_export_packs_with_its_checksum_manifest_inside(export: Path, tmp_path: Path):
    asset, checksum = _pack(export, tmp_path / "a")

    assert asset.name == f"export-UT-{TAG}.tar.gz"
    digest, name = checksum.read_text().split()
    assert name == asset.name
    assert digest == hashlib.sha256(asset.read_bytes()).hexdigest()
    members = _members(asset)
    names = list(members)
    assert names == sorted(names)
    files = sorted(p.relative_to(export).as_posix() for p in export.rglob("*") if p.is_file())
    assert set(names) == {*files, CHECKSUMS}
    manifest = Checksums.model_validate_json(members[CHECKSUMS])
    assert [f.path for f in manifest.files] == files
    for f in manifest.files:
        data = (export / f.path).read_bytes()
        assert members[f.path] == data
        assert (f.size, f.sha256) == (len(data), hashlib.sha256(data).hexdigest())
    # The manifest's bytes are its one canonical rendering, so the same export gives the same.
    assert members[CHECKSUMS] == checksums_bytes(manifest)


def test_the_tarball_is_byte_for_byte_reproducible(export: Path, tmp_path: Path) -> None:
    asset, _ = _pack(export, tmp_path / "a")
    (export / "meta.json").touch()  # a newer timestamp changes nothing
    (export / "bills.json").chmod(0o600)  # nor does a file's mode
    elsewhere = tmp_path / "elsewhere" / "UT"
    shutil.copytree(export, elsewhere)  # nor where it lies, or the order its files were made in
    again, _ = _pack(elsewhere, tmp_path / "b")

    assert asset.read_bytes() == again.read_bytes()
    with tarfile.open(asset) as tar:
        for m in tar.getmembers():
            assert (m.mtime, m.uid, m.gid, m.uname, m.gname, m.mode) == (0, 0, 0, "", "", 0o644)


def test_the_manifest_has_one_line_per_file(export: Path, tmp_path: Path) -> None:
    asset, _ = _pack(export, tmp_path / "a")
    manifest = read_checksums(asset)
    lines = _members(asset)[CHECKSUMS].decode().splitlines()

    assert lines[0] == '{"files": ['
    assert lines[-1] == "]}"
    meta = next(f for f in manifest.files if f.path == "meta.json")
    assert f'{{"path": "meta.json", "size": {meta.size}, "sha256": "{meta.sha256}"}},' in lines
    assert len(lines) == len(manifest.files) + 2


def test_two_releases_manifests_show_which_files_changed(export: Path, tmp_path: Path) -> None:
    before = read_checksums(_pack(export, tmp_path / "a")[0])
    page = export / "bills" / "ut-2023-sb-18.json"
    page.write_text(page.read_text().replace('"words_searched":', '"words_searched": '))
    gone = sorted((export / "compare").iterdir())[0]
    gone.unlink()
    shutil.copy(export / "texts" / "ut-2023-sb-18.json", export / "texts" / "zz-2099-hb-1.json")
    after = read_checksums(_pack(export, tmp_path / "b")[0])

    assert changed_files(before, after) == {
        "added": ["texts/zz-2099-hb-1.json"],
        "changed": ["bills/ut-2023-sb-18.json"],
        "removed": [f"compare/{gone.name}"],
    }


def test_a_release_without_a_manifest_is_named_as_such(tmp_path: Path) -> None:
    asset = tmp_path / "export-UT-data-ut-2026-11-12.tar.gz"
    with tarfile.open(asset, "w:gz") as tar:
        tar.add(COMMITTED / "meta.json", "meta.json")

    with pytest.raises(ReleaseError, match=r"holds no checksums\.json"):
        read_checksums(asset)


@pytest.mark.parametrize(
    "stray",
    [
        "notes.txt",
        ".DS_Store",
        "bills/ut-2023-sb-18.txt",
        "bills/nested/ut-2023-sb-18.json",
        "modeldocs/alec/doc.json",
        "texts/UT-2099-SB-1.json",  # upper case: not a slug
        "compare/ut-2023-sb-18.json",
        "release.json",
        CHECKSUMS,
    ],
)
def test_a_file_the_contract_does_not_name_is_refused(
    export: Path, tmp_path: Path, stray: str
) -> None:
    path = export / stray
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(export / "meta.json", path)

    with pytest.raises(
        ReleaseError, match=f"not a file the export contract names.*{re.escape(stray)}"
    ):
        _pack(export, tmp_path / "a")
    assert not (tmp_path / "a").exists()


def test_a_symlink_is_refused_even_to_a_valid_file(export: Path, tmp_path: Path) -> None:
    (export / "bills" / "ut-9999-sb-1.json").symlink_to(export / "bills" / "ut-2023-sb-18.json")

    with pytest.raises(ReleaseError, match=r"not a regular file.*ut-9999-sb-1\.json"):
        check_export(export, state="UT")


def _edit(path: Path, change: dict[str, object]) -> None:
    doc = json.loads(path.read_text())
    doc.update(change)
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("bills/ut-2023-sb-18.json", "{not json"),
        # A field the model does not have: model text riding along in a bill's page.
        ("bills/ut-2023-sb-18.json", {"model_text": "Section 1. Model act."}),
        ("meta.json", {"hosted_bills": "three"}),
        ("sources.json", {"sources": [{"key": "x"}]}),
    ],
)
def test_a_json_file_its_model_rejects_is_refused(
    export: Path, tmp_path: Path, name: str, content: str | dict[str, object]
) -> None:
    if isinstance(content, str):
        (export / name).write_text(content)
    else:
        _edit(export / name, content)

    with pytest.raises(ReleaseError, match=f"{name} is not a valid"):
        _pack(export, tmp_path / "a")
    assert not (tmp_path / "a").exists()


def test_a_file_in_the_wrong_place_is_refused(export: Path) -> None:
    # A valid comparison, but where a bill's page belongs.
    comparison = next((export / "compare").iterdir())
    shutil.copy(comparison, export / "bills" / "ut-2099-hb-1.json")

    with pytest.raises(ReleaseError, match=r"bills/ut-2099-hb-1\.json is not a valid BillPage"):
        check_export(export, state="UT")


@pytest.mark.parametrize("required", ["meta.json", "bills.json", "sources.json"])
def test_an_export_missing_a_required_file_is_refused(
    export: Path, tmp_path: Path, required: str
) -> None:
    (export / required).unlink()

    with pytest.raises(ReleaseError, match=f"no {required}"):
        _pack(export, tmp_path / "a")


def test_another_states_export_is_refused(export: Path, tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match=r"meta\.json is UT's export, not GA's"):
        _pack(export, tmp_path / "a", state="GA", tag="data-ga-2026-11-16")


def test_a_tag_the_deploy_would_refuse_is_refused(export: Path, tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="not a data release tag for UT"):
        _pack(export, tmp_path / "a", tag="data-ga-2026-11-16")
    with pytest.raises(ReleaseError, match="not a data release tag"):
        _pack(export, tmp_path / "a", tag="data-ut-1")
    with pytest.raises(ReleaseError, match="not a state code"):
        _pack(export, tmp_path / "a", state="Utah")
    with pytest.raises(ReleaseError, match=r"no meta\.json"):
        _pack(tmp_path / "empty", tmp_path / "a")
    assert not (tmp_path / "a").exists()


def test_the_release_manifest_pins_the_assets_own_digest(export: Path, tmp_path: Path) -> None:
    asset, checksum = _pack(export, tmp_path / "a")
    manifest = tmp_path / "site" / "data-release.json"

    write_release_manifest(asset, tag=TAG, path=manifest)

    pinned = Release.model_validate_json(manifest.read_text())
    assert pinned == Release(tag=TAG, asset=asset.name, sha256=checksum.read_text().split()[0])
    with pytest.raises(ReleaseError, match="is not release data-ut-2026-11-17's asset"):
        write_release_manifest(asset, tag="data-ut-2026-11-17", path=manifest)
    assert Release.model_validate_json(manifest.read_text()) == pinned


def test_the_tarball_holds_nothing_but_regular_files_at_relative_paths(
    export: Path, tmp_path: Path
) -> None:
    asset, _ = _pack(export, tmp_path / "a")
    with tarfile.open(asset) as tar:
        for m in tar.getmembers():
            assert m.isreg()
            assert not m.name.startswith("/")
            assert ".." not in m.name.split("/")
    # The gzip header carries no name and no time.
    header = asset.read_bytes()[:10]
    assert header[3] & 0x08 == 0
    assert header[4:8] == b"\0\0\0\0"


def _no_floor(path: Path, *, needed: int, floor: int) -> None:
    pass


def test_the_cli_packs_and_pins_through_the_release_module(
    export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI keeps the default floor; this checks the wiring, not the disk.
    monkeypatch.setattr(release, "require_free_space", _no_floor)
    pin = tmp_path / "site" / "data-release.json"
    args = ["export", "--state", "UT", "--out", str(export), "--package", TAG]
    result = CliRunner().invoke(app, [*args, "--release-manifest", str(pin)])

    assert result.exit_code == 0, result.output
    asset = tmp_path / f"export-UT-{TAG}.tar.gz"
    assert Release.model_validate_json(pin.read_text()).asset == asset.name
    assert CHECKSUMS in _members(asset)


def test_the_cli_pins_nothing_when_packing_is_refused(
    export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release, "require_free_space", _no_floor)
    (export / "notes.txt").write_text("stray")
    pin = tmp_path / "site" / "data-release.json"
    args = ["export", "--state", "UT", "--out", str(export), "--package", TAG]
    refused = CliRunner().invoke(app, [*args, "--release-manifest", str(pin)])

    assert refused.exit_code == 1
    assert not (tmp_path / f"export-UT-{TAG}.tar.gz").exists()
    assert not pin.exists()
