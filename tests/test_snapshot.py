import hashlib
import zipfile
from pathlib import Path

import pytest

from lineage.snapshot import (
    Archive,
    SnapshotError,
    load_manifest,
    read_archive,
    scan,
    verify,
    write_manifest,
)


def test_read_archive_takes_metadata_from_readme_and_hashes_the_file(raw_dir: Path) -> None:
    path = raw_dir / "GA" / "GA_2025_26_json_fixture.zip"

    archive = read_archive(raw_dir, path)

    assert archive == Archive(
        path="GA/GA_2025_26_json_fixture.zip",
        state="GA",
        session="2025_26",
        generated_at="2026-07-25 06:17:51.599960",
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_scan_lists_every_archive_in_path_order(raw_dir: Path) -> None:
    paths = [a.path for a in scan(raw_dir)]

    assert paths == sorted(paths)
    assert {p.split("/")[0] for p in paths} == {"AK", "AL", "GA", "HI"}


def test_manifest_round_trips(raw_dir: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    archives = scan(raw_dir)

    write_manifest(archives, manifest)

    assert load_manifest(manifest) == archives


def test_verify_accepts_files_that_match(raw_dir: Path) -> None:
    verify(raw_dir, scan(raw_dir))


def test_verify_names_changed_missing_and_unlisted_archives(raw_dir: Path) -> None:
    pinned = scan(raw_dir)
    (raw_dir / "AK" / "AK_33_json_fixture.zip").unlink()
    with (raw_dir / "GA" / "GA_2025_26_json_fixture.zip").open("ab") as f:
        f.write(b"\0")
    extra = raw_dir / "WY" / "WY_2026_json_fixture.zip"
    extra.parent.mkdir()
    extra.write_bytes((raw_dir / "HI" / "HI_2024_json_fixture.zip").read_bytes())

    with pytest.raises(SnapshotError) as excinfo:
        verify(raw_dir, pinned)

    message = str(excinfo.value)
    assert "missing: AK/AK_33_json_fixture.zip" in message
    assert "changed: GA/GA_2025_26_json_fixture.zip" in message
    assert "unlisted: WY/WY_2026_json_fixture.zip" in message


def test_rejects_an_unknown_export_format(tmp_path: Path) -> None:
    path = tmp_path / "XX" / "XX_1_json_x.zip"
    path.parent.mkdir()
    readme = "State: XX\nSession: 1\nGenerated At: 2026-01-01 00:00:00\nJSON Format Version: 2.0\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("README", readme)

    with pytest.raises(SnapshotError, match=r"format version 2\.0"):
        read_archive(tmp_path, path)
