from pathlib import Path

from typer.testing import CliRunner

from lineage import __version__
from lineage.cli import app

runner = CliRunner()


def test_version_prints_installed_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"lineage {__version__}\n"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "Usage" in result.output


def test_snapshot_then_ingest(raw_dir: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "tables"

    snap = runner.invoke(app, ["snapshot", "--raw", str(raw_dir), "--manifest", str(manifest)])
    ing = runner.invoke(
        app,
        [
            "ingest",
            "--raw",
            str(raw_dir),
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--min-free-gib",
            "0",
        ],
    )

    assert snap.exit_code == 0, snap.output
    assert "4 archives" in snap.output
    assert ing.exit_code == 0, ing.output
    assert "5 records from 4 archives" in ing.output


def test_ingest_fails_cleanly_on_a_mismatched_snapshot(raw_dir: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    runner.invoke(app, ["snapshot", "--raw", str(raw_dir), "--manifest", str(manifest)])
    (raw_dir / "HI" / "HI_2024_json_fixture.zip").unlink()

    result = runner.invoke(
        app,
        [
            "ingest",
            "--raw",
            str(raw_dir),
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "tables"),
            "--min-free-gib",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "missing: HI/HI_2024_json_fixture.zip" in result.output
