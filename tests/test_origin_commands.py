"""The origin eval commands (DESIGN.md#origin-matches): before matching has run, recall counts
nothing found and the sampler refuses, and a sealed key is refused without --final."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lineage.cli import app


def test_recall_before_matching_finds_nothing(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    run = CliRunner().invoke(
        app, ["eval", "origins", "--tables", str(tmp_path), "--results", str(results)]
    )

    assert run.exit_code == 0, run.output
    pooled = json.loads(results.read_text())["origins_development"]["recall"]["pooled"]
    assert pooled["found"] == 0
    assert pooled["enactments"] == 36


def test_a_sealed_key_needs_final(tmp_path: Path) -> None:
    run = CliRunner().invoke(
        app,
        ["eval", "origins", "--key", "held_out", "--tables", str(tmp_path), "--results",
         str(tmp_path / "results.json")],
    )  # fmt: skip

    assert run.exit_code == 1
    assert not (tmp_path / "results.json").exists()


def test_the_sampler_refuses_an_empty_table(tmp_path: Path) -> None:
    run = CliRunner().invoke(app, ["origin-sample", "--tables", str(tmp_path)])

    assert run.exit_code == 1
    assert "no origin matches" in run.output
