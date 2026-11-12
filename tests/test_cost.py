"""Cost and runtime per stage: the build's from the committed reports, a query's as measured."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lineage.cli import app
from lineage.cost import build_costs

ROOT = Path(__file__).parents[1]


def test_every_build_stage_is_read_from_its_committed_report() -> None:
    stages = build_costs(ROOT / "reports")

    assert list(stages) == [
        "near_duplicate_index",
        "embedding",
        "neighbors",
        "semantic_pairs",
        "state_boilerplate",
    ]
    index = json.loads((ROOT / "reports" / "index.json").read_text())
    assert stages["near_duplicate_index"]["seconds"] == index["build"]["seconds"]
    for stage in stages.values():
        assert stage["source"].startswith("reports/")
        assert stage["seconds"] > 0
        assert stage["dollars"] == 0  # retrieval makes no model call


def test_the_command_times_each_query_by_stage(corpus: Path, tmp_path: Path) -> None:
    from tests.test_acts import ACT

    acts = tmp_path / "acts"
    acts.mkdir()
    (acts / "one.txt").write_text(ACT)
    report = tmp_path / "cost.json"

    result = CliRunner().invoke(
        app,
        ["eval", "cost", "--tables", str(corpus), "--acts", str(acts), "--no-semantic",
         "--report", str(report)],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    doc = json.loads(report.read_text())
    timed = doc["query"]["one"]
    assert set(timed["first"]) == {"near_duplicate", "scoring"}
    assert set(timed["warm"]) == {"near_duplicate", "scoring"}
    assert "near_duplicate_index" in doc["build"]
