"""The README results table is generated from the committed results file and cannot drift."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.results_table import END, START, ResultsTableError, rendered_readme

ROOT = Path(__file__).parents[1]


def _section(found: int, enactments: int) -> dict[str, object]:
    cell = {"enactments": enactments, "found": found, "recall": None}
    both = {"all": cell, "with_text": cell}
    return {
        "command": "uv run lineage eval commission",
        "threshold": {"min_share": 0.0},
        "enactments_file": {"read": "ULC Bill List, read 2099-01-02"},
        "recall": {
            "pooled": both,
            "by_act": {"special-deposits": both},
            "by_stratum": {"enacted": both, "substantially_similar": both},
        },
        "miss_reasons": {"not_proposed": 2},
    }


def test_the_committed_readme_shows_the_committed_results() -> None:
    readme = (ROOT / "README.md").read_text()
    results = json.loads((ROOT / "reports" / "results.json").read_text())

    assert readme == rendered_readme(readme, results), "run `uv run lineage eval readme`"


def test_renders_counts_and_shares_from_the_results() -> None:
    readme = f"# Title\n\n{START}\nstale\n{END}\n\nMore.\n"

    out = rendered_readme(readme, {"commission": _section(1, 3)})

    assert "stale" not in out
    assert "| Pooled, three acts | 1 / 3 (33.3%) | 1 / 3 (33.3%) |" in out
    assert "| Special Deposits |" in out
    assert "read 2099-01-02" in out
    assert "never proposed 2" in out
    assert "at least one verified section" in out
    assert out.endswith(f"{END}\n\nMore.\n")


def test_a_readme_without_the_markers_is_refused() -> None:
    with pytest.raises(ResultsTableError, match="results:commission"):
        rendered_readme("# Title\n", {"commission": _section(1, 3)})


def test_the_command_rewrites_only_the_results_section(tmp_path: Path) -> None:
    readme, results = tmp_path / "README.md", tmp_path / "results.json"
    readme.write_text(f"Intro.\n{START}\nold\n{END}\nOutro.\n")
    results.write_text(json.dumps({"commission": _section(2, 4)}))

    out = CliRunner().invoke(
        app, ["eval", "readme", "--readme", str(readme), "--results", str(results)]
    )

    assert out.exit_code == 0, out.output
    text = readme.read_text()
    assert text.startswith("Intro.\n")
    assert text.endswith("Outro.\n")
    assert "2 / 4 (50.0%)" in text
