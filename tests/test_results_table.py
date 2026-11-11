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
        "stage_ablation": None,
    }


def _merged(found: int, near: int, enactments: int) -> dict[str, object]:
    """Both stages' results, with the near-duplicate stage alone in the ablation."""
    section = _section(found, enactments)
    near_alone = _section(near, enactments)["recall"]
    section["stage_ablation"] = {"near_duplicate": near_alone, "both": section["recall"]}
    return section


def test_the_committed_readme_shows_the_committed_results() -> None:
    readme = (ROOT / "README.md").read_text()
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    exported = json.loads((ROOT / "reports" / "results_exported_only.json").read_text())

    assert readme == rendered_readme(readme, results, exported), "run `uv run lineage eval readme`"


def test_renders_counts_and_shares_from_the_results() -> None:
    readme = f"# Title\n\n{START}\nstale\n{END}\n\nMore.\n"

    out = rendered_readme(readme, {"commission": _section(2, 3)}, {"commission": _section(1, 3)})

    assert "stale" not in out
    assert "| | With the text layer | Open States' text alone |" in out
    assert "| Pooled, three acts | 2 / 3 (66.7%) | 1 / 3 (33.3%) |" in out
    assert "| Special Deposits |" in out
    assert "read 2099-01-02" in out
    assert "never proposed 2" in out
    assert "at least one verified section" in out
    assert out.endswith(f"{END}\n\nMore.\n")


def test_both_stages_are_shown_beside_the_near_duplicate_stage_on_either_text() -> None:
    readme = f"{START}\n{END}\n"

    out = rendered_readme(readme, {"commission": _merged(3, 2, 4)}, {"commission": _section(1, 4)})

    assert (
        "| | Both stages | Near-duplicate stage | Near-duplicate stage, Open States' text alone |"
        in out
    )
    assert "| Pooled, three acts | 3 / 4 (75.0%) | 2 / 4 (50.0%) | 1 / 4 (25.0%) |" in out
    assert "by wording or by meaning" in out
    assert "Misses by both stages" in out


def test_export_only_results_must_be_the_near_duplicate_stage_alone() -> None:
    # the vectors are embedded from the fetched text, so the export's text has no semantic run
    readme = f"{START}\n{END}\n"

    with pytest.raises(ResultsTableError, match="--no-semantic"):
        rendered_readme(readme, {"commission": _merged(3, 2, 4)}, {"commission": _merged(2, 1, 4)})


def test_a_readme_without_the_markers_is_refused() -> None:
    with pytest.raises(ResultsTableError, match="results:commission"):
        rendered_readme("# Title\n", {"commission": _section(1, 3)}, {"commission": _section(1, 3)})


def test_results_from_different_thresholds_are_not_set_side_by_side() -> None:
    readme = f"{START}\n{END}\n"
    other = _section(1, 3) | {"threshold": {"min_share": 0.1}}

    with pytest.raises(ResultsTableError, match="threshold"):
        rendered_readme(readme, {"commission": _section(1, 3)}, {"commission": other})


def test_at_a_threshold_of_zero_a_changed_score_still_compares() -> None:
    # At 0, an enactment is found by any matched section, however the score weighs them.
    readme = f"{START}\n{END}\n"
    scored = _section(2, 3) | {"threshold": {"min_share": 0.0, "score": "by words"}}
    other = _section(1, 3) | {"threshold": {"min_share": 0.0, "score": "by sections"}}

    assert "2 / 3" in rendered_readme(readme, {"commission": scored}, {"commission": other})


def test_above_zero_a_changed_score_does_not_compare() -> None:
    readme = f"{START}\n{END}\n"
    scored = _section(2, 3) | {"threshold": {"min_share": 0.5, "score": "by words"}}
    other = _section(1, 3) | {"threshold": {"min_share": 0.5, "score": "by sections"}}

    with pytest.raises(ResultsTableError, match="threshold"):
        rendered_readme(readme, {"commission": scored}, {"commission": other})


def test_the_command_rewrites_only_the_results_section(tmp_path: Path) -> None:
    readme, results = tmp_path / "README.md", tmp_path / "results.json"
    exported = tmp_path / "exported.json"
    readme.write_text(f"Intro.\n{START}\nold\n{END}\nOutro.\n")
    results.write_text(json.dumps({"commission": _section(2, 4)}))
    exported.write_text(json.dumps({"commission": _section(1, 4)}))

    out = CliRunner().invoke(
        app,
        ["eval", "readme", "--readme", str(readme), "--results", str(results),
         "--exported-only", str(exported)],
    )  # fmt: skip

    assert out.exit_code == 0, out.output
    text = readme.read_text()
    assert text.startswith("Intro.\n")
    assert text.endswith("Outro.\n")
    assert "2 / 4 (50.0%)" in text
