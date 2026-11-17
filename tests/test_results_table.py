"""The README results table is generated from the committed results file and cannot drift."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.export import STORIES_RULES as STORIES
from lineage.results_table import (
    END,
    HELD_OUT_END,
    HELD_OUT_START,
    ORIGINS_END,
    ORIGINS_START,
    START,
    SUMMARY_END,
    SUMMARY_START,
    ResultsTableError,
    hosted_state,
    render_stories,
    rendered_readme,
)

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

    state = json.loads((ROOT / "reports" / "relatives_reporting.json").read_text())["state"]

    stories = render_stories(hosted_state(ROOT / "site" / "data-release.json"), ROOT / STORIES)
    assert readme == rendered_readme(readme, results, exported, state, stories=stories), (
        "run `uv run lineage eval readme`"
    )


def test_the_story_count_is_the_hosted_states_from_the_stories_rules(tmp_path: Path) -> None:
    rules = tmp_path / "stories.toml"
    rules.write_text(
        '[stories.a]\nstates = "all"\n[stories.b]\nstates = ["GA"]\n'
        '[stories.c]\nstates = ["UT"]\n[stories.origin]\nstates = "all"\n'
    )
    ga = render_stories("GA", rules)
    assert ga.startswith("two featured stories, and one on a passed bill")
    rules.write_text('[stories.a]\nstates = ["UT"]\n')
    assert render_stories("GA", rules) == "no featured stories."
    assert render_stories("UT", rules) == "one featured story."


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


def test_the_held_out_run_is_shown_by_stage_once_the_readme_has_its_section() -> None:
    readme = f"{START}\n{END}\n{HELD_OUT_START}\nstale\n{HELD_OUT_END}\n"
    held_out = _merged(3, 2, 4)
    near_alone = _section(2, 4)["recall"]
    held_out["stage_ablation"] = {
        "near_duplicate": near_alone,
        "semantic": held_out["recall"],
        "both": held_out["recall"],
    }
    first = {"semantic_threshold": 0.93, "recall": {"all": {"enactments": 4, "found": 3}}}
    held_out |= {
        "semantic_threshold": 0.94,
        "state_boilerplate_flag": {"sections": 1},
        "earlier_runs": [first],
    }
    results = {"commission": _merged(3, 2, 4), "commission_held_out": held_out}

    out = rendered_readme(readme, results, {"commission": _section(1, 4)})

    assert "stale" not in out
    assert "| | Both stages | Near-duplicate stage | Semantic stage |" in out
    assert "| Special Deposits | 3 / 4 (75.0%) | 2 / 4 (50.0%) | 3 / 4 (75.0%) |" in out
    assert "at 0.94 and with the state boilerplate flag" in out
    assert "first run at a threshold of 0.93, which found 3 of 4" in out
    assert "--held-out --final" in out
    with pytest.raises(ResultsTableError, match="held-out run has no results"):
        rendered_readme(readme, {"commission": _merged(3, 2, 4)}, {"commission": _section(1, 4)})


def test_the_summary_answers_each_question_from_the_results() -> None:
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    readme = f"# Title\n\n{SUMMARY_START}\nstale\n{SUMMARY_END}\n\n{START}\n{END}\n"
    exported = json.loads((ROOT / "reports" / "results_exported_only.json").read_text())

    out = rendered_readme(readme, results, exported, "GA")

    dev = results["commission"]["recall"]["pooled"]["all"]
    assert f"| {dev['found']} of {dev['enactments']} (" in out
    assert "sampled from Georgia's pages" in out
    judged = [line for line in out.splitlines() if "judged by a model" in line.lower()]
    # precision and faithfulness, said in the row itself, and origin precision once it is read
    assert len(judged) == 2 + ("origins_precision" in results)
    assert all(line.startswith("| Judged by a model:") for line in judged)
    assert "`uv run lineage eval faithfulness`" in out
    assert "stale" not in out
    with pytest.raises(ResultsTableError, match="precision sample"):
        rendered_readme(readme, results, exported)


def _rate(found: int, n: int, low: float, high: float) -> dict[str, object]:
    return {"enactments": n, "found": found, "recall": found / n, "interval_95": [low, high]}


def _origin_recall(
    key: str, found: int, n: int, *, sealed: bool = False, status: str | None = None
) -> dict[str, object]:
    """An origin recall section as `lineage eval origins` writes it, cut to what is read."""
    pooled = _rate(found, n, 0.5, 0.99)
    return {
        "command": "uv run lineage eval origins",
        "answer_key": {"name": key, "path": "reports/answer_key.csv", "read": "read 2099-01-02"},
        "sealed": sealed,
        "status": status,
        "threshold": {"min_share": 0.0, "floor": 0.25},
        "documents": {"act-a": ["commission/act-a"], "act-b": ["commission/act-b"]},
        "recall": {"pooled": pooled, "by_act": {"act-a": pooled, "act-b": pooled}},
    }


def _stratum(size: int, labeled: int, correct: int) -> dict[str, object]:
    return {
        "size": size,
        "labeled": labeled,
        "correct": correct,
        "precision": correct / labeled,
        "interval_95": [0.41, 0.97],
    }


def _origin_precision() -> dict[str, object]:
    """The reporting reading as `lineage eval origin-precision` writes it, cut to what is read."""
    return {
        "command": "uv run lineage eval origin-precision --part reporting_after_fix",
        "floor": 0.25,
        "overall": {
            "matches": 60,
            "labeled": 9,
            "correct": 8,
            "precision": {"precision": 0.8812, "interval_95": [0.5521, 0.9763]},
            "by_stage": {
                "near_duplicate_only": _stratum(10, 3, 2),
                "semantic_only": _stratum(20, 3, 3),
                "both": _stratum(30, 3, 3),
            },
        },
    }


def _origins() -> dict[str, object]:
    return {
        "origins_development": _origin_recall("development", 3, 4),
        "origins_held_out": _origin_recall("held_out", 2, 2, sealed=True),
        "origins_v1_held_out": _origin_recall(
            "v1_held_out", 4, 5, status="opened in v1; not a sealed test"
        ),
        "origins_precision": _origin_precision(),
    }


def _rendered_origins(results: dict[str, object], sealed_note: str = "") -> str:
    readme = f"{START}\n{END}\n{ORIGINS_START}\nstale\n{ORIGINS_END}\n"
    out = rendered_readme(
        readme,
        {"commission": _section(1, 3)} | results,
        {"commission": _section(1, 3)},
        sealed_note=sealed_note,
    )
    return out[out.index(ORIGINS_START) : out.index(ORIGINS_END)]


def test_origin_recall_and_precision_are_rendered_with_intervals_and_commands() -> None:
    block = _rendered_origins(_origins())

    assert "stale" not in block
    assert "listing floor of 0.25" in block
    rows = [line for line in block.splitlines() if line.startswith("| ")]
    dev = next(r for r in rows if "development" in r)
    assert "3 of 4, 0.75 [0.50, 0.99]" in dev
    assert "`uv run lineage eval origins`" in dev
    sealed = next(r for r in rows if "sealed key" in r)
    assert "2 of 2, 1.00 [0.50, 0.99]" in sealed
    assert "`uv run lineage eval origins --key held_out --final`" in sealed
    overall = next(r for r in rows if "all stages" in r)
    assert "0.88 [0.55, 0.98]" in overall
    assert "9 labels of 60 matches" in overall
    assert "`uv run lineage eval origin-precision --part reporting_after_fix`" in overall
    near = next(r for r in rows if "near-duplicate stage alone" in r)
    assert "0.67 [0.41, 0.97]" in near
    assert "3 labels of 10 matches" in near
    assert any("by meaning alone" in r for r in rows)
    assert any("both stages" in r for r in rows)


def test_v1_held_out_acts_are_labeled_exactly_and_kept_apart_from_the_sealed_row() -> None:
    lines = _rendered_origins(_origins()).splitlines()

    v1 = next(n for n, line in enumerate(lines) if "--key v1_held_out" in line)
    sealed = next(n for n, line in enumerate(lines) if "--key held_out --final" in line)
    assert "opened in v1; not a sealed test" in lines[v1]
    assert "4 of 5, 0.80 [0.50, 0.99]" in lines[v1]
    # a line that is not a table row stands between them, so they are two tables
    assert any(not line.startswith("|") for line in lines[min(v1, sealed) : max(v1, sealed)])
    assert "opened in v1" not in lines[sealed]


def test_an_absent_origin_section_leaves_no_row_and_nothing_in_its_place() -> None:
    results = _origins()
    del results["origins_precision"]

    block = _rendered_origins(results)

    assert "origin-precision" not in block
    assert "recision" not in block
    assert "--key held_out --final" in block

    for name in ("origins_held_out", "origins_v1_held_out"):
        partial = _origins()
        del partial[name]
        assert f"--key {name.removeprefix('origins_')}" not in _rendered_origins(partial)


def test_with_no_origin_section_the_block_is_left_empty() -> None:
    assert _rendered_origins({}).strip() == ORIGINS_START


def test_the_sealed_row_takes_one_footnote_only_when_one_is_given() -> None:
    plain = _rendered_origins(_origins())
    noted = _rendered_origins(_origins(), sealed_note="A note on the sealed run.")

    assert "[^sealed]" not in plain
    sealed = next(line for line in noted.splitlines() if "--key held_out --final" in line)
    assert "[^sealed]" in sealed
    assert "[^sealed]: A note on the sealed run." in noted.splitlines()
    assert noted.count("[^sealed]") == 2


def test_origin_evals_at_different_floors_are_not_set_side_by_side() -> None:
    results = _origins()
    results["origins_precision"] = _origin_precision() | {"floor": 0.1}

    with pytest.raises(ResultsTableError, match="floor"):
        _rendered_origins(results)


def test_the_summary_answers_the_origin_questions_only_from_sections_present() -> None:
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    exported = json.loads((ROOT / "reports" / "results_exported_only.json").read_text())
    readme = f"{SUMMARY_START}\n{SUMMARY_END}\n{START}\n{END}\n"
    results |= _origins()

    out = rendered_readme(readme, results, exported, "GA")

    origin_rows = [line for line in out.splitlines() if "model legislation" in line]
    assert len(origin_rows) == 2
    assert any("3 of 4 (75.0%)" in r and "`uv run lineage eval origins`" in r for r in origin_rows)
    assert any(
        r.startswith("| Judged by a model:") and "0.88 [0.55, 0.98]" in r for r in origin_rows
    )

    for name in ("origins_development", "origins_precision"):
        del results[name]
    out = rendered_readme(readme, results, exported, "GA")
    assert "model legislation" not in out
