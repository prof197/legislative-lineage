"""The text layer's rules file: the usable-text bar, per-state choices, the replacement rule."""

import json
from pathlib import Path
from typing import Any

import pytest

from lineage.text_rules import TextRules, load_rules, replaced_states, usable

RULES_FILE = """
[usable]
min_words = 5
min_real_word_share = 0.5
[fetch]
min_usable_rate = 0.8
html_margin = 0.05
[replace]
margin = 0.2
min_compared = 10
[states]
OH = { formats = ["html", "pdf"] }
GA = { format = "pdf", reading = "layout" }
CA = { source = "bulk" }
IN = { fetch = false, reason = "pages are built by scripts" }
[hosts]
"www.revisor.mn.gov" = { container = "div#document" }
"""


@pytest.fixture
def rules(tmp_path: Path) -> TextRules:
    path = tmp_path / "text_layer.toml"
    path.write_text(RULES_FILE)
    return load_rules(path)


def test_the_rules_file_names_each_states_format_reading_and_source(rules: TextRules) -> None:
    assert (rules.states["GA"].format, rules.states["GA"].reading) == ("pdf", "layout")
    assert rules.states["OH"].formats == ("html", "pdf")
    assert rules.states["CA"].source == "bulk"
    assert not rules.states["IN"].fetch
    assert rules.containers == {"www.revisor.mn.gov": "div#document"}


def test_a_state_left_out_must_say_why(tmp_path: Path) -> None:
    path = tmp_path / "text_layer.toml"
    path.write_text(RULES_FILE.replace(', reason = "pages are built by scripts"', ""))

    with pytest.raises(ValueError, match="says why"):
        load_rules(path)


def test_usable_text_is_long_enough_and_mostly_words(rules: TextRules) -> None:
    assert usable("Section 1. The board shall meet monthly.", rules)
    assert not usable("Section 1.", rules)
    assert not usable("x7 0x1F 3a 9 qz 22 ~~ 18 // 4", rules)


def _cell(ours: float, theirs: float, compared: int = 20) -> dict[str, Any]:
    return {
        "readings": {"plain": {"clean_rate": {"with_text": ours}}, "layout": {"clean_rate": {}}},
        "compared": compared,
        "theirs": {"clean_rate": theirs},
    }


def test_fetched_text_replaces_theirs_only_where_it_is_clearly_cleaner(rules: TextRules) -> None:
    report = {
        "states": {
            "OH": {"html": _cell(1.0, 0.0)},
            "GA": {"pdf": _cell(1.0, 0.0)},  # the layout reading, which GA uses, is unmeasured
        }
    }

    assert replaced_states(report, rules) == ["OH"]
    report["states"]["OH"]["html"] = _cell(1.0, 0.8)  # exactly the margin, 0.19999... in floats
    assert replaced_states(report, rules) == ["OH"]
    report["states"]["OH"]["html"] = _cell(0.95, 0.8)
    assert replaced_states(report, rules) == []
    report["states"]["OH"]["html"] = _cell(1.0, 0.0, compared=5)
    assert replaced_states(report, rules) == []


def test_the_committed_rules_agree_with_the_committed_quality_report() -> None:
    rules = load_rules()
    report = json.loads(Path("reports/text_quality.json").read_text())
    for state, rule in rules.states.items():
        if not rule.fetch or rule.source != "legislature":
            continue
        # The preferred format must be measured; a fallback, where the sample has it.
        for i, fmt in enumerate(rule.formats):
            cell = report["states"][state].get(fmt)
            if cell is None and i:
                continue
            assert cell is not None, (state, fmt)
            reading = rule.reading if fmt == "pdf" else "plain"
            rate = cell["readings"][reading]["usable_rate"]
            assert rate is not None, (state, fmt)
            assert rate >= rules.min_usable_rate, (state, fmt)


def test_full_run_jobs_follow_each_states_rule(rules: TextRules) -> None:
    from lineage.text_run import plan_jobs

    def row(i: str, state: str, kind: str, html: bool = True) -> dict[str, Any]:
        return {
            "id": i, "state": state, "kind": kind, "outcome": "planned", "note": "Enrolled",
            "date": "", "html_url": f"https://{state}.example/{i}.htm" if html else None,
            "pdf_url": f"https://{state}.example/{i}.pdf",
        }  # fmt: skip

    plan = [
        row("oh1", "OH", "blank"), row("oh2", "OH", "blank", html=False),
        row("oh3", "OH", "usable"), row("ga1", "GA", "blank"), row("ga2", "GA", "usable"),
        row("ca1", "CA", "missing"), row("in1", "IN", "blank"), row("tx1", "TX", "blank"),
    ]  # fmt: skip

    jobs, bulk, left = plan_jobs(plan, rules, replace={"OH"})

    assert [(j.id, j.format, j.layout) for j in jobs] == [
        ("oh1", "html", False), ("oh2", "pdf", False), ("oh3", "html", False),
        ("ga1", "pdf", True),
    ]  # fmt: skip
    assert bulk == ["ca1"]
    assert left == {("IN", "pages are built by scripts"): 1, ("TX", "no rule"): 1}
