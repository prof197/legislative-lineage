"""The origin precision sample: seeded strata, a tuning and a reporting part, and the budget
guard, which projects what labels would cost and asks no model."""

import json
import random
import re
from datetime import date
from pathlib import Path
from typing import Any

import anthropic
import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.evals import model_labels
from lineage.origins.matches import FLOOR, BillSpan, OriginMatch, Stage
from lineage.origins.modeldocs import ModelDoc, Source, sha256, store
from lineage.origins.precision import (
    FLOORS,
    PARTS,
    RESULTS_SECTIONS,
    STAND_IN,
    OriginPrecisionError,
    band_of,
    census_reading,
    doc_chars,
    draw_part,
    estimate,
    excluded_keys,
    load_sample,
    plan_labels,
    projected_cost,
    prompt_chars,
    reading,
    refuse_over_cap,
    retired,
    section_chars,
    without_intervals,
    write_part,
)

UTAH_SB_18 = "ocd-bill/7968b9d9-25ee-4684-a92e-c76167d6f917"


def _match(
    n: int, *stages: Stage, doc: str = "lib/a", model_share: float = 0.5, bill_share: float = 0.25
) -> OriginMatch:
    return OriginMatch(
        bill_id=f"bill-{n:03d}",
        doc_id=doc,
        stages=list(stages),
        model_share=model_share,
        bill_share=bill_share,
        spans=[BillSpan(ordinal=n % 3, start=0, end=10, kind="exact")],
    )


def _table() -> list[OriginMatch]:
    """30 matches by wording alone, 20 by meaning alone, 5 by both."""
    return (
        [_match(n, "near_duplicate") for n in range(30)]
        + [_match(n, "semantic") for n in range(30, 50)]
        + [_match(n, "semantic", "near_duplicate") for n in range(50, 55)]
    )


def test_each_stage_stratum_is_drawn_apart_under_the_seed_whatever_the_tables_order() -> None:
    table, reporting = _table(), PARTS["reporting"]
    rows, report = draw_part(table, reporting, final=True, per_stratum=10)
    shuffled = table[:]
    random.Random(7).shuffle(shuffled)

    assert draw_part(shuffled, reporting, final=True, per_stratum=10) == (rows, report)
    assert report["strata"] == {"near_duplicate_only": 30, "semantic_only": 20, "both": 5}
    assert report["drawn"] == {"near_duplicate_only": 10, "semantic_only": 10, "both": 5}
    first = reporting.first
    assert [r["pair"] for r in rows] == [f"o{n:04d}" for n in range(first, first + 25)]
    # Shuffled into one labeling order, not stratum by stratum.
    assert [r["stages"] for r in rows[:10]] != [["near_duplicate"]] * 10
    assert rows[0].keys() == {
        "pair", "stratum", "stages", "bill_id", "doc_id", "ordinals", "model_share", "bill_share"
    }  # fmt: skip


def _shares() -> list[OriginMatch]:
    """By the larger share: 30 under 0.05, 20 from 0.05, 10 from 0.10 and 40 from 0.25."""
    return (
        [_match(n, "semantic", model_share=0.02, bill_share=0.01) for n in range(30)]
        # A bill's share counts as a model's does, and a floor is in the band it opens.
        + [_match(n, "near_duplicate", model_share=0.01, bill_share=0.05) for n in range(30, 50)]
        + [
            _match(n, "semantic", "near_duplicate", model_share=0.2, bill_share=0.1)
            for n in range(50, 60)
        ]
        + [_match(n, "near_duplicate", model_share=0.9) for n in range(60, 100)]
    )


def test_the_tuning_sample_is_drawn_apart_from_each_band_between_the_candidate_floors() -> None:
    table = _shares()
    rows, report = draw_part(table, PARTS["tuning"], per_stratum=15)
    shuffled = table[:]
    random.Random(7).shuffle(shuffled)

    assert draw_part(shuffled, PARTS["tuning"], per_stratum=15) == (rows, report)
    assert FLOORS == (0.0, 0.05, 0.10, 0.25)
    assert report["floors"] == list(FLOORS)
    assert report["strata"] == {
        "share_from_0.00": 30, "share_from_0.05": 20, "share_from_0.10": 10,
        "share_from_0.25": 40,
    }  # fmt: skip
    assert report["drawn"] == {
        "share_from_0.00": 15, "share_from_0.05": 15, "share_from_0.10": 10,
        "share_from_0.25": 15,
    }  # fmt: skip
    assert all(band_of(max(r["model_share"], r["bill_share"])) == r["stratum"] for r in rows)
    assert [r["pair"] for r in rows] == [f"o{n:04d}" for n in range(1, 56)]
    # The stages are kept, for reading the labels by stage, though they do not stratify.
    assert {tuple(r["stages"]) for r in rows if r["stratum"] == "share_from_0.10"} == {
        ("near_duplicate", "semantic")
    }
    assert PARTS["tuning"].per_stratum == 50


def test_the_reporting_sample_is_drawn_only_when_final_from_matches_tuning_did_not_draw(
    tmp_path: Path,
) -> None:
    table = _table()
    reporting = PARTS["reporting"]
    with pytest.raises(OriginPrecisionError, match="final"):
        draw_part(table, reporting, per_stratum=10)
    with pytest.raises(OriginPrecisionError, match="draw the tuning sample"):
        excluded_keys(reporting, tmp_path)

    tuning_rows, tuning_report = draw_part(table, PARTS["tuning"], per_stratum=40)
    write_part(tuning_rows, tuning_report, PARTS["tuning"], root=tmp_path)
    excluded = excluded_keys(reporting, tmp_path)
    rows, report = draw_part(table, reporting, excluded=excluded, final=True, per_stratum=10)

    tuned = {(r["bill_id"], r["doc_id"]) for r in tuning_rows}
    assert not tuned & {(r["bill_id"], r["doc_id"]) for r in rows}
    # Every match of the table is in one band, so the tuning sample took 40 of the 55, and
    # the reporting sample, by stage, draws from the 15 left.
    left = {name: report["strata"][name] - report["excluded"][name] for name in report["strata"]}
    assert sum(report["excluded"].values()) == 40
    assert report["drawn"] == {name: min(10, n) for name, n in left.items()}
    assert rows[0]["pair"] == f"o{reporting.first:04d}"
    assert not {r["pair"] for r in tuning_rows} & {r["pair"] for r in rows}
    assert report["seed"] != tuning_report["seed"]

    write_part(rows, report, reporting, root=tmp_path)
    assert load_sample(tmp_path / reporting.sample) == rows
    with pytest.raises(OriginPrecisionError, match="drawn once"):
        write_part(rows, report, reporting, root=tmp_path)
    doc = json.loads((tmp_path / "reports" / "origin_precision_sample.json").read_text())
    assert doc == {"reporting": report, "tuning": tuning_report}


def test_the_reporting_sample_draws_what_a_page_lists_and_counts_each_source() -> None:
    """The published precision describes what a page lists, so the reporting part draws only
    matches at the floor, and records each source's matches by stage for a reading by source."""
    listed = [_match(n, "near_duplicate", doc="alec/x") for n in range(6)] + [
        _match(n, "semantic", doc="ncoil/y") for n in range(6, 10)
    ]
    unlisted = [
        _match(n, "near_duplicate", doc="alec/x", model_share=0.2, bill_share=0.1)
        for n in range(10, 30)
    ]
    rows, report = draw_part(listed + unlisted, PARTS["reporting"], final=True, per_stratum=10)

    assert PARTS["reporting"].floor == FLOOR
    assert {r["bill_id"] for r in rows} == {m.bill_id for m in listed}
    assert report["floor"] == FLOOR
    assert report["strata"] == {"near_duplicate_only": 6, "semantic_only": 4, "both": 0}
    assert report["by_source"] == {
        "alec": {"near_duplicate_only": 6, "semantic_only": 0, "both": 0},
        "ncoil": {"near_duplicate_only": 0, "semantic_only": 4, "both": 0},
    }
    assert "by_source" not in draw_part(unlisted, PARTS["tuning"], per_stratum=5)[1]


def test_a_match_listed_twice_is_refused() -> None:
    with pytest.raises(OriginPrecisionError, match="twice"):
        draw_part([_match(1, "semantic"), _match(1, "near_duplicate")], PARTS["tuning"])


ROWS: list[dict[str, Any]] = [
    {"pair": "o0001", "bill_id": "b1", "doc_id": "lib/a", "ordinals": [2], "stages": ["semantic"]},
    {"pair": "o0002", "bill_id": "b2", "doc_id": "lib/a", "ordinals": [], "stages": ["semantic"]},
]
SECTIONS = {"b1": {0: 100, 2: 300}, "b2": {0: 40, 1: 60}}


def test_a_prompt_counts_the_whole_model_and_the_bill_sections_its_spans_touch() -> None:
    system = len(STAND_IN.system)
    assert prompt_chars(ROWS, sections=SECTIONS, docs={"lib/a": 5000}) == [
        system + 5000 + 300,
        system + 5000 + 40 + 60,  # no spans: every matchable section
    ]
    with pytest.raises(OriginPrecisionError, match="no text for lib/a"):
        prompt_chars(ROWS, sections=SECTIONS, docs={})
    with pytest.raises(OriginPrecisionError, match="no matchable section 2"):
        prompt_chars(ROWS, sections={"b1": {0: 1}, "b2": {0: 1}}, docs={"lib/a": 1})


def test_the_projection_counts_as_v1s_label_runs_do() -> None:
    # 2,500 characters at 2.5 a token is 1,000 input tokens at $4 a million; each reply at the
    # stand-in's 2,048-token limit at $20 a million.
    one = (1000 * 4.0 + STAND_IN.max_tokens * 20.0) / 1e6
    assert projected_cost([2500] * 10, model="claude-opus-5-5") == pytest.approx(10 * one)
    assert projected_cost([2500] * 10, model="claude-opus-5-5", batch=True) == pytest.approx(
        5 * one
    )
    with pytest.raises(OriginPrecisionError, match="no price"):
        projected_cost([1], model="unpriced")


def _ledger(path: Path, dollars: float) -> Path:
    path.write_text(json.dumps({"budget": 30.0, "spent": dollars, "batches": [
        {"batch_id": "b", "dollars": dollars}
    ]}))  # fmt: skip
    return path


def test_a_run_over_its_cap_or_over_the_budget_with_what_is_spent_is_refused(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "spend.json", 20.19)
    chars = [10_000] * 100  # 4,000 tokens in, 2,048 out, 100 times: $5.70 at full price

    fits = plan_labels(chars, spend_path=ledger, max_cost=10.0, budget=30.0)
    assert fits.projected == pytest.approx(5.696)
    assert fits.spent == pytest.approx(20.19)
    refuse_over_cap(fits)

    with pytest.raises(OriginPrecisionError, match=re.escape("over the $5.00 cap")):
        refuse_over_cap(plan_labels(chars, spend_path=ledger, max_cost=5.0))
    with pytest.raises(OriginPrecisionError, match=re.escape("over the $25.00 budget")):
        refuse_over_cap(plan_labels(chars, spend_path=ledger, budget=25.0))
    assert plan_labels(chars, spend_path=tmp_path / "none.json").spent == 0


def test_the_sampler_and_the_guard_construct_no_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constructed: list[str] = []

    class NoClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructed.append(type(self).__name__)
            raise AssertionError("a model client was constructed")

    def no_calls(*args: object, **kwargs: object) -> None:
        constructed.append("call")
        raise AssertionError("a model was asked")

    monkeypatch.setattr(anthropic, "Anthropic", NoClient)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", NoClient)
    monkeypatch.setattr(model_labels, "anthropic_ask", no_calls)
    monkeypatch.setattr(model_labels, "anthropic_batches", no_calls)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    rows, report = draw_part(_shares(), PARTS["tuning"], per_stratum=10)
    write_part(rows, report, PARTS["tuning"], root=tmp_path)
    sections = {r["bill_id"]: {0: 500, 1: 500, 2: 500} for r in rows}
    chars = prompt_chars(rows, sections=sections, docs={"lib/a": 20_000})
    plan = plan_labels(chars, spend_path=tmp_path / "spend.json")
    refuse_over_cap(plan)

    assert plan.as_dict()["pairs"] == 40  # 10 from each of the four bands
    assert plan.prompt == "passages-v1-high"
    assert constructed == []


def test_an_estimate_weights_the_strata_by_their_size_in_the_table() -> None:
    rows = [
        {"pair": "o1", "stratum": "near_duplicate_only"},
        {"pair": "o2", "stratum": "near_duplicate_only"},
        {"pair": "o3", "stratum": "semantic_only"},
        {"pair": "o4", "stratum": "both"},
    ]
    sizes = {"near_duplicate_only": 300, "semantic_only": 100, "both": 0}
    out = estimate(rows, {"o1": True, "o2": False, "o3": True}, sizes)

    assert out["strata"]["near_duplicate_only"] | {"interval_95": None} == {
        "size": 300, "labeled": 2, "correct": 1, "precision": 0.5, "interval_95": None,
    }  # fmt: skip
    assert out["strata"]["both"]["labeled"] == 0  # o4 is not labeled; the stratum is empty
    assert out["overall"] is not None
    assert out["overall"]["precision"] == 0.625  # 0.75 x 0.5 + 0.25 x 1
    assert estimate(rows, {"o1": True}, sizes)["overall"] is None


def test_lengths_are_read_from_the_tables_and_the_document_store(
    commission_corpus: Path, tmp_path: Path
) -> None:
    lengths = section_chars(commission_corpus, [UTAH_SB_18, UTAH_SB_18, "no-such-bill"])
    assert list(lengths) == [UTAH_SB_18]
    assert all(n > 0 for n in lengths[UTAH_SB_18].values())

    source = Source(
        key="lib",
        organization="Example Library (test fixture)",
        descriptor="a test fixture",
        library_url="https://example.org/models",
        terms_note="test fixture",
        terms_read_on=date(2026, 11, 15),
    )
    text = "Section 1. A model provision.\n"
    doc = ModelDoc(
        id="lib/a", source="lib", title="A", url="https://example.org/models/a",
        collected_on=date(2026, 11, 15), robots="allowed", leaked=False,
        raw_sha256=sha256(text.encode()), media_type="text/plain",
        text_sha256=sha256(text.encode()),
    )  # fmt: skip
    store(source, doc, raw=text.encode(), text=text, root=tmp_path)
    assert doc_chars([doc], tmp_path) == {"lib/a": len(text)}


def test_the_reporting_reading_weights_each_source_by_stage_and_compares_one_with_the_rest() -> (
    None
):
    """Each source's precision is weighted by its own matches per stage, so a source whose few
    sampled near-duplicates are wrong is not hidden by the rest; nothing is read until every
    sampled match is labeled."""
    report = {
        "floor": FLOOR,
        "seed": 1,
        "strata": {"near_duplicate_only": 100, "semantic_only": 300, "both": 0},
        "by_source": {
            "alec": {"near_duplicate_only": 80, "semantic_only": 20, "both": 0},
            "ncoil": {"near_duplicate_only": 20, "semantic_only": 200, "both": 0},
            "eppc": {"near_duplicate_only": 0, "semantic_only": 80, "both": 0},
        },
    }
    rows = [
        {"pair": "o1", "stratum": "near_duplicate_only", "doc_id": "alec/a"},
        {"pair": "o2", "stratum": "near_duplicate_only", "doc_id": "alec/a"},
        {"pair": "o3", "stratum": "semantic_only", "doc_id": "alec/b"},
        {"pair": "o4", "stratum": "near_duplicate_only", "doc_id": "ncoil/c"},
        {"pair": "o5", "stratum": "semantic_only", "doc_id": "ncoil/c"},
        {"pair": "o6", "stratum": "semantic_only", "doc_id": "eppc/d"},
    ]
    correct = {"o1": False, "o2": True, "o3": True, "o4": True, "o5": True, "o6": False}

    with pytest.raises(OriginPrecisionError, match="read once"):
        reading(rows, {k: v for k, v in correct.items() if k != "o6"}, report)
    read = reading(rows, correct, report)

    alec = read["compared"]["alec"]
    assert (alec["matches"], alec["labeled"], alec["correct"]) == (100, 3, 2)
    # 80 near-duplicates at 1 of 2, 20 by meaning at 1 of 1.
    assert alec["precision"]["precision"] == round(0.8 * 0.5 + 0.2 * 1.0, 4)
    rest = read["compared"]["rest"]
    assert rest["sources"] == ["eppc", "ncoil"]
    assert (rest["matches"], rest["labeled"], rest["correct"]) == (300, 3, 2)
    # 20 near-duplicates at 1 of 1, 280 by meaning at 1 of 2.
    assert rest["precision"]["precision"] == round(20 / 300 * 1.0 + 280 / 300 * 0.5, 4)
    assert read["by_source"]["eppc"]["precision"]["precision"] == 0.0
    assert read["overall"]["matches"] == 400
    assert read["floor"] == FLOOR


def test_the_reporting_sample_is_read_once(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    stored: dict[str, dict[str, object]] = {
        "origins_precision": {
            "command": "uv run lineage eval origin-precision --part reporting_after_compact_fix",
            "overall": {},
        }
    }
    results.write_text(json.dumps(stored))

    run = CliRunner().invoke(app, ["eval", "origin-precision", "--results", str(results)])

    assert run.exit_code == 1
    assert "read once already" in run.output
    assert json.loads(results.read_text()) == stored


def test_only_its_intervals_may_be_rewritten_and_only_when_nothing_else_moves(
    tmp_path: Path,
) -> None:
    """Ruled at the step 47 review: the interval fix regenerates the reading's intervals; the
    reading itself, every point and count, stays as it was read."""
    assert without_intervals({"a": {"precision": 0.9, "interval_95": [0.8, 0.95]}, "b": [1]}) == {
        "a": {"precision": 0.9}, "b": [1],
    }  # fmt: skip
    stored = json.loads(Path("reports/results.json").read_text())["origins_precision_before_fix"]
    moved = json.loads(json.dumps(stored))
    moved["overall"]["correct"] -= 1
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"origins_precision_before_fix": moved}))

    args = ["eval", "origin-precision", "--part", "reporting", "--results", str(results),
            "--intervals-only"]  # fmt: skip
    run = CliRunner().invoke(app, args)
    assert run.exit_code == 1
    assert "other than its intervals" in run.output
    assert json.loads(results.read_text()) == {"origins_precision_before_fix": moved}

    widened = json.loads(json.dumps(stored))
    widened["overall"]["precision"]["interval_95"] = [0.0, 1.0]
    results.write_text(json.dumps({"origins_precision_before_fix": widened}))
    run = CliRunner().invoke(app, args)
    assert run.exit_code == 0, run.output
    assert without_intervals(json.loads(results.read_text())["origins_precision_before_fix"]) == (
        without_intervals(stored)
    )
    assert json.loads(results.read_text())["origins_precision_before_fix"]["overall"]["precision"][
        "interval_95"
    ] != [0.0, 1.0]


def test_a_census_takes_every_listed_match_of_its_source_once() -> None:
    """Ruled at the step 47 review: EPPC's matches at the floor are labeled whole, a census, so
    its precision is a count, not an estimate."""
    eppc = [_match(n, "near_duplicate", doc="eppc/x") for n in range(140)]
    low = [_match(n, "semantic", doc="eppc/y", model_share=0.2, bill_share=0.1) for n in range(3)]
    other = [_match(n, "semantic", doc="alec/z") for n in range(200, 210)]
    census = PARTS["census_eppc"]
    rows, report = draw_part(eppc + low + other, census)

    assert (census.source, census.census, census.floor, census.final_only) == (
        "eppc", True, FLOOR, False,
    )  # fmt: skip
    assert sorted(r["bill_id"] for r in rows) == sorted(m.bill_id for m in eppc)
    assert rows[0]["pair"] == "o0601"
    assert len({r["pair"] for r in rows}) == 140
    assert (report["source"], report["census"]) == ("eppc", True)
    assert report["strata"] == {"near_duplicate_only": 140, "semantic_only": 0, "both": 0}


def test_a_census_reading_counts_labels_from_any_part_and_lists_the_wrong_matches() -> None:
    rows = [
        {"pair": f"o06{n:02d}", "stratum": s, "bill_id": f"b{n}", "doc_id": d,
         "model_share": 0.3, "bill_share": 0.1}
        for n, (s, d) in enumerate(
            [("both", "eppc/x"), ("both", "eppc/x"), ("semantic_only", "eppc/x"),
             ("near_duplicate_only", "eppc/y")]
        )
    ]  # fmt: skip
    correct = {("b0", "eppc/x"): True, ("b1", "eppc/x"): False, ("b2", "eppc/x"): False,
               ("b3", "eppc/y"): True}  # fmt: skip
    sections = {("b1", "eppc/x"): [{"model_kind": "definitions"}],
                ("b2", "eppc/x"): [{"model_kind": None}]}  # fmt: skip
    out = census_reading(rows, correct, sections, source="eppc")

    assert (out["matches"], out["correct"], out["precision"]) == (4, 2, 0.5)
    assert out["by_stage"]["both"] == {"matches": 2, "correct": 1}
    assert out["by_document"] == {
        "eppc/x": {"matches": 3, "correct": 1}, "eppc/y": {"matches": 1, "correct": 1},
    }  # fmt: skip
    assert [w["bill_id"] for w in out["wrong"]] == ["b1", "b2"]
    assert out["wrong"][0]["sections"] == [{"model_kind": "definitions"}]
    with pytest.raises(OriginPrecisionError, match="no label"):
        census_reading(rows, {}, sections, source="eppc")


def test_the_reporting_sample_after_the_fix_is_drawn_once_apart_from_both_earlier_samples(
    tmp_path: Path,
) -> None:
    """Ruled at the step 47 review: with generic sections left out of matches, a fresh
    reporting sample is drawn and read once; the first reading is kept as the one before the
    fix."""
    after = PARTS["reporting_after_fix"]
    assert (after.final_only, after.floor, after.excludes) == (True, FLOOR, ("tuning", "reporting"))
    assert after.seed not in {p.seed for n, p in PARTS.items() if n != "reporting_after_fix"}
    # Ruled at the step 48 review: its reading is kept as the one before the compact fix.
    assert RESULTS_SECTIONS == {
        "reporting": "origins_precision_before_fix",
        "reporting_after_fix": "origins_precision_before_compact_fix",
        "reporting_after_compact_fix": "origins_precision",
    }
    table = _table()
    for name in ("tuning", "reporting"):
        (tmp_path / PARTS[name].sample).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / PARTS["tuning"].sample).write_text(
        json.dumps({"bill_id": "bill-000", "doc_id": "lib/a"}) + "\n"
    )
    (tmp_path / PARTS["reporting"].sample).write_text(
        json.dumps({"bill_id": "bill-030", "doc_id": "lib/a"}) + "\n"
    )
    rows, _ = draw_part(table, after, excluded=excluded_keys(after, tmp_path), final=True)

    drawn = {r["bill_id"] for r in rows}
    assert not drawn & {"bill-000", "bill-030"}
    assert rows
    assert min(int(r["pair"][1:]) for r in rows) == 1001
    with pytest.raises(OriginPrecisionError, match="final"):
        draw_part(table, after)


def test_a_published_reading_moves_aside_unchanged_when_a_later_part_is_read() -> None:
    """Ruled at the step 48 review: the reading after the first generic-section fix is kept as
    the reading before the compact fix, as it was read, and the fresh part's reading is the one
    published."""
    first = {"command": "uv run lineage eval origin-precision --part reporting_after_fix", "x": 1}
    oldest = {"command": "uv run lineage eval origin-precision --part reporting", "x": 0}
    results = {"origins_precision": first, "origins_precision_before_fix": oldest, "other": 2}
    moved = retired(results)
    assert moved == {
        "origins_precision_before_compact_fix": first,
        "origins_precision_before_fix": oldest,
        "other": 2,
    }
    assert retired(moved) == moved
    assert RESULTS_SECTIONS["reporting_after_compact_fix"] == "origins_precision"
    taken = results | {"origins_precision_before_compact_fix": oldest}
    with pytest.raises(OriginPrecisionError, match="is taken"):
        retired(taken)


def test_the_fresh_part_is_drawn_apart_from_every_earlier_sample() -> None:
    part = PARTS["reporting_after_compact_fix"]
    assert (part.final_only, part.by_sections, part.floor) == (True, True, FLOOR)
    assert set(part.excludes) == {"tuning", "reporting", "reporting_after_fix"}
    seeds = [p.seed for p in PARTS.values()]
    assert len(seeds) == len(set(seeds))
