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

from lineage import model_labels
from lineage.modeldocs import ModelDoc, Source, sha256, store
from lineage.origin_precision import (
    PARTS,
    STAND_IN,
    OriginPrecisionError,
    doc_chars,
    draw_part,
    estimate,
    excluded_keys,
    load_sample,
    plan_labels,
    projected_cost,
    prompt_chars,
    refuse_over_cap,
    section_chars,
    write_part,
)
from lineage.origins import BillSpan, OriginMatch, Stage

UTAH_SB_18 = "ocd-bill/7968b9d9-25ee-4684-a92e-c76167d6f917"


def _match(n: int, *stages: Stage, doc: str = "lib/a") -> OriginMatch:
    return OriginMatch(
        bill_id=f"bill-{n:03d}",
        doc_id=doc,
        stages=list(stages),
        model_share=0.5,
        bill_share=0.25,
        spans=[BillSpan(ordinal=n % 3, start=0, end=10, kind="exact")],
    )


def _table() -> list[OriginMatch]:
    """30 matches by wording alone, 20 by meaning alone, 5 by both."""
    return (
        [_match(n, "near_duplicate") for n in range(30)]
        + [_match(n, "semantic") for n in range(30, 50)]
        + [_match(n, "semantic", "near_duplicate") for n in range(50, 55)]
    )


def test_each_stratum_is_drawn_apart_under_the_seed_whatever_the_tables_order() -> None:
    table = _table()
    rows, report = draw_part(table, PARTS["tuning"], per_stratum=10)
    shuffled = table[:]
    random.Random(7).shuffle(shuffled)

    assert draw_part(shuffled, PARTS["tuning"], per_stratum=10) == (rows, report)
    assert report["strata"] == {"near_duplicate_only": 30, "semantic_only": 20, "both": 5}
    assert report["drawn"] == {"near_duplicate_only": 10, "semantic_only": 10, "both": 5}
    assert [r["pair"] for r in rows] == [f"o{n:04d}" for n in range(1, 26)]
    # Shuffled into one labeling order, not stratum by stratum.
    assert [r["stages"] for r in rows[:10]] != [["near_duplicate"]] * 10
    assert rows[0].keys() == {
        "pair", "stages", "bill_id", "doc_id", "ordinals", "model_share", "bill_share"
    }  # fmt: skip


def test_the_reporting_sample_is_drawn_only_when_final_from_matches_tuning_did_not_draw(
    tmp_path: Path,
) -> None:
    table = _table()
    reporting = PARTS["reporting"]
    with pytest.raises(OriginPrecisionError, match="final"):
        draw_part(table, reporting, per_stratum=10)
    with pytest.raises(OriginPrecisionError, match="draw the tuning sample"):
        excluded_keys(reporting, tmp_path)

    tuning_rows, tuning_report = draw_part(table, PARTS["tuning"], per_stratum=10)
    write_part(tuning_rows, tuning_report, PARTS["tuning"], root=tmp_path)
    excluded = excluded_keys(reporting, tmp_path)
    rows, report = draw_part(table, reporting, excluded=excluded, final=True, per_stratum=10)

    tuned = {(r["bill_id"], r["doc_id"]) for r in tuning_rows}
    assert not tuned & {(r["bill_id"], r["doc_id"]) for r in rows}
    assert report["excluded"] == {"near_duplicate_only": 10, "semantic_only": 10, "both": 5}
    assert report["drawn"] == {"near_duplicate_only": 10, "semantic_only": 10, "both": 0}
    assert rows[0]["pair"] == f"o{reporting.first:04d}"
    assert not {r["pair"] for r in tuning_rows} & {r["pair"] for r in rows}
    assert report["seed"] != tuning_report["seed"]

    write_part(rows, report, reporting, root=tmp_path)
    assert load_sample(tmp_path / reporting.sample) == rows
    with pytest.raises(OriginPrecisionError, match="drawn once"):
        write_part(rows, report, reporting, root=tmp_path)
    doc = json.loads((tmp_path / "reports" / "origin_precision_sample.json").read_text())
    assert doc == {"reporting": report, "tuning": tuning_report}


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

    rows, report = draw_part(_table(), PARTS["tuning"], per_stratum=10)
    write_part(rows, report, PARTS["tuning"], root=tmp_path)
    sections = {r["bill_id"]: {0: 500, 1: 500, 2: 500} for r in rows}
    chars = prompt_chars(rows, sections=sections, docs={"lib/a": 20_000})
    plan = plan_labels(chars, spend_path=tmp_path / "spend.json")
    refuse_over_cap(plan)

    assert plan.as_dict()["pairs"] == 25
    assert plan.prompt == "passages-v1-high"
    assert constructed == []


def test_an_estimate_weights_the_strata_by_their_size_in_the_table() -> None:
    rows = [
        {"pair": "o1", "stages": ["near_duplicate"]},
        {"pair": "o2", "stages": ["near_duplicate"]},
        {"pair": "o3", "stages": ["semantic"]},
        {"pair": "o4", "stages": ["near_duplicate", "semantic"]},
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
