"""Origin labels: a match judged on its strongest section pair, under v1's passage question, its
rows holding no text, and spend under v2's own ledger and cap."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from lineage.labels import Item, Pair, Section, SectionKey, SectionRef
from lineage.model_labels import PASSAGES_HIGH, BatchReply, SameProvision, Schema
from lineage.origin_labels import (
    PROMPT,
    OriginItem,
    label_in_batch,
    pending,
    plan,
    record,
    strongest_pair,
)
from lineage.origin_precision import BUDGET, OriginPrecisionError
from lineage.origins import OriginMatch, SectionPair


def _match(*pairs: SectionPair) -> OriginMatch:
    return OriginMatch(
        bill_id="ocd-bill/1",
        doc_id="lib/a",
        stages=["near_duplicate", "semantic"],
        model_share=0.5,
        bill_share=0.5,
        spans=[],
        pairs=list(pairs),
    )


def test_the_strongest_pair_is_the_most_agreements_else_the_best_cosine() -> None:
    by_meaning = SectionPair(model_section=0, ordinal=0, cosine=0.99)
    few = SectionPair(model_section=1, ordinal=4, agreements=30, cosine=0.95)
    most = SectionPair(model_section=2, ordinal=1, agreements=90)
    tie = SectionPair(model_section=3, ordinal=0, agreements=90)

    # Copied text is what a page highlights, so the near-duplicate stage's pair wins.
    assert strongest_pair(_match(by_meaning, few, most, tie)) == most
    second = SectionPair(model_section=0, ordinal=2, cosine=0.95)
    assert strongest_pair(_match(second, by_meaning)) == by_meaning
    with pytest.raises(OriginPrecisionError, match="run `lineage origins` again"):
        strongest_pair(_match())
    with pytest.raises(ValueError, match="at least one stage"):
        SectionPair(model_section=0, ordinal=0)


def test_the_question_is_v1s_passage_question_unchanged() -> None:
    assert PROMPT is PASSAGES_HIGH
    assert PROMPT.schema is SameProvision


def _item(
    n: int, model_text: str = "The model's rule.", bill_text: str = "The bill's rule."
) -> OriginItem:
    ref_m = SectionRef("lib/a", None, n, f"{n:064x}")
    ref_b = SectionRef("ocd-bill/1", None, n, f"{n + 100:064x}")
    item = Item(
        Pair(f"o{n:04d}", ("near_duplicate",), SectionKey("lib/a", n), SectionKey("ocd-bill/1", n)),
        Section(ref_m, model_text),
        Section(ref_b, bill_text),
        passages=((model_text, bill_text),),
    )
    return OriginItem({"pair": f"o{n:04d}"}, item)


class _Batches:
    def __init__(self, labels: dict[str, int | None]) -> None:
        self.labels = labels
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def submit(self, requests: list[tuple[str, dict[str, Any]]]) -> str:
        self.sent = requests
        return "msgbatch_test"

    def ended(self, batch_id: str) -> bool:
        return True

    def results(self, batch_id: str, schema: Schema = SameProvision) -> Iterable[BatchReply]:
        for custom_id, label in self.labels.items():
            served = None if label is None else "claude-opus-5-5"
            yield BatchReply(custom_id, label, served, None if served else "declined", 1000, 50)


def test_a_batch_labels_each_match_by_its_hashes_and_keeps_no_text(tmp_path: Path) -> None:
    items = [_item(1), _item(2), _item(3)]
    batches = _Batches({"o0001": 1, "o0002": 0, "o0003": None})
    labels = tmp_path / "labels.jsonl"

    done, unusable, sent = label_in_batch(
        items, batches=batches, model="claude-opus-5-5", labels_path=labels, today="2026-11-16",
        wait=lambda: None,
    )  # fmt: skip

    assert (done, unusable, sent) == (2, 1, "msgbatch_test")
    # The model's section is asked first, beside the matched passages.
    message = json.loads(batches.sent[0][1]["messages"][0]["content"])
    assert message["first"] == "The model's rule."
    assert message["passages"] == [{"first": "The model's rule.", "second": "The bill's rule."}]
    written = labels.read_text()
    assert "rule" not in written
    rows = [json.loads(line) for line in written.splitlines()]
    assert rows[0] == {
        "pair": "o0001", "bill_id": "ocd-bill/1", "doc_id": "lib/a",
        "model_section": f"{1:064x}", "bill_section": f"{101:064x}", "label": 1,
        "labeler": "model:claude-opus-5-5@passages-v1-high", "scale": PROMPT.scale,
        "at": "2026-11-16",
    }  # fmt: skip
    # Labeled matches are not asked again; the unusable one is.
    assert [i.pair for i in pending(items, rows)] == ["o0003"]


def test_the_plan_counts_v2s_own_ledger_and_refuses_over_its_cap(tmp_path: Path) -> None:
    assert BUDGET == 25.0
    ledger = tmp_path / "spend.json"
    items = [_item(n, "x" * 5000, "y" * 5000) for n in range(10)]
    fresh = plan(items, spend_path=ledger)
    assert fresh.spent == 0
    assert fresh.budget == 25.0
    assert fresh.refusal() is None

    record(_Batches({"o0001": 1}), "msgbatch_test", model="claude-opus-5-5",
           today="2026-11-16", spend_path=ledger)  # fmt: skip
    doc = json.loads(ledger.read_text())
    assert doc["budget"] == 25.0
    assert doc["command"] == "uv run lineage origin-label --live"
    assert plan(items, spend_path=ledger).spent == doc["spent"] > 0
    over = plan(items, spend_path=ledger, budget=doc["spent"])
    assert over.refusal() is not None
    assert "over v2's" in str(over.refusal())
    assert plan(items, spend_path=ledger, max_cost=0.0).refusal() is not None


def test_the_command_caps_a_run_at_v1s_cap_unless_raised_explicitly() -> None:
    """The reporting sample's projection, every reply at its token limit, came to $11.22, over
    the $10 cap; the M11 review allowed that one run over it, by an explicit --max-cost, with
    v2's budget still checked."""
    import inspect

    from lineage.cli import origin_label_command
    from lineage.model_labels import MAX_COST

    default = inspect.signature(origin_label_command).parameters["max_cost"].default
    assert default == MAX_COST == 10.0
