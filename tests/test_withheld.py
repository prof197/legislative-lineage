"""The withhold report: the rule's part written from labels, the reviewed part kept by hand."""

import json
from pathlib import Path

from lineage.origins.matches import BillSpan, OriginMatch
from lineage.origins.precision import PARTS
from lineage.origins.withheld import (
    REASON,
    REVIEW,
    TOPUP_PER_DOCUMENT,
    Reviewed,
    Tally,
    load_withheld,
    read_reviewed,
    topup,
    unsettled,
    write_report,
)

RULED = Tally(doc_id="lib/a", title="A", listed=5, labeled=3, correct=0)
KEPT = Tally(doc_id="lib/b", title="B", listed=9, labeled=4, correct=3)
REVIEWED = Reviewed(doc_id="lib/c", title="C", listed=2, note="wrong bills", on="2027-01-12")


def test_a_document_withheld_by_review_is_kept_when_the_rule_is_written_again(
    tmp_path: Path,
) -> None:
    """Ruled at the deploy review: bad claims on a held day are dealt with by withholding the
    documents behind them in the committed report, which `lineage withhold` must not undo."""
    path = tmp_path / "withheld.json"
    write_report([RULED, KEPT], path, [REVIEWED])
    write_report([RULED, KEPT], path)  # written again from labels alone

    assert read_reviewed(path) == [REVIEWED]
    got = load_withheld(path)
    assert dict(got.reasons) == {"lib/a": REASON, "lib/c": REVIEW}
    assert got.matches == 5 + 2
    doc = json.loads(path.read_text())
    assert [d["doc_id"] for d in doc["withheld"]] == ["lib/a"]
    assert doc["reviewed"][0]["note"] == "wrong bills"


def test_a_document_the_rule_withholds_is_listed_once(tmp_path: Path) -> None:
    path = tmp_path / "withheld.json"
    write_report([RULED], path, [Reviewed("lib/a", "A", 5, "also wrong", "2027-01-12")])
    got = load_withheld(path)
    assert dict(got.reasons) == {"lib/a": REASON}
    assert got.matches == 5


def test_no_report_withholds_nothing(tmp_path: Path) -> None:
    assert load_withheld(tmp_path / "none.json").ids == frozenset()


def _listed(bill: str, doc: str) -> OriginMatch:
    return OriginMatch(
        bill_id=bill,
        doc_id=doc,
        stages=["near_duplicate"],
        model_share=0.5,
        bill_share=0.5,
        spans=[BillSpan(ordinal=1, start=0, end=80, kind="exact")],
    )


def test_only_a_document_with_a_wrong_label_and_too_few_to_settle_is_topped_up() -> None:
    """Ruled for this week's release: a wrong label among fewer than the rule's minimum leaves a
    document neither withheld nor cleared; its remaining listed matches are labeled, up to ten,
    for every source alike."""
    matches = [_listed(f"b{i}", "lib/short") for i in range(14)]  # 1 labeled, wrong
    matches += [_listed(f"b{i}", "lib/right") for i in range(4)]  # 1 labeled, correct
    matches += [_listed(f"b{i}", "lib/settled") for i in range(5)]  # 3 labeled, 1 wrong
    answers = {("b0", "lib/short"): False, ("b0", "lib/right"): True}
    answers |= {("b0", "lib/settled"): False, ("b1", "lib/settled"): True}
    answers |= {("b2", "lib/settled"): True}

    assert unsettled(matches, answers) == ["lib/short"]
    rows, report = topup(matches, answers, PARTS["withhold_check"])
    assert len(rows) == TOPUP_PER_DOCUMENT
    assert {r["doc_id"] for r in rows} == {"lib/short"}
    assert "b0" not in {r["bill_id"] for r in rows}  # the labeled one is not asked again
    assert [r["pair"] for r in rows][:2] == ["o1901", "o1902"]
    assert report["documents"]["lib/short"] == {
        "listed": 14,
        "labeled": 1,
        "correct": 0,
        "drawn": TOPUP_PER_DOCUMENT,
    }
    assert topup(matches, answers, PARTS["withhold_check"])[0] == rows  # seeded


def test_a_match_no_page_lists_neither_counts_nor_is_drawn() -> None:
    hidden = _listed("b9", "lib/short").model_copy(update={"model_share": 0.01, "bill_share": 0.01})
    matches = [_listed("b0", "lib/short"), hidden]
    rows, _ = topup(matches, {("b0", "lib/short"): False}, PARTS["withhold_check"])
    assert rows == []


def test_a_document_whose_every_listed_match_is_labeled_is_judged_on_them_all() -> None:
    """Amended 2026-11-19: the minimum guards against judging from a small sample, and when
    every listed match is labeled there is no sample."""
    assert Tally(doc_id="x", title="X", listed=1, labeled=1, correct=0).withheld
    assert Tally(doc_id="x", title="X", listed=2, labeled=2, correct=0).withheld
    assert not Tally(doc_id="x", title="X", listed=2, labeled=2, correct=1).withheld  # half
    assert not Tally(doc_id="x", title="X", listed=3, labeled=2, correct=0).withheld  # a sample
    assert not Tally(doc_id="x", title="X", listed=0, labeled=0, correct=0).withheld


def test_a_document_with_every_listed_match_labeled_is_not_left_to_top_up() -> None:
    matches = [_listed("b0", "lib/all"), _listed("b1", "lib/all")]
    answers = {("b0", "lib/all"): False, ("b1", "lib/all"): False}
    assert unsettled(matches, answers) == []
