"""The withhold report: the rule's part written from labels, the reviewed part kept by hand."""

import json
from pathlib import Path

from lineage.origins.withheld import (
    REASON,
    REVIEW,
    Reviewed,
    Tally,
    load_withheld,
    read_reviewed,
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
