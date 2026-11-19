"""Origin matches (DESIGN.md#origin-matches): both shares, spans on the bill's side only, and a
table that is empty until matching has run."""

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lineage.origins.matches import (
    FLOOR,
    REACH_SCALE,
    BillSpan,
    OriginMatch,
    Weighed,
    is_listed,
    origins_report,
    read_origin_matches,
    share_of,
    weight,
    weighted_shares,
    write_origin_matches,
)
from lineage.origins.modeldocs import ModelDoc


def _match(bill: str, doc: str, model_share: float = 0.4, bill_share: float = 0.9) -> OriginMatch:
    return OriginMatch(
        bill_id=bill,
        doc_id=doc,
        stages=["near_duplicate"],
        model_share=model_share,
        bill_share=bill_share,
        spans=[BillSpan(ordinal=2, start=0, end=120, kind="exact")],
    )


def test_the_table_is_empty_until_matching_runs_and_sorted_after(tmp_path: Path) -> None:
    assert read_origin_matches(tmp_path) == []
    b, a = _match("ocd-bill/b", "commission/x"), _match("ocd-bill/a", "commission/y")

    write_origin_matches(tmp_path, [b, a])

    assert read_origin_matches(tmp_path) == [a, b]


def test_a_match_has_a_stage_and_shares_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        OriginMatch(bill_id="b", doc_id="d", stages=[], model_share=0.5, bill_share=0.5, spans=[])
    with pytest.raises(ValidationError):
        _match("b", "d", model_share=1.2)
    with pytest.raises(ValidationError):  # the model's text is never shown, so no span points in
        BillSpan.model_validate({"ordinal": 0, "start": 0, "end": 5, "kind": "exact", "side": "a"})


def _doc(doc_id: str, text: str) -> ModelDoc:
    return ModelDoc(
        id=doc_id,
        source=doc_id.partition("/")[0],
        title="Model Act",
        url="https://example.org/act",
        collected_on=date(2026, 11, 16),
        robots="allowed",
        leaked=False,
        raw_sha256="0" * 64,
        media_type="text/html",
        text_sha256=text * 64,
    )


def test_the_floor_reads_the_larger_share() -> None:
    def match(model_share: float, bill_share: float) -> OriginMatch:
        return OriginMatch(
            bill_id="b", doc_id="s/d", stages=["semantic"],
            model_share=model_share, bill_share=bill_share, spans=[],
        )  # fmt: skip

    assert FLOOR == 0.25
    assert share_of(match(0.1, 0.4)) == share_of(match(0.4, 0.1)) == 0.4
    assert is_listed(match(FLOOR, 0.0))
    assert is_listed(match(0.0, FLOOR))
    assert not is_listed(match(0.2499, 0.2499))
    assert is_listed(match(0.1, 0.1), floor=0.1)


def test_the_report_counts_by_source_and_stage_and_shared_texts_once() -> None:
    docs = [_doc("first/a", "1"), _doc("first/b", "2"), _doc("second/a", "1")]
    both = _match("ocd-bill/1", "first/a").model_copy(
        update={"stages": ["near_duplicate", "semantic"]}
    )
    by_meaning = _match("ocd-bill/2", "second/a").model_copy(update={"stages": ["semantic"]})
    matches = [both, _match("ocd-bill/2", "first/a"), by_meaning]

    report = origins_report(matches, docs, seconds=12.34)

    assert (report.documents, report.distinct_texts, report.matches, report.bills) == (3, 2, 3, 2)
    assert report.by_stage == {"near_duplicate": 1, "semantic": 1, "both": 1}
    assert report.by_source["first"].model_dump() == {
        "documents": 2,
        "documents_matched": 1,
        "matches": 2,
    }
    assert report.by_source["second"].matches == 1
    assert report.seconds == 12.3


def test_the_report_refuses_a_match_no_source_holds() -> None:
    with pytest.raises(ValueError, match="no source holds"):
        origins_report([_match("ocd-bill/1", "gone/a")], [_doc("first/a", "1")], seconds=None)


def _weighed(
    model: list[tuple[int, int]], bill: list[tuple[int, int]], words: tuple[int, int]
) -> dict[str, Any]:
    """A match's weighing fields: (words, reach) per matched section, and the shares' wholes."""
    return {
        "model_words": words[0],
        "bill_words": words[1],
        "weighed_model": [Weighed(section=i, words=w, reach=r) for i, (w, r) in enumerate(model)],
        "weighed_bill": [Weighed(section=i, words=w, reach=r) for i, (w, r) in enumerate(bill)],
    }


def test_a_passage_counts_in_full_up_to_the_scale_and_less_the_wider_it_recurs() -> None:
    assert weight(0) == weight(REACH_SCALE) == 1.0
    assert weight(2 * REACH_SCALE) == 0.5
    assert weight(10 * REACH_SCALE) == pytest.approx(0.1)


def test_the_floor_reads_the_weighted_shares_and_pages_keep_the_plain_ones() -> None:
    common = 4 * REACH_SCALE
    m = OriginMatch(
        bill_id="b",
        doc_id="lib/doc",
        stages=["near_duplicate"],
        model_share=0.5,
        bill_share=0.4,
        spans=[],
        **_weighed([(50, common)], [(40, common)], (100, 100)),
    )
    assert weighted_shares(m) == (0.125, 0.1)
    assert share_of(m) == 0.125
    assert not is_listed(m)
    assert (m.model_share, m.bill_share) == (0.5, 0.4)  # what a page shows is unchanged
    distinctive = m.model_copy(update=_weighed([(50, 3)], [(40, 3)], (100, 100)))
    assert weighted_shares(distinctive) == (0.5, 0.4)
    assert is_listed(distinctive)


def test_a_match_written_without_reach_is_read_by_its_plain_shares() -> None:
    m = OriginMatch(
        bill_id="b",
        doc_id="lib/doc",
        stages=["semantic"],
        model_share=0.3,
        bill_share=0.1,
        spans=[],
    )
    assert weighted_shares(m) == (0.3, 0.1)
    assert is_listed(m)
