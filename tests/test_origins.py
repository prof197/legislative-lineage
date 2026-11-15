"""Origin matches (DESIGN.md#origin-matches): both shares, spans on the bill's side only, and a
table that is empty until matching has run."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from lineage.origins import BillSpan, OriginMatch, read_origin_matches, write_origin_matches


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
