"""Grounded explanations: the model's reply schema, and the code that checks each claim cites
spans that exist and differ (DESIGN.md#explanations)."""

import pytest
from pydantic import ValidationError

from lineage.explanations import (
    ChangeExplanation,
    Claim,
    GroundingSpan,
    ground,
    grounding_spans,
    reasons,
)

SPANS = grounding_spans(
    [
        {"id": "s0", "kind": "copied", "origin": "The fund is", "descendant": "The fund is"},
        {
            "id": "s1",
            "kind": "paraphrased",
            "origin": "each fiscal year",
            "descendant": "every two years",
        },
        # removed from the origin, and added in the descendant
        {"id": "s2", "kind": "unmatched", "origin": "and audited", "descendant": None},
        {"id": "s3", "kind": "unmatched", "origin": None, "descendant": "by the treasurer"},
        # paired by meaning, but only punctuation and case tell its sides apart
        {"id": "s4", "kind": "paraphrased", "origin": "The Board;", "descendant": "the board"},
    ]
)


def _claim(
    kind: str = "changed", origin: list[str] | None = None, descendant: list[str] | None = None
) -> Claim:
    return Claim.model_validate(
        {
            "kind": kind,
            "summary": "The period changes from a fiscal year to a biennium.",
            "origin_span_ids": origin or [],
            "descendant_span_ids": descendant or [],
        }
    )


def test_the_reply_schema_bounds_what_the_model_may_say() -> None:
    schema = ChangeExplanation.model_json_schema()
    claim = schema["$defs"]["Claim"]["properties"]

    assert schema["properties"]["claims"]["maxItems"] == 4
    assert claim["summary"]["maxLength"] == 200
    assert claim["kind"]["enum"] == ["added", "removed", "changed"]


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "rewrote", "summary": "x", "origin_span_ids": [], "descendant_span_ids": []},
        {"kind": "added", "summary": "x" * 201, "origin_span_ids": [], "descendant_span_ids": []},
        {"kind": "added", "summary": "x", "origin_span_ids": []},
    ],
)
def test_a_malformed_claim_is_refused(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate(bad)


def test_more_than_four_claims_is_refused() -> None:
    claim = _claim(origin=["s1"]).model_dump()

    with pytest.raises(ValidationError):
        ChangeExplanation.model_validate({"claims": [claim] * 5})


def test_claims_citing_differences_that_exist_are_kept() -> None:
    reply = ChangeExplanation(
        claims=[
            _claim("changed", origin=["s1"], descendant=["s1"]),
            _claim("removed", origin=["s2"]),
            _claim("added", descendant=["s3"]),
        ]
    )

    grounded = ground(reply, SPANS)

    assert grounded.claims == reply.claims
    assert grounded.dropped == []
    assert reasons(reply, SPANS) == [None, None, None]


@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        (_claim(), "cites no span"),
        (_claim(origin=["s9"]), "s9 is not in this alignment"),
        (_claim(origin=["s0"], descendant=["s0"]), "s0 is an exact copy"),
        (_claim("added", origin=["s3"]), "s3 has no origin text"),
        (_claim("removed", descendant=["s2"]), "s2 has no descendant text"),
        (_claim(origin=["s4"], descendant=["s4"]), "s4 reads the same on both sides"),
    ],
)
def test_a_claim_without_a_real_difference_behind_it_is_dropped(claim: Claim, reason: str) -> None:
    grounded = ground(ChangeExplanation(claims=[claim, _claim(origin=["s1"])]), SPANS)

    assert grounded.claims == [_claim(origin=["s1"])]
    assert [why for _, why in grounded.dropped] == [reason]


def test_the_check_keeps_no_text_only_each_sides_normalized_hash() -> None:
    by_id = {g.id: g for g in SPANS}
    stored = [GroundingSpan.from_json(g.as_json()) for g in SPANS]

    assert by_id["s2"].descendant is None
    assert by_id["s1"].origin is not None
    assert len(by_id["s1"].origin) == 64
    assert "fiscal" not in str([g.as_json() for g in SPANS])
    assert stored == SPANS
