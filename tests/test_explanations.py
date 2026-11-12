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
    claim = _claim(origin=["s1"], descendant=["s1"]).model_dump()

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
    good = _claim(origin=["s1"], descendant=["s1"])
    grounded = ground(ChangeExplanation(claims=[claim, good]), SPANS)

    assert grounded.claims == [good]
    assert [why for _, why in grounded.dropped] == [reason]


@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        (_claim(origin=["s1"]), "a changed claim cites no span on the descendant's side"),
        (_claim(descendant=["s1"]), "a changed claim cites no span on the origin's side"),
        (_claim(origin=["s2"]), "a changed claim cites no span on the descendant's side"),
        (_claim(descendant=["s3"]), "a changed claim cites no span on the origin's side"),
    ],
)
def test_a_changed_claim_must_cite_text_from_both_texts(claim: Claim, reason: str) -> None:
    assert reasons(ChangeExplanation(claims=[claim]), SPANS) == [reason]


@pytest.mark.parametrize(
    "claim",
    [
        # Text the origin has where the descendant has other text: a removed span and an added
        # one, each counting for its own side (the M6 faithfulness ruling).
        _claim(origin=["s2"], descendant=["s3"]),
        _claim(origin=["s1", "s2"], descendant=["s1"]),
        _claim(origin=["s1"], descendant=["s1", "s3"]),
        _claim(origin=["s1"], descendant=["s1"]),
    ],
)
def test_a_span_on_one_side_only_counts_for_that_side_of_a_changed_claim(claim: Claim) -> None:
    assert reasons(ChangeExplanation(claims=[claim]), SPANS) == [None]


def test_a_one_sided_span_still_supports_a_claim_that_one_text_lacks_something() -> None:
    reply = ChangeExplanation(
        claims=[_claim("removed", origin=["s2"]), _claim("added", descendant=["s3", "s1"])]
    )

    assert reasons(reply, SPANS) == [None, None]


def test_the_check_keeps_no_text_only_each_sides_normalized_hash() -> None:
    by_id = {g.id: g for g in SPANS}
    stored = [GroundingSpan.from_json(g.as_json()) for g in SPANS]

    assert by_id["s2"].descendant is None
    assert by_id["s1"].origin is not None
    assert len(by_id["s1"].origin) == 64
    assert "fiscal" not in str([g.as_json() for g in SPANS])
    assert stored == SPANS
