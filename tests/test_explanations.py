"""Grounded explanations: the model's reply schema, and the code that checks each claim cites
spans that exist and differ (DESIGN.md#explanations)."""

import pytest
from pydantic import ValidationError

from lineage.align import Span
from lineage.explanations import ChangeExplanation, Claim, ground

# origin is side a, descendant side b
SPANS = [
    Span("s0", "copied", (0, 10), (0, 10)),
    Span("s1", "paraphrased", (11, 20), (11, 25)),
    Span("s2", "unmatched", (21, 30), None),  # removed from the origin
    Span("s3", "unmatched", None, (26, 40)),  # added in the descendant
]


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

    grounded = ground(reply, SPANS, origin_side="a")

    assert grounded.claims == reply.claims
    assert grounded.dropped == []


@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        (_claim(), "cites no span"),
        (_claim(origin=["s9"]), "s9 is not in this alignment"),
        (_claim(origin=["s0"], descendant=["s0"]), "s0 is an exact copy"),
        (_claim("added", origin=["s3"]), "s3 has no origin text"),
        (_claim("removed", descendant=["s2"]), "s2 has no descendant text"),
    ],
)
def test_a_claim_without_a_real_difference_behind_it_is_dropped(claim: Claim, reason: str) -> None:
    grounded = ground(
        ChangeExplanation(claims=[claim, _claim(origin=["s1"])]), SPANS, origin_side="a"
    )

    assert grounded.claims == [_claim(origin=["s1"])]
    assert [why for _, why in grounded.dropped] == [reason]


def test_the_origin_may_be_either_side() -> None:
    reply = ChangeExplanation(claims=[_claim("added", descendant=["s2"])])

    assert ground(reply, SPANS, origin_side="b").claims == reply.claims
