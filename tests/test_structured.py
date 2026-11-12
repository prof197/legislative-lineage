"""One structured request: the schema sent without the bounds the API does not take, and every
reply, usable or not, returned with its usage (DESIGN.md#explanations)."""

import json
from types import SimpleNamespace

import pytest

from lineage.explanations import ChangeExplanation
from lineage.structured import UnusableReplyError, reply_text, sent_schema, validated

CLAIM = {"kind": "added", "summary": "x", "origin_span_ids": [], "descendant_span_ids": ["s1"]}


def test_the_schema_sent_drops_length_bounds_and_closes_objects() -> None:
    schema = sent_schema(ChangeExplanation)
    text = json.dumps(schema)

    assert "maxLength" not in text
    assert "maxItems" not in text
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Claim"]["additionalProperties"] is False


def test_a_reply_over_a_bound_is_unusable_and_keeps_its_usage() -> None:
    reply = json.dumps({"claims": [{**CLAIM, "summary": "x" * 201}]})

    with pytest.raises(UnusableReplyError, match=r"off the schema: claims\.0\.summary") as e:
        validated(ChangeExplanation, reply, "end_turn", (900, 120), "claude-opus-5-5")
    assert (e.value.input_tokens, e.value.output_tokens) == (900, 120)
    assert e.value.served_by == "claude-opus-5-5"


@pytest.mark.parametrize(
    ("stop", "text", "why"),
    [
        ("refusal", None, "declined"),
        ("max_tokens", '{"claims": [', "cut off"),
        ("end_turn", "{", "not JSON"),
        ("end_turn", None, "no text"),
    ],
)
def test_an_unusable_reply_says_why(stop: str, text: str | None, why: str) -> None:
    with pytest.raises(UnusableReplyError, match=why):
        validated(ChangeExplanation, text, stop, (1, 1), "m")


def test_a_usable_reply_is_validated_and_the_text_after_a_fallback_is_read() -> None:
    content = [
        SimpleNamespace(type="fallback"),
        SimpleNamespace(type="text", text=json.dumps({"claims": [CLAIM]})),
    ]

    got = validated(ChangeExplanation, reply_text(content), "end_turn", (5, 6), "m")

    assert got.value.claims[0].descendant_span_ids == ["s1"]
    assert (got.input_tokens, got.output_tokens, got.served_by) == (5, 6, "m")
