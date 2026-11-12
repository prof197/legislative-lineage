"""Grounded explanations: what a model may say about a matched pair, and the check that it is
backed by the alignment.

A featured match is explained from its aligned spans (lineage.align), each with an id, never
from the raw bill texts, so the model sees exactly what is shared and what differs. Its reply
must fit ChangeExplanation: one Pydantic class gives the JSON schema sent to the model, the
validator for what comes back, and the types the rest of the code checks. A claim names a change
(added, removed, changed), summarizes it briefly, and cites the spans behind it on the origin's
side, the descendant's, or both.

Grounding is checked in code, not trusted. A claim is kept only if it cites at least one span,
every id it cites is a span of this alignment with text on the side it is cited for, none is
an exact copy, and none reads the same on both sides once normalized, since neither is a
change. A "changed" claim must also cite spans on both sides, each with text on both: a span
on one side only shows text one side lacks, which is an added or removed claim, never a
change (the M6 review's ruling). A failing claim is dropped with its reason. The check reads
each span as a GroundingSpan: its id, its kind, and the SHA-256 of each side's normalized
text, so it runs the same on a live alignment and on a cached explanation in CI, where no bill
text is kept. What happens when claims fail, a retry and then withholding the explanation,
belongs to the step that calls the model. See DESIGN.md#explanations.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lineage.sections import normalize

MAX_CLAIMS = 4
MAX_SUMMARY = 200

Side = Literal["a", "b"]  # which of an alignment's two texts is the origin


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["added", "removed", "changed"]
    summary: str = Field(max_length=MAX_SUMMARY)
    origin_span_ids: list[str]
    descendant_span_ids: list[str]


class ChangeExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[Claim] = Field(max_length=MAX_CLAIMS)


@dataclass(frozen=True, slots=True)
class GroundingSpan:
    """What the check needs of a span: its id, kind, and each side's normalized text as a
    SHA-256, None where the span has no text on that side."""

    id: str
    kind: str
    origin: str | None
    descendant: str | None

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "origin": self.origin,
            "descendant": self.descendant,
        }

    @staticmethod
    def from_json(row: Mapping[str, Any]) -> "GroundingSpan":
        return GroundingSpan(str(row["id"]), str(row["kind"]), row["origin"], row["descendant"])


def _digest(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(normalize(text).encode()).hexdigest()


def grounding_spans(spans_for_prompt: Sequence[Mapping[str, Any]]) -> list[GroundingSpan]:
    """The prompt's spans, each side's text replaced by its normalized text's SHA-256."""
    return [
        GroundingSpan(str(s["id"]), str(s["kind"]), _digest(s["origin"]), _digest(s["descendant"]))
        for s in spans_for_prompt
    ]


@dataclass(frozen=True, slots=True)
class Grounded:
    claims: list[Claim]  # the claims that passed, in the reply's order
    dropped: list[tuple[Claim, str]]  # each failing claim, with why


def _why_not(claim: Claim, spans: Mapping[str, GroundingSpan]) -> str | None:
    cited = [("origin", i) for i in claim.origin_span_ids]
    cited += [("descendant", i) for i in claim.descendant_span_ids]
    if not cited:
        return "cites no span"
    for role, span_id in cited:
        span = spans.get(span_id)
        if span is None:
            return f"{span_id} is not in this alignment"
        if span.kind == "copied":
            return f"{span_id} is an exact copy"
        if (span.origin if role == "origin" else span.descendant) is None:
            return f"{span_id} has no {role} text"
        if span.origin is not None and span.origin == span.descendant:
            return f"{span_id} reads the same on both sides"
    if claim.kind == "changed":
        for _, span_id in cited:
            span = spans[span_id]
            if span.origin is None or span.descendant is None:
                return (
                    f"{span_id} is on one side only, so it shows text one side lacks, not a change"
                )
        for side, ids in (
            ("origin", claim.origin_span_ids),
            ("descendant", claim.descendant_span_ids),
        ):
            if not ids:
                return f"a changed claim cites no span on the {side}'s side"
    return None


def reasons(reply: ChangeExplanation, spans: Sequence[GroundingSpan]) -> list[str | None]:
    """For each claim in order, why it fails grounding, or None when it passes."""
    by_id = {s.id: s for s in spans}
    return [_why_not(claim, by_id) for claim in reply.claims]


def ground(reply: ChangeExplanation, spans: Sequence[GroundingSpan]) -> Grounded:
    """Keep the claims backed by real differences in `spans`; drop the rest, saying why."""
    kept: list[Claim] = []
    dropped: list[tuple[Claim, str]] = []
    for claim, why in zip(reply.claims, reasons(reply, spans), strict=True):
        if why is None:
            kept.append(claim)
        else:
            dropped.append((claim, why))
    return Grounded(kept, dropped)
