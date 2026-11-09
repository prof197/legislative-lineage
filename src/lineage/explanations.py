"""Grounded explanations: what a model may say about a matched pair, and the check that it is
backed by the alignment.

A featured match is explained from its aligned spans (lineage.align), each with an id, never
from the raw bill texts, so the model sees exactly what is shared and what differs. Its reply
must fit ChangeExplanation: one Pydantic class gives the JSON schema sent to the model, the
validator for what comes back, and the types the rest of the code checks. A claim names a change
(added, removed, changed), summarizes it briefly, and cites the spans behind it on the origin's
side, the descendant's, or both.

Grounding is checked in code, not trusted. A claim is kept only if it cites at least one span,
every id it cites is a span of this alignment with text on the side it is cited for, and none
is an exact copy, since a copy is not a change. A failing claim is dropped with its reason.
What happens when claims fail, a retry and then withholding the explanation, belongs to the
step that calls the model. See DESIGN.md#explanations.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lineage.align import Span

MAX_CLAIMS = 4
MAX_SUMMARY = 200

Side = Literal["a", "b"]


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
class Grounded:
    claims: list[Claim]  # the claims that passed, in the reply's order
    dropped: list[tuple[Claim, str]]  # each failing claim, with why


def _why_not(claim: Claim, spans: dict[str, Span], origin_side: Side) -> str | None:
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
        side = origin_side if role == "origin" else ("b" if origin_side == "a" else "a")
        if (span.a if side == "a" else span.b) is None:
            return f"{span_id} has no {role} text"
    return None


def ground(reply: ChangeExplanation, spans: Sequence[Span], *, origin_side: Side) -> Grounded:
    """Keep the claims backed by real differences in `spans`; drop the rest, saying why.
    `origin_side` says which of the alignment's texts is the origin."""
    by_id = {s.id: s for s in spans}
    kept: list[Claim] = []
    dropped: list[tuple[Claim, str]] = []
    for claim in reply.claims:
        why = _why_not(claim, by_id, origin_side)
        if why is None:
            kept.append(claim)
        else:
            dropped.append((claim, why))
    return Grounded(kept, dropped)
