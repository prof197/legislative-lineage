"""`lineage explain`: a model's grounded account of what a descendant changed, cached.

The model sees a pair's aligned spans (lineage.align), each with its id, kind and text on the
origin's and descendant's sides, never the raw bills. Its reply must fit ChangeExplanation, and
each claim is checked against the spans in code (lineage.explanations.ground): a failing claim
is dropped. A reply that is unusable, refused, malformed, or making claims none of which is
grounded, is asked again once; if the retry is unusable too, the explanation is withheld. A
reply of no claims says nothing of substance changed, and is kept as such.

Every outcome, withheld ones included, is cached in one file named by the SHA-256 of the model
id, the prompt version and the spans, so a rerun pays only for pairs it has not seen, CI and
keyless users replay the cache, and a changed prompt or alignment can never be served a stale
answer. A cached reply is re-grounded each time it is read. Without --live, a pair with no
cached outcome is an error, never a call. See DESIGN.md#explanations.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from lineage import queries
from lineage.align import Span
from lineage.explanations import ChangeExplanation, Claim, Side, ground

MODEL = "claude-opus-5-5"
PROMPT_VERSION = "v1"
CACHE_DIR = Path("explanations/cache")
ATTEMPTS = 2  # the first ask and one retry
_MAX_TOKENS = 16000
# Declines are rerun on the fallback model the API picks for the refusal's category; the cache
# records which model served the reply.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """\
You compare two versions of one section of state legislation: an origin and a descendant \
adapted from it. You are given their aligned spans as JSON. Each span has an id, a kind and its \
text on each side: "copied" spans are word for word the same in both; "paraphrased" spans say \
the same thing in other words; "unmatched" spans appear on one side only.

Write at most four claims about what the descendant changed. Each claim has a kind ("added", \
"removed" or "changed"), a summary of at most 200 characters, and the ids of the spans that \
show it: origin_span_ids for spans you cite on the origin's side, descendant_span_ids for the \
descendant's. Cite only paraphrased or unmatched spans; a copied span is shared text, not a \
change. Say what the text does, plainly. Do not speculate about who wrote either version or \
why. If nothing of substance changed, return no claims."""


class NotCachedError(Exception):
    pass


class UnusableReplyError(Exception):
    """The model declined, or its reply did not fit the schema."""


@dataclass(frozen=True, slots=True)
class Reply:
    explanation: ChangeExplanation
    served_by: str  # the model that produced it, which a fallback can change


Ask = Callable[[str, str], Reply]  # (system, user) -> reply; raises UnusableReplyError


@dataclass(frozen=True, slots=True)
class Outcome:
    claims: list[Claim]  # grounded, in the reply's order
    dropped: list[tuple[Claim, str]]
    withheld: bool
    served_by: str | None


def prompt_spans(
    spans: Sequence[Span], a: str, b: str, *, origin_side: Side
) -> list[dict[str, Any]]:
    """Each span's id, kind and text on the origin's and descendant's sides."""

    def text(source: str, at: tuple[int, int] | None) -> str | None:
        return None if at is None else source[at[0] : at[1]]

    out: list[dict[str, Any]] = []
    for s in spans:
        on_a, on_b = text(a, s.a), text(b, s.b)
        origin, descendant = (on_a, on_b) if origin_side == "a" else (on_b, on_a)
        out.append({"id": s.id, "kind": s.kind, "origin": origin, "descendant": descendant})
    return out


def cache_key(model: str, prompt_version: str, spans: Sequence[dict[str, Any]]) -> str:
    doc = {"model": model, "prompt_version": prompt_version, "spans": list(spans)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _outcome(entry: dict[str, Any], spans: Sequence[Span], origin_side: Side) -> Outcome:
    if entry["explanation"] is None:
        return Outcome([], [], True, entry["served_by"])
    reply = ChangeExplanation.model_validate(entry["explanation"])
    grounded = ground(reply, spans, origin_side=origin_side)
    withheld = bool(reply.claims) and not grounded.claims
    return Outcome(grounded.claims, grounded.dropped, withheld, entry["served_by"])


def explain(
    spans_for_prompt: Sequence[dict[str, Any]],
    spans: Sequence[Span],
    *,
    origin_side: Side,
    ask: Ask | None,
    cache: Path = CACHE_DIR,
    model: str = MODEL,
) -> Outcome:
    """The cached outcome for these spans, or, with `ask`, a new one from the model."""
    path = cache / f"{cache_key(model, PROMPT_VERSION, spans_for_prompt)}.json"
    if path.exists():
        return _outcome(json.loads(path.read_text()), spans, origin_side)
    if ask is None:
        raise NotCachedError("no cached explanation for this pair; run with --live to ask")
    user = json.dumps({"spans": list(spans_for_prompt)}, ensure_ascii=False, indent=1)
    entry: dict[str, Any] = {"explanation": None, "served_by": None}
    for _ in range(ATTEMPTS):
        try:
            reply = ask(SYSTEM, user)
        except UnusableReplyError:
            continue
        claims = reply.explanation.claims
        if not claims or ground(reply.explanation, spans, origin_side=origin_side).claims:
            entry = {"explanation": reply.explanation.model_dump(), "served_by": reply.served_by}
            break
    cache.mkdir(parents=True, exist_ok=True)
    record = {"model": model, "prompt_version": PROMPT_VERSION} | entry
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return _outcome(entry, spans, origin_side)


def anthropic_ask(model: str = MODEL) -> Ask:
    """Ask through the Anthropic SDK with structured output from ChangeExplanation. The key
    comes from the environment, as every secret here does."""
    import anthropic

    client = anthropic.Anthropic()

    def ask(system: str, user: str) -> Reply:
        try:
            response = client.beta.messages.parse(
                model=model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=ChangeExplanation,
                output_config={"effort": "medium"},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except ValidationError as e:
            raise UnusableReplyError(f"the reply does not fit the schema: {e}") from e
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise UnusableReplyError(f"no usable reply (stop reason {response.stop_reason})")
        return Reply(response.parsed_output, response.model)

    return ask


@dataclass(frozen=True, slots=True)
class PairToExplain:
    origin: tuple[str, int]  # (bill id, ordinal) in the build the pairs were drawn from
    descendant: tuple[str, int]


def load_pairs(path: Path) -> list[PairToExplain]:
    """The pairs to explain, one JSON object per line, each naming its origin and descendant
    sections by bill id and ordinal: the scope, featured stories first."""
    out: list[PairToExplain] = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            out.append(
                PairToExplain(
                    (str(row["origin"]["bill_id"]), int(row["origin"]["ordinal"])),
                    (str(row["descendant"]["bill_id"]), int(row["descendant"]["ordinal"])),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}:{n}: not a pair to explain ({e})") from e
    return out


def pair_texts(tables_dir: Path, pairs: Sequence[PairToExplain]) -> list[tuple[str, str]]:
    """Each pair's origin and descendant display texts, read in one pass."""
    keys = sorted({k for p in pairs for k in (p.origin, p.descendant)})
    with queries.connect_quiet() as con:
        found = queries.sections_at(
            con, sections_glob=str(tables_dir / "sections" / "*.parquet"), keys=keys
        )
    missing = [k for k in keys if k not in found]
    if missing:
        bill, ordinal = missing[0]
        raise ValueError(
            f"{len(missing)} sections are not in {tables_dir / 'sections'}, {bill} section "
            f"{ordinal} first"
        )
    return [(found[p.origin][1], found[p.descendant][1]) for p in pairs]
