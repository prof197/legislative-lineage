"""`lineage explain`: a model's grounded account of what a descendant changed, cached.

The model sees a pair's aligned spans (lineage.align), each with its id, kind and text on the
origin's and descendant's sides, never the raw bills. Its reply must fit ChangeExplanation, and
each claim is checked against the spans in code (lineage.explanations.ground). A reply that is
unusable (refused, malformed, or making claims none of which is grounded) or has a claim that
fails grounding is asked again once, with why. The better of the usable replies is kept, one
whose every claim is grounded first, its failing claims dropped; with none, the explanation is
withheld. A reply of no claims says nothing of substance changed, and is kept as such.

Each cache file records the version of the grounding rules it was written under. Asked again
(`again`), a file from earlier rules is asked once more, with why: a withheld one afresh, with
its retry, and a kept one whose reply has a claim the current rules drop, once, the earlier reply
standing if the new one is no better. Asked afresh (`afresh`), every file from earlier rules
gets a new round of its own, a first ask and its retry, and its new reply stands whatever the
old one was: earlier rules shape how a reply is told, not only which claims it keeps. What a
file held before is kept in it (`earlier`).

Every outcome, withheld ones included, is cached in one file named by the SHA-256 of the model
id, the prompt version and the spans, so a rerun pays only for pairs it has not seen, keyless
users replay the cache, and a changed prompt or alignment can never be served a stale answer.
A cached reply is re-grounded each time it is read. Without --live, a pair with no cached
outcome is an error, never a call. A cache file holds no bill text: beside the reply it keeps
each span's grounding form (lineage.explanations.GroundingSpan) and what grounding kept and
dropped, so CI grounds every committed reply again with no corpus (check_cache). Every call is
metered against M6's budget (lineage.spend). See DESIGN.md#explanations.
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
from lineage.explanations import (
    ChangeExplanation,
    Claim,
    GroundingSpan,
    Side,
    ground,
    grounding_spans,
    reasons,
)
from lineage.spend import Meter, upper_bound
from lineage.structured import UnusableReplyError, ask_structured

MODEL = "claude-opus-5-5"
PROMPT_VERSION = "v1"
CACHE_DIR = Path("explanations/cache")
ATTEMPTS = 2  # the first ask and one retry
# The grounding rules' version, recorded in each cache file. 2 (ruled 2026-11-12): a changed
# claim cites spans on both sides, each with text on both. 3 (ruled 2026-11-12, correcting 2):
# a changed claim cites text from both texts, a span on one side only counting for that side.
GROUNDING_RULES = 3
# Four claims of at most 200 characters, and the thinking medium effort spends on them; a reply
# cut off at the limit is unusable and asked again.
MAX_TOKENS = 4096

SYSTEM = """\
You compare two versions of one section of state legislation. One is the story's anchor, \
called the origin only because the comparison starts from it; the other, a related text from \
another state, is called the descendant. Nothing is known about which was written first, who \
drew on whom, or why they differ. You are given their aligned spans as JSON. Each span has an \
id, a kind and its text on each side: "copied" spans are word for word the same in both; \
"paraphrased" spans say the same thing in other words; "unmatched" spans appear on one side \
only.

Write at most four claims, each describing one way the descendant's text differs from the \
origin's. Each claim has a kind, a summary of at most 200 characters, and the ids of the spans \
that show it: origin_span_ids for spans you cite on the origin's side, descendant_span_ids for \
the descendant's. The kind is "added" for text the descendant has and the origin lacks, \
"removed" for text the origin has and the descendant lacks, and "changed" for text both have \
that differs. Cite only paraphrased or unmatched spans; a copied span is shared text, not a \
difference.

Describe the texts, never people or history. Do not say that a state, legislature or drafter \
changed, added, removed, copied, adopted or borrowed anything, and do not say or imply which \
text came first or that one derives from the other. Write, for example: "The descendant sets \
the fee every biennium; the origin sets it each fiscal year." Say what the text does, plainly. \
If the two texts do not differ in substance, return no claims."""


class NotCachedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Reply:
    explanation: ChangeExplanation
    served_by: str  # the model that produced it, which a fallback can change
    input_tokens: int = 0
    output_tokens: int = 0


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


def explanation_key(spans_for_prompt: Sequence[dict[str, Any]], model: str = MODEL) -> str:
    """The cache key of a pair's explanation under the current prompt."""
    return cache_key(model, PROMPT_VERSION, spans_for_prompt)


def user_message(spans_for_prompt: Sequence[dict[str, Any]]) -> str:
    return json.dumps({"spans": list(spans_for_prompt)}, ensure_ascii=False, indent=1)


def retry_message(user: str, why: str) -> str:
    """The prompt asked again, with why the first reply could not be used."""
    return (
        f"{user}\n\nAn earlier reply to these spans could not be used: {why}. Reply again "
        "within the limits: at most four claims, each summary at most 200 characters, each "
        "citing only paraphrased or unmatched spans whose two sides differ. A changed claim "
        "cites text from both texts: at least one span in origin_span_ids and one in "
        "descendant_span_ids, each with text on the side it is cited for; an unmatched span "
        "counts for the side it has text on."
    )


# Room in a projection for the retry's added text and its reason.
RETRY_CHARS = len(retry_message("", "")) + 2000


def prompt_chars(spans_for_prompt: Sequence[dict[str, Any]]) -> int:
    return len(SYSTEM) + len(user_message(spans_for_prompt))


def _grounded_record(
    reply: ChangeExplanation | None, spans: Sequence[GroundingSpan]
) -> dict[str, Any] | None:
    if reply is None:
        return None
    why = reasons(reply, spans)
    return {
        "kept": [i for i, w in enumerate(why) if w is None],
        "dropped": [[i, w] for i, w in enumerate(why) if w is not None],
    }


def _outcome(entry: dict[str, Any], spans: Sequence[GroundingSpan]) -> Outcome:
    if entry["explanation"] is None:
        return Outcome([], [], True, entry["served_by"])
    reply = ChangeExplanation.model_validate(entry["explanation"])
    grounded = ground(reply, spans)
    withheld = bool(reply.claims) and not grounded.claims
    return Outcome(grounded.claims, grounded.dropped, withheld, entry["served_by"])


def cache_path(spans_for_prompt: Sequence[dict[str, Any]], cache: Path, model: str = MODEL) -> Path:
    return cache / f"{explanation_key(spans_for_prompt, model)}.json"


Kept = tuple[ChangeExplanation, str | None]  # a usable reply and the model that served it


def _usable(reply: ChangeExplanation, spans: Sequence[GroundingSpan]) -> bool:
    return not reply.claims or any(w is None for w in reasons(reply, spans))


def _rank(reply: ChangeExplanation, spans: Sequence[GroundingSpan]) -> tuple[bool, int]:
    why = reasons(reply, spans)
    return (all(w is None for w in why), sum(w is None for w in why))


def _failing(reply: ChangeExplanation, spans: Sequence[GroundingSpan]) -> str | None:
    """Why a reply is asked again: no claim grounded, or one that fails; None if all pass."""
    why = reasons(reply, spans)
    bad = [f"claim {i}: {w}" for i, w in enumerate(why, start=1) if w is not None]
    if not bad:
        return None
    if len(bad) == len(why):
        return "no claim cites spans that exist and differ (" + "; ".join(bad) + ")"
    return "a claim fails grounding (" + "; ".join(bad) + ")"


@dataclass(frozen=True, slots=True)
class _Round:
    attempts: int
    why: str | None  # what the first ask of the round says was wrong before
    kept: Kept | None  # the reply standing before the round


def _due(entry: dict[str, Any], spans: Sequence[GroundingSpan], *, afresh: bool) -> _Round | None:
    """The round a cache file from earlier grounding rules is asked again in, if any: afresh,
    a new round of its own whatever the file held."""
    if int(entry.get("grounding_rules", 1)) >= GROUNDING_RULES:
        return None
    if afresh:
        return _Round(ATTEMPTS, None, None)
    if entry["explanation"] is None:
        before = list(entry.get("unusable_attempts") or [])
        return _Round(ATTEMPTS, before[-1] if before else None, None)
    reply = ChangeExplanation.model_validate(entry["explanation"])
    why = _failing(reply, spans)
    if why is None:
        return None
    return _Round(1, why, (reply, entry["served_by"]) if _usable(reply, spans) else None)


def _ask_round(
    ask: Ask, user: str, spans: Sequence[GroundingSpan], plan: _Round, meter: Meter | None
) -> tuple[Kept | None, list[str], int]:
    """Ask up to `plan.attempts` times, each ask after the first saying why the last could not
    stand. Returns the best usable reply, why each ask but a last fully grounded one was not
    enough, and how many asks were made."""
    kept, why = plan.kept, plan.why
    unusable: list[str] = []
    asks = 0
    for _ in range(plan.attempts):
        asks += 1
        asked = retry_message(user, why) if why else user
        if meter is not None:
            meter.before_call(len(SYSTEM) + len(asked), MAX_TOKENS)
        try:
            reply = ask(SYSTEM, asked)
        except UnusableReplyError as e:
            if meter is not None:
                meter.add(e.input_tokens, e.output_tokens, e.served_by)
            why = str(e)
            unusable.append(why)
            continue
        if meter is not None:
            meter.add(reply.input_tokens, reply.output_tokens, reply.served_by)
        got = reply.explanation
        if _usable(got, spans) and (kept is None or _rank(got, spans) >= _rank(kept[0], spans)):
            kept = (got, reply.served_by)
        why = _failing(got, spans)
        if why is None:
            break
        unusable.append(why)
    return kept, unusable, asks


def asks_needed(
    spans_for_prompt: Sequence[dict[str, Any]],
    cache: Path,
    *,
    again: bool,
    afresh: bool = False,
    model: str = MODEL,
) -> int:
    """At most how many asks explaining this pair takes: none when its cached outcome stands."""
    path = cache_path(spans_for_prompt, cache, model)
    if not path.exists():
        return ATTEMPTS
    if not again:
        return 0
    plan = _due(json.loads(path.read_text()), grounding_spans(spans_for_prompt), afresh=afresh)
    return 0 if plan is None else plan.attempts


def explain(
    spans_for_prompt: Sequence[dict[str, Any]],
    *,
    ask: Ask | None,
    cache: Path = CACHE_DIR,
    model: str = MODEL,
    meter: Meter | None = None,
    again: bool = False,
    afresh: bool = False,
) -> Outcome:
    """The cached outcome for these spans, or, with `ask`, a new one from the model, each call
    metered by `meter`; with `again` too, a cache file from earlier grounding rules is asked
    once more, and with `afresh`, every such file is given a new round of its own."""
    spans = grounding_spans(spans_for_prompt)
    path = cache_path(spans_for_prompt, cache, model)
    earlier: list[dict[str, Any]] = []
    if path.exists():
        entry = json.loads(path.read_text())
        plan = _due(entry, spans, afresh=afresh) if again and ask is not None else None
        if plan is None or ask is None:
            return _outcome(entry, spans)
        before = ("explanation", "served_by", "grounded", "unusable_attempts", "attempts")
        earlier = [*entry.get("earlier", []), {k: entry[k] for k in before if k in entry}]
    elif ask is None:
        raise NotCachedError("no cached explanation for this pair; run with --live to ask")
    else:
        plan = _Round(ATTEMPTS, None, None)
    kept, unusable, asks = _ask_round(ask, user_message(spans_for_prompt), spans, plan, meter)
    new: dict[str, Any] = {
        "explanation": None if kept is None else kept[0].model_dump(),
        "served_by": None if kept is None else kept[1],
    }
    cache.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "grounding_rules": GROUNDING_RULES,
        "spans": [g.as_json() for g in spans],
        **new,
        "grounded": _grounded_record(None if kept is None else kept[0], spans),
        "attempts": asks,
        "unusable_attempts": unusable,
        **({"earlier": earlier} if earlier else {}),
    }
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return _outcome(new, spans)


def reground_cache(cache: Path = CACHE_DIR) -> int:
    """Rewrite each cache file's record of what grounding keeps under the current rules, with
    no call; a reply none of whose claims is now grounded becomes withheld, kept in `earlier`.
    The file's rules version is left as it was, so asking again still reaches it. Returns how
    many files changed."""
    changed = 0
    for path in sorted(cache.glob("*.json")) if cache.exists() else []:
        entry = json.loads(path.read_text())
        if entry["explanation"] is None:
            continue
        spans = [GroundingSpan.from_json(r) for r in entry["spans"]]
        reply = ChangeExplanation.model_validate(entry["explanation"])
        record = _grounded_record(reply, spans)
        if reply.claims and not ground(reply, spans).claims:
            before = {k: entry[k] for k in ("explanation", "served_by", "grounded")}
            entry["earlier"] = [*entry.get("earlier", []), before]
            entry |= {"explanation": None, "served_by": None, "grounded": None}
        elif record != entry["grounded"]:
            entry["grounded"] = record
        else:
            continue
        path.write_text(json.dumps(entry, indent=1, ensure_ascii=False) + "\n")
        changed += 1
    return changed


def check_cache(cache: Path = CACHE_DIR) -> list[str]:
    """Every problem with the committed explanations, grounded again from what each file
    records, with no corpus: a reply that no longer fits the schema, a kept claim that does not
    cite spans that exist and differ, a record of what was kept that grounding no longer gives,
    or a reply kept although none of its claims is grounded."""
    problems: list[str] = []
    for path in sorted(cache.glob("*.json")) if cache.exists() else []:
        name = path.name
        try:
            entry = json.loads(path.read_text())
            spans = [GroundingSpan.from_json(r) for r in entry["spans"]]
            if entry["explanation"] is None:
                if entry["grounded"] is not None:
                    problems.append(f"{name}: withheld, but records grounded claims")
                continue
            reply = ChangeExplanation.model_validate(entry["explanation"])
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as e:
            problems.append(f"{name}: not a cached explanation ({e})")
            continue
        if _grounded_record(reply, spans) != entry["grounded"]:
            problems.append(f"{name}: grounding now keeps a different set of claims")
        if reply.claims and not ground(reply, spans).claims:
            problems.append(f"{name}: kept, though none of its claims is grounded")
    return problems


def projected_cost(
    spans_for_prompts: Sequence[Sequence[dict[str, Any]]],
    asks: Sequence[int] | None = None,
    model: str = MODEL,
) -> float:
    """At most what explaining these pairs costs, each asked `asks` times (by default as often
    as it may be retried), every ask with room for a retry's reason."""
    counts = [ATTEMPTS] * len(spans_for_prompts) if asks is None else list(asks)
    return sum(
        upper_bound(model, [prompt_chars(s) + RETRY_CHARS], MAX_TOKENS, n)
        for s, n in zip(spans_for_prompts, counts, strict=True)
    )


def anthropic_ask(model: str = MODEL) -> Ask:
    """Ask through the Anthropic SDK for JSON fitting ChangeExplanation, at medium effort,
    every reply metered (lineage.structured). The key comes from the environment, as every
    secret here does."""
    import anthropic

    client = anthropic.Anthropic()

    def ask(system: str, user: str) -> Reply:
        got = ask_structured(
            client,
            ChangeExplanation,
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=MAX_TOKENS,
            effort="medium",
        )
        return Reply(got.value, got.served_by, got.input_tokens, got.output_tokens)

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
