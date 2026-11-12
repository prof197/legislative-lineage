"""The faithfulness eval: does each kept claim describe the spans it cites correctly?

Grounding in code (lineage.explanations) is the hard check: every kept claim cites spans that
exist and differ between the two texts, and CI checks it on every committed explanation. What
code cannot check is whether the claim's words are true of those spans. A model judges that,
one claim at a time, seeing what a reader of the comparison panel sees: each span the claim
cites, on the side it cites it, marked within the aligned text around it, CONTEXT_CHARS
characters either way cut at whole words. The claim must describe the marked text; the rest is
there only to read it by, and a claim resting on unmarked text is incorrect. The judge answers
yes or no through structured output. It is reported as model-judged, as precision is; no
person's labels check it (the M5 review's ruling). The first judge (judge-v1) saw the cited
spans' text alone, which judged a claim on less than a reader sees (the M6 faithfulness
ruling).

A verdict is keyed by the explanation's cache key, the claim's position among its kept claims,
and the SHA-256 of the claim with the grounding form of the spans it cites, so a regenerated
explanation is never counted against an old verdict. Verdicts are appended to a committed file
that holds no text, one per claim and judge prompt, so a rerun asks only for claims not yet
judged. A reply that is declined or off the schema is asked again once, then recorded as
unusable and counted apart. Every call is metered against M6's budget (lineage.spend). See
DESIGN.md#explanations.
"""

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from lineage.align import Span
from lineage.explain import CACHE_DIR, explain, explanation_key
from lineage.explanations import Claim, Side, grounding_spans
from lineage.intervals import wilson
from lineage.spend import Meter, upper_bound
from lineage.structured import UnusableReplyError, ask_structured

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_PROMPT = "judge-v2"
# Characters of aligned text shown either side of a cited span, cut back to whole words: about
# fifty words, enough to read a clause by, and few enough that the marked text stays the point.
CONTEXT_CHARS = 300
JUDGMENTS = Path("evals/faithfulness_judgments.jsonl")
ATTEMPTS = 2
MAX_TOKENS = 1024

JUDGE_SYSTEM = """\
You check one claim that a summarizer made about how a descendant section of state legislation \
differs from its origin. You are given, as JSON, the claim, the kind of each aligned span it \
cites ("paraphrased": the same thing said in other words; "unmatched": text on one side only) \
and the sides it cites it on, and passages of the origin's text and the descendant's around \
the cited spans. In each passage a cited span's text is marked, as ⟦s3⟧this⟦/s3⟧. A text the \
claim cites nothing in has no passages.

The claim describes its spans correctly when both hold. Its kind fits the marked text: "added" \
is text the descendant has and the origin lacks, "removed" text the origin has and the \
descendant lacks, "changed" a difference between marked text on both sides, the descendant \
saying otherwise what the origin says. And everything its summary asserts is borne out by the \
marked text: nothing misstated, reversed or exaggerated, and nothing asserted that the marked \
text does not show. The unmarked text is there only to read the marked text by. A claim that \
rests on unmarked text, asserting what only the unmarked text shows, does not describe its \
spans correctly. "Origin" and "descendant" only name the two texts: nothing is known about \
which came first or who drew on whom. A claim that says or implies a state, legislature or \
drafter changed, copied or adopted anything, or which text came first, does not describe its \
spans correctly. Judge only from what is given. Answer true or false."""


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    describes_spans_correctly: bool


@dataclass(frozen=True, slots=True)
class Verdict:
    correct: bool | None  # None when no usable reply came after one retry
    served_by: str | None
    input_tokens: int = 0
    output_tokens: int = 0


JudgeAsk = Callable[[str, str], Verdict]  # (system, user) -> verdict; raises UnusableReplyError


Offsets = tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class PairView:
    """What a reader sees of a pair: its two texts and where each span lies in them."""

    origin: str
    descendant: str
    at: dict[str, tuple[Offsets, Offsets]]  # span id -> its offsets in the origin, descendant


def pair_view(spans: Sequence[Span], a: str, b: str, *, origin_side: Side) -> PairView:
    """The pair's texts by role, as lineage.explain.prompt_spans gives its spans."""
    if origin_side == "a":
        return PairView(a, b, {s.id: (s.a, s.b) for s in spans})
    return PairView(b, a, {s.id: (s.b, s.a) for s in spans})


def marked_passages(
    text: str, cited: Sequence[tuple[str, tuple[int, int]]], width: int = CONTEXT_CHARS
) -> list[str]:
    """The passages of `text` around the cited spans, each span marked ⟦id⟧…⟦/id⟧ within
    `width` characters of text either way, cut back to whole words; spans whose windows meet
    share a passage."""
    windows: list[list[Any]] = []  # [start, end, spans inside]
    for span_id, (lo, hi) in sorted(cited, key=lambda c: c[1]):
        start, end = max(0, lo - width), min(len(text), hi + width)
        if start > 0 and not text[start - 1].isspace():
            while start < lo and not text[start].isspace():
                start += 1
        if end < len(text) and not text[end].isspace():
            while end > hi and not text[end - 1].isspace():
                end -= 1
        while start < lo and text[start].isspace():
            start += 1
        while end > hi and text[end - 1].isspace():
            end -= 1
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], end)
            windows[-1][2].append((span_id, lo, hi))
        else:
            windows.append([start, end, [(span_id, lo, hi)]])
    out: list[str] = []
    for start, end, inside in windows:
        parts = ["…" if start > 0 else ""]
        at = start
        for span_id, lo, hi in inside:
            parts += [text[at:lo], f"⟦{span_id}⟧", text[lo:hi], f"⟦/{span_id}⟧"]
            at = hi
        parts += [text[at:end], "…" if end < len(text) else ""]
        out.append("".join(parts))
    return out


@dataclass(frozen=True, slots=True)
class ClaimToJudge:
    key: str  # the explanation's cache key
    index: int  # the claim's position among the explanation's kept claims
    claim: Claim
    cited: list[dict[str, Any]]  # the prompt's spans the claim cites, with their text
    claim_sha256: str
    passages: dict[str, list[str]]  # by side, the aligned text around the cited spans, marked


def _passages(claim: Claim, view: PairView) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for side, text, ids in (
        ("origin", view.origin, claim.origin_span_ids),
        ("descendant", view.descendant, claim.descendant_span_ids),
    ):
        cited: list[tuple[str, tuple[int, int]]] = []
        for span_id in dict.fromkeys(ids):
            if span_id not in view.at:
                raise ValueError(f"{span_id} is not a span of the pair's view")
            at = view.at[span_id][0 if side == "origin" else 1]
            if at is None:
                raise ValueError(f"{span_id} has no {side} text in the pair's view")
            cited.append((span_id, at))
        out[side] = marked_passages(text, cited)
    return out


def claims_to_judge(
    prompts: Sequence[Sequence[dict[str, Any]]],
    views: Sequence[PairView],
    *,
    cache: Path = CACHE_DIR,
) -> tuple[list[ClaimToJudge], dict[str, Any]]:
    """Every kept claim of the pairs' cached explanations, with the spans it cites and the
    passages a reader sees them in; and what grounding did across them: explanations, withheld
    ones, claims kept and dropped by reason."""
    out: list[ClaimToJudge] = []
    counts: dict[str, Any] = {
        "explanations": 0,
        "withheld": 0,
        "with_no_claims": 0,
        "claims_kept": 0,
        "claims_dropped": {},
    }
    for for_prompt, view in zip(prompts, views, strict=True):
        outcome = explain(for_prompt, ask=None, cache=cache)
        counts["explanations"] += 1
        counts["withheld"] += outcome.withheld
        counts["with_no_claims"] += not outcome.withheld and not outcome.claims
        counts["claims_kept"] += len(outcome.claims)
        for _, why in outcome.dropped:
            reason = re.sub(r"^s\d+ ", "", why)  # the reason, without the span it names
            counts["claims_dropped"][reason] = counts["claims_dropped"].get(reason, 0) + 1
        by_id = {str(s["id"]): s for s in for_prompt}
        hashed = {g.id: g.as_json() for g in grounding_spans(for_prompt)}
        key = explanation_key(for_prompt)
        for index, claim in enumerate(outcome.claims):
            ids = list(dict.fromkeys([*claim.origin_span_ids, *claim.descendant_span_ids]))
            doc = {"claim": claim.model_dump(), "spans": [hashed[i] for i in ids]}
            digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
            out.append(
                ClaimToJudge(
                    key, index, claim, [by_id[i] for i in ids], digest, _passages(claim, view)
                )
            )
    counts["claims_dropped"] = dict(sorted(counts["claims_dropped"].items()))
    return out, counts


def judge_message(c: ClaimToJudge) -> str:
    origin = set(c.claim.origin_span_ids)
    descendant = set(c.claim.descendant_span_ids)
    spans = [
        {
            "id": s["id"],
            "kind": s["kind"],
            "cited_on": [
                side
                for side, ids in (("origin", origin), ("descendant", descendant))
                if s["id"] in ids
            ],
        }
        for s in c.cited
    ]
    claim = {"kind": c.claim.kind, "summary": c.claim.summary}
    doc = {"claim": claim, "spans": spans, **c.passages}
    return json.dumps(doc, ensure_ascii=False, indent=1)


def judge_name(model: str = JUDGE_MODEL, prompt: str = JUDGE_PROMPT) -> str:
    return f"model:{model}@{prompt}"


def _row_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (str(row["key"]), int(row["index"]), str(row["claim_sha256"]), str(row["judge"]))


def _claim_key(c: ClaimToJudge, judge: str) -> tuple[str, int, str, str]:
    return (c.key, c.index, c.claim_sha256, judge)


def load_judgments(path: Path = JUDGMENTS) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pending(
    claims: Sequence[ClaimToJudge], rows: Sequence[dict[str, Any]], judge: str
) -> list[ClaimToJudge]:
    done = {_row_key(r) for r in rows}
    return [c for c in claims if _claim_key(c, judge) not in done]


def projected_cost(claims: Sequence[ClaimToJudge], model: str = JUDGE_MODEL) -> float:
    chars = [len(JUDGE_SYSTEM) + len(judge_message(c)) for c in claims]
    return upper_bound(model, chars, MAX_TOKENS, ATTEMPTS)


def judge_claims(
    claims: Sequence[ClaimToJudge],
    *,
    ask: JudgeAsk,
    path: Path = JUDGMENTS,
    meter: Meter | None,
    today: str,
    judge: str = judge_name(),
) -> int:
    """Judge each claim not yet judged, appending each verdict as it comes. Returns how many
    were judged."""
    done = 0
    for c in pending(claims, load_judgments(path), judge):
        user = judge_message(c)
        verdict = Verdict(None, None)
        for _ in range(ATTEMPTS):
            if meter is not None:
                meter.before_call(len(JUDGE_SYSTEM) + len(user), MAX_TOKENS)
            try:
                verdict = ask(JUDGE_SYSTEM, user)
            except UnusableReplyError as e:
                if meter is not None:
                    meter.add(e.input_tokens, e.output_tokens, e.served_by)
                continue
            if meter is not None:
                meter.add(verdict.input_tokens, verdict.output_tokens, verdict.served_by)
            break
        row = {
            "key": c.key,
            "index": c.index,
            "claim_sha256": c.claim_sha256,
            "judge": judge,
            "describes_spans_correctly": verdict.correct,
            "served_by": verdict.served_by,
            "at": today,
        }
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done += 1
    return done


def faithfulness(
    claims: Sequence[ClaimToJudge],
    rows: Sequence[dict[str, Any]],
    counts: dict[str, Any],
    *,
    judge: str = judge_name(),
) -> dict[str, Any]:
    """The share of kept claims the judge found describe their spans correctly, with a 95%
    Wilson interval, overall and by kind; and what grounding did beside it."""
    latest = {_row_key(r): r for r in rows if r["judge"] == judge}
    verdicts = [(c, latest.get(_claim_key(c, judge))) for c in claims]

    def share(of: Sequence[tuple[ClaimToJudge, dict[str, Any] | None]]) -> dict[str, Any]:
        judged = [r for _, r in of if r is not None and r["describes_spans_correctly"] is not None]
        correct = sum(bool(r["describes_spans_correctly"]) for r in judged)
        lo, hi = wilson(correct, len(judged))
        return {
            "claims": len(of),
            "judged": len(judged),
            "unusable": sum(
                1 for _, r in of if r is not None and r["describes_spans_correctly"] is None
            ),
            "not_judged": sum(1 for _, r in of if r is None),
            "correct": correct,
            "share": round(correct / len(judged), 4) if judged else None,
            "interval_95": [round(lo, 4), round(hi, 4)],
        }

    kinds = sorted({c.claim.kind for c in claims})
    return {
        "judge": judge,
        "judge_context_chars": CONTEXT_CHARS,
        "question": "does the claim describe the spans it cites correctly?",
        "grounding": counts,
        "overall": share(verdicts),
        "by_kind": {k: share([v for v in verdicts if v[0].claim.kind == k]) for k in kinds},
    }


def run_name(grounding_rules: int) -> str:
    """A run is named by the grounding rules its explanations were kept under."""
    return f"grounding-rules-{grounding_rules}"


_SUMMARY = (
    "judge",
    "judge_context_chars",
    "pairs",
    "pairs_explained",
    "grounding",
    "overall",
    "by_kind",
    "hard_check",
)


def with_earlier_runs(
    section: dict[str, Any], previous: dict[str, Any] | None, *, run: str
) -> dict[str, Any]:
    """This run's results section; a previous run under other rules is kept beside it,
    summarized, and a previous run of the same name is replaced."""
    if previous is None:
        return section | {"run": run, "earlier_runs": []}
    earlier: list[dict[str, Any]] = list(previous.get("earlier_runs", []))
    before = str(previous.get("run", run_name(1)))  # the first run predates run names
    if before != run:
        earlier.append({"run": before} | {k: previous[k] for k in _SUMMARY if k in previous})
    return section | {"run": run, "earlier_runs": earlier}


def anthropic_judge(model: str = JUDGE_MODEL) -> JudgeAsk:
    """Ask through the Anthropic SDK for JSON fitting ClaimVerdict, at medium effort, every
    reply metered (lineage.structured)."""
    import anthropic

    client = anthropic.Anthropic()

    def ask(system: str, user: str) -> Verdict:
        got = ask_structured(
            client,
            ClaimVerdict,
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=MAX_TOKENS,
            effort="medium",
        )
        return Verdict(
            got.value.describes_spans_correctly, got.served_by, got.input_tokens, got.output_tokens
        )

    return ask
