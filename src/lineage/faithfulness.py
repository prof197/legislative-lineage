"""The faithfulness eval: does each kept claim describe the spans it cites correctly?

Grounding in code (lineage.explanations) is the hard check: every kept claim cites spans that
exist and differ between the two texts, and CI checks it on every committed explanation. What
code cannot check is whether the claim's words are true of those spans. A model judges that,
one claim at a time: it sees the claim's kind and summary and the text of each span it cites on
the side it cites it, and answers yes or no through structured output. Nothing else is shown, so
the verdict is about the cited evidence alone. It is reported as model-judged, as precision is;
no person's labels check it (the M5 review's ruling).

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

from pydantic import BaseModel, ConfigDict, ValidationError

from lineage.explain import CACHE_DIR, UnusableReplyError, explain, explanation_key
from lineage.explanations import Claim, grounding_spans
from lineage.intervals import wilson
from lineage.spend import Meter, upper_bound

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_PROMPT = "judge-v1"
JUDGMENTS = Path("evals/faithfulness_judgments.jsonl")
ATTEMPTS = 2
MAX_TOKENS = 1024
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

JUDGE_SYSTEM = """\
You check one claim that a summarizer made about how a descendant section of state legislation \
differs from its origin. You are given the claim and the aligned spans it cites, as JSON: each \
span's kind ("paraphrased": the same thing said in other words; "unmatched": text on one side \
only) and its text on the origin's side and the descendant's.

The claim describes its spans correctly when both hold. Its kind fits them: "added" is text the \
descendant has and the origin lacks, "removed" text the origin has and the descendant lacks, \
"changed" text both have that differs. And everything its summary asserts is borne out by the \
cited text: nothing misstated, reversed or exaggerated, and nothing asserted that the spans do \
not show. "Origin" and "descendant" only name the two texts: nothing is known about which came \
first or who drew on whom. A claim that says or implies a state, legislature or drafter \
changed, copied or adopted anything, or which text came first, does not describe its spans \
correctly. Judge only from the spans given. Answer true or false."""


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


@dataclass(frozen=True, slots=True)
class ClaimToJudge:
    key: str  # the explanation's cache key
    index: int  # the claim's position among the explanation's kept claims
    claim: Claim
    cited: list[dict[str, Any]]  # the prompt's spans the claim cites, with their text
    claim_sha256: str


def claims_to_judge(
    prompts: Sequence[Sequence[dict[str, Any]]], *, cache: Path = CACHE_DIR
) -> tuple[list[ClaimToJudge], dict[str, Any]]:
    """Every kept claim of the pairs' cached explanations, with the spans it cites; and what
    grounding did across them: explanations, withheld ones, claims kept and dropped by reason."""
    out: list[ClaimToJudge] = []
    counts: dict[str, Any] = {
        "explanations": 0,
        "withheld": 0,
        "with_no_claims": 0,
        "claims_kept": 0,
        "claims_dropped": {},
    }
    for for_prompt in prompts:
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
            out.append(ClaimToJudge(key, index, claim, [by_id[i] for i in ids], digest))
    counts["claims_dropped"] = dict(sorted(counts["claims_dropped"].items()))
    return out, counts


def judge_message(c: ClaimToJudge) -> str:
    origin = set(c.claim.origin_span_ids)
    descendant = set(c.claim.descendant_span_ids)
    spans = [
        {
            "id": s["id"],
            "kind": s["kind"],
            "origin": s["origin"],
            "descendant": s["descendant"],
            "cited_on": [
                side
                for side, ids in (("origin", origin), ("descendant", descendant))
                if s["id"] in ids
            ],
        }
        for s in c.cited
    ]
    claim = {"kind": c.claim.kind, "summary": c.claim.summary}
    return json.dumps({"claim": claim, "spans": spans}, ensure_ascii=False, indent=1)


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
        "question": "does the claim describe the spans it cites correctly?",
        "grounding": counts,
        "overall": share(verdicts),
        "by_kind": {k: share([v for v in verdicts if v[0].claim.kind == k]) for k in kinds},
    }


def anthropic_judge(model: str = JUDGE_MODEL) -> JudgeAsk:
    """Ask through the Anthropic SDK with structured output from ClaimVerdict, at medium
    effort, with the server-side fallback for a declined request."""
    import anthropic

    client = anthropic.Anthropic()

    def ask(system: str, user: str) -> Verdict:
        try:
            response = client.beta.messages.parse(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=ClaimVerdict,
                output_config={"effort": "medium"},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except ValidationError as e:
            raise UnusableReplyError(f"the reply does not fit the schema: {e}") from e
        tokens = (response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise UnusableReplyError(
                f"no usable reply (stop reason {response.stop_reason})", *tokens, response.model
            )
        return Verdict(response.parsed_output.describes_spans_correctly, response.model, *tokens)

    return ask
