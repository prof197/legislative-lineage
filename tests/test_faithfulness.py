"""The faithfulness eval: a model judges whether each kept claim describes the spans it cites,
one claim at a time, on the cited text alone; verdicts are cached without text and reported
with a Wilson interval. The judge is a scripted stand-in; nothing here calls the API."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest

from lineage.explain import UnusableReplyError, explain
from lineage.explanations import ChangeExplanation, Claim
from lineage.faithfulness import (
    JudgeAsk,
    Verdict,
    claims_to_judge,
    faithfulness,
    judge_claims,
    judge_message,
    judge_name,
    load_judgments,
    pending,
    projected_cost,
)
from lineage.spend import Meter, load_runs
from tests.test_explain import scripted

PROMPT = [
    {
        "id": "s0",
        "kind": "copied",
        "origin": "Records are kept.",
        "descendant": "Records are kept.",
    },
    {
        "id": "s1",
        "kind": "paraphrased",
        "origin": "The fee is set for each fiscal year.",
        "descendant": "The fee is set every biennium.",
    },
    {"id": "s2", "kind": "unmatched", "origin": None, "descendant": "Appeals go to the board."},
]


def _claim(kind: Literal["added", "removed", "changed"], *, o: list[str], d: list[str]) -> Claim:
    return Claim(kind=kind, summary=f"A {kind} claim.", origin_span_ids=o, descendant_span_ids=d)


ADDED = _claim("added", o=[], d=["s2"])
CHANGED = _claim("changed", o=["s1"], d=["s1"])
COPIED = _claim("changed", o=["s0"], d=["s0"])  # dropped by grounding


def _cached(tmp_path: Path, *claims: Claim) -> Path:
    cache = tmp_path / "cache"
    reply = ChangeExplanation(claims=list(claims))
    _, ask = scripted(reply, reply)  # a failing claim is asked about once more
    explain(PROMPT, ask=ask, cache=cache)
    return cache


def _judge(*answers: bool | Exception) -> tuple[list[str], JudgeAsk]:
    calls: list[str] = []
    queue: Iterator[bool | Exception] = iter(answers)

    def ask(system: str, user: str) -> Verdict:
        calls.append(user)
        nxt = next(queue)
        if isinstance(nxt, Exception):
            raise nxt
        return Verdict(nxt, "claude-opus-5-5", 800, 30)

    return calls, ask


def test_only_kept_claims_are_judged_with_the_text_of_the_spans_they_cite(tmp_path: Path) -> None:
    cache = _cached(tmp_path, ADDED, COPIED, CHANGED)

    claims, counts = claims_to_judge([PROMPT], cache=cache)

    assert [c.claim for c in claims] == [ADDED, CHANGED]
    assert counts == {
        "explanations": 1,
        "withheld": 0,
        "with_no_claims": 0,
        "claims_kept": 2,
        "claims_dropped": {"is an exact copy": 1},
    }
    shown = json.loads(judge_message(claims[1]))
    assert shown["claim"] == {"kind": "changed", "summary": "A changed claim."}
    assert shown["spans"] == [{**PROMPT[1], "cited_on": ["origin", "descendant"]}]
    assert "Records are kept" not in judge_message(claims[0])


def test_verdicts_are_cached_without_text_and_a_rerun_asks_nothing(tmp_path: Path) -> None:
    claims, _ = claims_to_judge([PROMPT], cache=_cached(tmp_path, ADDED, CHANGED))
    path = tmp_path / "judgments.jsonl"
    spend = tmp_path / "spend.json"
    calls, ask = _judge(True, False)

    meter = Meter(spend, "judge", "claude-opus-5-5", "judge-v1", "2026-11-12")
    judged = judge_claims(claims, ask=ask, path=path, meter=meter, today="2026-11-12")
    again = judge_claims(claims, ask=_judge()[1], path=path, meter=None, today="2026-11-12")

    assert (judged, again, len(calls)) == (2, 0, 2)
    assert "Appeals" not in path.read_text()
    assert [r["describes_spans_correctly"] for r in load_judgments(path)] == [True, False]
    assert load_runs(spend)[0]["calls"] == 2


def test_an_unusable_verdict_is_asked_once_more_then_counted_apart(tmp_path: Path) -> None:
    claims, counts = claims_to_judge([PROMPT], cache=_cached(tmp_path, ADDED, CHANGED))
    path = tmp_path / "judgments.jsonl"
    calls, ask = _judge(
        UnusableReplyError("refused"), True, UnusableReplyError("x"), UnusableReplyError("y")
    )

    judge_claims(claims, ask=ask, path=path, meter=None, today="2026-11-12")
    got = faithfulness(claims, load_judgments(path), counts)

    assert len(calls) == 4
    assert got["judge"] == judge_name()
    assert {k: got["overall"][k] for k in ("claims", "judged", "unusable", "correct")} == {
        "claims": 2,
        "judged": 1,
        "unusable": 1,
        "correct": 1,
    }
    assert got["by_kind"]["added"]["share"] == 1.0
    assert got["by_kind"]["changed"]["judged"] == 0


def test_a_regenerated_claim_is_never_scored_by_an_old_verdict(tmp_path: Path) -> None:
    claims, counts = claims_to_judge([PROMPT], cache=_cached(tmp_path, ADDED))
    path = tmp_path / "judgments.jsonl"
    judge_claims(claims, ask=_judge(True)[1], path=path, meter=None, today="2026-11-12")
    reworded = ADDED.model_copy(update={"summary": "Adds an appeal."})
    other, _ = claims_to_judge([PROMPT], cache=_cached(tmp_path / "b", reworded))

    assert pending(other, load_judgments(path), judge_name()) == other
    assert faithfulness(other, load_judgments(path), counts)["overall"]["not_judged"] == 1


def test_the_judge_is_projected_at_every_retry_at_its_reply_limit(tmp_path: Path) -> None:
    claims, _ = claims_to_judge([PROMPT], cache=_cached(tmp_path, ADDED, CHANGED))

    assert projected_cost(claims) == pytest.approx(
        projected_cost(claims[:1]) + projected_cost(claims[1:])
    )
    assert projected_cost(claims[:1]) > 2 * 1024 * 20 / 1e6
