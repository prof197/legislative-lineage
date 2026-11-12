"""The faithfulness eval: a model judges whether each kept claim describes the spans it cites,
one claim at a time, each cited span marked within the aligned text around it, as a reader
sees it; verdicts are cached without text and reported with a Wilson interval. The judge is a
scripted stand-in; nothing here calls the API."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pytest

from lineage.align import Span
from lineage.explain import UnusableReplyError, explain, prompt_spans
from lineage.explanations import ChangeExplanation, Claim
from lineage.faithfulness import (
    CONTEXT_CHARS,
    JUDGE_SYSTEM,
    JudgeAsk,
    PairView,
    Verdict,
    claims_to_judge,
    faithfulness,
    judge_claims,
    judge_message,
    judge_name,
    load_judgments,
    marked_passages,
    pair_view,
    pending,
    projected_cost,
    run_name,
    with_earlier_runs,
)
from lineage.spend import Meter, load_runs
from tests.test_explain import scripted

ORIGIN = "Records are kept. The fee is set for each fiscal year."
DESCENDANT = "Records are kept. The fee is set every biennium. Appeals go to the board."


def _at(text: str, part: str) -> tuple[int, int]:
    start = text.index(part)
    return (start, start + len(part))


SPANS = [
    Span("s0", "copied", _at(ORIGIN, "Records are kept."), _at(DESCENDANT, "Records are kept.")),
    Span(
        "s1",
        "paraphrased",
        _at(ORIGIN, "The fee is set for each fiscal year."),
        _at(DESCENDANT, "The fee is set every biennium."),
    ),
    Span("s2", "unmatched", None, _at(DESCENDANT, "Appeals go to the board.")),
]
PROMPT = prompt_spans(SPANS, ORIGIN, DESCENDANT, origin_side="a")
VIEW = pair_view(SPANS, ORIGIN, DESCENDANT, origin_side="a")


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


def test_only_kept_claims_are_judged_each_cited_span_marked_in_its_aligned_text(
    tmp_path: Path,
) -> None:
    cache = _cached(tmp_path, ADDED, COPIED, CHANGED)

    claims, counts = claims_to_judge([PROMPT], [VIEW], cache=cache)

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
    assert shown["spans"] == [
        {"id": "s1", "kind": "paraphrased", "cited_on": ["origin", "descendant"]}
    ]
    # The text around a cited span is shown, as the comparison panel shows it, but only the
    # cited text is marked.
    assert shown["origin"] == ["Records are kept. ⟦s1⟧The fee is set for each fiscal year.⟦/s1⟧"]
    assert shown["descendant"] == [
        "Records are kept. ⟦s1⟧The fee is set every biennium.⟦/s1⟧ Appeals go to the board."
    ]
    added = json.loads(judge_message(claims[0]))
    assert added["origin"] == []  # the claim cites nothing in the origin
    assert added["descendant"] == [
        "Records are kept. The fee is set every biennium. ⟦s2⟧Appeals go to the board.⟦/s2⟧"
    ]


WORDS = " ".join(f"w{i:03d}" for i in range(400))  # 400 words of four characters, 1,999 in all


def _word_at(n: int) -> tuple[int, int]:
    return _at(WORDS, f"w{n:03d}")


def test_a_cited_span_is_shown_in_a_fixed_window_cut_at_whole_words() -> None:
    [passage] = marked_passages(WORDS, [("s7", _word_at(200))])

    before, rest = passage.split("⟦s7⟧")
    marked, after = rest.split("⟦/s7⟧")
    assert marked == "w200"
    assert before.startswith("…")
    assert after.endswith("…")
    # Whole words only, each side at most the window and within a word of it.
    assert before[1:].split(" ")[0].startswith("w")
    assert len(before[1:].split(" ")[0]) == 4
    assert CONTEXT_CHARS - 5 <= len(before) - 1 <= CONTEXT_CHARS
    assert CONTEXT_CHARS - 5 <= len(after) - 1 <= CONTEXT_CHARS


def test_cited_spans_whose_windows_meet_share_one_passage_and_far_ones_do_not() -> None:
    near = marked_passages(WORDS, [("s1", _word_at(100)), ("s3", _word_at(110))])
    far = marked_passages(WORDS, [("s1", _word_at(10)), ("s3", _word_at(390))])

    assert len(near) == 1
    assert "⟦s1⟧w100⟦/s1⟧" in near[0]
    assert "⟦s3⟧w110⟦/s3⟧" in near[0]
    assert len(far) == 2
    assert far[0].startswith("w000")
    assert far[1].endswith("w399")
    assert "w200" not in "".join(far)


def test_the_judge_is_told_a_claim_must_rest_on_the_marked_text() -> None:
    assert judge_name().endswith("@judge-v2")
    assert "unmarked" in JUDGE_SYSTEM
    assert f"{CONTEXT_CHARS}" not in JUDGE_SYSTEM  # the window is the code's, not the judge's


def test_a_view_from_another_alignment_is_refused(tmp_path: Path) -> None:
    other = PairView(ORIGIN, DESCENDANT, {"s0": (None, None)})

    with pytest.raises(ValueError, match="s1"):
        claims_to_judge([PROMPT], [other], cache=_cached(tmp_path, CHANGED))


def test_verdicts_are_cached_without_text_and_a_rerun_asks_nothing(tmp_path: Path) -> None:
    claims, _ = claims_to_judge([PROMPT], [VIEW], cache=_cached(tmp_path, ADDED, CHANGED))
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
    claims, counts = claims_to_judge([PROMPT], [VIEW], cache=_cached(tmp_path, ADDED, CHANGED))
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
    claims, counts = claims_to_judge([PROMPT], [VIEW], cache=_cached(tmp_path, ADDED))
    path = tmp_path / "judgments.jsonl"
    judge_claims(claims, ask=_judge(True)[1], path=path, meter=None, today="2026-11-12")
    reworded = ADDED.model_copy(update={"summary": "Adds an appeal."})
    other, _ = claims_to_judge([PROMPT], [VIEW], cache=_cached(tmp_path / "b", reworded))

    assert pending(other, load_judgments(path), judge_name()) == other
    assert faithfulness(other, load_judgments(path), counts)["overall"]["not_judged"] == 1


def test_the_judge_is_projected_at_every_retry_at_its_reply_limit(tmp_path: Path) -> None:
    claims, _ = claims_to_judge([PROMPT], [VIEW], cache=_cached(tmp_path, ADDED, CHANGED))

    assert projected_cost(claims) == pytest.approx(
        projected_cost(claims[:1]) + projected_cost(claims[1:])
    )
    assert projected_cost(claims[:1]) > 2 * 1024 * 20 / 1e6


def test_a_run_under_new_grounding_rules_keeps_the_earlier_result_beside_it() -> None:
    first: dict[str, Any] = {
        "judge": "j",
        "pairs": "p",
        "pairs_explained": 137,
        "grounding": {"claims_kept": 319},
        "overall": {"share": 0.8},
        "by_kind": {"added": {"share": 0.88}},
        "hard_check": {"problems": []},
        "enactments": "not kept",
    }

    second = with_earlier_runs({"overall": {"share": 0.9}}, first, run=run_name(2))
    rerun = with_earlier_runs({"overall": {"share": 0.91}}, second, run=run_name(2))

    assert run_name(2) == "grounding-rules-2"
    [earlier] = second["earlier_runs"]
    assert earlier["run"] == "grounding-rules-1"  # a section from before runs were named
    assert earlier["overall"] == {"share": 0.8}
    assert "enactments" not in earlier
    assert (rerun["run"], rerun["overall"], rerun["earlier_runs"]) == (
        "grounding-rules-2",
        {"share": 0.91},
        second["earlier_runs"],
    )
    assert with_earlier_runs({}, None, run=run_name(2))["earlier_runs"] == []
