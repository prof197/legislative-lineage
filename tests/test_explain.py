"""Explaining a pair: the prompt is built from aligned spans, replies are cached by model,
prompt version and spans, and an unusable reply is retried once, then withheld. The model is
replaced by a scripted stand-in; nothing here calls the API."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
from typer.testing import CliRunner

from lineage import queries
from lineage.align import Span, align
from lineage.cli import app
from lineage.explain import (
    MODEL,
    PROMPT_VERSION,
    Ask,
    NotCachedError,
    Outcome,
    Reply,
    UnusableReplyError,
    cache_key,
    explain,
    load_pairs,
    pair_texts,
    prompt_spans,
)
from lineage.explanations import ChangeExplanation, Claim

ORIGIN = "The fee is set for each fiscal year. Records are kept."
DESCENDANT = "The fee is set every biennium. Records are kept. Appeals go to the board."


def _at(text: str, part: str) -> tuple[int, int]:
    start = text.index(part)
    return (start, start + len(part))


SPANS = [
    Span(
        "s0",
        "paraphrased",
        _at(ORIGIN, "The fee is set for each fiscal year."),
        _at(DESCENDANT, "The fee is set every biennium."),
    ),
    Span("s1", "copied", _at(ORIGIN, "Records are kept."), _at(DESCENDANT, "Records are kept.")),
    Span("s2", "unmatched", None, _at(DESCENDANT, "Appeals go to the board.")),
]


def _claim(span: str, kind: Literal["added", "removed", "changed"] = "added") -> Claim:
    return Claim(
        kind=kind,
        summary="Adds an appeal to the board.",
        origin_span_ids=[],
        descendant_span_ids=[span],
    )


def _scripted(*replies: ChangeExplanation | Exception) -> tuple[list[tuple[str, str]], Ask]:
    calls: list[tuple[str, str]] = []
    queue: Iterator[ChangeExplanation | Exception] = iter(replies)

    def ask(system: str, user: str) -> Reply:
        calls.append((system, user))
        nxt = next(queue)
        if isinstance(nxt, Exception):
            raise nxt
        return Reply(nxt, "claude-opus-5-5")

    return calls, ask


def _explain(ask: Ask | None, cache: Path) -> Outcome:
    return explain(
        prompt_spans(SPANS, ORIGIN, DESCENDANT, origin_side="a"),
        SPANS,
        origin_side="a",
        ask=ask,
        cache=cache,
    )


def test_the_prompt_carries_the_spans_texts_by_side_and_never_the_whole_bills() -> None:
    spans = prompt_spans(SPANS, ORIGIN, DESCENDANT, origin_side="a")

    assert spans[0] == {
        "id": "s0",
        "kind": "paraphrased",
        "origin": "The fee is set for each fiscal year.",
        "descendant": "The fee is set every biennium.",
    }
    assert spans[2] == {
        "id": "s2",
        "kind": "unmatched",
        "origin": None,
        "descendant": "Appeals go to the board.",
    }
    flipped = prompt_spans(SPANS, ORIGIN, DESCENDANT, origin_side="b")
    assert flipped[2]["origin"] == "Appeals go to the board."


def test_the_cache_key_changes_with_the_model_the_prompt_or_any_span() -> None:
    spans = prompt_spans(SPANS, ORIGIN, DESCENDANT, origin_side="a")
    key = cache_key(MODEL, PROMPT_VERSION, spans)

    assert key == cache_key(MODEL, PROMPT_VERSION, [dict(s) for s in spans])
    assert key != cache_key("another-model", PROMPT_VERSION, spans)
    assert key != cache_key(MODEL, PROMPT_VERSION + "x", spans)
    assert key != cache_key(MODEL, PROMPT_VERSION, [*spans[:2], {**spans[2], "descendant": "x"}])


def test_a_grounded_reply_is_kept_and_cached_so_a_rerun_makes_no_call(tmp_path: Path) -> None:
    calls, ask = _scripted(ChangeExplanation(claims=[_claim("s2")]))

    first = _explain(ask, tmp_path)
    again = _explain(None, tmp_path)

    assert len(calls) == 1
    assert "Appeals go to the board." in calls[0][1]
    assert first.claims == again.claims == [_claim("s2")]
    assert not first.withheld
    [entry] = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert (entry["model"], entry["prompt_version"], entry["served_by"]) == (
        MODEL,
        PROMPT_VERSION,
        "claude-opus-5-5",
    )


def test_failing_claims_are_dropped_and_the_rest_kept_without_a_retry(tmp_path: Path) -> None:
    calls, ask = _scripted(ChangeExplanation(claims=[_claim("s2"), _claim("s1", "changed")]))

    outcome = _explain(ask, tmp_path)

    assert len(calls) == 1
    assert outcome.claims == [_claim("s2")]
    assert [why for _, why in outcome.dropped] == ["s1 is an exact copy"]


def test_an_unusable_reply_is_retried_once(tmp_path: Path) -> None:
    calls, ask = _scripted(
        ChangeExplanation(claims=[_claim("s1", "changed")]),
        ChangeExplanation(claims=[_claim("s2")]),
    )

    outcome = _explain(ask, tmp_path)

    assert len(calls) == 2
    assert outcome.claims == [_claim("s2")]


def test_after_one_retry_the_explanation_is_withheld_and_that_is_cached(tmp_path: Path) -> None:
    calls, ask = _scripted(UnusableReplyError("refused"), ChangeExplanation(claims=[_claim("s9")]))

    outcome = _explain(ask, tmp_path)
    again = _explain(None, tmp_path)

    assert len(calls) == 2
    assert outcome.withheld
    assert again.withheld
    assert outcome.claims == []


def test_without_a_cached_reply_or_a_live_model_nothing_is_made_up(tmp_path: Path) -> None:
    with pytest.raises(NotCachedError, match="--live"):
        _explain(None, tmp_path)


def test_no_claims_means_nothing_changed_and_is_not_retried(tmp_path: Path) -> None:
    calls, ask = _scripted(ChangeExplanation(claims=[]))

    outcome = _explain(ask, tmp_path)

    assert len(calls) == 1
    assert (outcome.claims, outcome.withheld) == ([], False)


def test_the_command_replays_the_cache_and_asks_only_with_live(
    corpus: Path, tmp_path: Path
) -> None:
    with queries.connect_quiet() as con:
        rows: list[tuple[str, int]] = con.execute(
            "SELECT bill_id, ordinal FROM read_parquet($p) ORDER BY bill_id, ordinal LIMIT 2",
            {"p": str(corpus / "sections" / "*.parquet")},
        ).fetchall()
    (ob, oo), (db, do) = rows
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps(
            {"origin": {"bill_id": ob, "ordinal": oo}, "descendant": {"bill_id": db, "ordinal": do}}
        )
        + "\n"
    )
    cache = tmp_path / "cache"
    args = [
        "explain",
        str(pairs),
        "--tables",
        str(corpus),
        "--cache",
        str(cache),
        "--no-paraphrase",
    ]

    uncached = CliRunner().invoke(app, args)

    assert uncached.exit_code == 1
    assert "--live" in uncached.output
    [(origin, descendant)] = pair_texts(corpus, load_pairs(pairs))
    spans = align(origin, descendant).spans
    changed = next(s for s in spans if s.kind != "copied" and s.b is not None)
    claim = _claim(changed.id)
    _, ask = _scripted(ChangeExplanation(claims=[claim]))
    explain(
        prompt_spans(spans, origin, descendant, origin_side="a"),
        spans,
        origin_side="a",
        ask=ask,
        cache=cache,
    )

    cached = CliRunner().invoke(app, args)

    assert cached.exit_code == 0, cached.output
    assert f"added: {claim.summary} [{changed.id}]" in cached.output
