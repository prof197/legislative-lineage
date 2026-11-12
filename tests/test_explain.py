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
    check_cache,
    explain,
    load_pairs,
    pair_texts,
    projected_cost,
    prompt_spans,
)
from lineage.explanations import ChangeExplanation, Claim
from lineage.spend import BudgetError, Meter, load_runs

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


def scripted(*replies: ChangeExplanation | Exception) -> tuple[list[tuple[str, str]], Ask]:
    calls: list[tuple[str, str]] = []
    queue: Iterator[ChangeExplanation | Exception] = iter(replies)

    def ask(system: str, user: str) -> Reply:
        calls.append((system, user))
        nxt = next(queue)
        if isinstance(nxt, Exception):
            raise nxt
        return Reply(nxt, "claude-opus-5-5", 1000, 100)

    return calls, ask


def _explain(ask: Ask | None, cache: Path, meter: Meter | None = None) -> Outcome:
    return explain(
        prompt_spans(SPANS, ORIGIN, DESCENDANT, origin_side="a"),
        ask=ask,
        cache=cache,
        meter=meter,
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
    calls, ask = scripted(ChangeExplanation(claims=[_claim("s2")]))

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
    calls, ask = scripted(ChangeExplanation(claims=[_claim("s2"), _claim("s1", "changed")]))

    outcome = _explain(ask, tmp_path)

    assert len(calls) == 1
    assert outcome.claims == [_claim("s2")]
    assert [why for _, why in outcome.dropped] == ["s1 is an exact copy"]


def test_an_unusable_reply_is_retried_once(tmp_path: Path) -> None:
    calls, ask = scripted(
        ChangeExplanation(claims=[_claim("s1", "changed")]),
        ChangeExplanation(claims=[_claim("s2")]),
    )

    outcome = _explain(ask, tmp_path)

    assert len(calls) == 2
    assert outcome.claims == [_claim("s2")]


def test_after_one_retry_the_explanation_is_withheld_and_that_is_cached(tmp_path: Path) -> None:
    calls, ask = scripted(UnusableReplyError("refused"), ChangeExplanation(claims=[_claim("s9")]))

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
    calls, ask = scripted(ChangeExplanation(claims=[]))

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

    uncached = CliRunner().invoke(app, [*args, "--spend", str(tmp_path / "spend.json")])

    assert uncached.exit_code == 1
    assert "--live" in uncached.output
    [(origin, descendant)] = pair_texts(corpus, load_pairs(pairs))
    spans = align(origin, descendant).spans
    changed = next(s for s in spans if s.kind != "copied" and s.b is not None)
    claim = _claim(changed.id)
    _, ask = scripted(ChangeExplanation(claims=[claim]))
    explain(prompt_spans(spans, origin, descendant, origin_side="a"), ask=ask, cache=cache)

    cached = CliRunner().invoke(app, [*args, "--spend", str(tmp_path / "spend.json")])

    assert cached.exit_code == 0, cached.output
    assert f"added: {claim.summary} [{changed.id}]" in cached.output


def test_a_cache_file_holds_no_text_and_ci_grounds_it_again_from_what_it_records(
    tmp_path: Path,
) -> None:
    _, ask = scripted(ChangeExplanation(claims=[_claim("s2"), _claim("s1", "changed")]))
    _explain(ask, tmp_path)
    [path] = tmp_path.glob("*.json")
    entry = json.loads(path.read_text())

    assert "Appeals" not in path.read_text()
    assert [s["id"] for s in entry["spans"]] == ["s0", "s1", "s2"]
    assert entry["grounded"] == {"kept": [0], "dropped": [[1, "s1 is an exact copy"]]}
    assert check_cache(tmp_path) == []

    # A claim edited to cite a copied span, or a record of what was kept that no longer holds.
    entry["explanation"]["claims"] = [_claim("s1", "changed").model_dump()]
    path.write_text(json.dumps(entry))
    problems = check_cache(tmp_path)
    assert any("different set of claims" in p for p in problems)
    assert any("none of its claims is grounded" in p for p in problems)


def test_a_withheld_explanation_passes_the_check_and_an_unreadable_file_does_not(
    tmp_path: Path,
) -> None:
    _, ask = scripted(UnusableReplyError("refused"), UnusableReplyError("refused"))
    _explain(ask, tmp_path)
    assert check_cache(tmp_path) == []

    (tmp_path / "bad.json").write_text("{}")
    assert check_cache(tmp_path) == ["bad.json: not a cached explanation ('spans')"]


def test_every_call_is_metered_and_a_call_past_the_budget_is_refused(tmp_path: Path) -> None:
    spend = tmp_path / "spend.json"
    meter = Meter(spend, "explain", MODEL, PROMPT_VERSION, "2026-11-12")
    _, ask = scripted(
        UnusableReplyError("refused", 500, 20, "claude-opus-5-5"),
        ChangeExplanation(claims=[_claim("s2")]),
    )
    _explain(ask, tmp_path / "cache", meter)

    [run] = load_runs(spend)
    assert (run["calls"], run["input_tokens"], run["output_tokens"]) == (2, 1500, 120)
    assert run["dollars"] == round((1500 * 4 + 120 * 20) / 1e6, 4)

    spend.write_text(json.dumps({"runs": [{**run, "run": "earlier", "dollars": 29.99}]}))
    broke = Meter(spend, "explain", MODEL, PROMPT_VERSION, "2026-11-12")
    with pytest.raises(BudgetError, match="passes the"):
        _explain(scripted(ChangeExplanation(claims=[]))[1], tmp_path / "other", broke)


def test_the_projection_counts_every_retry_at_the_reply_limit() -> None:
    spans = prompt_spans(SPANS, ORIGIN, DESCENDANT, origin_side="a")

    assert projected_cost([spans, spans]) == pytest.approx(2 * projected_cost([spans]))
    assert projected_cost([spans]) > 2 * 4096 * 20 / 1e6


def test_the_retry_says_why_the_first_reply_was_unusable_and_the_cache_keeps_it(
    tmp_path: Path,
) -> None:
    calls, ask = scripted(
        UnusableReplyError("off the schema: claims.0.summary: too long"),
        ChangeExplanation(claims=[_claim("s2")]),
    )

    _explain(ask, tmp_path)

    assert "could not be used: off the schema: claims.0.summary: too long" in calls[1][1]
    assert calls[1][1].startswith(calls[0][1])
    [entry] = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert entry["unusable_attempts"] == ["off the schema: claims.0.summary: too long"]
