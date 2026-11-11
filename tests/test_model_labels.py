"""Provisional precision labels from a model: asked only with --live, recorded under the
model id and prompt version, never taking a pair off a person's list."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.labels import Item, Pair, Section, SectionKey, SectionRef, pending, text_sha256
from lineage.model_labels import (
    CHARS_PER_TOKEN,
    MAX_TOKENS,
    PRICES,
    PROMPT_VERSION,
    SYSTEM,
    BatchReply,
    ModelLabel,
    UnusableReplyError,
    label_in_batch,
    label_with_model,
    model_pending,
    projected_cost,
    user_message,
)


def _item(i: int) -> Item:
    def section(bill: str) -> Section:
        text = f"{bill} section text {i}"
        return Section(SectionRef(bill, None, i, text_sha256(text)), text)

    return Item(
        Pair(f"p{i}", ("semantic",), SectionKey("a", i), SectionKey("b", i)),
        *map(section, ("a", "b")),
    )


def test_the_reply_is_one_label_on_the_scale() -> None:
    assert ModelLabel.model_validate({"label": 3}).label == 3
    for bad in ({"label": 5}, {"label": 2, "reason": "x"}, {}):
        with pytest.raises(ValueError, match=r"label|reason|Field"):
            ModelLabel.model_validate(bad)


def test_the_prompt_gives_the_scale_and_both_texts_but_not_the_stages() -> None:
    item = _item(0)

    user = json.loads(user_message(item))

    assert user == {"first": item.query.text, "second": item.match.text}
    assert "the same provision in mostly the same words" in SYSTEM
    assert "semantic" not in SYSTEM + json.dumps(user)


def test_labels_are_recorded_by_model_and_prompt_version_and_resume(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    items = [_item(i) for i in range(3)]
    replies = iter([(2, "claude-x"), UnusableReplyError("declined"), (0, "claude-x")])

    def ask(system: str, user: str) -> tuple[int, str]:
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    labeled, unusable = label_with_model(items, ask=ask, labels_path=labels, today="2026-11-09")

    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert (labeled, unusable) == (2, 1)
    assert [(r["query"]["ordinal"], r["label"], r["labeler"]) for r in rows] == [
        (0, 2, f"model:claude-x@{PROMPT_VERSION}"),
        (2, 0, f"model:claude-x@{PROMPT_VERSION}"),
    ]
    assert [i.pair.pair for i in model_pending(items, rows)] == ["p1"]
    assert [i.pair.pair for i in pending(items, rows)] == ["p0", "p1", "p2"]  # still a person's


def test_a_label_under_an_older_prompt_is_asked_again() -> None:
    item = _item(0)
    row = {
        "query": item.query.ref.as_dict(),
        "match": item.match.ref.as_dict(),
        "label": 4,
        "labeler": "model:claude-x@v0",
        "at": "2026-11-01",
    }

    assert model_pending([item], [row]) == [item]


def _one_pair_sample(corpus: Path, tmp_path: Path) -> Path:
    from lineage import queries

    with queries.connect_quiet() as con:
        rows: list[tuple[str, int]] = con.execute(
            "SELECT bill_id, ordinal FROM read_parquet($p) ORDER BY bill_id, ordinal LIMIT 2",
            {"p": str(corpus / "sections" / "*.parquet")},
        ).fetchall()
    (qb, qo), (mb, mo) = rows
    sample = tmp_path / "sample.jsonl"
    row = {
        "pair": "p0001",
        "stages": ["semantic"],
        "query": {"bill_id": qb, "ordinal": qo},
        "match": {"bill_id": mb, "ordinal": mo},
    }
    sample.write_text(json.dumps(row) + "\n")
    return sample


def test_without_live_the_command_says_what_it_would_ask_and_asks_nothing(
    corpus: Path, tmp_path: Path
) -> None:
    sample = _one_pair_sample(corpus, tmp_path)
    labels = tmp_path / "labels.jsonl"

    result = CliRunner().invoke(
        app, ["label-model", str(sample), "--tables", str(corpus), "--labels", str(labels)]
    )

    assert result.exit_code == 0, result.output
    assert "1 pair would be asked" in result.output
    assert "--live" in result.output
    assert not labels.exists()
    assert "projected" in result.output


def test_the_projection_bounds_every_reply_at_its_token_limit() -> None:
    items = [_item(i) for i in range(3)]
    input_price, output_price = PRICES["claude-opus-5-5"]

    cost = projected_cost(items, model="claude-opus-5-5", max_tokens=1000)

    chars = sum(len(SYSTEM) + len(user_message(i)) for i in items)
    expected = (chars / CHARS_PER_TOKEN * input_price + 3 * 1000 * output_price) / 1e6
    assert cost == pytest.approx(expected)
    assert projected_cost(items, model="claude-opus-5-5", max_tokens=1000, batch=True) == (
        pytest.approx(expected / 2)
    )


def test_a_model_without_a_known_price_is_not_projected() -> None:
    with pytest.raises(ValueError, match="no price"):
        projected_cost([_item(0)], model="claude-unknown")


def test_live_over_the_cost_cap_asks_nothing(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lineage import cli

    def no_calls(*args: object, **kwargs: object) -> object:
        raise AssertionError("a model was asked")

    monkeypatch.setattr(cli, "anthropic_label_ask", no_calls)
    sample = _one_pair_sample(corpus, tmp_path)
    labels = tmp_path / "labels.jsonl"

    result = CliRunner().invoke(
        app,
        ["label-model", str(sample), "--tables", str(corpus), "--labels", str(labels), "--live",
         "--max-cost", "0.000001"],
    )  # fmt: skip

    assert result.exit_code == 1
    assert "over the" in result.output
    assert not labels.exists()


class _FakeBatches:
    """A stand-in for the Batches API: records what was sent, ends after two polls."""

    def __init__(self, replies: dict[str, tuple[int | None, str | None]]) -> None:
        self.replies = replies
        self.sent: list[tuple[str, dict[str, object]]] = []
        self.polls = 0

    def submit(self, requests: list[tuple[str, dict[str, object]]]) -> str:
        self.sent = requests
        return "batch-1"

    def ended(self, batch_id: str) -> bool:
        assert batch_id == "batch-1"
        self.polls += 1
        return self.polls >= 2

    def results(self, batch_id: str) -> list[BatchReply]:
        return [
            BatchReply(custom_id, label, served_by, None if label is not None else "errored")
            for custom_id, (label, served_by) in self.replies.items()
        ]


def test_a_batch_sends_each_pair_once_and_appends_what_comes_back(tmp_path: Path) -> None:
    items = [_item(i) for i in range(3)]
    batches = _FakeBatches(
        {"p0": (3, "claude-opus-5-5"), "p1": (None, None), "p2": (0, "claude-opus-5-5")}
    )
    labels = tmp_path / "labels.jsonl"

    labeled, unusable, batch_id = label_in_batch(
        items, batches=batches, model="claude-opus-5-5", labels_path=labels, today="2099-01-02",
        wait=lambda: None,
    )  # fmt: skip

    assert (labeled, unusable, batch_id) == (2, 1, "batch-1")
    assert [custom_id for custom_id, _ in batches.sent] == ["p0", "p1", "p2"]
    for (_, params), item in zip(batches.sent, items, strict=True):
        assert params["system"] == SYSTEM
        assert params["messages"] == [{"role": "user", "content": user_message(item)}]
        assert params["max_tokens"] == MAX_TOKENS
        config = params["output_config"]
        assert isinstance(config, dict)
        assert config["effort"] == "low"
        assert config["format"] == {"type": "json_schema", "schema": ModelLabel.model_json_schema()}
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert [r["label"] for r in rows] == [3, 0]
    assert all(r["labeler"] == f"model:claude-opus-5-5@{PROMPT_VERSION}" for r in rows)


def test_collecting_a_submitted_batch_sends_nothing(tmp_path: Path) -> None:
    items = [_item(0)]
    batches = _FakeBatches({"p0": (2, "claude-opus-5-5")})

    labeled, _, _ = label_in_batch(
        items, batches=batches, model="claude-opus-5-5", labels_path=tmp_path / "l.jsonl",
        today="2099-01-02", wait=lambda: None, batch_id="batch-1",
    )  # fmt: skip

    assert labeled == 1
    assert batches.sent == []


def test_a_reply_for_a_pair_not_asked_is_refused(tmp_path: Path) -> None:
    batches = _FakeBatches({"p9": (2, "claude-opus-5-5")})

    with pytest.raises(UnusableReplyError, match="p9"):
        label_in_batch(
            [_item(0)], batches=batches, model="claude-opus-5-5",
            labels_path=tmp_path / "l.jsonl", today="2099-01-02", wait=lambda: None,
        )  # fmt: skip
