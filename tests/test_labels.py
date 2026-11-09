"""`lineage label`: a person labels sampled matches, one pair at a time, resumable.

Labels are keyed by what survives a re-split: each side's Open States bill id and the hash of
its section's normalized text, with the heading and ordinal kept for reading."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage import queries
from lineage.cli import app
from lineage.labels import (
    CORRECT_FROM,
    Item,
    LabelError,
    Pair,
    Section,
    SectionKey,
    SectionRef,
    current_labels,
    load_labels,
    load_sample,
    marked,
    model_labeler,
    pending,
    person,
    session,
    text_sha256,
)
from lineage.pair_data import LABELS
from lineage.pair_eval import POSITIVE_FROM


def _section(bill: str, ordinal: int, text: str) -> Section:
    return Section(SectionRef(bill, f"Sec. {ordinal}.", ordinal, text_sha256(text)), text)


def _item(i: int) -> Item:
    pair = Pair(f"p{i}", ("semantic",), SectionKey("a", i), SectionKey("b", i))
    return Item(pair, _section("a", i, f"query text {i}"), _section("b", i, f"match text {i}"))


def _write_sample(path: Path, pairs: Sequence[Mapping[str, object]]) -> Path:
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    return path


def _row(item: Item, label: int, labeler: str) -> dict[str, object]:
    return {
        "query": item.query.ref.as_dict(),
        "match": item.match.ref.as_dict(),
        "label": label,
        "labeler": labeler,
        "at": "2026-11-09",
    }


def test_a_sample_is_read_and_a_repeated_pair_refused(tmp_path: Path) -> None:
    row = {
        "pair": "p0",
        "stages": ["near_duplicate", "semantic"],
        "query": {"bill_id": "a", "ordinal": 1},
        "match": {"bill_id": "b", "ordinal": 2},
    }

    sample = load_sample(_write_sample(tmp_path / "s.jsonl", [row]))

    assert sample == [
        Pair("p0", ("near_duplicate", "semantic"), SectionKey("a", 1), SectionKey("b", 2))
    ]
    with pytest.raises(LabelError, match="p0 appears twice"):
        load_sample(_write_sample(tmp_path / "d.jsonl", [row, row]))


def test_labelers_are_github_handles_or_model_ids_never_names() -> None:
    assert person("octo-cat") == "github:octo-cat"
    assert model_labeler("claude-opus-5-5", "v1") == "model:claude-opus-5-5@v1"
    for bad in ("Jane Doe", "", "-x", "a" * 40, "x--y"):
        with pytest.raises(LabelError, match="GitHub handle"):
            person(bad)


def test_only_a_persons_label_takes_a_pair_off_the_list() -> None:
    items = [_item(0), _item(1), _item(2)]
    labels = [_row(items[0], 3, "github:octo"), _row(items[1], 1, "model:x@v1")]

    assert [i.pair.pair for i in pending(items, labels)] == ["p1", "p2"]


def test_a_label_survives_renumbering_but_not_a_change_of_text() -> None:
    item = _item(0)
    q, m = item.query.ref, item.match.ref
    labels = [_row(item, 4, "github:octo")]

    kept, dropped = current_labels(labels, {("a", q.text_sha256), ("b", m.text_sha256)})
    renumbered = Section(SectionRef("a", "Sec. 9.", 9, q.text_sha256), item.query.text)
    assert (len(kept), dropped) == (1, 0)
    assert pending([Item(item.pair, renumbered, item.match)], kept) == []

    kept, dropped = current_labels(labels, {("a", text_sha256("rewritten")), ("b", m.text_sha256)})
    assert (len(kept), dropped) == (0, 1)


def test_a_session_records_labels_skips_reasks_and_stops(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    items = [_item(i) for i in range(4)]
    answers = iter(["3", "s", "nine", "1", "q"])
    shown: list[str] = []

    done = session(
        items,
        ask=lambda prompt: next(answers),
        show=shown.append,
        labels_path=labels,
        labeler="github:octo",
        today="2026-11-09",
    )

    got = load_labels(labels)
    assert done == 2
    assert [(r["query"]["ordinal"], r["label"], r["labeler"]) for r in got] == [
        (0, 3, "github:octo"),
        (2, 1, "github:octo"),
    ]
    assert got[0]["query"] == items[0].query.ref.as_dict()
    assert any("not a label" in line for line in shown)
    # Resuming asks only what is left: the skipped pair and the one never reached.
    assert [i.pair.pair for i in pending(items, got)] == ["p1", "p3"]


def test_a_correct_match_is_related_or_closer_as_the_threshold_was_set() -> None:
    assert len(LABELS) == 5
    assert CORRECT_FROM == POSITIVE_FROM == 2


def test_alignment_is_marked_in_plain_text() -> None:
    tail = "and the commissioner may adopt rules"
    a = f"the department shall publish the rates for each fiscal year {tail}"
    b = f"the department shall publish the rates every biennium {tail}"

    first, second = marked(a, b, color=False)

    assert first.startswith("[=the department shall publish the rates=]")
    assert "[+for each fiscal year+]" in first  # unmatched: no embedder pairs it
    assert "[+every biennium+]" in second


def _corpus_sample(corpus: Path, tmp_path: Path) -> Path:
    with queries.connect_quiet() as con:
        rows: list[tuple[str, int]] = con.execute(
            "SELECT bill_id, ordinal FROM read_parquet($p) ORDER BY bill_id, ordinal LIMIT 2",
            {"p": str(corpus / "sections" / "*.parquet")},
        ).fetchall()
    (qb, qo), (mb, mo) = rows
    return _write_sample(
        tmp_path / "sample.jsonl",
        [
            {
                "pair": "p0",
                "stages": ["near_duplicate"],
                "query": {"bill_id": qb, "ordinal": qo},
                "match": {"bill_id": mb, "ordinal": mo},
            }
        ],
    )


def _args(sample: Path, corpus: Path, labels: Path, labeler: str = "octo") -> list[str]:
    return [
        "label", str(sample), "--labeler", labeler, "--tables", str(corpus),
        "--labels", str(labels), "--no-color", "--no-paraphrase",
    ]  # fmt: skip


def test_the_command_labels_a_sample_from_the_built_tables(corpus: Path, tmp_path: Path) -> None:
    sample = _corpus_sample(corpus, tmp_path)
    labels = tmp_path / "labels.jsonl"

    first = CliRunner().invoke(app, _args(sample, corpus, labels), input="4\n")
    again = CliRunner().invoke(app, _args(sample, corpus, labels))

    assert first.exit_code == 0, first.output
    assert "1 labeled, 0 left" in first.output
    [row] = load_labels(labels)
    assert (row["label"], row["labeler"]) == (4, "github:octo")
    assert len(row["query"]["text_sha256"]) == 64
    assert "Every pair" in again.output


def test_the_command_reports_labels_whose_text_changed(corpus: Path, tmp_path: Path) -> None:
    sample = _corpus_sample(corpus, tmp_path)
    labels = tmp_path / "labels.jsonl"
    CliRunner().invoke(app, _args(sample, corpus, labels), input="4\n")
    [row] = load_labels(labels)
    row["query"]["text_sha256"] = text_sha256("a text the section no longer has")
    labels.write_text(json.dumps(row) + "\n")

    result = CliRunner().invoke(app, _args(sample, corpus, labels), input="q\n")

    assert "1 label no longer matches" in result.output


def test_a_name_is_refused_as_a_labeler(corpus: Path, tmp_path: Path) -> None:
    sample = _corpus_sample(corpus, tmp_path)

    result = CliRunner().invoke(app, _args(sample, corpus, tmp_path / "l", labeler="Jane Doe"))

    assert result.exit_code == 1
    assert "GitHub handle" in result.output


def test_a_sample_from_another_build_is_refused(corpus: Path, tmp_path: Path) -> None:
    sample = _write_sample(
        tmp_path / "sample.jsonl",
        [
            {
                "pair": "p0",
                "stages": ["semantic"],
                "query": {"bill_id": "nope", "ordinal": 1},
                "match": {"bill_id": "nope", "ordinal": 2},
            }
        ],
    )

    result = CliRunner().invoke(app, _args(sample, corpus, tmp_path / "l"))

    assert result.exit_code == 1
    assert "another build" in result.output
