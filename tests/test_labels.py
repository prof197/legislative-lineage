"""`lineage label`: a person labels sampled matches, one pair at a time, resumable."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage import queries
from lineage.cli import app
from lineage.labels import (
    HUMAN,
    LabelError,
    Pair,
    SectionKey,
    load_labels,
    load_sample,
    marked,
    pending,
    session,
)
from lineage.pair_data import LABELS


def _pair(i: int) -> Pair:
    return Pair(f"p{i}", ("semantic",), SectionKey("a", i), SectionKey("b", i))


def _write_sample(path: Path, pairs: Sequence[Mapping[str, object]]) -> Path:
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs))
    return path


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


def test_only_a_persons_label_takes_a_pair_off_the_list(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"pair": "p0", "label": 3, "labeler": HUMAN, "at": "2026-11-09"})
        + "\n"
        + json.dumps({"pair": "p1", "label": 1, "labeler": "model:x", "at": "2026-11-09"})
        + "\n"
    )

    left = pending([_pair(0), _pair(1), _pair(2)], load_labels(labels))

    assert [p.pair for p in left] == ["p1", "p2"]


def test_a_session_records_labels_skips_reasks_and_stops(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    answers = iter(["3", "s", "nine", "1", "q"])
    shown: list[str] = []

    done = session(
        [_pair(0), _pair(1), _pair(2), _pair(3)],
        texts=lambda key: f"text of {key.bill_id} {key.ordinal}",
        ask=lambda prompt: next(answers),
        show=shown.append,
        labels_path=labels,
        today="2026-11-09",
    )

    got = load_labels(labels)
    assert done == 2
    assert [(r["pair"], r["label"], r["labeler"]) for r in got] == [
        ("p0", 3, HUMAN),
        ("p2", 1, HUMAN),
    ]
    assert any("not a label" in line for line in shown)
    # Resuming asks only what is left: the skipped pair and the one never reached.
    assert [p.pair for p in pending([_pair(i) for i in range(4)], got)] == ["p1", "p3"]


def test_the_scale_is_the_pair_datasets() -> None:
    assert len(LABELS) == 5


def test_alignment_is_marked_in_plain_text() -> None:
    tail = "and the commissioner may adopt rules"
    a = f"the department shall publish the rates for each fiscal year {tail}"
    b = f"the department shall publish the rates every biennium {tail}"

    first, second = marked(a, b, meaning_match=False, color=False)

    assert first.startswith("[=the department shall publish the rates=]")
    assert "[~for each fiscal year~]" in first
    assert "[~every biennium~]" in second


def test_the_command_labels_a_sample_from_the_built_tables(corpus: Path, tmp_path: Path) -> None:
    with queries.connect_quiet() as con:
        rows: list[tuple[str, int]] = con.execute(
            "SELECT bill_id, ordinal FROM read_parquet($p) ORDER BY bill_id, ordinal LIMIT 2",
            {"p": str(corpus / "sections" / "*.parquet")},
        ).fetchall()
    (qb, qo), (mb, mo) = rows
    sample = _write_sample(
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
    labels = tmp_path / "labels.jsonl"
    args = ["label", str(sample), "--tables", str(corpus), "--labels", str(labels), "--no-color"]

    first = CliRunner().invoke(app, args, input="4\n")
    again = CliRunner().invoke(app, args)

    assert first.exit_code == 0, first.output
    assert "1 labeled, 0 left" in first.output
    assert [(r["pair"], r["label"]) for r in load_labels(labels)] == [("p0", 4)]
    assert "Every pair" in again.output


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

    result = CliRunner().invoke(
        app, ["label", str(sample), "--tables", str(corpus), "--labels", str(tmp_path / "l")]
    )

    assert result.exit_code == 1
    assert "another build" in result.output
