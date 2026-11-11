"""The labeling page: the same pairs, scale and label rows as `lineage label`, blind to stages."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lineage import queries
from lineage.cli import app
from lineage.label_page import (
    DATA,
    PAGE,
    TEMPLATE,
    imported_rows,
    load_saved,
    page_data,
    write_page,
)
from lineage.labels import (
    SCALE,
    Item,
    LabelError,
    Pair,
    Section,
    SectionKey,
    SectionRef,
    load_labels,
    marked,
    text_sha256,
)
from lineage.pair_data import LABELS

SHARED = "the board shall adopt rules to carry out this act within ninety days"


def _section(bill: str, ordinal: int, text: str) -> Section:
    return Section(SectionRef(bill, f"Sec. {ordinal}.", ordinal, text_sha256(text)), text)


def _item(i: int, stages: tuple[str, ...] = ("near_duplicate",)) -> Item:
    return Item(
        Pair(f"p{i:04d}", stages, SectionKey(f"a{i}", i), SectionKey(f"b{i}", i)),
        _section(f"a{i}", i, f"Georgia words here. {SHARED}. More {i}."),
        _section(f"b{i}", i, f"{SHARED}. Texas tail {i}."),
    )


def _saved(item: Item, label: int, handle: str = "octo", at: str = "2026-11-11") -> dict[str, Any]:
    return {
        "pair": item.pair.pair,
        "query": item.query.ref.as_dict(),
        "match": item.match.ref.as_dict(),
        "label": label,
        "labeler": f"github:{handle}",
        "at": at,
    }


def test_the_page_data_is_blind_to_stages_and_marks_what_the_terminal_marks() -> None:
    items = [_item(0, ("near_duplicate",)), _item(1, ("semantic",))]

    data = page_data(items, handle="octo", sample=Path("evals/s.jsonl"), embed=None)

    text = json.dumps(data)
    for leak in ("stage", "near_duplicate", "model:"):
        assert leak not in text
    assert data["labeler"] == "github:octo"
    assert [s["name"] for s in data["scale"]] == [n.replace("_", " ") for n in LABELS]
    assert [s["gloss"] for s in data["scale"]] == list(SCALE)
    for item, pair in zip(items, data["pairs"], strict=True):
        assert pair["query"]["text"] == item.query.text
        assert {k: v for k, v in pair["match"].items() if k != "text"} == item.match.ref.as_dict()
        # The page's spans, bracketed as the terminal brackets them, read the same.
        for side, text_of, key in (("a", item.query.text, 0), ("b", item.match.text, 1)):
            pieces = [
                {"copied": "[={}=]", "paraphrased": "[~{}~]", "unmatched": "[+{}+]"}[
                    s["kind"]
                ].format(text_of[s[side][0] : s[side][1]])
                for s in pair["spans"]
                if s[side] is not None
            ]
            assert " ".join(pieces) == marked(item.query.text, item.match.text, color=False)[key]
        assert any(s["kind"] == "copied" for s in pair["spans"])


def test_the_template_never_reads_a_stage() -> None:
    assert "stage" not in TEMPLATE.read_text()


def test_the_page_is_written_beside_its_data(tmp_path: Path) -> None:
    data = page_data([_item(0)], handle="octo", sample=Path("s.jsonl"), embed=None)

    page = write_page(data, tmp_path / "out")

    assert page == tmp_path / "out" / PAGE
    assert page.read_text() == TEMPLATE.read_text()
    assert json.loads((tmp_path / "out" / DATA).read_text()) == data


def test_a_name_is_refused_for_the_page() -> None:
    with pytest.raises(LabelError, match="GitHub handle"):
        page_data([_item(0)], handle="Jane Doe", sample=Path("s.jsonl"), embed=None)


def test_saved_labels_become_label_rows_in_sample_order_once() -> None:
    items = [_item(0), _item(1), _item(2)]
    saved = [_saved(items[2], 1), _saved(items[0], 4)]

    rows = imported_rows(saved, items, [], handle="octo")

    assert [r["label"] for r in rows] == [4, 1]
    assert rows[0] == {
        "query": items[0].query.ref.as_dict(),
        "match": items[0].match.ref.as_dict(),
        "label": 4,
        "labeler": "github:octo",
        "at": "2026-11-11",
    }
    assert imported_rows(saved, items, rows, handle="octo") == []
    # A changed mind is a new row, and the later row is the one counted.
    [changed] = imported_rows([_saved(items[0], 3, at="2026-11-12")], items, rows, handle="octo")
    assert changed["label"] == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"pair": "p9999"}, "not a pair of the sample"),
        ({"query": {**_item(0).query.ref.as_dict(), "text_sha256": "0" * 64}}, "current sections"),
        ({"label": 5}, "not a label"),
        ({"label": True}, "not a label"),
        ({"label": "2"}, "not a label"),
        ({"labeler": "github:someone-else"}, "not 'github:octo'"),
        ({"labeler": "model:claude-opus-5-5@v1"}, "not 'github:octo'"),
        ({"at": "yesterday"}, "not a date"),
    ],
)  # fmt: skip
def test_a_saved_label_that_does_not_fit_the_sample_refuses_the_import(
    change: dict[str, Any], message: str
) -> None:
    items = [_item(0)]

    with pytest.raises(LabelError, match=message):
        imported_rows([{**_saved(items[0], 2), **change}], items, [], handle="octo")


def test_saved_labels_are_read_one_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "saved.jsonl"
    path.write_text('{"pair": "p0"}\n\n{"pair": "p1"}\n')
    assert [d["pair"] for d in load_saved(path)] == ["p0", "p1"]
    path.write_text("[1, 2]\n")
    with pytest.raises(LabelError, match="not a saved label"):
        load_saved(path)


def _corpus_sample(corpus: Path, tmp_path: Path) -> Path:
    with queries.connect_quiet() as con:
        rows: list[tuple[str, int]] = con.execute(
            "SELECT bill_id, ordinal FROM read_parquet($p) ORDER BY bill_id, ordinal LIMIT 2",
            {"p": str(corpus / "sections" / "*.parquet")},
        ).fetchall()
    (qb, qo), (mb, mo) = rows
    path = tmp_path / "sample.jsonl"
    pair = {
        "pair": "p0",
        "stages": ["near_duplicate"],
        "query": {"bill_id": qb, "ordinal": qo},
        "match": {"bill_id": mb, "ordinal": mo},
    }
    path.write_text(json.dumps(pair) + "\n")
    return path


def test_the_commands_write_the_page_and_import_what_it_saved(corpus: Path, tmp_path: Path) -> None:
    sample = _corpus_sample(corpus, tmp_path)
    labels = tmp_path / "labels.jsonl"
    out = tmp_path / "page"

    built = CliRunner().invoke(
        app,
        ["label-page", str(sample), "--labeler", "octo", "--tables", str(corpus),
         "--labels", str(labels), "--out", str(out), "--no-paraphrase"],
    )  # fmt: skip

    assert built.exit_code == 0, built.output
    [pair] = json.loads((out / DATA).read_text())["pairs"]
    doc = {
        "pair": pair["pair"],
        "query": {k: v for k, v in pair["query"].items() if k != "text"},
        "match": {k: v for k, v in pair["match"].items() if k != "text"},
        "label": 2,
        "labeler": "github:octo",
        "at": "2026-11-11",
    }
    saved = tmp_path / "saved.jsonl"
    saved.write_text(json.dumps(doc) + "\n")
    args = ["label-import", str(saved), "--sample", str(sample), "--labeler", "octo",
            "--tables", str(corpus), "--labels", str(labels)]  # fmt: skip

    imported = CliRunner().invoke(app, args)
    again = CliRunner().invoke(app, args)

    assert imported.exit_code == 0, imported.output
    assert "1 labels imported, 0 pairs" in imported.output
    assert "0 labels imported" in again.output
    [row] = load_labels(labels)
    assert (row["label"], row["labeler"]) == (2, "github:octo")
    rebuilt = CliRunner().invoke(
        app,
        ["label-page", str(sample), "--labeler", "octo", "--tables", str(corpus),
         "--labels", str(labels), "--out", str(out), "--no-paraphrase"],
    )  # fmt: skip
    assert "0 pending pairs" in rebuilt.output
