"""The precision sample: the build's cross-state section pairs stratified by the stages that
found them, a seeded draw from each stratum, and precision estimated from the labels, per
stratum and weighted by stratum size for each stage and overall."""

import json
from pathlib import Path

import numpy as np
import pytest

from lineage.intervals import wilson
from lineage.labels import SAME_PROVISION, Item, Pair, Section, SectionKey, SectionRef, text_sha256
from lineage.precision import (
    STRATA,
    PrecisionError,
    draw,
    estimate,
    latest_correct,
    latest_labels,
    pair_keys,
    sample_rows,
    strata,
    stratified,
    stratum_of,
)


def _keys(*pairs: tuple[int, int]) -> np.ndarray:
    left = np.array([p[0] for p in pairs], dtype=np.uint32)
    right = np.array([p[1] for p in pairs], dtype=np.uint32)
    return pair_keys(left, right)


def test_pairs_fall_into_one_stratum_each_by_the_stages_that_found_them() -> None:
    near = _keys((0, 1), (0, 2), (3, 4))
    semantic = _keys((0, 2), (5, 6))

    got = strata(near, semantic)

    assert list(got) == list(STRATA)
    assert got["near_duplicate_only"].tolist() == _keys((0, 1), (3, 4)).tolist()
    assert got["both"].tolist() == _keys((0, 2)).tolist()
    assert got["semantic_only"].tolist() == _keys((5, 6)).tolist()
    assert stratum_of(("semantic", "near_duplicate")) == "both"
    assert stratum_of(("near_duplicate",)) == "near_duplicate_only"


def test_a_pair_key_keeps_both_sections() -> None:
    key = _keys((7, 4_000_000_000))[0]

    assert (int(key) >> 32, int(key) & 0xFFFFFFFF) == (7, 4_000_000_000)


def test_the_draw_is_seeded_without_replacement_and_capped_at_the_stratum() -> None:
    population = {
        "near_duplicate_only": np.arange(1000, dtype=np.uint64),
        "semantic_only": np.arange(5000, 5003, dtype=np.uint64),
        "both": np.arange(9000, 9100, dtype=np.uint64),
    }

    first = draw(population, per_stratum=50, seed=7)
    again = draw(population, per_stratum=50, seed=7)
    other = draw(population, per_stratum=50, seed=8)

    assert first == again
    assert first != other
    assert [s for s, _ in first].count("near_duplicate_only") == 50
    assert sorted(k for s, k in first if s == "semantic_only") == [5000, 5001, 5002]
    assert len({k for _, k in first}) == len(first)
    # Strata are interleaved, so a session stopped part way has labeled every stratum.
    assert {"near_duplicate_only", "both"} <= {s for s, _ in first[:20]}
    assert draw({"both": np.arange(0, dtype=np.uint64)}, per_stratum=5) == []


def test_a_drawn_row_names_each_section_by_bill_and_ordinal() -> None:
    drawn = [("both", (3 << 32) | 9)]
    where = {3: SectionKey("ga-1", 2), 9: SectionKey("tx-5", 1)}

    [row] = sample_rows(drawn, where.__getitem__)

    assert row == {
        "pair": "p0001",
        "stages": ["near_duplicate", "semantic"],
        "query": {"bill_id": "ga-1", "ordinal": 2},
        "match": {"bill_id": "tx-5", "ordinal": 1},
    }


def test_one_stratum_gets_the_wilson_interval() -> None:
    got = stratified([(1000, 20, 18)])

    assert got is not None
    assert got["precision"] == 0.9
    assert got["interval_95"] == [round(v, 4) for v in wilson(18, 20)]


def test_strata_are_weighted_by_size_and_an_unlabeled_one_leaves_no_estimate() -> None:
    got = stratified([(900, 10, 10), (100, 10, 5)])

    assert got is not None
    assert got["precision"] == pytest.approx(0.95)
    lo, hi = got["interval_95"]
    assert lo < 0.95 < hi <= 1.0
    # Every pair in the first stratum was correct, and the interval is still not zero wide.
    assert stratified([(500, 10, 10), (500, 10, 10)])["interval_95"][0] < 1.0  # type: ignore[index]
    assert stratified([(900, 10, 10), (100, 0, 0)]) is None
    assert stratified([(900, 10, 10), (0, 0, 0)])["precision"] == 1.0  # type: ignore[index]


def _item(i: int, stages: tuple[str, ...]) -> Item:
    def section(bill: str) -> Section:
        text = f"{bill} text {i}"
        return Section(SectionRef(bill, None, i, text_sha256(text)), text)

    pair = Pair(f"p{i}", stages, SectionKey("a", i), SectionKey("b", i))
    return Item(pair, section("a"), section("b"))


def _label(item: Item, label: int, labeler: str, at: str = "2026-11-09") -> dict[str, object]:
    return {
        "query": item.query.ref.as_dict(),
        "match": item.match.ref.as_dict(),
        "label": label,
        "labeler": labeler,
        "at": at,
    }


def test_the_estimate_counts_labels_from_two_up_as_correct_by_stratum_and_stage() -> None:
    near, both, sem = ("near_duplicate",), ("near_duplicate", "semantic"), ("semantic",)
    items = [_item(0, near), _item(1, near), _item(2, both), _item(3, sem), _item(4, sem)]
    labels = [
        _label(items[0], 4, "github:a"),
        _label(items[1], 1, "github:a"),
        _label(items[2], 2, "github:a"),
        _label(items[3], 0, "github:a", "2026-11-01"),
        _label(items[3], 3, "github:a", "2026-11-02"),  # the later label counts
        _label(items[4], 4, "model:m@v1"),  # provisional, counted apart
    ]
    sizes = {"near_duplicate_only": 600, "semantic_only": 300, "both": 100}

    report = estimate(items, labels, sizes)

    people = report["person"]
    assert people["strata"]["near_duplicate_only"] == {
        "size": 600,
        "sampled": 2,
        "labeled": 2,
        "correct": 1,
        "precision": 0.5,
        "interval_95": [round(v, 4) for v in wilson(1, 2)],
    }
    assert people["strata"]["semantic_only"]["labeled"] == 1
    assert people["stages"]["near_duplicate"]["precision"] == pytest.approx(
        (600 * 0.5 + 100 * 1.0) / 700, abs=1e-4
    )
    assert people["stages"]["semantic"]["precision"] == 1.0
    assert people["overall"]["precision"] == pytest.approx(0.7, abs=1e-4)
    model = report["provisional_model"]
    assert model["strata"]["semantic_only"]["labeled"] == 1
    assert model["overall"] is None  # the model has not labeled every stratum
    assert report["correct_from"] == 2


def test_agreement_is_reported_where_a_person_and_the_model_both_labeled() -> None:
    items = [_item(i, ("semantic",)) for i in range(3)]
    labels = [
        _label(items[0], 3, "github:a"),
        _label(items[0], 2, "model:m@v1"),  # both correct: agree on correctness, not the label
        _label(items[1], 1, "github:a"),
        _label(items[1], 2, "model:m@v1"),  # disagree
        _label(items[2], 4, "model:m@v1"),
    ]

    agreement = estimate(items, labels, {"semantic_only": 3})["agreement"]

    assert agreement == {"pairs": 2, "same_label": 0, "same_correctness": 1}


def test_a_label_from_neither_a_person_nor_a_model_is_refused() -> None:
    item = _item(0, ("semantic",))

    with pytest.raises(PrecisionError, match="labeler"):
        estimate([item], [_label(item, 2, "Jane Doe")], {"semantic_only": 1})


def test_a_sample_pair_from_an_unknown_stratum_is_refused(tmp_path: Path) -> None:
    item = _item(0, ("lexical",))

    with pytest.raises(PrecisionError, match="stages"):
        estimate([item], [], {"semantic_only": 1})
    assert json.dumps(STRATA)  # the strata are plain data, written into reports as is


def test_the_command_writes_the_estimate_into_the_results(corpus: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from lineage import queries
    from lineage.cli import app

    with queries.connect_quiet() as con:
        rows: list[tuple[str, int]] = con.execute(
            "SELECT bill_id, ordinal FROM read_parquet($p) ORDER BY bill_id, ordinal LIMIT 2",
            {"p": str(corpus / "sections" / "*.parquet")},
        ).fetchall()
    (qb, qo), (mb, mo) = rows
    sample = tmp_path / "sample.jsonl"
    pair = {
        "pair": "p0001",
        "stages": ["semantic"],
        "query": {"bill_id": qb, "ordinal": qo},
        "match": {"bill_id": mb, "ordinal": mo},
    }
    sample.write_text(json.dumps(pair) + "\n")
    sizes = {name: {"size": 10 if name == "semantic_only" else 0} for name in STRATA}
    strata_file = tmp_path / "draw.json"
    strata_file.write_text(json.dumps({"strata": sizes}))
    labels, results = tmp_path / "labels.jsonl", tmp_path / "results.json"
    common = ["--tables", str(corpus), "--labels", str(labels)]
    labeled = CliRunner().invoke(
        app,
        ["label", str(sample), "--labeler", "octo", *common, "--no-color", "--no-paraphrase"],
        input="3\n",
    )

    result = CliRunner().invoke(
        app,
        ["eval", "precision", str(sample), *common, "--strata", str(strata_file),
         "--results", str(results)],
    )  # fmt: skip

    assert labeled.exit_code == 0, labeled.output
    assert result.exit_code == 0, result.output
    section = json.loads(results.read_text())["precision"]
    assert section["command"] == "uv run lineage eval precision"
    assert section["person"]["overall"]["precision"] == 1.0
    assert section["provisional_model"]["overall"] is None
    assert "not every stratum labeled" in result.output


def test_the_sample_is_drawn_from_both_stages_stored_pairs(embedded_commission: Path) -> None:
    from lineage.pairs import load_cross_state_pairs
    from lineage.precision import build_sample
    from lineage.semantic_pairs import build_semantic_pairs, load_semantic_pairs, section_keys
    from tests.conftest import FAKE_MODEL

    tables = embedded_commission
    build_semantic_pairs(tables, model=FAKE_MODEL, floor_bytes=0)

    rows, report = build_sample(tables, model=FAKE_MODEL, per_stratum=5)

    near = load_cross_state_pairs(tables)
    meaning = load_semantic_pairs(tables, FAKE_MODEL)
    bills, ordinals = section_keys(tables)
    key = {(bills[s], int(ordinals[s])): s for s in range(len(bills))}

    def pairs_of(left: np.ndarray, right: np.ndarray) -> set[tuple[int, int]]:
        return set(zip(left.tolist(), right.tolist(), strict=True))

    by_stage = {
        "near_duplicate": pairs_of(np.asarray(near.left), np.asarray(near.right)),
        "semantic": pairs_of(np.asarray(meaning.left), np.asarray(meaning.right)),
    }
    sizes = {name: report["strata"][name]["size"] for name in STRATA}
    assert sizes["both"] == len(by_stage["near_duplicate"] & by_stage["semantic"])
    assert sizes["semantic_only"] == len(by_stage["semantic"] - by_stage["near_duplicate"])
    assert sizes["near_duplicate_only"] == len(by_stage["near_duplicate"] - by_stage["semantic"])
    assert len(rows) == sum(min(5, n) for n in sizes.values())
    for row in rows:
        a = key[(row["query"]["bill_id"], row["query"]["ordinal"])]
        b = key[(row["match"]["bill_id"], row["match"]["ordinal"])]
        found = {stage for stage, pairs in by_stage.items() if (a, b) in pairs}
        assert set(row["stages"]) == found
    assert build_sample(tables, model=FAKE_MODEL, per_stratum=5)[0] == rows  # seeded


def test_a_yes_to_the_passage_question_counts_correct_and_is_never_read_as_a_grade() -> None:
    items = [_item(i, ("semantic",)) for i in range(3)]
    asker = "model:claude-opus-5-5@passages-v1-high"
    labels = [
        {**_label(items[0], 1, asker), "scale": SAME_PROVISION},
        {**_label(items[1], 0, asker), "scale": SAME_PROVISION},
        {**_label(items[2], 1, asker), "scale": SAME_PROVISION},
    ]
    sizes = {"near_duplicate_only": 0, "semantic_only": 10, "both": 0}

    got = estimate(items, labels, sizes, model_labeler=asker)

    assert got["provisional_model"]["strata"]["semantic_only"]["correct"] == 2
    assert latest_correct(items, labels, asker, exact=True) == {
        "p0": True, "p1": False, "p2": True
    }  # fmt: skip
    with pytest.raises(PrecisionError, match="not on the 0-4 scale"):
        latest_labels(items, labels, asker, exact=True)
