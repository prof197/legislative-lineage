"""A person's labels against the model's on a stratified subsample of the precision sample."""

from pathlib import Path

import pytest

from lineage.agreement import (
    CUTOFF_AGREEMENT_FROM,
    KAPPA_FROM,
    PAIRS,
    AgreementError,
    agreement,
    allocation,
    draw_subsample,
    subsample_rows,
    weighted_kappa,
    write_subsample,
)
from lineage.labels import (
    Item,
    Pair,
    Section,
    SectionKey,
    SectionRef,
    load_sample,
    model_labeler,
    text_sha256,
)
from lineage.precision import STRATA, stratified, stratum_of

MODEL = model_labeler("claude-opus-5-5", "v1")


def _pair(n: int, stages: tuple[str, ...]) -> Pair:
    return Pair(f"p{n:04d}", stages, SectionKey(f"q{n}", 1), SectionKey(f"m{n}", 2))


def _sample(per_stratum: int = 100) -> list[Pair]:
    return [
        _pair(i * per_stratum + n, stages)
        for i, stages in enumerate(STRATA.values())
        for n in range(per_stratum)
    ]


def test_linear_weighted_kappa_matches_a_hand_count() -> None:
    # Observed weighted disagreement 1/8 against 7/16 by chance: 1 - (1/8)/(7/16) = 5/7.
    assert weighted_kappa([0, 0, 1, 2], [0, 1, 1, 2], 3) == pytest.approx(5 / 7)
    assert weighted_kappa([0, 2, 4], [0, 2, 4], 5) == pytest.approx(1.0)
    assert weighted_kappa([3, 3], [3, 3], 5) is None
    assert weighted_kappa([], [], 5) is None


def test_the_bars_are_the_ones_set_before_labeling() -> None:
    assert (KAPPA_FROM, CUTOFF_AGREEMENT_FROM, PAIRS) == (0.6, 0.85, 50)
    assert allocation(50) == {"near_duplicate_only": 17, "semantic_only": 17, "both": 16}


def test_the_subsample_is_stratified_seeded_and_keeps_the_sample_rows(tmp_path: Path) -> None:
    sample = _sample()

    drawn = draw_subsample(sample)

    assert drawn == draw_subsample(sample)
    assert drawn != draw_subsample(sample, seed=1)
    assert len({p.pair for p in drawn}) == PAIRS
    assert set(drawn) <= set(sample)
    counts = {name: sum(stratum_of(p.stages) == name for p in drawn) for name in STRATA}
    assert counts == allocation(PAIRS)
    # Interleaved, so the order tells nothing of the stage.
    assert len({stratum_of(p.stages) for p in drawn[:10]}) > 1
    path = tmp_path / "agreement.jsonl"
    write_subsample(subsample_rows(drawn), {}, sample_path=path, report_path=tmp_path / "r.json")
    assert load_sample(path) == drawn


def test_a_sample_short_of_a_stratum_is_refused() -> None:
    # 17 near-duplicate-only and semantic-only pairs are wanted; each stratum has 16.
    with pytest.raises(AgreementError, match="near_duplicate_only"):
        draw_subsample(_sample(per_stratum=16))


def _item(pair: Pair) -> Item:
    def section(key: SectionKey) -> Section:
        text = f"{key.bill_id} text"
        return Section(SectionRef(key.bill_id, None, key.ordinal, text_sha256(text)), text)

    return Item(pair, section(pair.query), section(pair.match))


def _label(item: Item, label: int, labeler: str, at: str = "2026-11-11") -> dict[str, object]:
    return {
        "query": item.query.ref.as_dict(),
        "match": item.match.ref.as_dict(),
        "label": label,
        "labeler": labeler,
        "at": at,
    }


def test_agreement_counts_the_pairs_both_labeled_against_the_named_model() -> None:
    items = [_item(_pair(n, stages)) for n, stages in enumerate(STRATA.values())]
    person = "github:someone"
    labels = [
        _label(items[0], 4, person),
        _label(items[0], 3, MODEL),
        _label(items[1], 1, person),
        _label(items[1], 2, MODEL),  # across the cutoff
        _label(items[2], 2, person),
        _label(items[2], 2, MODEL),
        # Another prompt version is another labeler, and is not compared.
        _label(items[1], 0, model_labeler("claude-opus-5-5", "v2"), at="2026-11-12"),
    ]

    got = agreement(items, labels, model_labeler=MODEL)

    assert got["pairs_labeled_by_both"] == 3
    assert got["same_label"] == 1
    assert got["same_side_of_cutoff"] == 2
    assert got["cutoff_agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert got["confusion_person_by_model"][1][2] == 1
    assert got["mean_model_minus_person"] == pytest.approx(0.0)
    assert got["strata"]["semantic_only"] == {
        "pairs": 1, "same_label": 0, "same_side_of_cutoff": 0,
        "person_correct": 0, "model_correct": 1,
    }  # fmt: skip
    assert got["complete"] is True
    assert got["passes"] is False


def test_agreement_passes_only_when_every_pair_is_labeled_and_both_bars_are_met() -> None:
    items = [_item(_pair(n, ("semantic",))) for n in range(5)]
    labels = [_label(i, n % 5, "github:someone") for n, i in enumerate(items)]
    labels += [_label(i, n % 5, MODEL) for n, i in enumerate(items)]

    assert agreement(items, labels, model_labeler=MODEL)["passes"] is True
    partial = agreement(items, labels[:4] + labels[5:], model_labeler=MODEL)
    assert partial["complete"] is False
    assert partial["passes"] is False


def test_each_labelers_precision_is_weighted_on_the_same_pairs() -> None:
    items = [_item(_pair(n, stages)) for n, stages in enumerate(STRATA.values())]
    person = "github:someone"
    labels = [
        _label(items[0], 4, person), _label(items[0], 3, MODEL),
        _label(items[1], 2, person), _label(items[1], 1, MODEL),
        _label(items[2], 2, person), _label(items[2], 2, MODEL),
    ]  # fmt: skip
    sizes = {"near_duplicate_only": 10, "semantic_only": 30, "both": 10}

    got = agreement(items, labels, model_labeler=MODEL, sizes=sizes)["precision_on_these_pairs"]

    # Each stratum holds one pair; the model calls the semantic-only one wrong.
    assert got["person"] == stratified([(10, 1, 1), (30, 1, 1), (10, 1, 1)])
    assert got["model"] == stratified([(10, 1, 1), (30, 1, 0), (10, 1, 1)])
    assert agreement(items, labels, model_labeler=MODEL)["precision_on_these_pairs"] is None
