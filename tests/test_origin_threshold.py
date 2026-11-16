"""The origin threshold proposal: each candidate floor's precision over the bands it keeps,
weighted by their sizes, beside its development recall from the same table."""

import json
from pathlib import Path
from typing import Any

import pytest

from lineage.acts import ACTS
from lineage.answer_key import ANSWER_KEY
from lineage.commission import RESULTS
from lineage.labels import SAME_PROVISION
from lineage.origin_precision import BANDS, FLOORS, share_of
from lineage.origin_recall import load_key, matches_from_commission, same_slug
from lineage.origin_threshold import correct_by_pair, proposal

ROOT = Path(__file__).parents[1]
LABELER = "model:claude-opus-5-5@passages-v1-high"


def _sample() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Four labels a band; 1, 2, 3 and then 4 of them correct, from the lowest band up."""
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for b, band in enumerate(BANDS):
        for k in range(4):
            pair = f"o{4 * b + k + 1:04d}"
            stages = ["semantic"] if k % 2 else ["near_duplicate", "semantic"]
            rows.append({"pair": pair, "stratum": band, "stages": stages})
            labels.append(
                {"pair": pair, "label": int(k <= b), "labeler": LABELER, "scale": SAME_PROVISION}
            )
    return rows, labels


def test_each_floor_weighs_the_bands_it_keeps_and_reads_recall_from_the_same_table() -> None:
    rows, labels = _sample()
    v1 = json.loads((ROOT / RESULTS).read_text())["commission"]
    matches = matches_from_commission(v1, lambda act: f"commission/{act}")
    sizes = dict.fromkeys(BANDS, 100)

    doc = proposal(
        rows, labels, sizes, matches, load_key(ROOT / ANSWER_KEY), same_slug(ACTS, "commission")
    )

    floors = {f["floor"]: f for f in doc["floors"]}
    assert list(floors) == list(FLOORS)
    # Equal bands: the mean of 1/4, 2/4, 3/4 and 4/4, then of the bands each floor keeps.
    assert floors[0.0]["precision"]["precision"] == 0.625
    assert floors[0.10]["precision"]["precision"] == 0.875
    assert floors[0.25]["precision"]["precision"] == 1.0
    assert floors[0.05]["band_above"]["correct"] == 2
    assert floors[0.0]["development_recall"]["found"] == 33
    found = [f["development_recall"]["found"] for f in doc["floors"]]
    assert found == sorted(found, reverse=True)
    assert floors[0.25]["matches_listed"] == sum(share_of(m) >= 0.25 for m in matches)
    # By stage, counted unweighted over the floor's labels.
    assert floors[0.0]["by_stage"]["semantic_only"]["labeled"] == 8
    assert doc["labeled"] == doc["sampled"] == 16


def test_labels_under_another_prompt_are_not_read_and_a_pair_labeled_twice_is_refused() -> None:
    rows, labels = _sample()
    other = [r | {"labeler": "model:claude-opus-5-5@v1"} for r in labels]
    assert correct_by_pair(other) == {}
    with pytest.raises(ValueError, match="labeled twice"):
        correct_by_pair(labels + labels[:1])
    # A sample stratified by stage is not a tuning sample.
    with pytest.raises(ValueError, match="not the bands"):
        proposal(rows, labels, {"both": 1}, [], load_key(ROOT / ANSWER_KEY), {})
