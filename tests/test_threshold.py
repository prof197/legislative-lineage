"""The threshold proposal: the share correct of the sampled relatives each option keeps."""

from typing import Any

import pytest

from lineage.intervals import wilson
from lineage.semantic import BUILD_THRESHOLD
from lineage.threshold import (
    CANDIDATES,
    Listed,
    ThresholdError,
    at_threshold,
    by_band,
    by_meaning_only,
    near_duplicate_only,
    page_precision,
    proposal,
)

# Six sampled pages, each with its drawn relative: (found by the near-duplicate stage, its best
# cosine by meaning, correct), and the best cosines of what each page lists by meaning alone.
DRAWN = [
    (True, None, True, (0.89,)),
    (True, 0.95, True, ()),
    (False, 0.8906, False, (0.8906, 0.95)),
    (False, 0.912, True, (0.912,)),
    (False, 0.931, False, (0.931, 0.89)),
    (False, 0.97, True, (0.97, 0.90)),
]
PAGES = [Listed(len(m) + n, n, c, m) for n, c, _, m in DRAWN]
CORRECT: list[bool | None] = [r for _, _, r, _ in DRAWN]


@pytest.mark.parametrize("t", CANDIDATES)
def test_an_option_counts_the_sampled_relatives_that_clear_it(t: float) -> None:
    kept = [n or (c is not None and c >= t) for n, c, _, _ in DRAWN]
    right = [r for (_, _, r, _), k in zip(DRAWN, kept, strict=True) if k]

    got = page_precision(PAGES, CORRECT, at_threshold(t))

    assert (got["labeled"], got["correct"]) == (len(right), sum(right))
    assert got["interval_95"] == [round(v, 4) for v in wilson(sum(right), len(right))]
    emptied = sum(not n and all(m < t for m in ms) for n, _, _, ms in DRAWN)
    assert got["pages_emptied"] == emptied


def test_listing_apart_splits_the_sample_by_the_stage_that_found_each_relative() -> None:
    wording = page_precision(PAGES, CORRECT, near_duplicate_only)
    meaning = page_precision(PAGES, CORRECT, by_meaning_only(BUILD_THRESHOLD))

    assert (wording["labeled"], wording["correct"]) == (2, 2)
    assert (meaning["labeled"], meaning["correct"]) == (4, 2)
    # Pages listing only relatives found by meaning have no list found by wording.
    assert wording["pages_emptied"] == 4


def test_an_unlabeled_relative_is_counted_and_left_out() -> None:
    got = page_precision(PAGES, [None, *CORRECT[1:]], at_threshold(BUILD_THRESHOLD))

    assert (got["labeled"], got["unlabeled"]) == (5, 1)


def test_the_bands_cover_every_labeled_relative_once() -> None:
    bands = by_band(PAGES, CORRECT)

    assert sum(b["labeled"] for b in bands) == len(DRAWN)
    assert bands[-1] == {**bands[-1], "band": "by wording", "labeled": 2, "correct": 2}
    assert [b["labeled"] for b in bands[:-1]] == [1, 0, 1, 0, 1, 0, 0, 0, 1]


def _pooled(n: int) -> dict[str, Any]:
    return {"pooled": {"all": {"enactments": 36, "found": n}}}


def _commission(found: dict[float, int]) -> dict[str, Any]:
    return {
        "recall_by_semantic_threshold": [{"threshold": t, "found": n} for t, n in found.items()],
        "stage_ablation": {"near_duplicate": _pooled(31), "both": _pooled(33)},
    }


def test_the_proposal_needs_recall_at_every_option() -> None:
    with pytest.raises(ThresholdError, match="no recall"):
        proposal([], [], {}, _commission({BUILD_THRESHOLD: 33}), labeler="model:x@y")

    got = proposal([], [], {}, _commission(dict.fromkeys(CANDIDATES, 33)), labeler="model:x@y")

    assert [o.get("threshold") for o in got["options"]][: len(CANDIDATES)] == list(CANDIDATES)
    assert got["options"][-2]["development_recall"]["found"] == 31
    assert got["options"][-1]["development_recall"]["on_the_page"]["found"] == 33
