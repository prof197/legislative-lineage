"""The threshold proposal: a page's precision under each option, from one drawn relative a page."""

from typing import Any

import pytest

from lineage.semantic import THRESHOLD
from lineage.threshold import (
    CANDIDATES,
    Listed,
    ThresholdError,
    at_threshold,
    by_meaning_only,
    near_duplicate_only,
    page_precision,
    proposal,
)

# One page's whole list: (found by the near-duplicate stage, best cosine by meaning, correct).
PAGE = [
    (True, None, True),
    (True, 0.95, True),
    (False, 0.89, False),
    (False, 0.91, True),
    (False, 0.93, False),
    (False, 0.97, True),
]


def _drawn(i: int) -> Listed:
    near, cosine, _ = PAGE[i]
    by_meaning = tuple(sorted(c for n, c, _ in PAGE if not n and c is not None))
    return Listed(len(PAGE), near, cosine, by_meaning)


@pytest.mark.parametrize("t", [THRESHOLD, 0.90, 0.92, 0.94, 0.96, 0.98])
def test_the_weighted_draw_averages_to_the_pages_precision_under_each_option(t: float) -> None:
    for keep, kept in (
        (at_threshold(t), [n or (c is not None and c >= t) for n, c, _ in PAGE]),
        (by_meaning_only(t), [not n and c is not None and c >= t for n, c, _ in PAGE]),
        (near_duplicate_only, [n for n, _, _ in PAGE]),
    ):
        right = [r for (_, _, r), k in zip(PAGE, kept, strict=True) if k]
        if not right:
            assert page_precision([_drawn(0)], [True], keep)["pages_emptied"] == 1
            continue
        # Every relative the page could have drawn, equally likely.
        values = [
            page_precision([_drawn(i)], [PAGE[i][2]], keep)["precision"] for i in range(len(PAGE))
        ]
        assert sum(values) / len(values) == pytest.approx(sum(right) / len(right), abs=1e-3)


def test_at_the_builds_threshold_the_estimate_is_the_share_labeled_correct() -> None:
    pages = [_drawn(i) for i in range(len(PAGE))]
    correct: list[bool | None] = [r for _, _, r in PAGE]

    got = page_precision(pages, correct, at_threshold(0.89))

    assert got["precision"] == pytest.approx(4 / 6, abs=1e-4)
    low, high = got["interval_95"]
    assert low <= got["precision"] <= high
    assert got == page_precision(pages, correct, at_threshold(0.89))  # seeded


def test_an_emptied_page_and_an_unlabeled_one_are_counted_apart() -> None:
    only_meaning = Listed(2, False, 0.90, (0.90, 0.91))

    got = page_precision([only_meaning, _drawn(0)], [True, None], at_threshold(0.95))

    assert (got["pages_emptied"], got["pages_unlabeled"], got["pages_listing"]) == (1, 1, 1)
    assert got["precision"] is None


def _pooled(n: int) -> dict[str, Any]:
    return {"pooled": {"all": {"enactments": 36, "found": n}}}


def _commission(found: dict[float, int]) -> dict[str, Any]:
    pooled = _pooled
    return {
        "recall_by_semantic_threshold": [{"threshold": t, "found": n} for t, n in found.items()],
        "stage_ablation": {"near_duplicate": pooled(31), "both": pooled(33)},
    }


def test_the_proposal_needs_recall_at_every_option() -> None:
    with pytest.raises(ThresholdError, match="no recall"):
        proposal([], [], {}, _commission({THRESHOLD: 33}), labeler="model:x@y")

    got = proposal([], [], {}, _commission(dict.fromkeys(CANDIDATES, 33)), labeler="model:x@y")

    assert [o.get("threshold") for o in got["options"]][: len(CANDIDATES)] == list(CANDIDATES)
    assert got["options"][-2]["development_recall"]["found"] == 31
    assert got["options"][-1]["development_recall"]["on_the_page"]["found"] == 33
