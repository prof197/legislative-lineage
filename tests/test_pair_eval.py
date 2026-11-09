"""Scoring the stages on labeled pairs: the near-duplicate scorer, thresholds and metrics."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from lineage.pair_data import Pair
from lineage.pair_eval import (
    best_cosines,
    near_duplicate_scores,
    precision_recall,
    roc_auc,
    threshold_at_precision,
)

TEXT = (
    "the department of revenue shall publish the rates for each fiscal year and the "
    "commissioner may adopt rules to carry out this section after notice and hearing"
)


def test_a_copied_subsection_is_found_and_an_unrelated_one_is_not() -> None:
    other = "a person who operates a motor vehicle without a license commits a misdemeanor"
    pairs = [Pair(TEXT, TEXT, 4), Pair(TEXT, other, 0), Pair("effective date", TEXT, 0)]

    got = near_duplicate_scores(pairs, np.empty(0, dtype=np.uint32))

    assert (got[0].agreements, got[0].proposed) == (128, True)
    assert got[1].agreements < 64
    assert (got[2].agreements, got[2].proposed, got[2].signed) == (0, False, False)  # too short
    assert got[0].signed
    assert got[1].signed


def test_precision_and_recall_at_a_threshold() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    positive = np.array([True, True, False, True, False])

    assert precision_recall(scores, positive, 0.6) == (0.75, 1.0)
    assert precision_recall(scores, positive, 0.95) == (1.0, 0.0)


def test_roc_auc_counts_ties_as_half() -> None:
    positive = np.array([True, False, True, False])

    assert roc_auc(np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32), positive) == 1.0
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), positive) == 0.5


def test_the_threshold_is_the_lowest_score_that_keeps_the_precision() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    positive = np.array([True, True, False, True, False])

    assert threshold_at_precision(scores, positive, 0.75) == np.float32(0.6)
    assert threshold_at_precision(scores, positive, 1.0) == np.float32(0.8)
    assert threshold_at_precision(scores, ~positive, 1.0) is None


def _by_first_word(texts: Sequence[str]) -> NDArray[np.float32]:
    # A unit vector per window, along an axis named by its first word.
    axes = {"alpha": 0, "beta": 1, "gamma": 2}
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i, axes[text.split()[0]]] = 1.0
    return out


def test_a_pair_scores_its_best_cosine_between_windows_of_94_words() -> None:
    alpha = " ".join(["alpha"] * 94)
    beta = " ".join(["beta"] * 94)
    gamma = " ".join(["gamma"] * 94)
    seen: list[list[str]] = []

    def embed(texts: Sequence[str]) -> NDArray[np.float32]:
        seen.append(list(texts))
        return _by_first_word(texts)

    pairs = [Pair(alpha + " " + beta, beta, 3), Pair(alpha, gamma, 0), Pair(beta, beta, 4)]

    assert best_cosines(pairs, embed).tolist() == [1.0, 0.0, 1.0]
    # The 188-word text is two windows without overlap, and each distinct window is embedded once.
    assert len(seen) == 1
    assert sorted(seen[0]) == [alpha, beta, gamma]
