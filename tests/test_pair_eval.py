"""Scoring the stages on labeled pairs: the near-duplicate scorer, thresholds and metrics."""

import numpy as np

from lineage.pair_data import Pair
from lineage.pair_eval import near_duplicate_scores, precision_recall, roc_auc

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
