"""Intervals for a share measured on a sample."""

import pytest

from lineage.intervals import weighted_share, wilson


def test_the_wilson_interval_stays_inside_zero_to_one() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(5, 5)
    assert 0.0 < lo < hi == 1.0


def test_the_weighted_share_is_weighted_hits_over_total_weight_and_cannot_pass_one() -> None:
    weights = [5.0, 1.0, 1.0, 2.5]
    hits = [True, False, True, True]

    share, (lo, hi) = weighted_share(weights, hits)

    assert share is not None
    assert share == pytest.approx(8.5 / 9.5)
    assert 0.0 <= lo <= share <= hi <= 1.0
    assert weighted_share(weights, hits) == (share, (lo, hi))  # seeded
    assert weighted_share([], []) == (None, (0.0, 1.0))
