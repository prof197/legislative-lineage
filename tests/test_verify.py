import numpy as np
import pytest

from lineage.minhash import NUM_PERM
from lineage.verify import agreements, min_agreements


def test_counts_agreeing_positions_per_pair() -> None:
    values = np.zeros((3, NUM_PERM), dtype=np.uint32)
    values[1, :32] = 7  # row 1 differs from row 0 at 32 positions
    values[2] = 9  # row 2 agrees with neither

    counts = agreements(
        values, np.array([0, 0, 1], dtype=np.uint32), np.array([1, 2, 2], dtype=np.uint32)
    )

    assert counts.tolist() == [NUM_PERM - 32, 0, 0]
    assert counts.dtype == np.uint8


def test_rejects_signatures_of_another_width() -> None:
    with pytest.raises(ValueError, match="columns"):
        agreements(
            np.zeros((2, 64), dtype=np.uint32), np.array([0], np.uint32), np.array([1], np.uint32)
        )


@pytest.mark.parametrize(("threshold", "count"), [(0.5, 64), (0.501, 65), (1.0, 128), (0.0, 0)])
def test_min_agreements_is_the_smallest_count_that_reaches_the_threshold(
    threshold: float, count: int
) -> None:
    assert min_agreements(threshold) == count
