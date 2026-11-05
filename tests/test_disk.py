import pytest

from lineage.disk import InsufficientDiskSpaceError, check_free_space

GiB = 1 << 30


def test_allows_a_write_that_leaves_the_floor_intact() -> None:
    check_free_space(free=40 * GiB, needed=7 * GiB, floor=20 * GiB)


def test_allows_a_write_that_lands_exactly_on_the_floor() -> None:
    check_free_space(free=27 * GiB, needed=7 * GiB, floor=20 * GiB)


def test_refuses_a_write_that_would_cross_the_floor() -> None:
    with pytest.raises(InsufficientDiskSpaceError, match=r"would leave 13\.0 GiB"):
        check_free_space(free=20 * GiB, needed=7 * GiB, floor=20 * GiB)
