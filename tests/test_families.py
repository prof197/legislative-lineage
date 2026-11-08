import numpy as np
import pytest

from lineage.families import FAMILY_SHARE, bill_pairs, components, families


def _rows(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.uint32)


def test_shares_count_each_section_once_in_both_directions() -> None:
    # Bill 0 has rows 0-3, bill 1 rows 4-5, bill 2 row 6.
    bill_of_section = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int32)
    # Row 0 matches both of bill 1's rows; row 1 matches row 4. Rows 2 and 3 match nothing.
    left, right = _rows(0, 0, 1, 2), _rows(4, 5, 4, 3)

    pairs = bill_pairs(bill_of_section, left, right)

    # The pair inside bill 0 (rows 2 and 3) is ignored.
    assert pairs.first.tolist() == [0]
    assert pairs.second.tolist() == [1]
    assert pairs.first_share.tolist() == [0.5]  # rows 0 and 1 of four
    assert pairs.second_share.tolist() == [1.0]  # rows 4 and 5 of two


def test_a_pair_listed_either_way_round_gives_the_same_shares() -> None:
    bill_of_section = np.array([0, 1, 1], dtype=np.int32)

    forward = bill_pairs(bill_of_section, _rows(0), _rows(1))
    backward = bill_pairs(bill_of_section, _rows(1), _rows(0))

    assert (forward.first_share.tolist(), forward.second_share.tolist()) == ([1.0], [0.5])
    assert (backward.first_share.tolist(), backward.second_share.tolist()) == ([1.0], [0.5])


@pytest.mark.parametrize(
    ("n", "edges", "expected"),
    [
        (5, [], [0, 1, 2, 3, 4]),
        (5, [(3, 4), (1, 2)], [0, 1, 1, 3, 3]),
        # A chain listed from its far end still resolves to its smallest node.
        (6, [(4, 5), (3, 4), (2, 3), (1, 2)], [0, 1, 1, 1, 1, 1]),
        (4, [(0, 3), (2, 3), (1, 2)], [0, 0, 0, 0]),
    ],
)
def test_components_are_labelled_by_their_smallest_node(
    n: int, edges: list[tuple[int, int]], expected: list[int]
) -> None:
    first = np.array([a for a, _ in edges], dtype=np.int32)
    second = np.array([b for _, b in edges], dtype=np.int32)

    assert components(n, first, second).tolist() == expected


def test_one_shared_clause_does_not_join_two_bills() -> None:
    # Bills 0 and 1 have four sections each and share one, a clause such as an effective date.
    bill_of_section = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

    assert families(2, bill_of_section, _rows(3), _rows(7)).tolist() == [0, 1]


def test_an_omnibus_bill_does_not_absorb_a_small_bill_it_contains() -> None:
    # Bill 1's only section is one of bill 0's five: a share of 1 one way, 0.2 the other.
    bill_of_section = np.array([0, 0, 0, 0, 0, 1], dtype=np.int32)

    assert families(2, bill_of_section, _rows(2), _rows(5)).tolist() == [0, 1]


def test_companion_bills_and_a_refiling_form_one_family() -> None:
    # Bills 0, 1 and 2 have the same two sections; bill 2 matches bill 1 only.
    bill_of_section = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    left, right = _rows(0, 1, 2, 3), _rows(2, 3, 4, 5)

    assert families(3, bill_of_section, left, right).tolist() == [0, 0, 0]


def test_the_share_is_required_in_both_directions_at_the_threshold() -> None:
    # Five sections each, four shared: a share of 0.8 both ways, exactly the threshold.
    bill_of_section = np.array([0] * 5 + [1] * 5, dtype=np.int32)
    left, right = _rows(0, 1, 2, 3), _rows(5, 6, 7, 8)

    assert FAMILY_SHARE == 0.8
    assert families(2, bill_of_section, left, right).tolist() == [0, 0]
    assert families(2, bill_of_section, left[:3], right[:3]).tolist() == [0, 1]
