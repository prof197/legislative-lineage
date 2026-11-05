import math
from typing import Any, cast

import numpy as np
import pytest
from datasketch import MinHash  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from lineage.minhash import (
    NUM_PERM,
    PRIME,
    _splitmix64,  # pyright: ignore[reportPrivateUsage]
    permutations,
    signatures,
    similarity,
    universal_hash,
)

U32 = (1 << 32) - 1


def _u32(values: list[int]) -> NDArray[np.uint32]:
    return np.array(values, dtype=np.uint32)


@pytest.mark.parametrize(
    ("a", "b"),
    [(1, 0), (PRIME - 1, PRIME - 1), (0x1234_5678_9ABC_DEF, 0xFED_CBA9_8765_4321 % PRIME)],
)
def test_universal_hash_is_exact_modular_arithmetic(a: int, b: int) -> None:
    # Edge inputs plus random ones, against Python's arbitrary-precision integers: the product
    # a·x reaches 2^93, so any 64-bit wraparound would show up here.
    xs = [0, 1, 2, U32 - 1, U32, *np.random.default_rng(7).integers(0, U32 + 1, 500).tolist()]

    got = universal_hash(np.array(xs, dtype=np.uint64), np.uint64(a), np.uint64(b))

    assert got.tolist() == [(a * x + b) % PRIME for x in xs]


def test_the_parameter_generator_is_splitmix64() -> None:
    # The reference sequence for seed 0, as published with the algorithm.
    draws = _splitmix64(0)

    assert [next(draws), next(draws)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]


def test_permutations_are_fixed_and_in_range() -> None:
    perms = permutations()

    assert perms.a.shape == perms.b.shape == (NUM_PERM,)
    assert perms.a.min() >= 1
    assert perms.a.max() < PRIME
    assert perms.b.max() < PRIME
    assert len(set(perms.a.tolist())) == NUM_PERM
    assert np.array_equal(permutations().a, perms.a)
    # Pinned: every stored signature depends on these values.
    assert int(perms.a[0]) == 2_150_242_486_686_805_653
    assert int(perms.b[0]) == 643_983_082_913_198_339


def test_a_signature_is_the_per_permutation_minimum_over_the_set() -> None:
    perms = permutations()
    values = [3, 1_000_003, U32]

    (row,) = signatures([_u32(values)], perms)

    expected = [
        min(((int(a) * x + int(b)) % PRIME) & U32 for x in values)
        for a, b in zip(perms.a, perms.b, strict=True)
    ]
    assert row.dtype == np.uint32
    assert row.tolist() == expected


def test_signature_values_are_pinned() -> None:
    (row,) = signatures([_u32([1, 2, 3])], permutations())

    assert row[:4].tolist() == [306_040_367, 559_579_878, 1_623_785_209, 688_720_479]


def test_the_signature_of_a_union_is_the_minimum_of_the_signatures() -> None:
    rng = np.random.default_rng(1)
    a = np.unique(rng.integers(0, U32, 300, dtype=np.uint32))
    b = np.unique(rng.integers(0, U32, 50, dtype=np.uint32))

    sig_a, sig_b, sig_union = signatures([a, b, np.union1d(a, b)], permutations())

    assert np.array_equal(sig_union, np.minimum(sig_a, sig_b))


def test_order_and_repeats_do_not_change_a_signature() -> None:
    perms = permutations()

    first, second = signatures([_u32([5, 9, 2]), _u32([2, 9, 9, 5, 2])], perms)

    assert np.array_equal(first, second)
    assert similarity(first, second) == 1.0


def test_rows_follow_the_input_order() -> None:
    perms = permutations()
    sets = [_u32([1, 2]), _u32([7]), _u32([1, 2, 3, 4])]

    together = signatures(sets, perms)

    for i, s in enumerate(sets):
        assert np.array_equal(together[i], signatures([s], perms)[0])


def test_sets_hashed_in_separate_runs_get_the_same_signatures() -> None:
    # Large enough to cross several runs, with one set longer than a run on its own.
    rng = np.random.default_rng(5)
    sets = [
        np.unique(rng.integers(0, U32, n, dtype=np.uint32))
        for n in (90_000, 300_000, *([7_000] * 40), 3)
    ]
    perms = permutations()

    together = signatures(sets, perms)

    for row, s in zip(together, sets, strict=True):
        x = s.astype(np.uint64)
        reference = [
            int((universal_hash(x, a, b) & np.uint64(U32)).min())
            for a, b in zip(perms.a, perms.b, strict=True)
        ]
        assert row.tolist() == reference


def test_an_empty_set_has_no_signature() -> None:
    with pytest.raises(ValueError, match="empty"):
        signatures([_u32([1]), _u32([])], permutations())


def test_similarity_is_the_share_of_agreeing_positions() -> None:
    x = np.arange(8, dtype=np.uint32)
    y = x.copy()
    y[:2] += 1

    assert similarity(x, y) == 0.75


# The oracle. Both engines sign the same pairs of sets, whose exact Jaccard similarity is known
# by construction, and both must estimate it without bias and with the variance theory
# predicts. They use different hash families, so the test compares estimates, not values.

_PAIRS = 200
_SET_SIZE = 300


def _pairs_with_jaccard(
    target: float, rng: np.random.Generator
) -> list[tuple[NDArray[np.uint32], NDArray[np.uint32], float]]:
    # Two sets of equal size n sharing k elements have Jaccard k / (2n - k).
    shared = round(2 * _SET_SIZE * target / (1 + target))
    pairs: list[tuple[NDArray[np.uint32], NDArray[np.uint32], float]] = []
    while len(pairs) < _PAIRS:
        distinct = np.unique(rng.integers(0, U32, 2 * _SET_SIZE, dtype=np.uint32))
        if distinct.size < 2 * _SET_SIZE - shared:
            continue
        values = rng.permutation(distinct)[: 2 * _SET_SIZE - shared]
        a = values[:_SET_SIZE]
        b = np.concatenate((values[:shared], values[_SET_SIZE:]))
        exact = np.intersect1d(a, b).size / np.union1d(a, b).size
        pairs.append((a, b, exact))
    return pairs


def _datasketch_similarity(a: NDArray[np.uint32], b: NDArray[np.uint32]) -> float:
    def identity(value: int) -> int:
        return value

    # The library's batch path, with its default (current) permutation scheme.
    minhash = cast("Any", MinHash)
    sketches: list[Any] = minhash.bulk(
        [a.tolist(), b.tolist()], num_perm=NUM_PERM, seed=1, hashfunc=identity
    )
    return float(sketches[0].jaccard(sketches[1]))


@pytest.mark.parametrize("target", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_estimates_agree_with_datasketch_and_with_the_exact_jaccard(target: float) -> None:
    rng = np.random.default_rng(int(target * 100))
    pairs = _pairs_with_jaccard(target, rng)
    perms = permutations()
    ours = np.array([similarity(*signatures([a, b], perms)) for a, b, _ in pairs], dtype=np.float64)
    theirs = np.array([_datasketch_similarity(a, b) for a, b, _ in pairs], dtype=np.float64)
    exact = np.array([j for *_, j in pairs], dtype=np.float64)

    # One estimate's standard error is sqrt(J(1 - J) / k) for k permutations.
    sd = math.sqrt(target * (1 - target) / NUM_PERM)
    mean_bound = 4 * sd / math.sqrt(_PAIRS)
    assert abs(float(np.mean(ours - exact))) < mean_bound
    assert abs(float(np.mean(theirs - exact))) < mean_bound
    assert abs(float(np.mean(ours - theirs))) < math.sqrt(2) * mean_bound
    # Too small an error would mean correlated permutations as surely as too large a one.
    for estimates in (ours, theirs):
        rmse = math.sqrt(float(np.mean((estimates - exact) ** 2)))
        assert 0.8 * sd < rmse < 1.2 * sd


def test_identical_and_disjoint_sets_agree_with_datasketch() -> None:
    rng = np.random.default_rng(3)
    a = np.unique(rng.integers(0, U32, 300, dtype=np.uint32))
    b = np.setdiff1d(np.unique(rng.integers(0, U32, 300, dtype=np.uint32)), a)
    perms = permutations()

    assert similarity(*signatures([a, a.copy()], perms)) == 1.0 == _datasketch_similarity(a, a)
    assert similarity(*signatures([a, b], perms)) == 0.0 == _datasketch_similarity(a, b)
