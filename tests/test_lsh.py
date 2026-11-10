import numpy as np
import pytest
from numpy.typing import NDArray

from lineage.lsh import (
    BANDS,
    ROWS,
    band_keys,
    build_band_index,
    candidate_pairs,
    candidate_probability,
    cap_sweep,
    threshold,
)
from lineage.minhash import NUM_PERM, permutations, signatures

_MASK64 = (1 << 64) - 1


def _mix(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _reference_key(band: list[int]) -> int:
    v0, v1, v2, v3 = band
    return _mix(_mix(v0 | v1 << 32) ^ (v2 | v3 << 32))


def _random_signatures(n: int, seed: int) -> NDArray[np.uint32]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << 32, size=(n, NUM_PERM), dtype=np.uint32)


def _pairs(candidates_left: NDArray[np.uint32], right: NDArray[np.uint32]) -> set[tuple[int, int]]:
    return set(zip(candidates_left.tolist(), right.tolist(), strict=True))


def test_bands_cover_the_signature() -> None:
    assert BANDS * ROWS == NUM_PERM


def test_band_keys_match_the_reference_mix() -> None:
    values = _random_signatures(5, seed=1)
    values[0, :4] = [0, 0, 0, 0]
    values[1, :4] = [0xFFFF_FFFF] * 4

    keys = band_keys(values)

    assert keys.shape == (5, BANDS)
    assert keys.dtype == np.uint64
    for i in range(5):
        for b in range(BANDS):
            expected = _reference_key(values[i, b * ROWS : (b + 1) * ROWS].tolist())
            assert int(keys[i, b]) == expected


def test_band_keys_are_pinned() -> None:
    # Every stored index depends on these values. A change to the mix must rebuild the index.
    values = np.arange(NUM_PERM, dtype=np.uint32).reshape(1, NUM_PERM)

    assert band_keys(values)[0, :2].tolist() == [8331255074758805912, 7110555104232845293]


def test_equal_bands_share_a_key_and_any_change_moves_it() -> None:
    values = _random_signatures(2, seed=2)
    values[1, 8:12] = values[0, 8:12]

    keys = band_keys(values)

    assert keys[0, 2] == keys[1, 2]
    assert np.count_nonzero(keys[0] == keys[1]) == 1
    values[1, 11] ^= 1
    assert band_keys(values)[0, 2] != band_keys(values)[1, 2]


def test_the_index_sorts_each_band_and_remembers_the_rows() -> None:
    values = _random_signatures(50, seed=3)
    keys = band_keys(values)

    index = build_band_index(values)

    assert index.keys.shape == index.rows.shape == (BANDS, 50)
    assert index.rows.dtype == np.uint32
    for b in range(BANDS):
        assert np.all(index.keys[b, 1:] >= index.keys[b, :-1])
        assert np.array_equal(index.keys[b], keys[index.rows[b], b])
        assert sorted(index.rows[b].tolist()) == list(range(50))


def _share_band(values: NDArray[np.uint32], i: int, j: int, band: int) -> None:
    values[j, band * ROWS : (band + 1) * ROWS] = values[i, band * ROWS : (band + 1) * ROWS]


def test_a_pair_is_a_candidate_when_any_band_matches() -> None:
    values = _random_signatures(5, seed=4)
    _share_band(values, 0, 1, band=0)
    _share_band(values, 0, 2, band=31)
    for band in range(BANDS):  # rows 3 and 4 are identical: one pair, not 32
        _share_band(values, 3, 4, band)

    found = candidate_pairs(build_band_index(values), cap=100)

    assert _pairs(found.left, found.right) == {(0, 1), (0, 2), (3, 4)}
    assert np.all(found.left < found.right)
    assert found.buckets_over_cap == 0


def test_candidates_come_back_sorted_and_distinct() -> None:
    values = _random_signatures(40, seed=5)
    rng = np.random.default_rng(5)
    for _ in range(60):
        i, j = rng.choice(40, size=2, replace=False)
        _share_band(values, int(i), int(j), band=int(rng.integers(BANDS)))

    found = candidate_pairs(build_band_index(values), cap=100)

    codes = found.left.astype(np.uint64) << np.uint64(32) | found.right
    assert np.all(codes[1:] > codes[:-1])


def test_a_pair_shared_by_many_bands_is_one_candidate() -> None:
    values = _random_signatures(30, seed=8)
    for i in range(0, 30, 3):  # ten triples identical in every band: 3 pairs each, 32 times over
        for band in range(BANDS):
            _share_band(values, i, i + 1, band)
            _share_band(values, i, i + 2, band)
    # Row 4 takes the first triple's band 7, so it pairs with all three of them in that band.
    _share_band(values, 1, 4, band=7)
    index = build_band_index(values)
    keys = band_keys(values)
    expected = sorted(
        (a, b) for a in range(30) for b in range(a + 1, 30) if np.any(keys[a] == keys[b])
    )

    found = candidate_pairs(index, cap=100)

    assert list(zip(found.left.tolist(), found.right.tolist(), strict=True)) == expected
    assert len(expected) == 33


def test_a_bucket_over_the_cap_is_skipped_and_counted() -> None:
    values = _random_signatures(12, seed=6)
    for j in range(1, 5):  # a bucket of 5 in band 0
        _share_band(values, 0, j, band=0)
    for j in range(6, 9):  # a bucket of 4 in band 1, within the cap
        _share_band(values, 5, j, band=1)

    found = candidate_pairs(build_band_index(values), cap=4)

    assert _pairs(found.left, found.right) == {
        (a, b) for a in range(5, 9) for b in range(5, 9) if a < b
    }
    assert found.buckets_over_cap == 1
    assert found.rows_in_capped_buckets == 5
    assert found.pairs_in_capped_buckets == 10
    assert found.largest_bucket == 5


def test_a_capped_pair_is_still_found_through_another_band() -> None:
    values = _random_signatures(6, seed=7)
    for j in range(1, 6):
        _share_band(values, 0, j, band=0)
    _share_band(values, 1, 2, band=9)

    found = candidate_pairs(build_band_index(values), cap=3)

    assert _pairs(found.left, found.right) == {(1, 2)}


def test_cap_sweep_counts_the_cross_group_pairs_each_cap_loses() -> None:
    values = _random_signatures(9, seed=8)
    # Band 0: a bucket of six, rows 0-5, three in group 0 and three in group 1.
    for j in range(1, 6):
        _share_band(values, 0, j, band=0)
    # Rows 0 and 3 (different groups) also share band 4, in a bucket of two.
    _share_band(values, 0, 3, band=4)
    # Band 2: a bucket of three, all in group 2.
    for j in (7, 8):
        _share_band(values, 6, j, band=2)
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)

    sweep = cap_sweep(build_band_index(values), groups, caps=[2, 5, 6])

    # Cap 2 skips both large buckets. Of band 0's nine cross-group pairs, (0, 3) survives
    # through band 4, so eight are lost; band 2's bucket is single-group and loses none.
    assert sweep[2].buckets_over_cap == 2
    assert sweep[2].multi_group_buckets_over_cap == 1
    assert sweep[2].most_groups_in_a_capped_bucket == 2
    assert sweep[2].cross_group_pairs_lost == 8
    assert sweep[5].buckets_over_cap == 1
    assert sweep[5].cross_group_pairs_lost == 8
    assert sweep[6].buckets_over_cap == 0
    assert sweep[6].cross_group_pairs_lost == 0


@pytest.mark.parametrize(
    ("s", "expected"),
    [(0.0, 0.0), (1.0, 1.0), (0.5, 1 - (1 - 0.5**4) ** 32)],
)
def test_candidate_probability_is_the_s_curve(s: float, expected: float) -> None:
    assert candidate_probability(s) == pytest.approx(expected)


def _pair_with_jaccard(
    rng: np.random.Generator, s: float, union: int
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    common = round(s * union)
    each = (union - common) // 2
    values = rng.choice(1 << 32, size=common + 2 * each, replace=False).astype(np.uint32)
    shared, a, b = values[:common], values[common : common + each], values[common + each :]
    return np.concatenate([shared, a]), np.concatenate([shared, b])


# The S-curve assumes each band's rows are independent permutations. The hash family only
# approximates that, so the empirical rate of real signatures is what this checks.
@pytest.mark.parametrize("s", [0.2, 0.3, 0.4, 0.5, 0.6])
def test_the_candidate_rate_follows_the_s_curve(s: float) -> None:
    pairs = 500
    rng = np.random.default_rng(int(s * 100))
    sets: list[NDArray[np.uint32]] = []
    for _ in range(pairs):
        sets += _pair_with_jaccard(rng, s, union=200)
    values = signatures(sets, permutations())

    found = candidate_pairs(build_band_index(values), cap=1_000)

    # Rows 2k and 2k + 1 are the constructed pair; random sets collide in no band.
    same_pair = (found.left % 2 == 0) & (found.right == found.left + 1)
    assert np.all(same_pair)
    p = candidate_probability(s)
    sd = np.sqrt(p * (1 - p) / pairs)
    assert abs(found.left.size / pairs - p) < 4 * sd


# For the same 128 values, more bands lower the threshold and more rows raise it, and a pair at
# a setting's threshold is proposed with about the same chance, 1 - (1 - 1/b)^b, near 1 - 1/e.
@pytest.mark.parametrize(
    ("bands", "rows", "expected"), [(16, 8, 0.7071), (32, 4, 0.4204), (64, 2, 0.125)]
)
def test_bands_and_rows_move_the_threshold(bands: int, rows: int, expected: float) -> None:
    t = threshold(bands, rows)

    assert t == pytest.approx(expected, abs=1e-4)
    assert candidate_probability(t, bands, rows) == pytest.approx(1 - (1 - 1 / bands) ** bands)
    assert 0.63 < candidate_probability(t, bands, rows) < 0.65
