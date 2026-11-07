"""LSH banding: candidate pairs from signatures without comparing every pair.

Each signature is cut into BANDS bands of ROWS values, and each band is hashed to one 64-bit
key. Two sections are candidates when any band's keys are equal. Buckets of equal keys are
found by sorting each band's keys, not by a dict, so the index is a few dense arrays whose
memory is known in advance. A bucket larger than the cap is boilerplate the stoplist missed;
it yields no pairs, and the counters record what was skipped. See DESIGN.md#dr-3.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lineage.minhash import NUM_PERM

BANDS = 32
ROWS = 4
BUCKET_CAP = 1000

_U27 = np.uint64(27)
_U30 = np.uint64(30)
_U31 = np.uint64(31)
_U32 = np.uint64(32)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_LOW32 = np.uint64(0xFFFF_FFFF)
# Pairs gathered per pass when comparing band keys row by row.
_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class BandIndex:
    keys: NDArray[np.uint64]  # (BANDS, n): each band's keys, sorted
    rows: NDArray[np.uint32]  # (BANDS, n): the signature row behind each key


@dataclass(frozen=True, slots=True)
class Candidates:
    left: NDArray[np.uint32]  # signature rows, left < right, pairs sorted and distinct
    right: NDArray[np.uint32]
    buckets_over_cap: int  # summed over bands, as are the next two
    sections_in_capped_buckets: int
    pairs_in_capped_buckets: int
    largest_bucket: int


@dataclass(frozen=True, slots=True)
class CapEffect:
    buckets_over_cap: int
    multi_group_buckets_over_cap: int
    most_groups_in_a_capped_bucket: int
    cross_group_pairs_lost: int  # candidates without the cap that the cap removes


def _mix(z: NDArray[np.uint64]) -> NDArray[np.uint64]:
    # SplitMix64's finalizer. Multiplication wraps modulo 2^64, which is intended here.
    z = (z ^ (z >> _U30)) * _M1
    z = (z ^ (z >> _U27)) * _M2
    return z ^ (z >> _U31)


def _band_key(values: NDArray[np.uint32], band: int) -> NDArray[np.uint64]:
    w = values[:, band * ROWS : (band + 1) * ROWS].astype(np.uint64)
    # Equal keys for unequal bands only add a candidate, which verification removes. Among
    # two million sections, a band expects about n^2 / 2^65 < 10^-6 such collisions.
    return _mix(_mix(w[:, 0] | (w[:, 1] << _U32)) ^ (w[:, 2] | (w[:, 3] << _U32)))


def band_keys(values: NDArray[np.uint32]) -> NDArray[np.uint64]:
    """One 64-bit key per band of each signature: shape (n, BANDS)."""
    if values.shape[1] != NUM_PERM:
        raise ValueError(f"signatures must have {NUM_PERM} values")
    return np.stack([_band_key(values, b) for b in range(BANDS)], axis=1)


def build_band_index(values: NDArray[np.uint32]) -> BandIndex:
    n = values.shape[0]
    keys = np.empty((BANDS, n), dtype=np.uint64)
    rows = np.empty((BANDS, n), dtype=np.uint32)
    for b in range(BANDS):
        k = _band_key(values, b)
        # A stable sort keeps rows with equal keys in row order, so the stored index is
        # reproducible byte for byte.
        order = np.argsort(k, kind="stable")
        keys[b] = k[order]
        rows[b] = order
    return BandIndex(keys, rows)


def candidate_probability(s: float, bands: int = BANDS, rows: int = ROWS) -> float:
    """The S-curve: the chance that a pair with Jaccard similarity s shares a band."""
    return 1 - (1 - s**rows) ** bands


def threshold(bands: int = BANDS, rows: int = ROWS) -> float:
    """The similarity near the S-curve's steepest point."""
    return (1 / bands) ** (1 / rows)


def _buckets(keys: NDArray[np.uint64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Start and size of each run of equal keys in one sorted band."""
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return starts, np.diff(np.r_[starts, keys.size])


def _pairs_of_size(
    rows: NDArray[np.uint32], starts: NDArray[np.int64], size: int
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Every pair within each bucket of `size` members starting at `starts`, left < right."""
    members = np.sort(rows[starts[:, None] + np.arange(size)], axis=1)
    i, j = np.triu_indices(size, 1)
    return members[:, i].ravel(), members[:, j].ravel()


def _codes(left: NDArray[np.uint32], right: NDArray[np.uint32]) -> NDArray[np.uint64]:
    return (left.astype(np.uint64) << _U32) | right


def _distinct(codes: NDArray[np.uint64]) -> NDArray[np.uint64]:
    codes.sort()
    return codes[np.r_[True, codes[1:] != codes[:-1]]] if codes.size else codes


def candidate_pairs(index: BandIndex, *, cap: int = BUCKET_CAP) -> Candidates:
    """Every pair that shares a band key in a bucket of at most `cap` sections."""
    buckets = [_buckets(index.keys[b]) for b in range(BANDS)]
    # The pair count is known before any pair is formed, so one array of exactly that size is
    # filled in place; collecting per-band arrays and joining them would hold two copies.
    pairs = sum(int((s[s <= cap] * (s[s <= cap] - 1) // 2).sum()) for _, s in buckets)
    codes = np.empty(pairs, dtype=np.uint64)
    filled = over = sections = skipped = largest = 0
    for b, (starts, sizes) in enumerate(buckets):
        capped = sizes[sizes > cap]
        over += capped.size
        sections += int(capped.sum())
        skipped += int((capped * (capped - 1) // 2).sum())
        largest = max(largest, int(sizes.max()) if sizes.size else 0)
        kept = (sizes >= 2) & (sizes <= cap)
        # Buckets of one size form one matrix, so pairs come from a few array operations
        # per distinct size rather than a loop per bucket.
        for size in np.unique(sizes[kept]).tolist():
            part = _codes(*_pairs_of_size(index.rows[b], starts[sizes == size], size))
            codes[filled : filled + part.size] = part
            filled += part.size
    codes = _distinct(codes)
    left = (codes >> _U32).astype(np.uint32)
    right = (codes & _LOW32).astype(np.uint32)
    return Candidates(left, right, over, sections, skipped, largest)


def cap_sweep(
    index: BandIndex, groups: NDArray[np.int32], caps: Sequence[int]
) -> dict[int, CapEffect]:
    """What each cap would skip, and the cross-group candidates it would lose for good.

    `groups` labels each signature row (a state, say). A pair in a capped bucket is lost only
    if no other band puts it in a bucket within the cap, so each pair is checked across all
    of its shared bands.
    """
    lowest = min(caps)
    n = index.keys.shape[1]
    key_of = np.empty((n, BANDS), dtype=np.uint64)
    size_of = np.empty((n, BANDS), dtype=np.int32)
    over = dict.fromkeys(caps, 0)
    multi = dict.fromkeys(caps, 0)
    most = dict.fromkeys(caps, 0)
    parts: list[NDArray[np.uint64]] = []
    for b in range(BANDS):
        starts, sizes = _buckets(index.keys[b])
        key_of[index.rows[b], b] = index.keys[b]
        size_of[index.rows[b], b] = np.repeat(sizes, sizes)
        for start, size in zip(
            starts[sizes > lowest].tolist(), sizes[sizes > lowest].tolist(), strict=True
        ):
            labels = np.unique(groups[index.rows[b][start : start + size]]).size
            for cap in caps:
                if size > cap:
                    over[cap] += 1
                    multi[cap] += labels > 1
                    most[cap] = max(most[cap], labels)
            if labels > 1:
                left, right = _pairs_of_size(index.rows[b], np.array([start]), size)
                cross = groups[left] != groups[right]
                parts.append(_codes(left[cross], right[cross]))
    codes = _distinct(np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64))
    # The smallest bucket each pair shares in any band decides whether a cap keeps it.
    smallest = np.empty(codes.size, dtype=np.int32)
    for i in range(0, codes.size, _CHUNK):
        chunk = codes[i : i + _CHUNK]
        left, right = (chunk >> _U32).astype(np.intp), (chunk & _LOW32).astype(np.intp)
        shared = key_of[left] == key_of[right]
        smallest[i : i + _CHUNK] = np.where(shared, size_of[left], np.iinfo(np.int32).max).min(1)
    return {
        cap: CapEffect(over[cap], multi[cap], most[cap], int(np.count_nonzero(smallest > cap)))
        for cap in caps
    }
