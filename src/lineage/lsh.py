"""LSH banding: candidate pairs from signatures without comparing every pair.

Each signature is cut into BANDS bands of ROWS values, and each band is hashed to one 64-bit
key. Two sections are candidates when any band's keys are equal. Buckets of equal keys are
found by sorting each band's keys, not by a dict, so the index is a few dense arrays whose
memory is known in advance. A bucket larger than the cap is boilerplate the stoplist missed;
it yields no pairs, and the counters record what was skipped. See DESIGN.md#dr-3.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lineage.minhash import NUM_PERM

BANDS = 32
ROWS = 4
# The smallest cap in the sweep (lineage.bands.SWEEP_CAPS) that removes cross-state candidates
# from no more than one bucket (reports/index.json, lsh.cap_sweep). Signed by window, one
# interstate compact's sections fill buckets of over 1,000 windows in 47 to 49 states.
BUCKET_CAP = 2000

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
    rows_in_capped_buckets: int
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
    # 16 million windows, a band expects about n^2 / 2^65, under 10^-5, such collisions.
    return _mix(_mix(w[:, 0] | (w[:, 1] << _U32)) ^ (w[:, 2] | (w[:, 3] << _U32)))


def band_keys(values: NDArray[np.uint32]) -> NDArray[np.uint64]:
    """One 64-bit key per band of each signature: shape (n, BANDS)."""
    if values.shape[1] != NUM_PERM:
        raise ValueError(f"signatures must have {NUM_PERM} values")
    return np.stack([_band_key(values, b) for b in range(BANDS)], axis=1)


def build_band_index(values: NDArray[np.uint32]) -> BandIndex:
    """The band index of signatures held in memory."""
    n = values.shape[0]
    index = BandIndex(np.empty((BANDS, n), dtype=np.uint64), np.empty((BANDS, n), dtype=np.uint32))
    fill_band_index(index, [values])
    return index


def fill_band_index(index: BandIndex, chunks: Iterable[NDArray[np.uint32]]) -> None:
    """Fill `index`, whose arrays may be memory-mapped files, from the signatures in row order,
    given in chunks, so the signatures never need to be in memory at once."""
    at = 0
    for chunk in chunks:
        index.keys[:, at : at + chunk.shape[0]] = band_keys(chunk).T
        at += chunk.shape[0]
    if at != index.keys.shape[1]:
        raise ValueError(f"{at} signature rows for an index of {index.keys.shape[1]}")
    for b in range(BANDS):
        k = np.array(index.keys[b])
        # A stable sort keeps rows with equal keys in row order, so the stored index is
        # reproducible byte for byte.
        order = np.argsort(k, kind="stable")
        index.keys[b] = k[order]
        index.rows[b] = order


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


@dataclass(frozen=True, slots=True)
class BandPairs:
    codes: NDArray[np.uint64]  # (left << 32) | right, left < right, sorted and distinct
    buckets_over_cap: int
    rows_in_capped_buckets: int
    pairs_in_capped_buckets: int
    largest_bucket: int


def band_pairs(index: BandIndex, band: int, *, cap: int = BUCKET_CAP) -> BandPairs:
    """Every pair that shares band `band`'s key in a bucket of at most `cap` rows."""
    keys = np.asarray(index.keys[band], dtype=np.uint64)
    rows = np.asarray(index.rows[band], dtype=np.uint32)
    starts, sizes = _buckets(keys)
    capped = sizes[sizes > cap]
    kept = (sizes >= 2) & (sizes <= cap)
    # Buckets of one size form one matrix, so pairs come from a few array operations per
    # distinct size rather than a loop per bucket.
    parts = [
        _codes(*_pairs_of_size(rows, starts[sizes == size], size))
        for size in np.unique(sizes[kept]).tolist()
    ]
    # A pair falls in one bucket of a band at most, so a band's pairs are already distinct.
    codes = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
    codes.sort()
    return BandPairs(
        codes,
        int(capped.size),
        int(capped.sum()),
        int((capped * (capped - 1) // 2).sum()),
        int(sizes.max()) if sizes.size else 0,
    )


def unseen(codes: NDArray[np.uint64], seen: NDArray[np.uint64]) -> NDArray[np.bool_]:
    """Which of the sorted `codes` are absent from the sorted `seen`."""
    if not seen.size:
        return np.ones(codes.size, dtype=bool)
    at = np.minimum(np.searchsorted(seen, codes), seen.size - 1)
    return seen[at] != codes


def merge_sorted(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """The sorted union of two sorted arrays with nothing in common."""
    out = np.concatenate([a, b])
    # Timsort finds the two runs and merges them in linear time.
    out.sort(kind="stable")
    return out


def candidate_pairs(index: BandIndex, *, cap: int = BUCKET_CAP) -> Candidates:
    """Every pair that shares a band key in a bucket of at most `cap` rows, all in memory.

    The build forms the same pairs band by band (lineage.pairs), verifying each band's new
    ones before the next; this is the reference it is tested against.
    """
    seen = np.empty(0, dtype=np.uint64)
    over = rows = skipped = largest = 0
    for b in range(BANDS):
        found = band_pairs(index, b, cap=cap)
        seen = merge_sorted(seen, found.codes[unseen(found.codes, seen)])
        over += found.buckets_over_cap
        rows += found.rows_in_capped_buckets
        skipped += found.pairs_in_capped_buckets
        largest = max(largest, found.largest_bucket)
    left = (seen >> _U32).astype(np.uint32)
    right = (seen & _LOW32).astype(np.uint32)
    return Candidates(left, right, over, rows, skipped, largest)


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
    over = dict.fromkeys(caps, 0)
    multi = dict.fromkeys(caps, 0)
    most = dict.fromkeys(caps, 0)
    parts: list[NDArray[np.uint64]] = []
    for b in range(BANDS):
        keys = np.asarray(index.keys[b], dtype=np.uint64)
        rows = np.asarray(index.rows[b], dtype=np.uint32)
        starts, sizes = _buckets(keys)
        for start, size in zip(
            starts[sizes > lowest].tolist(), sizes[sizes > lowest].tolist(), strict=True
        ):
            labels = np.unique(groups[rows[start : start + size]]).size
            for cap in caps:
                if size > cap:
                    over[cap] += 1
                    multi[cap] += labels > 1
                    most[cap] = max(most[cap], labels)
            if labels > 1:
                left, right = _pairs_of_size(rows, np.array([start]), size)
                cross = groups[left] != groups[right]
                parts.append(_codes(left[cross], right[cross]))
    codes = _distinct(np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64))
    del parts
    # Only the rows of these pairs need their key and bucket size in every band. Gathering
    # them band by band keeps 12 bytes per band for those rows alone, where a table for every
    # row would be 12 bytes per band for all of them, or a file written at random.
    involved = np.unique(np.concatenate([codes >> _U32, codes & _LOW32])).astype(np.intp)
    key_of = np.empty((involved.size, BANDS), dtype=np.uint64)
    size_of = np.empty((involved.size, BANDS), dtype=np.int32)
    for b in range(BANDS):
        keys = np.asarray(index.keys[b], dtype=np.uint64)
        rows = np.asarray(index.rows[b], dtype=np.uint32)
        starts, sizes = _buckets(keys)
        position = np.empty(n, dtype=np.intp)
        position[rows] = np.arange(n, dtype=np.intp)
        at = position[involved]
        del position
        key_of[:, b] = keys[at]
        size_of[:, b] = sizes[np.searchsorted(starts, at, side="right") - 1]
    # The smallest bucket each pair shares in any band decides whether a cap keeps it.
    smallest = np.empty(codes.size, dtype=np.int32)
    for i in range(0, codes.size, _CHUNK):
        chunk = codes[i : i + _CHUNK]
        left = np.searchsorted(involved, (chunk >> _U32).astype(np.intp))
        right = np.searchsorted(involved, (chunk & _LOW32).astype(np.intp))
        shared = key_of[left] == key_of[right]
        smallest[i : i + _CHUNK] = np.where(shared, size_of[left], np.iinfo(np.int32).max).min(1)
    return {
        cap: CapEffect(over[cap], multi[cap], most[cap], int(np.count_nonzero(smallest > cap)))
        for cap in caps
    }
