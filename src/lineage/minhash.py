"""MinHash signatures: 128 values per section whose agreement rate estimates Jaccard similarity.

Permutation k maps a 32-bit shingle hash x to ((a_k·x + b_k) mod p) mod 2^32 with
p = 2^61 - 1, the universal family of Carter and Wegman. A section's signature holds, for each
k, the minimum over its shingles, and two sections agree at position k with probability equal
to the Jaccard similarity of their shingle sets. Each permutation runs over every shingle of a
batch in one vectorized pass. See DESIGN.md#dr-3.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

NUM_PERM = 128
SEED = 42
PRIME = (1 << 61) - 1

_P = np.uint64(PRIME)
_MASK64 = (1 << 64) - 1
_LOW32 = np.uint64(0xFFFF_FFFF)
_LOW29 = np.uint64((1 << 29) - 1)
_U29 = np.uint64(29)
_U32 = np.uint64(32)
_U61 = np.uint64(61)
# Shingles hashed per pass; three arrays of this many uint64 values fit in a core's cache.
_RUN = 1 << 17


@dataclass(frozen=True, slots=True)
class Permutations:
    a: NDArray[np.uint64]  # each in [1, p)
    b: NDArray[np.uint64]  # each in [0, p)


def _splitmix64(seed: int) -> Iterator[int]:
    state = seed
    while True:
        state = (state + 0x9E3779B97F4A7C15) & _MASK64
        z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        yield z ^ (z >> 31)


def permutations(num_perm: int = NUM_PERM, seed: int = SEED) -> Permutations:
    """The fixed permutation parameters, drawn uniformly from the family's ranges."""
    # SplitMix64 rather than a NumPy generator: NumPy does not promise its streams stay the
    # same across versions, and every stored signature depends on these numbers.
    draws = (z & PRIME for z in _splitmix64(seed))
    a: list[int] = []
    b: list[int] = []
    while len(b) < num_perm:
        x, y = next(draws), next(draws)
        if 0 < x < PRIME and y < PRIME:
            a.append(x)
            b.append(y)
    return Permutations(np.array(a, dtype=np.uint64), np.array(b, dtype=np.uint64))


def universal_hash(x: NDArray[np.uint64], a: np.uint64, b: np.uint64) -> NDArray[np.uint64]:
    """(a·x + b) mod p exactly, for x < 2^32 and a, b < p, without 64-bit overflow."""
    h = np.empty_like(x)
    _hash_into(x, a, b, h, np.empty_like(x))
    return h


def _hash_into(
    x: NDArray[np.uint64], a: np.uint64, b: np.uint64, h: NDArray[np.uint64], t: NDArray[np.uint64]
) -> None:
    """Write (a·x + b) mod p into `h`, using `t` as scratch; both are the length of `x`."""
    # a·x reaches 2^93, so a is split at bit 32 and each partial product is reduced using
    # 2^61 ≡ 1 (mod p): a value q·2^61 + r is congruent to q + r.
    np.multiply(x, a >> _U32, out=t)  # high part, below 2^61
    # high·2^32 mod p: writing high = q·2^29 + r gives high·2^32 ≡ q + r·2^32.
    np.right_shift(t, _U29, out=h)
    t &= _LOW29
    t <<= _U32
    h += t
    np.multiply(x, a & _LOW32, out=t)  # low part, below 2^64
    h += t & _P
    t >>= _U61
    h += t
    h += b  # every term is below 2^61 + 2^32, so the sum stays below 2^63
    np.bitwise_and(h, _P, out=t)
    h >>= _U61
    h += t  # now below p + 4
    np.subtract(h, _P, out=h, where=h >= _P)


def signatures(sets: Sequence[NDArray[np.uint32]], perms: Permutations) -> NDArray[np.uint32]:
    """One signature row per shingle set, in input order. Every set must be non-empty."""
    sizes = np.fromiter((s.size for s in sets), dtype=np.int64, count=len(sets))
    if (sizes == 0).any():
        raise ValueError("an empty set has no signature")
    out = np.empty((len(sets), perms.a.size), dtype=np.uint32)
    if not sets:
        return out
    flat = np.concatenate(sets).astype(np.uint64)
    ends = np.cumsum(sizes)
    starts = ends - sizes
    # Sets are hashed in runs of about _RUN shingles, so the working arrays stay in cache and
    # are reused rather than reallocated for every permutation. A larger set is a run alone.
    h = np.empty(max(_RUN, int(sizes.max())), dtype=np.uint64)
    t = np.empty_like(h)
    first = 0
    while first < len(sets):
        offset = int(starts[first])
        last = max(first + 1, int(np.searchsorted(ends, offset + _RUN, side="right")))
        x = flat[offset : ends[last - 1]]
        run_starts = starts[first:last] - offset
        hk, tk = h[: x.size], t[: x.size]
        for k, (a, b) in enumerate(zip(perms.a, perms.b, strict=True)):
            _hash_into(x, a, b, hk, tk)
            hk &= _LOW32
            out[first:last, k] = np.minimum.reduceat(hk, run_starts)
        first = last
    return out


def similarity(x: NDArray[np.uint32], y: NDArray[np.uint32]) -> float:
    """The estimated Jaccard similarity of two signatures: the share of positions that agree."""
    return np.count_nonzero(x == y) / x.size
