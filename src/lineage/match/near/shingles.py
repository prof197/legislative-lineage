"""Word 5-gram shingles of normalized section text, hashed to 32 bits with NumPy.

Each word is hashed with 64-bit FNV-1a over its UTF-8 bytes; five consecutive word hashes are
combined positionally and finished with MurmurHash3's 64-bit mixer, keeping the top 32 bits.
Everything runs as whole-array operations: a Python loop per shingle would take hours over
the corpus. The values are part of the data contract, since the stoplist and every MinHash
signature are built from them; a test pins them. See DESIGN.md#boilerplate.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

SHINGLE_WORDS = 5

_FNV_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV_PRIME = np.uint64(0x100000001B3)
_COMBINE = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xFF51AFD7ED558CCD)
_MIX2 = np.uint64(0xC4CEB9FE1A85EC53)
_SPACE = ord(" ")


def _word_hashes(data: NDArray[np.uint8]) -> tuple[NDArray[np.uint64], NDArray[np.int64]]:
    """FNV-1a of every space-separated word in `data`, with each word's start offset."""
    space = data == _SPACE
    word = ~space
    starts = np.flatnonzero(word & np.concatenate(([True], space[:-1])))
    ends = np.flatnonzero(word & np.concatenate((space[1:], [True]))) + 1
    lengths = ends - starts
    # Longest words first, so the words still being hashed at byte k are always a prefix and
    # the loop touches each byte once, however long the longest word is.
    order = np.argsort(-lengths, kind="stable")
    sorted_starts = starts[order]
    descending = lengths[order]
    hashes = np.full(starts.size, _FNV_OFFSET, dtype=np.uint64)
    longest = int(descending[0]) if descending.size else 0
    for k in range(longest):
        active = int(np.searchsorted(-descending, -k, side="left"))
        prefix = hashes[:active]
        prefix ^= data[sorted_starts[:active] + k].astype(np.uint64)
        prefix *= _FNV_PRIME
    out = np.empty_like(hashes)
    out[order] = hashes
    return out, starts


def _fmix64(h: NDArray[np.uint64]) -> NDArray[np.uint64]:
    h ^= h >> np.uint64(33)
    h *= _MIX1
    h ^= h >> np.uint64(33)
    h *= _MIX2
    h ^= h >> np.uint64(33)
    return h


def shingle_hashes(texts: Sequence[str]) -> tuple[NDArray[np.uint32], NDArray[np.int64]]:
    """Every shingle hash of the normalized `texts`, in order, with the index of its text."""
    encoded = [t.encode() for t in texts]
    data = np.frombuffer(b" ".join(encoded), dtype=np.uint8)
    words, word_starts = _word_hashes(data)
    # Byte offset where each text begins in the joined buffer, to find each word's text.
    text_starts = np.cumsum([0, *(len(e) + 1 for e in encoded[:-1])], dtype=np.int64)
    owner = np.searchsorted(text_starts, word_starts, side="right") - 1
    last = SHINGLE_WORDS - 1
    if words.size <= last:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.int64)
    # A shingle starts at word i only if word i + 4 belongs to the same text.
    valid = np.flatnonzero(owner[:-last] == owner[last:])
    combined = words[valid].copy()
    for j in range(1, SHINGLE_WORDS):
        combined *= _COMBINE
        combined += words[valid + j]
    return (_fmix64(combined) >> np.uint64(32)).astype(np.uint32), owner[valid]


def section_shingles(texts: Sequence[str]) -> list[NDArray[np.uint32]]:
    """The set of shingle hashes of each text, sorted, as one array per text."""
    hashes, owners = shingle_hashes(texts)
    keys = np.unique((owners.astype(np.uint64) << np.uint64(32)) | hashes.astype(np.uint64))
    owner_of_key = (keys >> np.uint64(32)).astype(np.int64)
    bounds = np.searchsorted(owner_of_key, np.arange(len(texts) + 1))
    values = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return [values[bounds[i] : bounds[i + 1]] for i in range(len(texts))]
