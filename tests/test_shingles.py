import numpy as np

from lineage.shingles import SHINGLE_WORDS, section_shingles, shingle_hashes

MASK64 = (1 << 64) - 1


def _fnv1a64(word: str) -> int:
    h = 0xCBF29CE484222325
    for byte in word.encode():
        h = ((h ^ byte) * 0x100000001B3) & MASK64
    return h


def _fmix64(h: int) -> int:
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & MASK64
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & MASK64
    return h ^ (h >> 33)


def _reference(text: str) -> list[int]:
    """The specification, one shingle at a time."""
    words = text.split()
    out: list[int] = []
    for i in range(len(words) - SHINGLE_WORDS + 1):
        h = 0
        for w in words[i : i + SHINGLE_WORDS]:
            h = (h * 0x9E3779B97F4A7C15 + _fnv1a64(w)) & MASK64
        out.append(_fmix64(h) >> 32)
    return out


TEXTS = [
    "be it enacted by the general assembly of georgia",
    "short text",
    "",
    "the département shall publish a résumé of each § report within 30 days",
    "a " + "x" * 300 + " word that is very long indeed",
]


def test_matches_the_reference_for_every_shingle() -> None:
    hashes, owners = shingle_hashes(TEXTS)

    expected = [(i, h) for i, text in enumerate(TEXTS) for h in _reference(text)]
    assert list(zip(owners.tolist(), hashes.tolist(), strict=True)) == expected
    assert hashes.dtype == np.uint32


def test_hashes_are_pinned() -> None:
    # The stoplist and every signature depend on these values; changing the hash invalidates
    # them all, so a change must be deliberate.
    hashes, _ = shingle_hashes(["be it enacted by the"])

    assert hashes.tolist() == [_reference("be it enacted by the")[0]] == [702324083]


def test_shingles_never_span_two_texts() -> None:
    _, owners = shingle_hashes(["one two three four", "five six seven eight nine"])

    assert owners.tolist() == [1]


def test_a_text_shorter_than_a_shingle_has_none() -> None:
    hashes, owners = shingle_hashes(["four words only here", ""])

    assert hashes.size == owners.size == 0


def test_section_shingles_are_distinct_per_section() -> None:
    texts = ["a b c d e a b c d e", "a b c d e"]

    per_section = section_shingles(texts)

    assert [len(s) for s in per_section] == [5, 1]
    assert per_section[1].tolist() == _reference("a b c d e")
    assert all(np.all(np.diff(s.astype(np.int64)) > 0) for s in per_section)
