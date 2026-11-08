from itertools import pairwise

import pytest

from lineage.sections import normalize
from lineage.shingles import section_shingles
from lineage.windows import (
    WHOLE_MAX,
    WINDOW_STRIDE,
    WINDOW_WORDS,
    display_windows,
    normalized_windows,
    window_starts,
)


def _words(n: int, prefix: str = "w") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _jaccard(a: str, b: str) -> float:
    x, y = (set(s.tolist()) for s in section_shingles([a, b]))
    return len(x & y) / len(x | y)


def test_windows_overlap_by_half() -> None:
    assert WINDOW_STRIDE * 2 == WINDOW_WORDS
    assert WHOLE_MAX == 2 * WINDOW_WORDS


@pytest.mark.parametrize("n", [1, 50, WHOLE_MAX])
def test_a_section_up_to_twice_the_window_is_kept_whole(n: int) -> None:
    text = "Sec. 1.\n" + " ".join(_words(n - 2))

    assert window_starts(n) == [0]
    assert display_windows(text) == [text]
    assert normalized_windows(text, "stored") == ["stored"]


@pytest.mark.parametrize("n", [WHOLE_MAX + 1, 500, 2003])
def test_a_longer_section_is_cut_into_windows_that_cover_it(n: int) -> None:
    words = _words(n)

    starts = window_starts(n)
    windows = display_windows(" ".join(words))

    assert starts[0] == 0
    assert starts[-1] + WINDOW_WORDS == n
    assert all(b - a <= WINDOW_STRIDE for a, b in pairwise(starts))
    assert all(len(w.split()) == WINDOW_WORDS for w in windows)
    # Every shingle of the section is in some window, so cutting loses no shared text.
    whole = set(section_shingles([" ".join(words)])[0].tolist())
    cut = set[int]().union(*(s.tolist() for s in section_shingles(windows)))
    assert cut == whole


def test_a_window_normalizes_like_the_section_it_comes_from() -> None:
    text = " ".join(["Under § 1-6-105.", *_words(WHOLE_MAX), "THE END."])

    windows = normalized_windows(text, normalize(text))

    assert windows == [normalize(w) for w in display_windows(text)]
    assert windows[0].startswith("under section 1 6 105 w0")


@pytest.mark.parametrize("offset", range(0, 1900, 37))
def test_a_provision_inside_a_long_section_has_a_window_like_it_wherever_it_sits(
    offset: int,
) -> None:
    provision = _words(150, "p")
    filler = _words(1900, "f")
    section = " ".join(filler[:offset] + provision + filler[offset:])
    act = " ".join(provision)

    # Whole, the act section is under a tenth of the state's: no threshold of 0.5 sees it.
    assert _jaccard(act, section) < 0.1
    best = max(_jaccard(act, w) for w in normalized_windows(section, normalize(section)))
    assert best >= 0.5
