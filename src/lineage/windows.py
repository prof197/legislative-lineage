"""Windows: the units each matching stage signs or embeds, cut inside long sections.

Jaccard similarity penalizes a difference in size. A 200-word act section wholly contained in a
2,000-word state section shares at most a tenth of their union, so at a threshold of 0.5
neither banding nor verification ever sees the pair. What matching wants is containment, and
windows of comparable size recover it with standard MinHash. A section of at most `WINDOW_WORDS`
words is one window, itself. A longer section is cut into windows of `WINDOW_WORDS` words, each
starting `WINDOW_STRIDE` words after the one before, so neighbors share half their words and a
provision's position in the section does not decide whether one window holds it. No unit is
larger than a window, so two copies of one section on either side of the limit, one kept whole
and one cut, still compare at their own size. The last
window ends where the section ends. A section matches when any of its windows does; scores,
families and display stay with the section.

Words are the display text's whitespace-separated words, which is what the semantic stage
will embed. See DESIGN.md#windows.
"""

import numpy as np
from numpy.typing import NDArray

from lineage.sections import normalize

# The size of a typical section: the median section split at headings has 97 words
# (reports/index.json, section_words), and 94 keeps the stride at exactly half. Every unit is
# well within the embedding models' 256 tokens, about 190 words.
WINDOW_WORDS = 94
WINDOW_STRIDE = WINDOW_WORDS // 2


def window_starts(
    words: int, *, size: int = WINDOW_WORDS, stride: int = WINDOW_STRIDE
) -> list[int]:
    """The first word of each window of a section of `words` words.

    `size` and `stride` default to the scheme both stages use; other values serve only the
    sizing benchmark, which compares schemes."""
    if words <= size:
        return [0]
    starts = list(range(0, words - size + 1, stride))
    if starts[-1] + size < words:
        starts.append(words - size)
    return starts


def window_counts(
    words: NDArray[np.int64], *, size: int = WINDOW_WORDS, stride: int = WINDOW_STRIDE
) -> NDArray[np.int64]:
    """How many windows each section of `words` words makes: `len(window_starts(n))`, for an
    array of word counts at once."""
    over = np.maximum(words - size, 0)
    return 1 + -(-over // stride)


def display_windows(
    text: str, *, size: int = WINDOW_WORDS, stride: int = WINDOW_STRIDE
) -> list[str]:
    """A section's windows as display text: the section itself, or its windows' words."""
    words = text.split()
    if len(words) <= size:
        return [text]
    return [
        " ".join(words[s : s + size]) for s in window_starts(len(words), size=size, stride=stride)
    ]


def normalized_windows(text: str, normalized: str) -> list[str]:
    """A section's windows as matching text, given its display and normalized text.

    Normalizing never joins two display words, so a window's normalized text is the
    normalized text of its display words, and a section kept whole is its stored normalized
    text unchanged.
    """
    if len(text.split()) <= WINDOW_WORDS:
        return [normalized]
    return [normalize(window) for window in display_windows(text)]
