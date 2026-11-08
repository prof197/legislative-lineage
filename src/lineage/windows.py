"""Windows: the units each matching stage signs or embeds, cut inside long sections.

Jaccard similarity penalizes a difference in size. A 200-word act section wholly contained in a
2,000-word state section shares at most a tenth of their union, so at a threshold of 0.5
neither banding nor verification ever sees the pair. What matching wants is containment, and
windows of comparable size recover it with standard MinHash. A section of at most `WHOLE_MAX`
words is one window, itself. A longer section is cut into windows of `WINDOW_WORDS` words, each
starting `WINDOW_STRIDE` words after the one before, so neighbors share half their words and a
provision's position in the section does not decide whether one window holds it. The last
window ends where the section ends. A section matches when any of its windows does; scores,
families and display stay with the section.

Words are the display text's whitespace-separated words, which is what the semantic stage
will embed. See DESIGN.md#windows.
"""

from lineage.sections import normalize

# The size of a typical section: the median section split at headings has 97 words
# (reports/index.json, windows). Twice it, the largest section kept whole, is within the
# embedding models' 256 tokens, about 190 words.
WINDOW_WORDS = 94
WINDOW_STRIDE = WINDOW_WORDS // 2
WHOLE_MAX = 2 * WINDOW_WORDS


def window_starts(words: int) -> list[int]:
    """The first word of each window of a section of `words` words."""
    if words <= WHOLE_MAX:
        return [0]
    starts = list(range(0, words - WINDOW_WORDS + 1, WINDOW_STRIDE))
    if starts[-1] + WINDOW_WORDS < words:
        starts.append(words - WINDOW_WORDS)
    return starts


def display_windows(text: str) -> list[str]:
    """A section's windows as display text: the section itself, or its windows' words."""
    words = text.split()
    if len(words) <= WHOLE_MAX:
        return [text]
    return [" ".join(words[s : s + WINDOW_WORDS]) for s in window_starts(len(words))]


def normalized_windows(text: str, normalized: str) -> list[str]:
    """A section's windows as matching text, given its display and normalized text.

    Normalizing never joins two display words, so a window's normalized text is the
    normalized text of its display words, and a section kept whole is its stored normalized
    text unchanged.
    """
    if len(text.split()) <= WHOLE_MAX:
        return [normalized]
    return [normalize(window) for window in display_windows(text)]
