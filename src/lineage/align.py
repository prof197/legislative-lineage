"""Word-level alignment of a matched section pair: what was copied, paraphrased or unmatched.

Both sections' display texts are read as words, each word keyed by its normalized form, so case
and punctuation do not break a copy. `difflib` finds the blocks of words the two share in order,
and a block of at least MIN_COPIED words, the shingle length, is a copy; a shorter one is a
coincidence of common phrasing and is not kept. The copies are anchors. Between two anchors, or
between an anchor and either end, text on both sides is the same passage put in other words:
paraphrased. Text on one side only is unmatched. A pair with no copy at all is paraphrased whole
when the semantic stage matched it, and otherwise each side is unmatched.

Every span carries an id, numbered in reading order, and its character offsets in each display
text, so the site can highlight it and an explanation's claims can cite it, which code then
checks (DESIGN.md#constraints). Gap scoring as in Smith-Waterman would add nothing to marking
shared blocks (DESIGN.md#alternatives-considered).
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from lineage.sections import normalize

MIN_COPIED = 5  # words; the shingle length, below which shared wording is a coincidence

Kind = Literal["copied", "paraphrased", "unmatched"]
_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Span:
    id: str
    kind: Kind
    a: tuple[int, int] | None  # character offsets in the first text, or None
    b: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class Alignment:
    spans: list[Span]
    copied_share_a: float  # of the first text's words, the share inside copied spans
    copied_share_b: float


@dataclass(frozen=True, slots=True)
class _Words:
    offsets: list[tuple[int, int]]  # every word's characters
    keys: list[str]  # the words that normalize to something, keyed
    at: list[int]  # each key's word


def _words(text: str) -> _Words:
    offsets: list[tuple[int, int]] = []
    keys: list[str] = []
    at: list[int] = []
    for m in _WORD.finditer(text):
        key = normalize(m.group())
        if key:
            keys.append(key)
            at.append(len(offsets))
        offsets.append((m.start(), m.end()))
    return _Words(offsets, keys, at)


def _chars(words: _Words, lo: int, hi: int) -> tuple[int, int] | None:
    """The characters of words lo..hi-1, or None when there are none."""
    return (words.offsets[lo][0], words.offsets[hi - 1][1]) if hi > lo else None


def align(a: str, b: str, *, meaning_match: bool = False) -> Alignment:
    wa, wb = _words(a), _words(b)
    # Junk detection would treat frequent words such as "the" as unmatchable in long sections.
    blocks = [
        block
        for block in SequenceMatcher(None, wa.keys, wb.keys, autojunk=False).get_matching_blocks()
        if block.size >= MIN_COPIED
    ]
    spans: list[tuple[Kind, tuple[int, int] | None, tuple[int, int] | None]] = []
    copied_a = copied_b = 0

    def gap(a_lo: int, a_hi: int, b_lo: int, b_hi: int, *, anchored: bool) -> None:
        sa, sb = _chars(wa, a_lo, a_hi), _chars(wb, b_lo, b_hi)
        if sa is not None and sb is not None and (anchored or meaning_match):
            spans.append(("paraphrased", sa, sb))
            return
        if sa is not None:
            spans.append(("unmatched", sa, None))
        if sb is not None:
            spans.append(("unmatched", None, sb))

    pa = pb = 0
    for block in blocks:
        a_lo, b_lo = wa.at[block.a], wb.at[block.b]
        a_hi, b_hi = wa.at[block.a + block.size - 1] + 1, wb.at[block.b + block.size - 1] + 1
        gap(pa, a_lo, pb, b_lo, anchored=True)
        spans.append(("copied", _chars(wa, a_lo, a_hi), _chars(wb, b_lo, b_hi)))
        copied_a += block.size
        copied_b += block.size
        pa, pb = a_hi, b_hi
    gap(pa, len(wa.offsets), pb, len(wb.offsets), anchored=bool(blocks))
    return Alignment(
        [Span(f"s{i}", kind, sa, sb) for i, (kind, sa, sb) in enumerate(spans)],
        copied_a / len(wa.keys) if wa.keys else 0.0,
        copied_b / len(wb.keys) if wb.keys else 0.0,
    )
