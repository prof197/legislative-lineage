"""Word-level alignment of a matched section pair: what was copied, paraphrased or unmatched.

Both sections' display texts are read as words, each word keyed by its normalized form, so case
and punctuation do not break a copy. `difflib` finds the blocks of words the two share in order,
and a block of at least MIN_COPIED words, the shingle length, is an exact copy; a shorter one is
a coincidence of common phrasing and is not kept.

Between the copies, and over the whole pair when nothing was copied, lie unmatched stretches.
Inside each stretch that has text on both sides, each side is cut into sentences, and sentences
are paired by embedding similarity: a pair at or above THRESHOLD, the semantic stage's cosine
(lineage.semantic), is a paraphrase. Pairs are taken best first and never cross, so each text
still reads in order. Every sentence left unpaired is unmatched. Without an embedder nothing is
marked as paraphrase.

Every span carries an id, numbered in reading order, and its character offsets in each display
text, so the site can highlight it and an explanation's claims can cite it, which code then
checks (DESIGN.md#alignment). Gap scoring as in Smith-Waterman would add nothing to marking
shared blocks (DESIGN.md#alternatives-considered).
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from lineage.sections import normalize
from lineage.semantic import THRESHOLD

MIN_COPIED = 5  # words; the shingle length, below which shared wording is a coincidence

Kind = Literal["copied", "paraphrased", "unmatched"]
Embedder = Callable[[Sequence[str]], NDArray[np.float32]]  # unit-length rows
_WORD = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.;:?!][\"')\]]*$")


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


def _sentences(text: str, words: _Words, lo: int, hi: int) -> list[tuple[int, int]]:
    """Words lo..hi-1 cut after each word that ends a sentence or clause, as word ranges."""
    out: list[tuple[int, int]] = []
    start = lo
    for i in range(lo, hi):
        s, e = words.offsets[i]
        if _SENTENCE_END.search(text[s:e]) or i == hi - 1:
            out.append((start, i + 1))
            start = i + 1
    return out


def pair_sentences(similarity: NDArray[np.float32], threshold: float) -> list[tuple[int, int]]:
    """Pairs (row, column) at or above the threshold, best first, each row and column used
    once, and no pair crossing another, so both sides keep their order."""
    rows, cols = np.nonzero(similarity >= threshold)
    order = np.argsort(-similarity[rows, cols], kind="stable")
    kept: list[tuple[int, int]] = []
    for r, c in zip(rows[order].tolist(), cols[order].tolist(), strict=True):
        if all((r - kr) * (c - kc) > 0 for kr, kc in kept):
            kept.append((r, c))
    return sorted(kept)


def align(
    a: str, b: str, *, embed: Embedder | None = None, threshold: float = THRESHOLD
) -> Alignment:
    wa, wb = _words(a), _words(b)
    # Junk detection would treat frequent words such as "the" as unmatchable in long sections.
    blocks = [
        block
        for block in SequenceMatcher(None, wa.keys, wb.keys, autojunk=False).get_matching_blocks()
        if block.size >= MIN_COPIED
    ]
    spans: list[tuple[Kind, tuple[int, int] | None, tuple[int, int] | None]] = []
    copied = 0

    def stretch(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> None:
        """An unmatched stretch: paraphrased sentence pairs, and the rest unmatched."""
        sa = _sentences(a, wa, a_lo, a_hi)
        sb = _sentences(b, wb, b_lo, b_hi)
        pairs: list[tuple[int, int]] = []
        if embed is not None and sa and sb:
            texts_a = [a[wa.offsets[lo][0] : wa.offsets[hi - 1][1]] for lo, hi in sa]
            texts_b = [b[wb.offsets[lo][0] : wb.offsets[hi - 1][1]] for lo, hi in sb]
            vectors = embed([*texts_a, *texts_b])
            pairs = pair_sentences(vectors[: len(sa)] @ vectors[len(sa) :].T, threshold)
        ia = ib = 0

        def unmatched(upto_a: int, upto_b: int) -> None:
            nonlocal ia, ib
            if upto_a > ia:
                spans.append(("unmatched", _chars(wa, sa[ia][0], sa[upto_a - 1][1]), None))
            if upto_b > ib:
                spans.append(("unmatched", None, _chars(wb, sb[ib][0], sb[upto_b - 1][1])))
            ia, ib = upto_a, upto_b

        for ra, rb in pairs:
            unmatched(ra, rb)
            spans.append(("paraphrased", _chars(wa, *sa[ra]), _chars(wb, *sb[rb])))
            ia, ib = ra + 1, rb + 1
        unmatched(len(sa), len(sb))

    pa = pb = 0
    for block in blocks:
        a_lo, b_lo = wa.at[block.a], wb.at[block.b]
        a_hi, b_hi = wa.at[block.a + block.size - 1] + 1, wb.at[block.b + block.size - 1] + 1
        stretch(pa, a_lo, pb, b_lo)
        spans.append(("copied", _chars(wa, a_lo, a_hi), _chars(wb, b_lo, b_hi)))
        copied += block.size
        pa, pb = a_hi, b_hi
    stretch(pa, len(wa.offsets), pb, len(wb.offsets))
    return Alignment(
        [Span(f"s{i}", kind, sa, sb) for i, (kind, sa, sb) in enumerate(spans)],
        copied / len(wa.keys) if wa.keys else 0.0,
        copied / len(wb.keys) if wb.keys else 0.0,
    )
