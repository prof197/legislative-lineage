"""Word-level alignment of a matched section pair (DESIGN.md#alignment).

The embedder is replaced by a lookup: texts listed as the same in SAME get one vector, every
other text its own, so a paraphrase is whatever the test says it is."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from lineage.align import MIN_COPIED, Span, align, pair_sentences

A = (
    "The department of revenue shall publish the rates for each fiscal year, and the "
    "commissioner may adopt rules to carry out this section after notice and hearing."
)
SAME = {
    "every biennium,": "for each fiscal year,",
    "Every biennium, the fee is set.": "For each fiscal year, the fee is set.",
}


def _embed(texts: Sequence[str]) -> NDArray[np.float32]:
    keys = [SAME.get(t, t) for t in texts]
    index = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    out = np.zeros((len(texts), len(index)), dtype=np.float32)
    for row, k in enumerate(keys):
        out[row, index[k]] = 1.0
    return out


def _text(source: str, at: tuple[int, int] | None) -> str | None:
    return None if at is None else source[at[0] : at[1]]


def _read(a: str, b: str, spans: list[Span]) -> list[tuple[str, str | None, str | None]]:
    return [(s.kind, _text(a, s.a), _text(b, s.b)) for s in spans]


def test_an_identical_pair_is_one_copied_span() -> None:
    result = align(A, A, embed=_embed)

    assert _read(A, A, result.spans) == [("copied", A, A)]
    assert (result.copied_share_a, result.copied_share_b) == (1.0, 1.0)


def test_case_and_punctuation_do_not_break_a_copy() -> None:
    b = A.upper().replace(",", "").replace(".", ";")

    assert [s.kind for s in align(A, b).spans] == ["copied"]


def test_a_reworded_phrase_is_paraphrased_when_its_meaning_pairs() -> None:
    b = A.replace("for each fiscal year,", "every biennium,")

    got = _read(A, b, align(A, b, embed=_embed).spans)

    assert [kind for kind, _, _ in got] == ["copied", "paraphrased", "copied"]
    assert got[1][1:] == ("for each fiscal year,", "every biennium,")


def test_a_reworded_phrase_whose_meaning_does_not_pair_is_unmatched() -> None:
    b = A.replace("for each fiscal year,", "on demand,")

    got = _read(A, b, align(A, b, embed=_embed).spans)

    assert got[1:3] == [
        ("unmatched", "for each fiscal year,", None),
        ("unmatched", None, "on demand,"),
    ]


def test_without_an_embedder_nothing_is_paraphrased() -> None:
    b = A.replace("for each fiscal year,", "every biennium,")

    assert "paraphrased" not in {s.kind for s in align(A, b).spans}


def test_text_on_one_side_only_is_unmatched() -> None:
    added = " The commissioner shall report to the legislature each January."
    b = A + added

    got = _read(A, b, align(A, b, embed=_embed).spans)

    assert got[-1] == ("unmatched", None, added.strip())
    assert all(kind == "copied" for kind, _, _ in got[:-1])


def test_a_short_coincidence_is_not_a_copy() -> None:
    shared = ["the", "state", "shall"]
    assert len(shared) < MIN_COPIED
    a = " ".join(shared + [f"alpha{i}" for i in range(20)])
    b = " ".join(shared + [f"beta{i}" for i in range(20)])

    assert "copied" not in {s.kind for s in align(a, b).spans}


def test_sentences_pair_inside_a_pair_with_no_copy_and_the_rest_stay_unmatched() -> None:
    a = "Alpha one. Every biennium, the fee is set. Gamma three."
    b = "Delta four. For each fiscal year, the fee is set. Epsilon five."

    got = _read(a, b, align(a, b, embed=_embed).spans)

    assert got == [
        ("unmatched", "Alpha one.", None),
        ("unmatched", None, "Delta four."),
        ("paraphrased", "Every biennium, the fee is set.", "For each fiscal year, the fee is set."),
        ("unmatched", "Gamma three.", None),
        ("unmatched", None, "Epsilon five."),
    ]


def test_sentence_pairs_are_taken_best_first_and_never_cross() -> None:
    similarity = np.array([[0.2, 0.95], [0.97, 0.1]], dtype=np.float32)

    # The best pair (1, 0) wins; (0, 1) would cross it, so it is left unpaired.
    assert pair_sentences(similarity, 0.9) == [(1, 0)]
    assert pair_sentences(np.eye(3, dtype=np.float32), 0.9) == [(0, 0), (1, 1), (2, 2)]


def test_spans_are_numbered_in_reading_order_and_cover_both_texts() -> None:
    b = "Section 1. " + A.replace("after notice and hearing", "without a hearing") + " Done."

    spans = align(A, b, embed=_embed).spans

    assert [s.id for s in spans] == [f"s{i}" for i in range(len(spans))]
    for side, text in (("a", A), ("b", b)):
        covered = " ".join(
            text[at[0] : at[1]] for s in spans if (at := getattr(s, side)) is not None
        )
        assert covered.split() == text.split()


def test_empty_text_aligns_to_nothing() -> None:
    assert align("", "").spans == []
    assert _read("", A, align("", A).spans) == [("unmatched", None, A)]
