"""Word-level alignment of a matched section pair (DESIGN.md#pipeline-overview, step 7)."""

from lineage.align import MIN_COPIED, Span, align

A = (
    "The department of revenue shall publish the rates for each fiscal year, and the "
    "commissioner may adopt rules to carry out this section after notice and hearing."
)


def _text(source: str, at: tuple[int, int] | None) -> str | None:
    return None if at is None else source[at[0] : at[1]]


def _read(a: str, b: str, spans: list[Span]) -> list[tuple[str, str | None, str | None]]:
    return [(s.kind, _text(a, s.a), _text(b, s.b)) for s in spans]


def test_an_identical_pair_is_one_copied_span() -> None:
    result = align(A, A)

    assert _read(A, A, result.spans) == [("copied", A, A)]
    assert (result.copied_share_a, result.copied_share_b) == (1.0, 1.0)


def test_case_and_punctuation_do_not_break_a_copy() -> None:
    b = A.upper().replace(",", "").replace(".", ";")

    assert [s.kind for s in align(A, b).spans] == ["copied"]


def test_a_changed_phrase_between_copies_is_paraphrased() -> None:
    b = A.replace("for each fiscal year", "every biennium")

    got = _read(A, b, align(A, b).spans)

    assert [kind for kind, _, _ in got] == ["copied", "paraphrased", "copied"]
    assert got[1][1:] == ("for each fiscal year,", "every biennium,")


def test_text_on_one_side_only_is_unmatched() -> None:
    added = " The commissioner shall report to the legislature each January."
    b = A + added

    got = _read(A, b, align(A, b).spans)

    assert got[-1] == ("unmatched", None, added.strip())
    assert all(kind == "copied" for kind, _, _ in got[:-1])


def test_a_short_coincidence_is_not_a_copy() -> None:
    shared = ["the", "state", "shall"]
    assert len(shared) < MIN_COPIED
    a = " ".join(shared + [f"alpha{i}" for i in range(20)])
    b = " ".join(shared + [f"beta{i}" for i in range(20)])

    assert "copied" not in {s.kind for s in align(a, b).spans}


def test_without_a_copy_a_meaning_match_is_paraphrased_whole_and_any_other_is_unmatched() -> None:
    a = " ".join(f"alpha{i}" for i in range(30))
    b = " ".join(f"beta{i}" for i in range(30))

    assert _read(a, b, align(a, b, meaning_match=True).spans) == [("paraphrased", a, b)]
    assert _read(a, b, align(a, b).spans) == [("unmatched", a, None), ("unmatched", None, b)]


def test_spans_are_numbered_in_reading_order_and_cover_both_texts() -> None:
    b = "Section 1. " + A.replace("after notice and hearing", "without a hearing") + " Done."

    spans = align(A, b).spans

    assert [s.id for s in spans] == [f"s{i}" for i in range(len(spans))]
    for side, text in (("a", A), ("b", b)):
        covered = " ".join(
            text[at[0] : at[1]] for s in spans if (at := getattr(s, side)) is not None
        )
        assert covered.split() == text.split()


def test_empty_text_aligns_to_nothing() -> None:
    assert align("", "").spans == []
    assert _read("", A, align("", A).spans) == [("unmatched", None, A)]
