import pytest

from lineage.sections import WINDOW_WORDS, clean, normalize, split_sections


def _words(n: int, word: str = "word") -> str:
    return " ".join([word] * n)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Section 1.", "Section 2."),
        ("SECTION 1.", "SECTION 2."),
        ("Sec. 1.", "Sec. 2."),
        ("SEC. 1.", "SEC. 2."),
        ("§ 1.", "§ 2."),
        ("Section 1:", "Section 2:"),
        ("Section 1-101.", "Section 1-102."),
        ("Section 2a.", "Section 2b."),
    ],
)
def test_splits_on_each_heading_pattern(first: str, second: str) -> None:
    text = f"AN ACT relating to water.\n{first} The first provision.\n{second} The second one.\n"

    split = split_sections(text)

    assert split.by_heading
    assert [(s.ordinal, s.heading, s.text) for s in split.sections] == [
        (0, None, "AN ACT relating to water."),
        (1, first, f"{first} The first provision."),
        (2, second, f"{second} The second one."),
    ]


def test_a_single_heading_still_splits_off_the_preamble() -> None:
    split = split_sections("AN ACT relating to water.\nSection 1. The only provision.")

    assert split.by_heading
    assert [s.heading for s in split.sections] == [None, "Section 1."]


def test_no_preamble_section_when_the_text_opens_with_a_heading() -> None:
    split = split_sections("Section 1. One.\nSection 2. Two.")

    assert [s.ordinal for s in split.sections] == [1, 2]


def test_a_reference_inside_a_sentence_is_not_a_heading() -> None:
    text = "Section 1. As provided in\nsection 5. of the code, and Section 6. of the act.\n"

    assert [s.heading for s in split_sections(text).sections] == ["Section 1."]


def test_washington_style_new_section_marker_is_part_of_the_heading_line() -> None:
    text = "NEW SECTION. Sec. 1. A new section is added.\nSec. 2. RCW 1.2.3 is amended."

    assert [s.heading for s in split_sections(text).sections] == ["Sec. 1.", "Sec. 2."]


def test_line_number_gutters_and_markers_do_not_hide_headings() -> None:
    # Alaska prints a line-number gutter, starts each section with an asterisk, and ends
    # lines with carriage returns alone.
    text = (
        "01 AN ACT relating to boards.\r02    * Section 1. AS 08.03 is amended.   \r"
        "03 more\r04    * Sec. 2. AS 08.04 is amended.\r"
    )

    split = split_sections(text)

    assert [s.heading for s in split.sections] == [None, "Section 1.", "Sec. 2."]
    assert split.sections[1].text == "* Section 1. AS 08.03 is amended.\nmore"


def test_clean_drops_private_use_and_format_characters() -> None:
    # Kentucky's export puts a private-use glyph before each heading.
    raw = "\uf0e2Section 1. KRS 1.1\u00a0is\u200b amended."

    assert clean(raw) == "Section 1. KRS 1.1 is amended."


def test_clean_keeps_one_blank_line_between_paragraphs() -> None:
    assert clean("First.\r\n\r\n\r\n\n  Second.  \n") == "First.\n\nSecond."


def test_clean_keeps_numbered_lines_that_are_not_gutters() -> None:
    # New Jersey numbers its sections "1." at the start of a line; that is text, not a gutter.
    nj = "     1.    (New section)  All of the\n     2.    The functions"
    assert clean(nj) == "1. (New section) All of the\n2. The functions"
    assert clean("2023 Regular Session") == "2023 Regular Session"


def test_falls_back_to_windows_on_paragraph_boundaries() -> None:
    paragraphs = [_words(120, f"p{i}") for i in range(5)]

    split = split_sections("\n\n".join(paragraphs))

    assert not split.by_heading
    assert [s.heading for s in split.sections] == [None, None]
    # 120 words is short of a window, so paragraphs pair up; the fifth joins the last window.
    assert [len(s.text.split()) for s in split.sections] == [240, 360]
    assert all(s.text.split()[0] in {"p0", "p2"} for s in split.sections)


def test_falls_back_to_word_windows_when_the_text_has_no_breaks() -> None:
    split = split_sections(_words(3 * WINDOW_WORDS + 30))

    assert [len(s.text.split()) for s in split.sections] == [200, 200, 230]


def test_a_short_text_without_headings_is_one_section() -> None:
    split = split_sections("A resolution recognizing a week.")

    assert [(s.ordinal, s.text) for s in split.sections] == [
        (1, "A resolution recognizing a week.")
    ]


@pytest.mark.parametrize("text", [None, "", " \n\t\u00a0\r\n "])
def test_blank_text_has_no_sections(text: str | None) -> None:
    split = split_sections(text)

    assert split.sections == []
    assert not split.by_heading


def test_normalize_folds_case_punctuation_and_whitespace() -> None:
    assert normalize("Sec. 2.  The STATE'S\n“Board” shall—by rule—act.") == (
        "sec 2 the state s board shall by rule act"
    )


def test_a_quoted_heading_inside_amended_text_is_not_a_heading() -> None:
    text = 'SECTION 2.\nSaid Act is amended as follows:\n"SECTION 3.\nElection of commissioners."'

    assert [s.heading for s in split_sections(text).sections] == ["SECTION 2."]


def test_clean_drops_running_page_headers() -> None:
    page = "LC 47 3375\nshall be elected at the general election\n"
    text = "SECTION 1.\n" + page * 3 + "SENATE BILL 12\n"

    body = "\n".join(["shall be elected at the general election"] * 3)
    # The header repeats on every page; the bill number appears once and stays.
    assert clean(text) == f"SECTION 1.\n{body}\nSENATE BILL 12"


def test_clean_keeps_repeated_lines_that_read_as_text() -> None:
    # Amending phrases and enumerators repeat legitimately; the stoplist handles them instead.
    text = "\n".join(["is amended to read as follows:", "Subd. 2.", "* * *"] * 3)

    assert clean(text) == text


def test_normalize_unifies_section_symbols() -> None:
    assert (
        normalize("§ 2 and §§ 3-4")
        == normalize("section 2 and sections 3-4")
        == ("section 2 and sections 3 4")
    )
