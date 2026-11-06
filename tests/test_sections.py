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
        ("§ 680-a.", "§ 680-b."),  # New York's inserted sections
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


# Style sheets left in the text when the source's HTML was converted, trimmed from real records.
WORD_STYLE_SHEET = """\n\n\n<!--\n /* Font Definitions */\n @font-face\n {font-family:Calibri;\n \
panose-1:2 15 5 2 2 2 4 3 2 4;\n mso-font-charset:0;\n mso-font-signature:-520092929 1073786111 9 \
0 415 0;}\n /* Style Definitions */\n p.MsoNormal, li.MsoNormal, div.MsoNormal\n \
{mso-style-unhide:no;\n mso-style-parent:"";\n margin-top:0in;}\n@page WordSection1\n \
{size:8.5in 11.0in;\n mso-paper-source:0;}\ndiv.WordSection1\n {page:WordSection1;}\n-->\n\n"""
TEXAS_TABLE_STYLE = (
    "      89(R) HB 4135 - Introduced version - Bill Text\n"
    "      td { font-family: Courier, Arial, sans-serif; font-size: 10pt; } "
    "table { empty-cells:show; } \n   \n            89R9039 JRR-D\n  By: Zwiener"
)
WEST_VIRGINIA_CLASSES = (
    "\n\n\n \n  .cs2516C423{text-align:center;text-indent:0pt;margin:0pt 0pt 0pt 0pt;"
    "line-height:2}\n"
    "  .cs6D637943{color:#000000;background-color:transparent;font-family:'Microsoft Sans Serif';"
    "font-size:10pt;font-weight:normal;font-style:normal;}\n \n WEST VIRGINIA LEGISLATURE"
)
TEXAS_NO_TEXT = (
    "This version is not currently available, but is coming\nsoon.  In the meantime, please "
    "see the PDF\nfor the content of this bill.\n\n"
)


def test_clean_drops_a_word_style_sheet() -> None:
    assert clean(WORD_STYLE_SHEET + "Section 1. The text.") == "Section 1. The text."


def test_clean_drops_style_rules_and_keeps_the_text_around_them() -> None:
    expected = "89(R) HB 4135 - Introduced version - Bill Text\n\n89R9039 JRR-D\nBy: Zwiener"
    assert clean(TEXAS_TABLE_STYLE) == expected
    assert clean(WEST_VIRGINIA_CLASSES) == "WEST VIRGINIA LEGISLATURE"


@pytest.mark.parametrize(
    "text",
    [
        # Kansas marks amendments with braces, and other states brace parts of forms.
        "Sec. 26. Text.\n{Sec. 27. K.S.A. 25-1115 is hereby amended to read as follows:\n1115.}",
        "Nothing in this subsection shall{:\n(A) }Prohibit disclosure",
        "publication in the statute book {Kansas register}.",
        '1. "Do you speak and understand English?"\n{If no, an interpreter must be present}',
        "{1} Name of any debtor that is not an addressee: (Name of each debtor)",
    ],
)
def test_clean_keeps_braced_text_that_is_not_a_style_rule(text: str) -> None:
    assert clean(text) == text


@pytest.mark.parametrize(
    "notice",
    [
        TEXAS_NO_TEXT,
        "An HTML version of this bill is not available. Please see the PDF for the content of "
        "this bill.\nFor additional information, consult the Legislative Budget Board's website.",
    ],
)
def test_a_notice_that_the_text_is_unavailable_is_no_text(notice: str) -> None:
    assert split_sections(WORD_STYLE_SHEET + notice).sections == []


def test_the_words_of_a_notice_inside_a_bill_are_kept() -> None:
    bill = "Section 1. Projects must develop water supply sources not currently available."
    assert split_sections(bill).sections[0].text == bill


# A chapter enacted under one section, each code section headed by its citation alone, as
# Colorado, Idaho, Rhode Island, Arizona and New York print new chapters. Invented text.
CHAPTER = """\
SECTION 1. In the revised statutes, add article 9 to title 4 as follows:
ARTICLE 9
4-9-101. Short title. This article may be cited as the Widget Act.
4-9-102. Definitions. In this article, "widget" has the meaning in section
4-1-201. A widget is not a gadget.
4-9-102.5. Scope. This article applies to widgets.
SECTION 2. Effective date. This act takes effect July 1.
"""


def test_a_citation_after_a_title_line_or_a_struck_clause_is_a_heading() -> None:
    for before in ("CHAPTER 35", "THE WIDGET ACT", "Part 2. Widget Permits", "[as follows:]"):
        text = f"SECTION 1. Adds a chapter.\n{before}\n4-9-101. Short title."
        assert [s.heading for s in split_sections(text).sections] == ["SECTION 1.", "4-9-101."]


def test_a_citation_heading_starting_a_line_splits_a_chapter() -> None:
    split = split_sections(CHAPTER)

    assert split.by_heading
    assert [s.heading for s in split.sections] == [
        "SECTION 1.",
        "4-9-101.",
        "4-9-102.",
        "4-9-102.5.",
        "SECTION 2.",
    ]
    assert split.sections[1].text == (
        "4-9-101. Short title. This article may be cited as the Widget Act."
    )


def test_a_citation_wrapped_after_the_word_section_is_a_cross_reference() -> None:
    sections = split_sections(CHAPTER).sections

    assert sections[2].text.endswith("in section\n4-1-201. A widget is not a gadget.")
    for before in (
        "sections 4-1-200 and",
        "Section",
        "SECTIONS",
        "§",
        "§§",
        "§\n",  # a blank line between
        "as provided by",
        "the requirements of",
        "under W.S.",
        "K.S.A. 4-",  # the citation itself broken at a hyphen
        "Section [4-1-200]",  # a struck citation, replaced by the next one
    ):
        text = f"SECTION 1. Adds.\n4-9-101. Text that cites {before}\n4-1-201. And goes on."
        headings = [s.heading for s in split_sections(text).sections]
        assert headings == ["SECTION 1.", "4-9-101."], before


@pytest.mark.parametrize(
    "heading",
    ["19-35-1.", "6-3901.", "7-16.1-708.", "9-33.2-10.", "15-24-101.", "39-4A-2.", "26B-9-214."],
)
def test_each_citation_form_is_a_heading(heading: str) -> None:
    text = f"SECTION 1. Adds a chapter.\n{heading} Short title. The Widget Act."

    assert [s.heading for s in split_sections(text).sections] == ["SECTION 1.", heading]


@pytest.mark.parametrize(
    "line",
    [
        "6-3903, Idaho Code, applies.",  # a wrapped reference, followed by a comma
        "11-111-113 MAY BE REVISED BY AGREEMENT.",  # no period after the number
        "2. The second numbered paragraph.",  # no hyphen: an enumerator
        "31.150. (1) A special motion.",  # a dotted citation: also an outline number
        "in 4-9-101. of the code",  # not at the start of a line
        "2025-2026. This subsection expires.",  # a year range
        "2-b. The trustees shall establish a policy.",  # New York's inserted subdivision
        "680-a. Definitions.",  # New York's table of contents; its sections start with §
    ],
)
def test_lines_that_are_not_citation_headings(line: str) -> None:
    text = f"SECTION 1. Adds a chapter.\n{line}"

    assert [s.heading for s in split_sections(text).sections] == ["SECTION 1."]


def test_clean_drops_arizona_statute_markers() -> None:
    raw = "ARTICLE 5.1. WIDGETS\n\nSTART_STATUTE6-251. Short title\n\nThe Widget Act. END_STATUTE"

    assert clean(raw) == "ARTICLE 5.1. WIDGETS\n\n6-251. Short title\n\nThe Widget Act."


def test_a_citation_never_replaces_the_window_fallback() -> None:
    # Without a section heading, a citation at a line start is a wrapped reference ("Public
    # Law\n116-260."), and taking it would turn a long bill's windows into two sections.
    text = _words(250) + "\n" + "under Public Law\n116-260. " + _words(250)

    split = split_sections(text)

    assert not split.by_heading
    assert len(split.sections) == 2
    assert all(s.heading is None for s in split.sections)


def test_a_citation_after_a_statute_abbreviation_is_a_cross_reference() -> None:
    for before in ("RSA", "Public Law", "consumer protection law"):
        text = f"SECTION 1. Adds a chapter. It applies {before}\n4-1-201. And goes on."
        assert [s.heading for s in split_sections(text).sections] == ["SECTION 1."], before
