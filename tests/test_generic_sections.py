"""Which kind of generic section a section is, by its opening: the census of a source's matches
lists the sections each wrong match rests on by kind (DESIGN.md#origin-matches)."""

import pytest

from lineage.text.generic_sections import (
    MAX_WORDS,
    Inside,
    generic_inside,
    generic_kind,
    words_outside,
)
from lineage.text.sections import normalize


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('Section 2. Definitions. As used in this act: (1) "Minor" means...', "definitions"),
        ('Sec. 3. In this act: (a) "Website" means an Internet site.', "definitions"),
        (
            'As used in this chapter, unless the context otherwise requires, "x" means y.',
            "definitions",
        ),
        (
            "For purposes of this article, the following words have the meanings ascribed in "
            'this section: (1) "Lamp" means a light.',
            "definitions",
        ),
        (
            "NEW SECTION. Sec. 2. The definitions in this section apply throughout this chapter "
            'unless the context clearly requires otherwise. (1) "Lamp" means a light.',
            "definitions",
        ),
        (
            "NEW SECTION. Sec. 2. DEFINITIONS. (1) Lamp. A light that is lit at dusk.",
            "definitions",
        ),
        ("Section 1. Short Title. This act may be cited as the Lamp Act.", "short title"),
        ("This Act shall be known and may be cited as the Lantern Act.", "short title"),
        ("Section 2. Legislative Findings. The legislature finds that lamps...", "findings"),
        ("Section 1. Purpose. The purpose of this act is to protect lamps.", "findings"),
        (
            "Section 9. Severability. If any provision of this act is held invalid, the "
            "remainder is not affected.",
            "severability",
        ),
        ("Section 10. Effective Date. This act takes effect July 1.", "effective date"),
        ("Section 4. Liability. A commercial entity that fails to verify age is liable.", None),
    ],
)
def test_a_section_is_named_by_its_opening(text: str, kind: str | None) -> None:
    assert generic_kind(normalize(text)) == kind


def test_a_definition_deep_in_a_section_is_not_a_definitions_section() -> None:
    text = "Section 4. Duties. " + "A lamp shall be lit at dusk. " * 12 + '"Dusk" means sunset.'
    assert generic_kind(normalize(text)) is None


LONG = " ".join(["The court shall consider the legislature's intent in each case."] * 12)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        # A long severability section, the case-law recital EPPC's models carry, is generic by
        # its heading, though too long for lineage.text.boilerplate's whole-section patterns.
        (f"Section 6. Severability. Mindful of Leavitt v. Jane L. {LONG}", "severability"),
        (f"Section 501. Severability. (1) {LONG}", "severability"),
        (f"Section 12. Effective Date. {LONG}", "effective date"),
        (f"Section 9. Rules of Construction. {LONG}", "construction"),
        (f"Article X. Withdrawal, Default and Expulsion. {LONG}", "compact clause"),
        (f"Article XII. Binding Effect of Compact and Other Laws. {LONG}", "compact clause"),
        (f"Section 3. Definitions. {LONG}", "definitions"),
    ],
)
def test_a_long_section_is_named_by_its_heading(text: str, kind: str) -> None:
    assert generic_kind(normalize(text)) == kind


@pytest.mark.parametrize(
    "text",
    [
        f"Section 4. Construction Contracts. A contractor shall post a bond. {LONG}",
        f"Section 5. Liability. A commercial entity is liable for damages. {LONG}",
        f"Section 7. Default Judgments. A court may enter a judgment. {LONG}",
        f"Section 1. Title IX of the Education Code is amended to read. {LONG}",
        # A heading's word deep in a provision names nothing.
        f"Section 8. Enforcement. The attorney general may sue. {LONG} Severability applies.",
    ],
)
def test_a_provision_is_not_generic_for_a_word_its_heading_shares(text: str) -> None:
    assert generic_kind(normalize(text)) is None


@pytest.mark.parametrize(
    "text",
    [
        # Terms defined for the section itself serve its own provisions.
        'Sec. 122.16. (A) As used in this section, "distressed area" means a municipal '
        "corporation. (B) The director may grant a credit against the tax.",
        'Section 818. 1. For purposes of this section, "evidence-based" means rigorous. '
        "2. The department shall develop a list of programs.",
        # An act codified as one section: its short title, then its definitions, then the act.
        "537.529. 1. This section shall be known and may be cited as the Lamp Act. 2. As used "
        'in this section, "lamp" means a light. 3. A person shall light a lamp.',
        "Section 40-1-730. (A) The purpose of this section is to offer loans. (B) The "
        "department shall develop a program.",
    ],
)
def test_terms_defined_for_their_own_section_make_no_definitions_section(text: str) -> None:
    assert generic_kind(normalize(text)) is None


SENTENCE = "The court shall consider the legislature's intent."
SENTENCE_WORDS = len(normalize(SENTENCE).split())


def _sentences(n: int) -> str:
    return " ".join([SENTENCE] * n)


@pytest.mark.parametrize(
    ("heading", "kind"),
    [
        ("Section 12. Construction and Severability.", "severability"),
        ("Section 1. Short Title.", "short title"),
        ("Section 8. Effective Date.", "effective date"),
        ("Section 9. Rules of Construction.", "construction"),
        ("Section 2. Legislative Findings.", "findings"),
        ("Article XII. Binding Effect of Compact and Other Laws.", "compact clause"),
    ],
)
def test_a_section_past_its_kinds_bound_is_a_provision(heading: str, kind: str) -> None:
    bound = MAX_WORDS[kind]
    within = normalize(f"{heading} {_sentences((bound - 12) // SENTENCE_WORDS)}")
    past = normalize(f"{heading} {_sentences(bound // SENTENCE_WORDS + 1)}")
    assert len(within.split()) <= bound < len(past.split())
    assert generic_kind(within) == kind
    assert generic_kind(past) is None


def test_a_definitions_section_has_no_bound() -> None:
    text = 'Section 2. Definitions. As used in this act: "x" means y. ' + _sentences(800)
    assert generic_kind(normalize(text)) == "definitions"


def test_a_heading_past_its_bound_does_not_fall_through_to_an_opening() -> None:
    text = (
        'Section 1. Short Title. This act may be cited as the Lamp Act. As used in this act, "x" '
        f"means y. {_sentences(200)}"
    )
    assert generic_kind(normalize(text)) is None


# --- Generic text inside a provision (ruled at the step 48 review) ---

# An invented compact in the shape interstate compacts share: substantive articles, then the
# articles that set up and run its commission and close it. Each article is a few sentences.
LAMP = "The lamp keeper shall light each lamp at dusk and record the hour."
COMMISSION = "The commission shall meet once a year and keep minutes of each meeting."
COMPACT_ARTICLES = [
    ("ARTICLE III. LAMP KEEPING.", LAMP, False),
    ("ARTICLE IV. LAMP INSPECTION.", "Each member state shall inspect its lamps in spring.", False),
    (
        "ARTICLE V. INTERSTATE COMMISSION FOR LAMP KEEPING.",
        "The member states hereby establish the Interstate Commission for Lamp Keeping. "
        + COMMISSION,
        True,
    ),
    ("ARTICLE VI. POWERS AND DUTIES OF THE INTERSTATE COMMISSION.", COMMISSION, True),
    ("ARTICLE VII. ORGANIZATION AND OPERATION OF THE INTERSTATE COMMISSION.", COMMISSION, True),
    ("ARTICLE VIII. RULEMAKING FUNCTIONS OF THE INTERSTATE COMMISSION.", COMMISSION, True),
    ("ARTICLE IX. OVERSIGHT, ENFORCEMENT, AND DISPUTE RESOLUTION.", COMMISSION, True),
    ("ARTICLE X. FINANCING OF THE COMMISSION.", COMMISSION, True),
    ("ARTICLE XI. MEMBER STATES, EFFECTIVE DATE AND AMENDMENT.", COMMISSION, True),
    ("ARTICLE XII. WITHDRAWAL AND DISSOLUTION.", COMMISSION, True),
    ("ARTICLE XIII. SEVERABILITY AND CONSTRUCTION.", COMMISSION, True),
    ("ARTICLE XIV. BINDING EFFECT OF COMPACT AND OTHER LAWS.", COMMISSION, True),
    ("ARTICLE XV. LAMPS ON TRIBAL LANDS.", LAMP, False),
]


def _compact() -> str:
    return "Section 8. The Lamp Compact is enacted as follows:\n" + "\n".join(
        f"{heading}\n{body}" for heading, body, _ in COMPACT_ARTICLES
    )


def _generic_text(text: str, inside: Inside | None) -> str:
    return " ".join(text[a:b] for a, b in (inside.ranges if inside else ()))


def test_a_compacts_standard_articles_inside_a_section_are_generic_text() -> None:
    """A whole compact sectioned as one section: its commission's and closing articles are
    generic text inside it, each heading to the next, and its substance is not."""
    text = _compact()
    (inside,) = generic_inside([text], continuous=False)
    assert inside is not None
    generic = _generic_text(text, inside)
    for heading, _body, is_generic in COMPACT_ARTICLES:
        assert (heading in generic) == is_generic, heading
    assert generic_kind(normalize(text)) is None  # the section itself stays a provision
    assert LAMP not in generic
    assert inside.words == sum(
        len(normalize(f"{h} {b}").split()) for h, b, g in COMPACT_ARTICLES if g
    )


def test_a_cross_reference_is_no_article_heading() -> None:
    text = (
        "ARTICLE XII. WITHDRAWAL AND DISSOLUTION.\n" + COMMISSION + "\n"
        "ARTICLE XV. LAMPS ON TRIBAL LANDS.\nA tribe may keep lamps as specified in "
        "Article XII. The tribe shall act pursuant to\nArticle XII of this compact. " + LAMP
    )
    (inside,) = generic_inside([text], continuous=False)
    assert inside is not None
    generic = _generic_text(text, inside)
    assert "WITHDRAWAL" in generic
    assert "TRIBAL" not in generic
    assert "tribe" not in generic


def test_an_article_runs_on_through_the_fallbacks_windows() -> None:
    """Windows of one undivided text: an article begun in one window is generic into the next,
    up to the next heading; sections split at headings are read apart."""
    windows = [
        LAMP + " ARTICLE XII: WITHDRAWAL AND DISSOLUTION " + COMMISSION,
        COMMISSION + " " + COMMISSION,
        "ARTICLE XV: LAMPS ON TRIBAL LANDS " + LAMP,
    ]
    first, second, third = generic_inside(windows, continuous=True)
    assert first is not None
    assert LAMP not in _generic_text(windows[0], first)
    assert second is not None
    assert second.words == len(normalize(windows[1]).split())
    assert third is None
    assert generic_inside(windows, continuous=False)[1] is None


@pytest.mark.parametrize(
    ("number", "title"),
    [("Section 5.", "Financing of the commission."),
     ("Sec. 4.", "Powers and duties of the interstate commission.")],
)  # fmt: skip
def test_a_commissions_articles_are_compact_clauses_only_under_an_articles_number(
    number: str, title: str
) -> None:
    assert generic_kind(normalize(f"{number} {title} {COMMISSION}")) is None
    assert generic_kind(normalize(f"Article 5. {title} {COMMISSION}")) == "compact clause"


# An invented Article V application: the shared wording around a subject.
SUBJECT = "that require each state to keep its lamps lit from dusk until dawn"
APPLICATION = [
    "Section 1. The legislature of the State of Lampland hereby applies to Congress, under the "
    "provisions of Article V of the Constitution of the United States, for the calling of a "
    "convention of the states limited to proposing amendments to the Constitution of the "
    f"United States {SUBJECT}.",
    "Section 2. The secretary of state is hereby directed to transmit copies of this "
    "application to the President and Secretary of the United States Senate and to the "
    "Speaker and Clerk of the United States House of Representatives, and copies to the "
    "members of the said Senate and House of Representatives from this state; also to "
    "transmit copies hereof to the presiding officers of each of the legislative houses in "
    "the several states, requesting their cooperation.",
    "Section 3. This application constitutes a continuing application in accordance with "
    "Article V of the Constitution of the United States until the legislatures of at least "
    "two-thirds of the several states have made applications on the same subject.",
]


def test_an_article_v_applications_shared_wording_is_generic_and_its_subject_is_not() -> None:
    first, second, third = generic_inside(APPLICATION, continuous=False)
    assert first is not None
    assert second is not None
    assert third is not None
    kept = APPLICATION[0]
    for a, b in reversed(first.ranges):
        kept = kept[:a] + kept[b:]
    assert SUBJECT in kept
    assert "Congress" not in kept
    assert "convention" not in kept
    # The closings hold nothing outside their shared wording but their numbers.
    assert len(normalize(APPLICATION[1]).split()) - second.words < 5
    assert len(normalize(APPLICATION[2]).split()) - third.words < 5


def test_the_application_wording_is_generic_only_where_article_v_is_applied_under() -> None:
    memorial = [
        "Section 1. The legislature urges Congress to fund lamps.",
        APPLICATION[1],
    ]
    assert generic_inside(memorial, continuous=False) == [None, None]


def test_words_outside_count_a_spans_words_that_start_outside_the_ranges() -> None:
    text = "one two three four five six"
    assert words_outside(text, (0, len(text)), [(4, 13)]) == 4  # two and three are inside
    assert words_outside(text, (4, 13), [(4, 13)]) == 0
