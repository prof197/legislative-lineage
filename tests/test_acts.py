"""Act texts as query documents: only the enacted provisions are kept.

The act below is invented, laid out the way the commission's final acts are: a title page and
committee list, a table of contents, a prefatory note, then sections, each followed by an
official comment or a legislative note of one or more paragraphs. No real act text is committed
(DESIGN.md#commission-recall).
"""

from pathlib import Path

import pytest

from lineage.acts import ACTS, ActTextError, enacted_text, load_act
from lineage.sections import normalize

ACT = """\
Uniform Widget Registration Act

drafted by the

NATIONAL CONFERENCE OF COMMISSIONERS ON UNIFORM STATE LAWS

and by it

APPROVED AND RECOMMENDED FOR ENACTMENT IN ALL THE STATES

The Committee appointed by and representing the commission consists of the following:

A Commissioner, Chair

Copyright © 2099 by the commission

Table of Contents

Section 1. Short Title\t1
Section 2. Definitions\t1
[Section 3. Registration]\t3
Section 4. Effective Date\t4

Uniform Widget Registration Act

Prefatory Note

Section 1 of this act names it. Widgets are registered so that their owners can be found.

Section 1. Short Title

This [act] may be cited as the Uniform Widget Registration Act.

Comment

The short title follows the commission's usual form. Section 2 defines its terms.

Section 2. Definitions

In this [act]:

(1) "Widget" means a device registered under Section 3.

(2) "Owner" means a person that holds title to a widget.

Legislative Note: A state that already defines "owner" should cross-reference its definition.

Comment

1. The definition of widget is deliberately broad.

2. Section 3 sets out the registration duty.

[Section 3. Registration

(a) An owner shall register a widget with the [secretary of state] not later than [30] days
after acquiring it.

(b) A registration expires after [five] years.]

Legislative Note: Bracketed text may be adapted to the state's own offices and periods.

A state may set a shorter period for an owner that registers many widgets.

Section 4. Effective Date

This [act] takes effect [effective date].
"""


def test_keeps_the_sections_and_drops_everything_else() -> None:
    text = enacted_text(ACT)

    assert text.startswith("Section 1. Short Title")
    assert text.rstrip().endswith("This [act] takes effect [effective date].")
    for kept in ("may be cited as", '"Widget" means', "(b) A registration expires", "[30] days"):
        assert kept in text
    for dropped in (
        "Commissioner, Chair",
        "Table of Contents",
        "Prefatory Note",
        "Widgets are registered so that",
        "Comment",
        "deliberately broad",
        "Legislative Note",
        "cross-reference its definition",
        "Bracketed text may be adapted",
        "A state may set a shorter period",
    ):
        assert dropped not in text


def test_a_bracketed_optional_section_is_kept_as_a_section() -> None:
    text = enacted_text(ACT)

    assert "[Section 3. Registration" in text
    assert "Section 3 sets out" not in text  # from a comment, not the act


# Longer acts are divided into articles, with sections numbered by article; an optional article
# can be no more than a bracketed heading and a note.
ARTICLES = """\
Uniform Widget Registration Act

Copyright © 2099 by the commission

Uniform Widget Registration Act
[Article] 1
General Provisions
\tSection 101. Title
\tThis [act] may be cited as the Uniform Widget Registration Act.
[[Article] 2
Widget Registry]
Legislative Note: A state that wishes to keep a registry has two options:
(1) The state may insert a registry as Article 2 of this act.
(2) The state may omit Article 2.
[Article] 3
Miscellaneous Provisions
\tSection 301. Effective Date
\tThis [act] takes effect [effective date].
"""


def test_an_act_in_articles_starts_at_its_first_article() -> None:
    text = enacted_text(ARTICLES)

    assert text.startswith("[Article] 1\nGeneral Provisions\n\tSection 101. Title")
    for kept in ("[[Article] 2", "Widget Registry]", "[Article] 3", "Section 301. Effective Date"):
        assert kept in text
    for dropped in ("Copyright", "Legislative Note", "two options", "(1) The state may insert"):
        assert dropped not in text


def test_an_act_with_no_numbered_sections_is_refused() -> None:
    with pytest.raises(ActTextError, match="no section or article heading"):
        enacted_text("Uniform Widget Registration Act\n\nPrefatory Note\n\nWidgets matter.\n")


def test_loads_a_named_act_from_the_local_directory(tmp_path: Path) -> None:
    (tmp_path / f"{ACTS[0]}.txt").write_text(ACT, encoding="utf-8")

    assert load_act(ACTS[0], tmp_path).startswith("Section 1. Short Title")
    with pytest.raises(ActTextError, match=ACTS[1]):
        load_act(ACTS[1], tmp_path)


# The commission marks text a state adapts in brackets: a placeholder such as "[act]" or
# "[name of state]", or an optional clause. The parser keeps them as written, and what they
# cost in matching is parked until a miss traces to one (DESIGN.md#parked); these tests pin
# what reaches the query meanwhile.
PLACEHOLDERS = """\
Section 1. Short Title
This [act] may be cited as the Uniform Widget Registration Act.

Section 2. Registry
The [name of state agency] shall keep a registry of widgets[, and may charge a fee].
"""


def test_bracketed_placeholders_and_optional_text_are_kept_as_written() -> None:
    text = enacted_text(PLACEHOLDERS)

    assert "This [act] may be cited" in text
    assert "The [name of state agency] shall keep" in text
    assert "widgets[, and may charge a fee]." in text


def test_a_generic_placeholder_normalizes_like_its_word_and_a_filled_one_does_not() -> None:
    act_line = normalize("This [act] may be cited as the Uniform Widget Registration Act.")
    agency_line = normalize("The [name of state agency] shall keep a registry of widgets.")

    assert act_line == normalize("This act may be cited as the Uniform Widget Registration Act.")
    assert agency_line != normalize("The Department of Revenue shall keep a registry of widgets.")


# An invented section as the commission drafts one, with its placeholders about as dense as the
# densest sections of the development acts, and two ways a state might enact it.
DRAFTED = """\
Section 4. Registration of Widgets
(a) A person that sells widgets in [this state] shall register with the [name of state agency]
before the first sale. The registration must state the person's name, its principal office and
each place in [this state] where it keeps widgets for sale.
(b) The [name of state agency] [shall][may] issue a certificate of registration within 30 days
after receiving a complete application[, and may charge a reasonable fee].
(c) A person that fails to register is subject to a civil penalty under [insert citation].
"""
# Every placeholder filled in, one alternative chosen and the optional clause dropped.
FILLED = """\
Section 4. Registration of Widgets
(a) A person that sells widgets in this Commonwealth shall register with the Department of
Revenue before the first sale. The registration must state the person's name, its principal
office and each place in this Commonwealth where it keeps widgets for sale.
(b) The Department of Revenue shall issue a certificate of registration within 30 days after
receiving a complete application.
(c) A person that fails to register is subject to a civil penalty under section 12-401 of the
Revised Statutes.
"""
# The generic placeholders kept as their words, the agency named, the optional clause kept.
LIGHTLY_FILLED = """\
Section 4. Registration of Widgets
(a) A person that sells widgets in this state shall register with the Department of Revenue
before the first sale. The registration must state the person's name, its principal office and
each place in this state where it keeps widgets for sale.
(b) The Department of Revenue shall issue a certificate of registration within 30 days after
receiving a complete application, and may charge a reasonable fee.
(c) A person that fails to register is subject to a civil penalty under section 12-401.
"""


def _proposed_and_estimate(act: str, bill: str) -> tuple[bool, float]:
    """The near-duplicate stage on one section pair, as the index treats one window."""
    from lineage.lsh import band_keys
    from lineage.minhash import permutations, signatures, similarity
    from lineage.shingles import section_shingles

    texts = [normalize(act), normalize(bill)]
    assert all(len(t.split()) <= 94 for t in texts)  # one window each
    signed = signatures(section_shingles(texts), permutations())
    keys = band_keys(signed)
    return bool((keys[0] == keys[1]).any()), similarity(signed[0], signed[1])


def test_the_query_keeps_the_placeholders_alternatives_and_optional_clause() -> None:
    drafted = enacted_text(DRAFTED)

    for kept in ("[this state]", "[insert citation]", "[shall][may]", "[, and may charge"):
        assert kept in drafted


def test_a_lightly_filled_enactment_is_proposed_and_verified() -> None:
    from lineage.verify import THRESHOLD

    proposed, estimate = _proposed_and_estimate(enacted_text(DRAFTED), LIGHTLY_FILLED)

    assert proposed
    assert estimate >= THRESHOLD


def test_a_fully_filled_enactment_is_proposed_but_lost_at_verification() -> None:
    # Each filled placeholder changes the shingles around it, and at this density they add up:
    # the section is the semantic stage's to find (DESIGN.md#commission-recall). Pinned so that a
    # change to matching that recovers it is seen, not assumed.
    from lineage.verify import THRESHOLD

    proposed, estimate = _proposed_and_estimate(enacted_text(DRAFTED), FILLED)

    assert proposed
    assert estimate < THRESHOLD
