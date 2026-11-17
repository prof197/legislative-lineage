"""Which kind of generic section a section is, by its opening: the census of a source's matches
lists the sections each wrong match rests on by kind (DESIGN.md#origin-matches)."""

import pytest

from lineage.generic_sections import generic_kind
from lineage.sections import normalize


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('Section 2. Definitions. As used in this act: (1) "Minor" means...', "definitions"),
        ('Sec. 3. In this act: (a) "Website" means an Internet site.', "definitions"),
        (
            'As used in this section, unless the context otherwise requires, "x" means y.',
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
        # its heading, though too long for lineage.boilerplate's whole-section patterns.
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
