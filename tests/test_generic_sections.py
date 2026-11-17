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
