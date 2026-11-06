import pytest

from lineage.boilerplate import boilerplate_kind
from lineage.sections import normalize


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("BE IT ENACTED BY THE GENERAL ASSEMBLY OF GEORGIA:", "enacting clause"),
        (
            "Be it enacted by the Senate and House of Representatives in General Court "
            "assembled, and by the authority of the same, as follows:",
            "enacting clause",
        ),
        ("The People of the State of California do enact as follows:", "enacting clause"),
        (
            "SECTION 4. This Act shall become effective upon its approval by the Governor or "
            "upon its becoming law without such approval.",
            "effective date",
        ),
        ("Section 9. This act takes effect July 1, 2025.", "effective date"),
        ("SECTION 3. This act shall take effect immediately.", "effective date"),
        (
            "Sec. 12. If any provision of this act or its application to any person or "
            "circumstance is held invalid, the invalidity does not affect other provisions or "
            "applications of the act that can be given effect without the invalid provision or "
            "application, and to this end the provisions of this act are severable.",
            "severability",
        ),
        ("Section 5. The provisions of this act are severable.", "severability"),
        (
            "SECTION 5. All laws and parts of laws in conflict with this Act are repealed.",
            "repealer",
        ),
        # The Uniform Law Commission's standard sections, in the act's wording and in states'.
        (
            "SECTION 12. UNIFORMITY OF APPLICATION AND CONSTRUCTION. In applying and construing "
            "this uniform act, consideration must be given to the need to promote uniformity of "
            "the law with respect to its subject matter among states that enact it.",
            "uniformity clause",
        ),
        (
            "6-265. Uniformity of application and construction\nIn applying and construing this "
            "article, a court shall consider the promotion of uniformity of the law among "
            "jurisdictions that enact it.",
            "uniformity clause",
        ),
        (
            "Section 402. Relation to Electronic Signatures in Global and National Commerce Act\n"
            "This [act] modifies, limits, or supersedes the Electronic Signatures in Global and "
            "National Commerce Act, 15 U.S.C. Section 7001 et seq.[, as amended], but does not "
            "modify, limit, or supersede 15 U.S.C. Section 7001(c), or authorize electronic "
            "delivery of any of the notices described in 15 U.S.C. Section 7003(b).",
            "federal e-sign relation",
        ),
    ],
)
def test_flags_sections_that_are_only_boilerplate(text: str, kind: str) -> None:
    assert boilerplate_kind(normalize(text)) == kind


def test_a_heading_plus_two_boilerplate_clauses_is_boilerplate() -> None:
    text = (
        "SECTION 6. This Act shall become effective upon its approval by the Governor. "
        "All laws and parts of laws in conflict with this Act are repealed."
    )

    assert boilerplate_kind(normalize(text)) is not None


@pytest.mark.parametrize(
    "text",
    [
        # Substance with an effective date attached is a provision, not boilerplate.
        "Section 2. A person who distributes a synthetic image of another person without "
        "consent commits a misdemeanor punishable by a fine of not more than $1,000. "
        "This section takes effect January 1, 2026.",
        # A preamble carries the title, which is not boilerplate, along with the clause.
        "AN ACT relating to the licensing of electricians and the duties of the board. "
        "BE IT ENACTED BY THE LEGISLATURE OF THE STATE OF IDAHO:",
        "SECTION 1. The department shall publish the report on its website.",
        # An act's own construction rule is substance, unlike the uniformity clause.
        "SECTION 11. CONSTRUCTION. This [act] must be broadly construed and applied to protect "
        "the exercise of the right of freedom of speech and of the press, the right to assemble "
        "and petition, and the right of association.",
        "",
    ],
)
def test_leaves_substantive_sections_alone(text: str) -> None:
    assert boilerplate_kind(normalize(text)) is None
