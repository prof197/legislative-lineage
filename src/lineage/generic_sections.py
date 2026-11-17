"""Which kind of generic section a section is: the clauses any act carries whatever it enacts,
which say nothing of where its substance came from.

Origin matching leaves them out of a match on both sides (lineage.origin_matching), as ruled at
the step 47 review: of EPPC's 71 listed matches, a census, 54 were wrong, and 52 of those rested
on two models' severability sections, too long for lineage.boilerplate's whole-section patterns
(a case-law recital runs 500 to 700 words). A section is generic when it is boilerplate as
lineage.boilerplate flags it, or when its heading, right after its number, names a generic kind:
definitions, a short title, findings or purpose, severability, an effective date, rules of
construction, or a standard article of an interstate compact. A definitions section is also
known by its opening ("as used in this act", "unless the context otherwise requires"). A
heading's word met deeper in a provision names nothing, and "general provisions" is not a kind,
since some acts put substance under it.

Patterns run on normalized text (lineage.sections.normalize: lowercase words, no punctuation).
See DESIGN.md#origin-matches.
"""

import re

from lineage.boilerplate import boilerplate_kind

# How far into a section its opening runs, in words, for a definitions section's phrasing.
OPENING_WORDS = 30
# A section's number: "section 6", "sec 3", "section 501", "article xii", "article 3 section 2".
_NUMBER = r"^(?:new )?(?:(?:section|sec|article|art) [0-9a-z]{1,6} ){1,2}"
HEADINGS: dict[str, re.Pattern[str]] = {
    "definitions": re.compile(_NUMBER + r"(?:general )?definitions?\b"),
    "short title": re.compile(_NUMBER + r"(?:short title|title this (?:act|chapter))\b"),
    "findings": re.compile(
        _NUMBER + r"(?:legislative )?(?:findings?|purposes?|intent|declarations? of policy|"
        r"statement of (?:policy|purpose))\b"
    ),
    "severability": re.compile(_NUMBER + r"(?:\w+ (?:and )?)?severab(?:ility|le)\b"),
    "effective date": re.compile(_NUMBER + r"(?:\w+ (?:and )?)?effective dates?\b"),
    "construction": re.compile(
        _NUMBER + r"(?:rules? of construction|liberal construction|construction and "
        r"(?:interpretation|severability)|interpretation)\b"
    ),
    # The articles every interstate compact closes with, as the Council of State Governments'
    # model compacts head them.
    "compact clause": re.compile(
        _NUMBER + r"(?:withdrawal(?: default)?(?: and)? (?:expulsion|termination|dissolution)|"
        r"withdrawal (?:from|and) (?:the )?(?:compact|amendment)|dissolution of (?:the )?compact|"
        r"binding effect of (?:the )?compact|default technical assistance|entry into force|"
        r"oversight dispute resolution(?: and)? enforcement|enactment(?: and)? withdrawal)\b"
    ),
}
_DEFINITIONS_OPENING = re.compile(
    r"\b(?:as used in|in|for (?:the )?purposes of) this (?:act|section|chapter|article|"
    r"subchapter|part|compact)(?: unless the context (?:otherwise|clearly) (?:requires|"
    r"indicates))? (?:the following|(?:\w+ ){1,4}means|the terms?|a term|unless)"
    r"|\bunless the context (?:otherwise|clearly) (?:requires|indicates)\b"
)
_SHORT_TITLE_OPENING = re.compile(
    r"\bthis (?:act|chapter|article|subchapter) (?:shall be known and )?(?:may|shall) be "
    r"(?:cited|known|referred to) as\b"
)
_FINDINGS_OPENING = re.compile(
    r"\bthe purposes? of this (?:act|chapter|section) (?:is|are)\b"
    r"|^(?:\w+ ){0,4}(?:the )?(?:legislature|general assembly) (?:hereby )?(?:finds|declares)\b"
)


def generic_kind(normalized: str) -> str | None:
    """The generic kind a section is, or None for a provision: boilerplate first, as
    lineage.boilerplate flags it, then by heading, then by a definitions section's opening."""
    if (kind := boilerplate_kind(normalized)) is not None:
        return kind
    for kind, pattern in HEADINGS.items():
        if pattern.search(normalized):
            return kind
    opening = " ".join(normalized.split()[:OPENING_WORDS])
    if _DEFINITIONS_OPENING.search(opening):
        return "definitions"
    if _SHORT_TITLE_OPENING.search(opening):
        return "short title"
    if _FINDINGS_OPENING.search(opening):
        return "findings"
    return None
