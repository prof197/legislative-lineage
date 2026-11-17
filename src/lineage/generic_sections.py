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

Two limits keep a provision from passing for a generic section, both found reading what the
first run with this rule set aside, before any of its matches were sampled. Terms defined "as
used in this section" or "for purposes of this section" serve the section's own provisions, as
in an Ohio tax credit or a Missouri act codified as one section, so they make no definitions
section, unless the section says they apply throughout a chapter or act, as Washington's do.
And a section far longer than its kind runs has substance merged into it where sectioning found
no break, as a Florida bill's 50,949-word "construction and severability" section does:
severability, a short title, an effective date and construction are bounded by MAX_WORDS at 750
words (the longest real one read, a severability recital, ran 697), findings and a compact's
articles at 1,200 (findings ran to 1,114), and past its bound a section is a provision.
Definitions have no bound: real ones run to thousands of words.

Patterns run on normalized text (lineage.sections.normalize: lowercase words, no punctuation).
See DESIGN.md#origin-matches.
"""

import re

from lineage.boilerplate import boilerplate_kind

# How far into a section its opening runs, in words, for a definitions section's phrasing.
OPENING_WORDS = 30
# The longest a section of each kind runs before substance must be merged into it, in words.
MAX_WORDS: dict[str, int] = {
    "short title": 750,
    "severability": 750,
    "effective date": 750,
    "construction": 750,
    "findings": 1200,
    "compact clause": 1200,
}
# A section's number: "section 6", "sec 3", "section 501", "article xii", "article 3 section 2",
# and Washington's "new section sec 2".
_NUMBER = r"^(?:new (?:section )?)?(?:(?:section|sec|article|art) [0-9a-z]{1,6} ){1,2}"
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
# What an opening's terms or purpose are for: its first scope names it, so "for purposes of this
# article ... the meanings ascribed in this section" defines an article's terms.
_SCOPE = re.compile(
    r"\b(?:as used in|in|for (?:the )?purposes of|the purposes? of) this (act|section|chapter|"
    r"article|subchapter|part|compact)\b"
)
# Washington's form names the wider unit after the section: "the definitions in this section
# apply throughout this chapter".
_APPLIES = re.compile(
    r"\bdefinitions? (?:set forth |contained )?in this section (?:shall )?appl(?:y|ies) "
    r"(?:throughout|to|in) (?:all of )?this (act|chapter|article|subchapter|part|compact|title)\b"
)
_SHORT_TITLE_OPENING = re.compile(
    r"\bthis (?:act|chapter|article|subchapter) (?:shall be known and )?(?:may|shall) be "
    r"(?:cited|known|referred to) as\b"
)
_FINDINGS_OPENING = re.compile(
    r"\bthe purposes? of this (?:act|chapter) (?:is|are)\b"
    r"|^(?:\w+ ){0,4}(?:the )?(?:legislature|general assembly) (?:hereby )?(?:finds|declares)\b"
)


def generic_kind(normalized: str) -> str | None:
    """The generic kind a section is, or None for a provision: boilerplate first, as
    lineage.boilerplate flags it, then by heading, then by its opening, each kind within its
    bound (MAX_WORDS)."""
    if (kind := boilerplate_kind(normalized)) is not None:
        return kind
    words = normalized.split()
    for kind, pattern in HEADINGS.items():
        if pattern.search(normalized):
            return _bounded(kind, len(words))
    opening = " ".join(words[:OPENING_WORDS])
    scope = _SCOPE.search(opening)
    if scope is not None and scope.group(1) == "section" and not _APPLIES.search(opening):
        return None
    for kind, pattern in (
        ("definitions", _DEFINITIONS_OPENING),
        ("definitions", _APPLIES),
        ("short title", _SHORT_TITLE_OPENING),
        ("findings", _FINDINGS_OPENING),
    ):
        if pattern.search(opening):
            return _bounded(kind, len(words))
    return None


def _bounded(kind: str, words: int) -> str | None:
    """The kind, unless the section runs past its kind's bound: then substance is merged into it,
    and it is a provision whatever a later pattern would name it."""
    return kind if words <= MAX_WORDS.get(kind, words) else None
