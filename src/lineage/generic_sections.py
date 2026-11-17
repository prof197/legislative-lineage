"""Which kind of generic section a section is: definitions, a short title, findings or a
statement of purpose, or the boilerplate lineage.boilerplate already strips.

A census of one source's matches (lineage.origin_precision) lists the sections each wrong match
rests on by kind, to see whether the errors rest on generic passages, which the parked fix
would leave out of the shares on both sides (DESIGN.md#parked). A kind is read from a section's
opening, its heading and first words, since a definitions section announces itself there; a
definition met deep in a provision does not make the provision generic. Patterns run on
normalized text (lineage.sections.normalize: lowercase words, no punctuation). See
DESIGN.md#origin-matches.
"""

import re

from lineage.boilerplate import boilerplate_kind

# How far into a section its opening runs, in words.
OPENING_WORDS = 30
KINDS: dict[str, re.Pattern[str]] = {
    "definitions": re.compile(
        r"\bdefinitions?\b"
        r"|\b(?:as used in|in|for (?:the )?purposes of) this (?:act|section|chapter|article|"
        r"subchapter|part|compact)(?: unless the context (?:otherwise|clearly) (?:requires|"
        r"indicates))? (?:the following|(?:\w+ ){1,4}means|the terms?|a term|unless)"
        r"|\bunless the context (?:otherwise|clearly) (?:requires|indicates)\b"
    ),
    "short title": re.compile(
        r"\bshort title\b|\bthis (?:act|chapter|article|subchapter) (?:shall be known and )?"
        r"(?:may|shall) be (?:cited|known|referred to) as\b"
    ),
    "findings": re.compile(
        r"\b(?:legislative )?(?:findings|declarations?|statement of policy|intent|purposes?)\b"
        r"(?: \w+){0,3}? (?:the )?(?:legislature|general assembly|congress|state) (?:hereby )?"
        r"(?:finds|declares)\b"
        r"|^(?:\w+ ){0,3}(?:legislative )?(?:findings|purposes?|intent)\b"
        r"|\bthe purposes? of this (?:act|chapter|section) (?:is|are)\b"
        r"|\b(?:legislature|general assembly) (?:hereby )?(?:finds|declares)\b"
    ),
}


def generic_kind(normalized: str) -> str | None:
    """The generic kind a section is, or None for a provision: boilerplate first, as
    lineage.boilerplate flags it, then by the section's opening."""
    if (kind := boilerplate_kind(normalized)) is not None:
        return kind
    opening = " ".join(normalized.split()[:OPENING_WORDS])
    for kind, pattern in KINDS.items():
        if pattern.search(opening):
            return kind
    return None
