"""Which kind of generic section a section is: the clauses any act carries whatever it enacts,
which say nothing of where its substance came from.

Origin matching leaves them out of a match on both sides (lineage.origin_matching), as ruled at
the step 47 review: of EPPC's 71 listed matches, a census, 54 were wrong, and 52 of those rested
on two models' severability sections, too long for lineage.text.boilerplate's whole-section patterns
(a case-law recital runs 500 to 700 words). A section is generic when it is boilerplate as
lineage.text.boilerplate flags it, or when its heading, right after its number, names a generic
kind: definitions, a short title, findings or purpose, severability, an effective date, rules of
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

Patterns run on normalized text (lineage.text.sections.normalize: lowercase words, no punctuation).
See DESIGN.md#origin-matches.

See DESIGN.md#dr-11.
"""

import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from lineage.text.boilerplate import boilerplate_kind
from lineage.text.sections import normalize

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
# An article's number alone, as a compact numbers its articles.
_ARTICLE = r"^(?:article|art) [0-9a-z]{1,6} "
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
        r"oversight dispute resolution(?: and)? enforcement|oversight enforcement(?: and)? dispute "
        r"resolution|enactment(?: and)? withdrawal)\b"
        # The articles that set up and run a compact's commission, under an article's number
        # only, since a section headed "financing" in an ordinary act is substance. Its
        # establishing article is known by the states creating it, since its name varies.
        r"|" + _ARTICLE + r"(?:(?:establishment of )?(?:the )?interstate (?:\w+ ){0,6}?"
        r"commission(?: for(?: \w+){1,6}?)?(?= the (?:member|compacting|party) states (?:hereby )?"
        r"(?:establish|create))|powers and duties of (?:the )?(?:\w+ ){0,4}?commission|"
        r"organization and operation of (?:the )?(?:\w+ ){0,4}?commission|"
        r"rule ?making functions of (?:the )?(?:\w+ ){0,4}?commission|"
        r"financ(?:e|ing)(?: and administration| of (?:the )?(?:\w+ ){0,4}?commission)?|"
        r"(?:member states )?effective date and amendment)\b"
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
    lineage.text.boilerplate flags it, then by heading, then by its opening, each kind within its
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


# --- Generic text inside a provision (ruled at the step 48 review) ---

# The least of a pair's shared text, on each side, that must lie outside generic text for the
# pair to count: the near-duplicate stage's shingle, five words.
MIN_WORDS = 5

# An article's heading inside a section, where sectioning found no break: "ARTICLE XVI.
# WITHDRAWAL AND DISSOLUTION" at a line's start, or after a sentence's end where a long
# paragraph lost its lines, with its title starting in capitals or on the next line. A
# cross-reference ("as specified in Article I. The commission") follows a word, not a line's
# start or a sentence's end, and one wrapped to a line's start ("pursuant to\nArticle IX of")
# runs on in lowercase.
_ARTICLE_HEADING = re.compile(
    r"(?:^|(?<=[.;:] ))[ \t]*[\"'\u201c\u2018]?(?:ARTICLE|Article)[ \t]+"
    r"(?:[IVXLC]{1,7}|[0-9]{1,3})\b"
    r"(?=[ \t]*[.:\u2013\u2014-]?[ \t]*(?:$|[A-Z]))"
    # In capitals with its number punctuated, as after a title on the same line.
    r"|(?<=\s)ARTICLE[ \t]+(?:[IVXLC]{1,7}|[0-9]{1,3})(?=[ \t]*[.:\u2013\u2014-])",
    re.MULTILINE,
)
_CONSTITUTION = (
    r"(?:constitution of the united states|united states constitution|u s constitution)"
    r"(?: of america)?"
)
# A document or bill that applies to Congress for a convention under Article V.
_ARTICLE_V = re.compile(r"\barticle v of the " + _CONSTITUTION)
# The wording every Article V application shares, whatever amendment it seeks: the application
# to Congress under Article V for a convention, and the closing that transmits it, aggregates it
# with other states' applications on its subject and keeps it standing. The subject, which says
# what the convention may propose, is left as substance.
_APPLICATION = re.compile(
    "|".join(
        (
            r"\b(?:hereby )?(?:makes? (?:an |its )?applications?|applies|petitions|requests)"
            r"(?: to)?(?: the)? congress(?: of the united states)?(?: of america)?"
            r"(?: at its (?:next )?session)?",
            r"\b(?:under|pursuant to|as provided (?:in|by|for in)|in accordance with) "
            r"(?:the provisions of )?article v of the " + _CONSTITUTION,
            r"\b(?:to call|for the calling of|calling for|for|to convene) (?:a|an) "
            r"(?:article v )?convention(?: of (?:the )?(?:several )?states)?"
            r"(?: limited to| for the (?:sole and exclusive |sole |limited |specific |exclusive )?"
            r"purposes? of)? proposing (?:an )?amendments?(?: to the " + _CONSTITUTION + ")?",
            r"\b(?:the secretary of state (?:of (?:\w+ ){1,3})?(?:is hereby directed to|is "
            r"directed to|shall) |(?:and )?also to )?transmit (?:certified |duly authenticated |"
            r"authenticated )?copies (?:of this (?:joint |concurrent )?(?:application|resolution|"
            r"memorial)|hereof)(?: \w+){0,110}? (?:in the several states(?: requesting their "
            r"cooperation)?|requesting their cooperation|from this state)",
            r"\b(?:it|this (?:joint |concurrent )?(?:application|resolution)) supersedes all "
            r"previous applications by this legislature on the same subject",
            r"\bthis (?:joint |concurrent )?(?:application|resolution|memorial) constitutes a "
            r"continuing application in accordance with article v of the " + _CONSTITUTION + r" "
            r"until the legislatures of at least two thirds(?: 2 3)? of the several states "
            r"(?:have )?made applications? on the same subject",
            r"\bthis (?:joint |concurrent )?(?:application|resolution) (?:is to|shall) be "
            r"(?:considered|deemed) as covering the same subject matter as the (?:presently "
            r"outstanding )?applications? (?:from|of) other states(?: to (?:the )?congress)?"
            r"(?: to call a convention)?",
            r"\b(?:shall|is to) be aggregated with (?:same|such applications|those applications|"
            r"the applications of other states) for the purpose of attaining the two thirds"
            r"(?: 2 3)? of (?:the )?states necessary to require (?:the )?congress to call a "
            r"(?:limited )?convention on this subject(?: but shall not be aggregated with any "
            r"other applications? on any other subjects?)?",
        )
    )
)


@dataclass(frozen=True, slots=True)
class Inside:
    """A provision's generic text: its characters in the section's display text, merged and
    sorted, and how many of the section's normalized words they hold."""

    ranges: tuple[tuple[int, int], ...]
    words: int


@dataclass(frozen=True, slots=True)
class _Tokens:
    words: list[str]  # normalized words, in order
    spans: list[tuple[int, int]]  # each word's whitespace token, in characters
    starts: list[int]  # each word's start in " ".join(words)


def _tokens(text: str) -> _Tokens:
    words: list[str] = []
    spans: list[tuple[int, int]] = []
    starts: list[int] = []
    at = 0
    for m in re.finditer(r"\S+", text):
        for word in normalize(m.group()).split():
            words.append(word)
            spans.append((m.start(), m.end()))
            starts.append(at)
            at += len(word) + 1
    return _Tokens(words, spans, starts)


def _merged(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _articles(text: str) -> list[tuple[int, int]]:
    """The characters of each generic article in `text`: from its heading to the next
    heading or the text's end, judged as a section is (generic_kind), within its bound."""
    heads = [
        m.start() + len(m.group()) - len(m.group().lstrip())
        for m in _ARTICLE_HEADING.finditer(text)
    ]
    out: list[tuple[int, int]] = []
    if not heads:
        return out
    for start, end in zip(heads, [*heads[1:], len(text)], strict=True):
        if generic_kind(normalize(text[start:end])) is not None:
            out.append((start, end))
    return out


def _application(text: str, tokens: _Tokens) -> list[tuple[int, int]]:
    """The characters of an Article V application's shared wording in `text`."""
    joined = " ".join(tokens.words)
    out: list[tuple[int, int]] = []
    for m in _APPLICATION.finditer(joined):
        first = bisect_right(tokens.starts, m.start()) - 1
        last = bisect_left(tokens.starts, m.end()) - 1
        if first <= last:
            out.append((tokens.spans[first][0], tokens.spans[last][1]))
    return out


def _inside(text: str, *, application: bool) -> tuple[list[tuple[int, int]], _Tokens]:
    tokens = _tokens(text)
    ranges = _articles(text) + (_application(text, tokens) if application else [])
    return _merged(ranges), tokens


def _counted(ranges: Sequence[tuple[int, int]], tokens: _Tokens, lo: int, hi: int) -> Inside | None:
    """The ranges within lo..hi, shifted to start at lo, and the words they hold."""
    local = [(max(s, lo) - lo, min(e, hi) - lo) for s, e in ranges if s < hi and e > lo]
    if not local:
        return None
    words = sum(
        1 for s, _ in tokens.spans if lo <= s < hi and any(a <= s - lo < b for a, b in local)
    )
    return Inside(tuple(local), words)


def generic_inside(displays: Sequence[str], *, continuous: bool) -> list[Inside | None]:
    """Each section's generic text inside it, or None when it has none: a compact's standard
    articles, where sectioning left them inside a longer section, and an Article V
    application's shared wording, in a text that applies under Article V. `continuous` says the
    sections are windows of one undivided text (lineage.text.sections' fallback), so an article
    begun in one window runs on into the next; sections split at headings are read apart."""
    application = any(_ARTICLE_V.search(normalize(d)) for d in displays)
    if not continuous:
        out: list[Inside | None] = []
        for display in displays:
            ranges, tokens = _inside(display, application=application)
            out.append(_counted(ranges, tokens, 0, len(display)))
        return out
    joined = "\n\n".join(displays)
    ranges, tokens = _inside(joined, application=application)
    out = []
    at = 0
    for display in displays:
        out.append(_counted(ranges, tokens, at, at + len(display)))
        at += len(display) + 2
    return out


def words_outside(text: str, span: tuple[int, int], ranges: Sequence[tuple[int, int]]) -> int:
    """The whitespace-separated words of text[span] that start outside every range."""
    lo, hi = span
    return sum(
        1
        for m in re.finditer(r"\S+", text[lo:hi])
        if not any(a <= lo + m.start() < b for a, b in ranges)
    )
