"""Split a bill's text into sections, the unit every later stage matches on.

Provisions travel between states section by section, often inside omnibus bills, so a whole
bill is too coarse to compare. A bill is split at its section headings. When none are found,
the text is cut into windows of about `WINDOW_WORDS` words on paragraph boundaries, so every
bill with text still yields comparable units. See DESIGN.md#sections.

A bill that enacts a new chapter often does so under one "SECTION 1.", heading each code
section with its citation alone ("4-9-101. Short title."). Those citations are headings too,
or the whole chapter would be one section that cannot match the provisions it carries.

Some states number bill or code sections in a form no general pattern can take without
misreading other states' lines, so those forms are rules for that state's bills alone
(`STATE_HEADINGS`).
"""

import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache

WINDOW_WORDS = 200

# Headings start a line: "Section 1.", "SECTION 1.", "Sec. 1.", "SEC. 1.", "§ 1.", with a colon
# allowed for the period and code-style numbers such as "1-101", "2a" or New York's "680-a".
# Alaska prefixes an asterisk and Washington "NEW SECTION."; both are skipped. Lowercase
# "section" is left out on purpose: at the start of a line it is a wrapped cross-reference far
# more often than a heading.
_HEADING = re.compile(
    r"^(?:\*[ \t]*)?(?:NEW SECTION\.[ \t]*)?"
    r"(?P<heading>(?:SECTION|Section|SEC\.|Sec\.|§)[ \t]*"
    r"\d+[A-Za-z]?(?:[.-]\d+[A-Za-z]?|-[A-Za-z]{1,2})*[ \t]*[.:])(?=\s|$)",
    re.MULTILINE,
)
# A code citation alone at the start of a line, then a period: "19-35-1.", "6-3901.",
# "9-33.2-10.", "39-4A-2.". At least one hyphen is required, so enumerators ("2.") and outline
# numbers ("31.150.") are not taken for citations, and a year range ("2025-2026.") is not one.
# Each part after a hyphen starts with a digit: New York labels inserted subdivisions "2-b."
# and lists an article's contents as "680-a. Definitions.", while its sections start with §.
_CITATION = re.compile(
    r"^(?!(?:19|20)\d\d-\d\d(?:\d\d)?\.)"
    r"(?P<heading>\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*(?:-\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*)+\.)"
    r"(?=\s|$)",
    re.MULTILINE,
)
# A citation is a wrapped cross-reference, not a heading, when the line before ends mid-sentence
# on a word that introduces one ("as provided by\n1-6-105.", "Pub. L.\n109-248."), on a hyphen
# that breaks the citation itself ("16a-\n4-109."), or on a struck citation that Utah replaces
# with the new one ("Section [76-10-1236]\n76-5c-209."). A blank line between them does not
# change that, and neither does a line number a PDF left after a lowercase "section" ("under
# section 3"). Only that word: a page footer ("Page 2 of 3") or a capitalized title line
# ("CHAPTER 35") also ends in a number, and a heading follows them.
_CITES = re.compile(
    r"(?:[§-]|\d\]|\b(?:sections?|subsections?|chapters?|titles?|articles?|items?|w\.s\.|and|or|to"
    r"|through|in|of|by|under|law|rsa|pub\.[ \t]*l\.)"
    r"|(?-i:\b(?:sections?|subsections?)[ \t]+\d{1,3}))\n\n?\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StateHeadings:
    """One state's own heading formats, each in the numbering its bills and code use."""

    sections: tuple[re.Pattern[str], ...] = ()  # headings that split a bill by themselves
    citations: tuple[re.Pattern[str], ...] = ()  # headings only within a bill already split
    page_header: re.Pattern[str] | None = None  # a running header that changes on each page


# Each rule applies only to its state's bills, since elsewhere a line of the same shape is an
# outline number or a wrapped reference. A citation that follows a word introducing a reference
# is skipped, as for the general citation rule.
STATE_HEADINGS = {
    # North Dakota Century Code sections are title-chapter-section ("59-22-01.", "6-08.6-04."),
    # and the legislature's PDFs space the hyphens ("59 - 22 - 01.").
    "ND": StateHeadings(
        citations=(
            re.compile(
                r"^(?P<heading>\d+(?:\.\d+)?(?:[ \t]*-[ \t]*\d+(?:\.\d+)?){2}\.)(?=\s|$)",
                re.MULTILINE,
            ),
        ),
    ),
    # North Carolina letters a bill section's subsections ("SECTION 45.(a)", "SECTION 2.1.(b)"),
    # and quotes each General Statutes section a bill enacts ('"§ 50A-411.', '"§ 48-3-230.').
    "NC": StateHeadings(
        sections=(
            re.compile(
                r"^(?P<heading>SECTION[ \t]+\d+[A-Z]?(?:\.\d+[A-Z]?)*\.\([a-z]{1,2}\))(?=\s|$)",
                re.MULTILINE,
            ),
        ),
        citations=(
            re.compile(
                r'^"(?P<heading>§[ \t]*\d+[A-Z]*(?:-\d+[A-Z]*(?:\.\d+[A-Z]*)?)+\.)(?=\s|$)',
                re.MULTILINE,
            ),
        ),
    ),
    # New Hampshire numbers bill sections and gives each a caption ending in a period ("1 New
    # Chapter; Uniform Special Deposits Act.", "228:1 ..." once chaptered); RSA sections are
    # chapter:section ("384-H:1 Short Title.", "21-I:7-a"). Its paragraphs are numbered in
    # roman numerals and letters, so a line opening with an arabic number and a caption is not
    # one; a caption needs three characters before its period, so "15 U.S.C." is not one.
    "NH": StateHeadings(
        sections=(
            re.compile(
                r"^(?P<heading>(?:\d+:)?\d+)(?=[ \t][A-Z][^.\n]{2,}\.(?:\s|$))", re.MULTILINE
            ),
        ),
        citations=(
            re.compile(
                r"^(?P<heading>\d+(?:-[A-Z]{1,2})?:\d+(?:-[a-z])?)(?=[ \t][A-Z])", re.MULTILINE
            ),
        ),
    ),
    # Missouri letters a bill's sections ("Section A.", "Section B."), RSMo sections are
    # chapter.section ("474.540.", "208.1050."), and each page after the first opens with the
    # bill's number and the page ("SB 289 2", "CCS SS SCS HCS HB 4 12").
    "MO": StateHeadings(
        sections=(re.compile(r"^(?P<heading>Section[ \t]+[A-Z]\.)(?=\s|$)", re.MULTILINE),),
        citations=(re.compile(r"^(?P<heading>\d+[A-Z]?\.\d+\.)(?=\s|$)", re.MULTILINE),),
        page_header=re.compile(
            r"^(?:[A-Z]{1,4} )*(?:[HS]B|[HS]J?R|[HS]C?R) \d+ \d+\n?", re.MULTILINE
        ),
    ),
}
_NO_STATE_HEADINGS = StateHeadings()
# Arizona's export wraps each code section in markers glued to the citation:
# "START_STATUTE6-251. Short title ... END_STATUTE".
_STATUTE_MARKERS = re.compile(r"START_STATUTE|END_STATUTE")
# A line-number gutter: a short number alone or followed by whitespace, never by "." or ")".
_GUTTER = re.compile(r"^\d{1,3}(?=[ \t]|$)[ \t]*")
# Treat leading numbers as a gutter only when most lines carry one, as printed bills do.
_GUTTER_MIN_LINES = 3
_GUTTER_MIN_SHARE = 0.5
# Line numbers a PDF prints after the text, or in a column of their own, come out at the end of
# each line (glued to the last word or after a space) or on lines alone. They are recognized as
# a run of lines whose numbers count up by one, restarting at 1 on each page; a number that only
# looks like one ends a run within a few lines.
_LINE_RUN_MIN = 5
_LINE_RUN_MAX_GAP = 3
# A running page header repeats on every page: short, with a number, and without lowercase
# ("LC 47 3375", "STATE OF WYOMING 25LSO-0123"). Repeated lines that read as text, such as
# "is amended to read as follows:", stay; the document-frequency stoplist handles those.
_HEADER_MIN_REPEATS = 3
_HEADER_MAX_WORDS = 8
_NOT_WORD = re.compile(r"[\W_]+")
# Style sheets survive the source's conversion from HTML: Texas's Word exports carry one inside
# an HTML comment, and Texas's and West Virginia's other pages put style rules before the text.
# A rule's body must be nothing but lowercase `property: value` declarations, and a rule is
# dropped only on lines that hold nothing else, so braces in bill text (Kansas marks amendments
# with them) are never taken for a rule. Items end at a mandatory separator, so a failing match
# cannot backtrack exponentially.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SELECTOR = r"[\w.#*@:-]+(?:(?:[ \t]*,\s*|[ \t]+)[\w.#*@:-]+)*"
_DECLARATION = r"\s*[a-z][a-z0-9-]*\s*:[^;{}]*"
_DECLARATIONS = rf"(?=\s*[a-z])(?:{_DECLARATION};)*(?:{_DECLARATION})?\s*"
_RULE = rf"{_SELECTOR}\s*\{{{_DECLARATIONS}\}}"
_STYLE_RULES = re.compile(rf"^[ \t]*{_RULE}(?:\s*{_RULE})*[ \t]*$", re.MULTILINE)
# What Texas publishes in place of a version's text, normalized.
_NO_TEXT_NOTICES = frozenset(
    {
        "this version is not currently available but is coming soon in the meantime please see "
        "the pdf for the content of this bill",
        "an html version of this bill is not available please see the pdf for the content of "
        "this bill for additional information consult the legislative budget board s website",
    }
)


@dataclass(frozen=True, slots=True)
class Section:
    ordinal: int  # 0 is the text before the first heading; windows start at 1
    heading: str | None
    text: str


@dataclass(frozen=True, slots=True)
class Split:
    sections: list[Section]
    by_heading: bool  # False when the window fallback produced the sections


@cache
def _invisible() -> re.Pattern[str]:
    """Format (Cf), private-use (Co) and control (Cc) characters, except tab and newline.

    They render as nothing or as one font's private glyph. One character class built from the
    Unicode tables replaces a per-character category test, which dominated the run time.
    """
    codepoints = [
        cp
        for cp in range(sys.maxunicode + 1)
        if unicodedata.category(chr(cp)) in {"Cf", "Co", "Cc"} and chr(cp) not in "\t\n"
    ]
    ranges: list[str] = []
    start = prev = codepoints[0]
    for cp in [*codepoints[1:], -1]:
        if cp != prev + 1:
            ranges.append(
                re.escape(chr(start)) + ("-" + re.escape(chr(prev)) if prev > start else "")
            )
            start = cp
        prev = cp
    return re.compile("[" + "".join(ranges) + "]")


def _strip_gutter(lines: list[str]) -> list[str]:
    numbered = sum(1 for line in lines if line and _GUTTER.match(line))
    nonblank = sum(1 for line in lines if line)
    if numbered < _GUTTER_MIN_LINES or numbered < _GUTTER_MIN_SHARE * nonblank:
        return lines
    return [_GUTTER.sub("", line, count=1) for line in lines]


def _without_line_number(line: str, n: int) -> str | None:
    """`line` without the line number `n` at its end, or None if it does not end with it."""
    tag = str(n)
    return line[: -len(tag)].rstrip() if line.endswith(tag) else None


def _strip_line_numbers(lines: list[str]) -> list[str]:
    out = list(lines)
    dropped: set[int] = set()
    run: list[tuple[int, str]] = []
    expected = gap = 0

    def close() -> None:
        if len(run) >= _LINE_RUN_MIN:
            for i, rest in run:
                out[i] = rest
                if not rest:
                    dropped.add(i)

    for i, line in enumerate(lines):
        rest = _without_line_number(line, expected) if run else None
        if rest is not None:
            run.append((i, rest))
            expected, gap = expected + 1, 0
            continue
        first = _without_line_number(line, 1)
        if first is not None:
            close()
            run, expected, gap = [(i, first)], 2, 0
        elif run and line:
            gap += 1
            if gap > _LINE_RUN_MAX_GAP:
                close()
                run = []
    close()
    return [line for i, line in enumerate(out) if i not in dropped]


def _is_running_header(line: str, repeats: int) -> bool:
    return (
        repeats >= _HEADER_MIN_REPEATS
        and len(line.split()) <= _HEADER_MAX_WORDS
        and any(ch.isdigit() for ch in line)
        and not any(ch.islower() for ch in line)
    )


def _strip_running_headers(lines: list[str]) -> list[str]:
    counts = Counter(line for line in lines if line)
    return [line for line in lines if not (line and _is_running_header(line, counts[line]))]


def clean(raw: str, state: str | None = None) -> str:
    """Display text: one newline convention; no style sheet, invisible character, gutter, or
    the state's running page header."""
    text = unicodedata.normalize("NFKC", raw.replace("\r\n", "\n").replace("\r", "\n"))
    text = _STYLE_RULES.sub("", _HTML_COMMENT.sub("", text))
    text = _STATUTE_MARKERS.sub("", text)
    text = _invisible().sub("", text)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    lines = _strip_gutter(_strip_line_numbers(lines))
    lines = _strip_running_headers(lines)
    text = "\n".join(lines)
    header = STATE_HEADINGS.get(state or "", _NO_STATE_HEADINGS).page_header
    if header is not None:
        text = header.sub("", text)
    # Keep paragraph breaks, which the window fallback uses, but never more than one.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize(text: str) -> str:
    """Matching text: case-folded words and numbers separated by single spaces.

    The section sign is spelled out, so "§ 2" and "Section 2" normalize alike.
    """
    spelled = text.replace("§§", " sections ").replace("§", " section ")
    return _NOT_WORD.sub(" ", spelled.casefold()).strip()


def _windows(text: str) -> list[Section]:
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        words = paragraph.split()
        if len(words) <= WINDOW_WORDS:
            units.append(paragraph)
        else:
            units.extend(
                " ".join(words[i : i + WINDOW_WORDS]) for i in range(0, len(words), WINDOW_WORDS)
            )
    windows: list[list[str]] = []
    current: list[str] = []
    count = 0
    for unit in units:
        current.append(unit)
        count += len(unit.split())
        if count >= WINDOW_WORDS:
            windows.append(current)
            current, count = [], 0
    if current:
        # A short remainder joins the window before it rather than standing alone.
        if windows:
            windows[-1].extend(current)
        else:
            windows.append(current)
    return [Section(i, None, "\n\n".join(w)) for i, w in enumerate(windows, start=1)]


def _unwrapped(patterns: tuple[re.Pattern[str], ...], text: str) -> list[re.Match[str]]:
    """Matches that do not continue a reference wrapped from the line before."""
    return [
        m
        for pattern in patterns
        for m in pattern.finditer(text)
        if not _CITES.search(text[max(0, m.start() - 16) : m.start()])
    ]


def _headings(text: str, state: str | None) -> list[re.Match[str]]:
    """Section headings, and the citation headings within a bill they already split.

    Every state seen printing citation headings does so under a section heading, so a bill
    with none keeps the window fallback: there, a citation starting a line is a wrapped
    reference far more often than a heading.
    """
    rules = STATE_HEADINGS.get(state or "", _NO_STATE_HEADINGS)
    sections = [*_HEADING.finditer(text), *_unwrapped(rules.sections, text)]
    if not sections:
        return []
    citations = _unwrapped((_CITATION, *rules.citations), text)
    # Two rules can take the same line, as New Hampshire's chaptered "228:1" fits both.
    first: dict[int, re.Match[str]] = {}
    for m in [*sections, *citations]:
        first.setdefault(m.start(), m)
    return sorted(first.values(), key=lambda m: m.start())


def split_sections(raw: str | None, state: str | None = None) -> Split:
    """Sections of a bill's text, with `state`'s own heading rules if it has any."""
    text = clean(raw or "", state)
    if not text or normalize(text) in _NO_TEXT_NOTICES:
        return Split([], by_heading=False)
    matches = _headings(text, state)
    if not matches:
        return Split(_windows(text), by_heading=False)
    sections: list[Section] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(0, None, preamble))
    bounds = [m.start() for m in matches] + [len(text)]
    for i, m in enumerate(matches):
        sections.append(Section(i + 1, m["heading"], text[bounds[i] : bounds[i + 1]].strip()))
    return Split(sections, by_heading=True)
