"""Split a bill's text into sections, the unit every later stage matches on.

Provisions travel between states section by section, often inside omnibus bills, so a whole
bill is too coarse to compare. A bill is split at its section headings. When none are found,
the text is cut into windows of about `WINDOW_WORDS` words on paragraph boundaries, so every
bill with text still yields comparable units. See DESIGN.md#sections.
"""

import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache

WINDOW_WORDS = 200

# Headings start a line: "Section 1.", "SECTION 1.", "Sec. 1.", "SEC. 1.", "§ 1.", with a colon
# allowed for the period and code-style numbers such as "1-101" or "2a". Alaska prefixes an
# asterisk and Washington "NEW SECTION."; both are skipped. Lowercase "section" is left out on
# purpose: at the start of a line it is a wrapped cross-reference far more often than a heading.
_HEADING = re.compile(
    r"^(?:\*[ \t]*)?(?:NEW SECTION\.[ \t]*)?"
    r"(?P<heading>(?:SECTION|Section|SEC\.|Sec\.|§)[ \t]*"
    r"\d+[A-Za-z]?(?:[.-]\d+[A-Za-z]?)*[ \t]*[.:])(?=\s|$)",
    re.MULTILINE,
)
# A line-number gutter: a short number alone or followed by whitespace, never by "." or ")".
_GUTTER = re.compile(r"^\d{1,3}(?=[ \t]|$)[ \t]*")
# Treat leading numbers as a gutter only when most lines carry one, as printed bills do.
_GUTTER_MIN_LINES = 3
_GUTTER_MIN_SHARE = 0.5
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


def clean(raw: str) -> str:
    """Display text: one newline convention; no style sheet, invisible character or gutter."""
    text = unicodedata.normalize("NFKC", raw.replace("\r\n", "\n").replace("\r", "\n"))
    text = _STYLE_RULES.sub("", _HTML_COMMENT.sub("", text))
    text = _invisible().sub("", text)
    lines = _strip_gutter([" ".join(line.split()) for line in text.split("\n")])
    lines = _strip_running_headers(lines)
    # Keep paragraph breaks, which the window fallback uses, but never more than one.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


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


def split_sections(raw: str | None) -> Split:
    text = clean(raw or "")
    if not text or normalize(text) in _NO_TEXT_NOTICES:
        return Split([], by_heading=False)
    matches = list(_HEADING.finditer(text))
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
