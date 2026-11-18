"""NCOIL's model laws: the National Council of Insurance Legislators' Model Laws Library.

The library (ncoil.org/model-laws-resolutions/model-laws-library/) lists every model law in
force, ten to a page, newest first, through a `?_paged=` parameter; a page past the last answers
with the first page again, so paging stops at the first page that lists nothing new. Each entry
links a document page under /resource/ whose "Open PDF" button is the model's file under
/wp-content/uploads/. Read on 2026-11-15: 96 entries on ten pages, and the document pages read
each held one PDF.

What is collected: every entry in the Model Laws Library, as NCOIL's current text of that model.
Resolutions and working drafts are separate libraries and are not read; an entry in this one
whose title calls it a draft or a resolution is passed over. So is a model agreement: the
library holds one, another association's reciprocal agreement for state workers' compensation
agencies to sign, which NCOIL supports; it is not legislation, and its unnumbered layout of
background, then agreement, would be read from the middle. A model is not listed once per
adoption: amendments and readoptions are recorded in its one entry's title ("Originally Adopted
11/18/12; ... Readopted 7/16/22; Amended on 7/19/25") and its file is replaced, so a readopted
or amended text arrives as a new version of the same document, keyed by the model's name with
that history stripped. No archive of earlier or sunset versions is published.

A document's date is the latest adoption the title records, readoption and amendment included,
since that is the text collected. A title that gives only a month and year ("Adopted November,
2025") takes the day from the entry's listing date when that falls in the same month; otherwise
the date is left out rather than guessed.

Fetching is on (`SITE_FETCH`): the site publishes no terms of use, its privacy policy speaks
only to what NCOIL collects from visitors, robots.txt disallows only /wp-admin/, and the files
open without a login or a click-through. The texts stay local, as every model text does.

Cleaning: NCOIL's models put a title block, adoption and sponsor lines, sometimes a preamble and
a table of contents before their text, and drafting notes in the middle of a section, between
its lettered subsections. `lineage.evals.acts.enacted_text` would drop everything from such a note
to the next section, law included, and keep a table of contents that has no page numbers, so this
module reads its own layouts (`clean_model`) and shares only the bracket rule,
`drop_instructions`. Forty years of models write their headings many ways ("Section 1.",
"Section 1:", "Section 1 Short Title", "Section 1 - text", "Section I.", "Section A." within
"Part I.", "Chapter 1.", "TITLE I", "I. DISMISSAL"); two have none and open at "(A)" or at
lettered parts. Measured on the 84 models collected on 2026-11-15 and the 12 that failed then.
See DESIGN.md#model-documents.
"""

import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import PurePosixPath

from lineage.origins.collect import CollectContext, CollectReport
from lineage.origins.collect_base import (
    CollectFetchError,
    Collection,
    Fetch,
    Fetched,
    ReadRobots,
    fetch_url,
    provenance,
    read_robots,
    slugify,
)
from lineage.origins.modeldocs import Source
from lineage.origins.modeltext import ModelTextError, docx_text, drop_instructions
from lineage.text.extract import ExtractError, pdf_text

SITE = "https://ncoil.org"
LIBRARY = f"{SITE}/model-laws-resolutions/model-laws-library/"
SITE_FETCH = True
# Ten entries a page and 96 entries on 2026-11-15; far more pages than that means paging broke.
MAX_PAGES = 60

# The descriptor's sources, read 2026-11-15:
# - kind and members: https://ncoil.org/about/ (a national legislative organization whose
#   members are the states, represented principally by legislators on state insurance and
#   financial institutions committees) and https://ncoil.org/membership/member-states/ ("All
#   states are NCOIL members").
# - IRS: ProPublica's Nonprofit Explorer lists National Council of Insurance Legislators, EIN
#   23-7412297, as a 501(c)(4), latest Form 990 for 2025:
#   https://projects.propublica.org/nonprofits/organizations/237412297
#   Its affiliate the Insurance Legislators Foundation, EIN 39-1781696, files as a 501(c)(3)
#   and is described on NCOIL's about page as separate from it:
#   https://projects.propublica.org/nonprofits/organizations/391781696
# - terms: no terms-of-use page exists (none linked from any page; /terms-of-use/ answers 404);
#   the privacy policy, effective 2026-03-01, is the only policy page.
SOURCE = Source(
    key="ncoil",
    organization="National Council of Insurance Legislators (NCOIL)",
    descriptor="nonprofit organization of state legislators that adopts model insurance and "
    "financial services laws; files with the IRS as a 501(c)(4) social welfare organization",
    members="Every state is a member, represented principally by legislators who serve on "
    "state insurance and financial institutions committees",
    members_source="https://ncoil.org/about/",
    library_url=LIBRARY,
    terms_url=f"{SITE}/privacy-policy/",
    terms_note="No terms of use are published. The only policy page, the privacy policy, "
    "covers the data NCOIL gathers from visitors and says nothing about copying, reuse or "
    "automated reading. robots.txt disallows only /wp-admin/, and each model's PDF is linked "
    "openly from its page, with no login or terms to accept first.",
    terms_read_on=date(2026, 11, 15),
)

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MEDIA = {".pdf": "application/pdf", ".docx": DOCX}
_UPLOADS = "/wp-content/uploads/"
_RESOURCE = "/resource/"

# ---- the library ----------------------------------------------------------------------------

_MONTHS: tuple[str, ...] = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip
_MONTH = r"(?:" + "|".join(m.capitalize() for m in _MONTHS) + r")"
# "July 23, 2026": the date an entry's card shows under its committee.
_CARD_DATE = re.compile(rf"^\s*({_MONTH})\s+(\d{{1,2}}),\s+(\d{{4}})\s*$")
# Where a title's adoption history begins, after an en dash, a semicolon or a hyphen: "Adopted
# 4-27-25", "Adopted November, 2021", "Originally Adopted 11/17/00", or "(...) Supported 2/27/04".
_HISTORY = re.compile(
    r"\s*(?:[\u2013\u2014;,]|\s-)?\s+(?:Originally\s+|Last\s+)?(?:Re-?adopted|Adopted|Supported)\b",
    re.IGNORECASE,
)
# A working draft ("Draft as of 10/21/2026"), or a resolution by its title's first word; a model
# can be named for dispute resolution, so the word alone is not enough. A model agreement is an
# agreement for state agencies to sign, not legislation (see the docstring).
_PASSED_OVER = re.compile(
    r"\bdraft\b|^\s*(?:NCOIL\s+)?(?:resolution|model\s+agreement)\b", re.IGNORECASE
)
# A sunset, not an adoption: "Readopted temporarily until November 2026 Meeting".
_UNTIL = re.compile(r"\buntil\b[^;)]*", re.IGNORECASE)
_FULL_DATE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})\b")
_MONTH_YEAR = re.compile(rf"\b({_MONTH}),?\s+(\d{{4}})\b")


@dataclass(frozen=True, slots=True)
class Entry:
    url: str  # the document's page under /resource/
    title: str  # as the library lists it, adoption history included
    listed_on: date | None  # the card's date


class _Library(HTMLParser):
    """A library page's links to document pages, each with its text and the first card date
    after it. The parser keys on links and dates, not on the theme's markup."""

    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.entries: dict[str, tuple[str, date | None]] = {}
        self._href: str | None = None
        self._text: list[str] = []
        self._last: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            url = urllib.parse.urljoin(self.base, href)
            if urllib.parse.urlsplit(url).path.startswith(_RESOURCE):
                self._href, self._text = url, []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            text = " ".join("".join(self._text).split())
            known, listed = self.entries.get(self._href, ("", None))
            # A card can link its document twice, by image and by title; the title is longer.
            self.entries[self._href] = (max(known, text, key=len), listed)
            self._last, self._href = self._href, None

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)
        elif self._last is not None and (m := _CARD_DATE.match(data)):
            title, listed = self.entries[self._last]
            if listed is None:
                month = _MONTHS.index(m.group(1).lower()) + 1
                self.entries[self._last] = (title, date(int(m.group(3)), month, int(m.group(2))))


def library_entries(page: Fetched) -> list[Entry]:
    """The model laws a library page lists, in its order; drafts and resolutions passed over."""
    parser = _Library(page.url)
    parser.feed(page.body.decode("utf-8", errors="replace"))
    parser.close()
    return [
        Entry(url, title, listed)
        for url, (title, listed) in parser.entries.items()
        if title and not _PASSED_OVER.search(title)
    ]


def library_page(n: int) -> str:
    return LIBRARY if n == 1 else f"{LIBRARY}?_paged={n}"


def model_name(title: str) -> str:
    """The model's name without its adoption history, which changes at every readoption."""
    name = _HISTORY.split(title, maxsplit=1)[0].strip(" \u2013\u2014;,-")
    return name or title


def _year(text: str) -> int:
    year = int(text)
    if len(text) == 4:
        return year
    # NCOIL's oldest models date from the 1990s; two-digit years after 69 are 19xx.
    return 1900 + year if year > 69 else 2000 + year


def adopted_on(title: str, listed_on: date | None) -> date | None:
    """The latest adoption, readoption or amendment the title records (see the docstring)."""
    history = _UNTIL.sub("", title)
    marks: list[tuple[int, int, int]] = []
    for m in _FULL_DATE.finditer(history):
        month, day, year = int(m.group(1)), int(m.group(2)), _year(m.group(3))
        try:
            date(year, month, day)
        except ValueError:
            continue
        marks.append((year, month, day))
    for m in _MONTH_YEAR.finditer(history):
        marks.append((int(m.group(2)), _MONTHS.index(m.group(1).lower()) + 1, 0))
    if not marks:
        return None
    year, month, day = max(marks)
    if day:
        return date(year, month, day)
    if listed_on is not None and (listed_on.year, listed_on.month) == (year, month):
        return listed_on
    return None


# ---- a document's page ------------------------------------------------------------------------


class _Files(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.files: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href") if tag == "a" else None
        if href:
            url = urllib.parse.urljoin(self.base, href)
            if urllib.parse.urlsplit(url).path.startswith(_UPLOADS) and url not in self.files:
                self.files.append(url)


def model_file(page: Fetched) -> str | None:
    """The model's file on its document page: the first upload linked, the "Open PDF" button.
    Related documents link other document pages, never their files."""
    parser = _Files(page.url)
    parser.feed(page.body.decode("utf-8", errors="replace"))
    parser.close()
    return parser.files[0] if parser.files else None


def file_suffix(url: str) -> str:
    return PurePosixPath(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).suffix.lower()


# ---- cleaning ---------------------------------------------------------------------------------

# A section heading as NCOIL's models write it, across forty years of layouts: "Section 1. Title",
# "Section 1: Title", "Section 1 Title", "Section 1 - text", "SECTION 1.", "Section I.",
# "Section A." (sections lettered within parts), "Section __." (an amendment's blank number).
# The word is matched in any case but what follows in its own, so "Section 3 may recover" in a
# wrapped sentence is not a heading; nor is "Section C.2 of this Act", a reference.
_SECTION_LABEL = r"(?i:section|sec\.?)\s+(?:\d+[A-Za-z]?|[IVXL]+|[A-Z]|_+)"
# A container heading: "Part I.", "Article 2", "TITLE I - ANTI-RUNNER", "Chapter 1. Short Title".
_CONTAINER_LABEL = r"(?i:article|part|title|chapter)\s+(?:\d+|[IVXL]+)"
_HEADING = re.compile(
    rf"^\s*\[?(?P<label>{_SECTION_LABEL}|{_CONTAINER_LABEL})"
    r"(?:\s*[.:](?!\d)|\s+[\u2013\u2014-]|\s+(?=[A-Z(\[\u201c\"])|\s*$)"
    # A bare capital roman numeral before a heading in capitals: "I. DISMISSAL OR ABATEMENT".
    r"|^\s*(?P<roman>[IVX]+)\.\s+[A-Z]{2,}"
)
# An unnumbered section some models open with: "Purpose.", "Short Title", "Purpose - The ...".
_NAMED = re.compile(
    r"^\s*(?:Purpose|Short Title|Title|Findings|Legislative Findings|Definitions|Scope)"
    r"\s*(?:[.:\u2013\u2014-]|\s*$)"
)
# The start of a subsection or list item: "B. ", "(2) ", "(a) ", "1. ". Three digits at most,
# so a wrapped adoption line that starts "2010. Re-adopted" is not one.
_ITEM = re.compile(r"^\s*(?:\(?(?:[A-Za-z]|\d{1,3}|[ivx]+)\)|[A-Z]\.|\d{1,3}\.)\s")
# A drafting note, bracketed or not, and its kin: "Drafting Note:", "[Drafting note -",
# "Legislative Note", "Drafting option:", "Note:", "[Note that ...".
_NOTE = re.compile(
    r"^\s*([\[(])?\s*(?:(?:drafting|legislative)\s+(?:notes?|options?|comments?)\b"
    r"|notes?\s*:|note\s+that\b)",
    re.IGNORECASE,
)
# A footnote, or a starred line of front matter ("*Sponsored by ..."); never law.
_FOOTNOTE = re.compile(r"^\s*\*{1,2}\s*[\w(\u201c\"]")
# A definition opens a provision of its own: "\u201cLessee\u201d means ...".
_DEFINITION = re.compile(r"^\s*[\u201c\"][^\u201d\"]{1,80}[\u201d\"].*\bmeans\b")
_CONTENTS = re.compile(
    r"^\s*(?:table\s+of\s+)?contents\b[\s:]*(?:page\s+numbers?)?\s*$", re.IGNORECASE
)
# A contents entry's page reference: "(1-9)", "(23)", ".... 4", a wide gap then "12".
_PAGE_REFERENCE = re.compile(
    r"(?:\(\s*\d[\d\s,\u2013-]*\)|\.{2,}\s*\d+|(?:\s{2,}|\t)\d+(?:\s*[\u2013-]\s*\d+)?)\s*$"
)
# Lines that are layout, not text: page numbers, the running header or footer naming the
# organization, rules of asterisks or underscores.
_LAYOUT = re.compile(
    r"^\s*(?:\d+|\W*national council of insurance legislators(?:\s*\(ncoil\))?\W*"
    r"|[*_=-]{4,}[\s_*]*)\s*$",
    re.IGNORECASE,
)


# A braced title: "Section 4. {Severability clause}" with no text under it is a placeholder for
# the state's own clause; over text, as some models brace every title, it is only a title.
_PLACEHOLDER = re.compile(r"^\{[^{}]*\}\.?$")
# A note's line that ends mid-sentence, so the next line continues it even if it starts with a
# heading's words: "... enforcement provisions for" / "Section 6.".
_OPEN = re.compile(r"[^.:;?!)\]\"”]\s*$")
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50}


class NcoilTextError(Exception):
    pass


def _arabic(word: str) -> str:
    """A roman numeral as its number ("iv" -> "4"); any other word as it is. A contents list
    can number its sections 1-8 while the text numbers them I-VIII."""
    if not word or any(c not in _ROMAN for c in word):
        return word
    values = [_ROMAN[c] for c in word]
    total = sum(-v if v < after else v for v, after in zip(values, [*values[1:], 0], strict=True))
    return str(total)


def _label(line: str) -> str | None:
    m = _HEADING.match(line)
    if m is None:
        return None
    if m.group("roman"):
        return f"roman {_arabic(m.group('roman').lower())}"
    kind, number = m.group("label").lower().split(maxsplit=1)
    return f"{kind} {_arabic(number)}"


def _is_section(line: str) -> bool:
    """A section heading, as against a part, article or title holding sections."""
    label = _label(line)
    return label is not None and (label.startswith("sec") or label.startswith("roman"))


def _heading_rest(line: str) -> str:
    """What follows a heading's label: its title, and any text set on the same line."""
    m = _HEADING.match(line)
    return line[m.end() :].strip(" .:\u2013\u2014-") if m else ""


def _continues(line: str) -> bool:
    """Whether a heading-like line could be a sentence's wrapped tail: a bare label ("Section
    6."), or one followed by lowercase words. A label with a title is a heading."""
    rest = _heading_rest(line)
    return not rest or rest[0].islower()


def _has_inline_text(line: str) -> bool:
    """A heading carrying its section's text on the same line: "Section 4 - This law is
    effective upon enactment." A title alone is a few words without a closing period."""
    m = _HEADING.match(line)
    rest = line[m.end() :].strip() if m else ""
    return len(rest.split()) >= 5 and rest.endswith((".", ";", ":"))


@dataclass(slots=True)
class _Notes:
    """A pass through a model's lines: the lines kept, whether a note is being read, and the
    section the pass is in."""

    out: list[str] = field(default_factory=list[str])
    in_note: bool = False
    bracketed: bool = False
    previous: str = ""  # the note's last line
    section_at: int | None = None  # where the current section's heading is in `out`
    body: bool = False
    noted: bool = False
    placeholder: bool = False

    def is_heading(self, line: str) -> bool:
        """A heading, unless it could be the wrapped tail of the note's open sentence."""
        if _HEADING.match(line) is None:
            return False
        return not (self.in_note and _OPEN.search(self.previous) and _continues(line))

    def continues_note(self, line: str, heading: bool) -> bool:
        """Whether `line` belongs to the note being read; the note ends where it does not."""
        if self.in_note and _CONTENTS.match(line):
            self.in_note = False
        if heading or not self.in_note:
            self.in_note = False
            return False
        if self.bracketed and "]" in line:
            self.in_note = False
            return True
        if not line.strip() or _ITEM.match(line) or _DEFINITION.match(line):
            self.in_note = False
            return False
        self.previous = line
        return True

    def opens_note(self, line: str) -> bool:
        """Whether `line` opens a note or a footnote, which is then being read."""
        m = _NOTE.match(line) or _FOOTNOTE.match(line)
        if m is None:
            return False
        self.noted = True
        self.bracketed = m.re is _NOTE and m.group(1) == "["
        self.in_note = not (self.bracketed and "]" in line)
        self.previous = line
        return True

    def keep(self, line: str, heading: bool) -> None:
        if heading:
            self.close_section()
            self.section_at, self.body, self.noted = None, False, False
            self.placeholder = _PLACEHOLDER.match(_heading_rest(line)) is not None
            if _is_section(line) and not _has_inline_text(line):
                self.section_at = len(self.out)
        elif line.strip():
            self.body = True
        self.out.append(line)

    def close_section(self) -> None:
        # A section with no text of its own, only a note or a placeholder title, is left to
        # each state to write. A braced title over text is only a title: "Section 1. {Short
        # Title}" in some models.
        if self.section_at is not None and not self.body and (self.noted or self.placeholder):
            del self.out[self.section_at]


def _drop_notes(lines: list[str]) -> list[str]:
    """`lines` without drafting notes or footnotes, and without the heading of a section whose
    only text was a note: the model leaves that section for each state to write.

    A note is one paragraph: it ends at a blank line, a heading or the next subsection, and a
    bracketed one at its closing bracket. Ending early keeps a note's tail rather than dropping
    law; a PDF's text layer often has no blank lines, so a page break (a page number, now blank)
    ends one too."""
    notes = _Notes()
    for line in lines:
        heading = notes.is_heading(line)
        if notes.continues_note(line, heading) or (not heading and notes.opens_note(line)):
            continue
        notes.keep(line, heading)
    notes.close_section()
    return notes.out


def _key(line: str) -> list[str]:
    """A contents entry or a heading as words, without a page reference, its number arabic."""
    words = re.findall(r"[a-z0-9]+", _PAGE_REFERENCE.sub("", line).lower())
    if len(words) >= 2 and _HEADING.match(line):
        words[1] = _arabic(words[1])
    return words


def _is_wrapped_entry(line: str) -> bool:
    """A contents entry's second line: a few words, not a sentence."""
    words = line.split()
    return 0 < len(words) <= 8 and not line.rstrip().endswith((".", ":", ";"))


def _same_entry(entry: list[str], line: list[str]) -> bool:
    if len(entry) < 2 or len(line) < 2:
        return entry == line
    shorter = min(len(entry), len(line))
    return entry[:shorter] == line[:shorter]


def _repeated_entry(lines: list[str], header: int) -> int | None:
    """Where the text repeats the first entry of the contents list under `header`, if it does.

    A heading's entry is found again by its number, since the list may shorten its title
    ("Section 1. Title" for "Section 1. Short Title"); any other by its words."""
    entry = next((i for i in range(header + 1, len(lines)) if lines[i].strip()), None)
    if entry is None:
        return None
    label, key = _label(lines[entry]), _key(lines[entry])
    for j in range(entry + 1, len(lines)):
        if not lines[j].strip():
            continue
        if label is not None and _label(lines[j]) == label:
            return j
        if label is None and _same_entry(key, _key(lines[j])):
            return j
    return None


def _contents_end(lines: list[str], header: int) -> int:
    """Where the contents list under `header` ends: past its entries, headings and the short
    lines an entry wraps onto."""
    end = header + 1
    while end < len(lines) and (
        not lines[end].strip() or _HEADING.match(lines[end]) or _is_wrapped_entry(lines[end])
    ):
        end += 1
    return end


def _past_unheaded_contents(lines: list[str], headings: list[int], first: int) -> int:
    """`first`, or, when it opens a run of headings with nothing between, a contents list
    without a header, where its first label comes back later."""
    label = _label(lines[first])
    if label is None:
        return first
    run = [first]
    for i in headings:
        if i > run[-1]:
            if any(line.strip() for line in lines[run[-1] + 1 : i]):
                break
            run.append(i)
    if len(run) >= 2:
        return next((i for i in headings if i > first and _label(lines[i]) == label), first)
    return first


def _body_start(lines: list[str]) -> int | None:
    """Where the model's text begins, after its title, adoption and sponsor lines, preamble and
    contents.

    It begins at its first heading, numbered provision or named opening section ("Purpose").
    Two layouts put headings before that point: a contents list under its own header, whose
    first entry the text then repeats, and a contents list without one, a run of headings with
    nothing between whose first label comes back later."""
    headings = [i for i, line in enumerate(lines) if _HEADING.match(line)]
    starts = sorted(
        set(headings)
        | {i for i, line in enumerate(lines) if _ITEM.match(line) or _NAMED.match(line)}
    )
    header = next((i for i, line in enumerate(lines) if _CONTENTS.match(line)), None)
    if header is not None and (not starts or header < starts[0]):
        if (repeated := _repeated_entry(lines, header)) is not None:
            return repeated
        # The text never repeats the first entry: the list ends where its entries do.
        end = _contents_end(lines, header)
        starts = [i for i in starts if i >= end]
    if not starts:
        return None
    return _past_unheaded_contents(lines, headings, starts[0])


def clean_model(raw: str) -> str:
    """An NCOIL model as matching reads it: its sections, without front matter, the table of
    contents, page numbers, drafting notes, footnotes or bracketed instructions to the drafter."""
    lines = [
        "" if _LAYOUT.match(line) else line.rstrip()
        for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    lines = _drop_notes(lines)
    start = _body_start(lines)
    if start is None:
        raise NcoilTextError("no heading or numbered provision: is this the model's text?")
    text = drop_instructions("\n".join(lines[start:]))
    text = "\n".join(line if line.strip() else "" for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def file_text(data: bytes, media_type: str) -> str:
    return docx_text(data) if media_type == DOCX else pdf_text(data)


# ---- collecting -------------------------------------------------------------------------------


def _fetch_step(c: Collection, doc_url: str, url: str) -> Fetched | None:
    """One request on the way to a model's file; a refusal or failure settles the model."""
    if c.permission(url) == "blocked":
        c.block(url)
        return None
    try:
        return c.get(url)
    except CollectFetchError as e:
        c.fail(doc_url, f"{url}: {e}")
        return None


def _list(c: Collection) -> list[Entry]:
    """Every entry the library lists, page by page until a page lists nothing new."""
    entries: dict[str, Entry] = {}
    for n in range(1, MAX_PAGES + 1):
        url = library_page(n)
        if (page := _fetch_step(c, url, url)) is None:
            break
        new = [e for e in library_entries(page) if e.url not in entries]
        if not new:
            break
        entries |= {e.url: e for e in new}
    else:
        c.fail(LIBRARY, f"still listing new entries after {MAX_PAGES} pages; paging stopped")
    return list(entries.values())


def _collect_entry(c: Collection, entry: Entry, slug: str) -> None:
    if (page := _fetch_step(c, entry.url, entry.url)) is None:
        return
    if (file_url := model_file(page)) is None:
        c.fail(entry.url, "no PDF or Word file on the document's page")
        return
    suffix = file_suffix(file_url)
    if suffix not in _MEDIA:
        c.fail(entry.url, f"{file_url}: not a PDF or a Word (.docx) file")
        return
    if (robots := c.check(file_url)) is None:
        return
    if (got := _fetch_step(c, entry.url, file_url)) is None:
        return
    media_type = _MEDIA[suffix]
    try:
        text = clean_model(file_text(got.body, media_type))
    except (ModelTextError, ExtractError, NcoilTextError) as e:
        c.fail(entry.url, f"{file_url}: {e}")
        return
    doc = provenance(
        SOURCE,
        slug=slug,
        title=model_name(entry.title),
        url=entry.url,
        raw=got.body,
        media_type=media_type,
        text=text,
        robots=robots,
        collected_on=c.context.on,
        published=adopted_on(entry.title, entry.listed_on),
    )
    c.keep(doc, raw=got.body, text=text)


def collect_from(
    context: CollectContext,
    *,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    site: bool = SITE_FETCH,
) -> CollectReport:
    """Every model law in the library, each from its document page's file."""
    c = Collection(SOURCE, context, fetch=fetch, robots_reader=robots_reader)
    if not site:
        c.fail(LIBRARY, "fetching from the site is off")
        return c.report()
    seen: dict[str, str] = {}
    for entry in _list(c):
        slug = slugify(model_name(entry.title))
        if slug in seen:
            c.fail(entry.url, f"the same model name as {seen[slug]}")
            continue
        seen[slug] = entry.url
        _collect_entry(c, entry, slug)
    return c.report()


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network.
    return collect_from(context, fetch=fetch_url, robots_reader=read_robots, site=SITE_FETCH)
