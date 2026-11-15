"""NCOIL's model laws: the National Council of Insurance Legislators' Model Laws Library.

The library (ncoil.org/model-laws-resolutions/model-laws-library/) lists every model law in
force, ten to a page, newest first, through a `?_paged=` parameter; a page past the last answers
with the first page again, so paging stops at the first page that lists nothing new. Each entry
links a document page under /resource/ whose "Open PDF" button is the model's file under
/wp-content/uploads/. Read on 2026-11-15: 96 entries on ten pages, and the document pages read
each held one PDF.

What is collected: every entry in the Model Laws Library, as NCOIL's current text of that model.
Resolutions and working drafts are separate libraries and are not read; an entry in this one
whose title calls it a draft or a resolution is passed over. A model is not listed once per
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

Cleaning: NCOIL's models put a title block, a sponsor and adoption line and a table of contents
before Section 1, and drafting notes in the middle of a section, between its lettered
subsections. `lineage.acts.enacted_text` would drop everything from such a note to the next
section, law included, and keep a table of contents that has no page numbers, so this module
reads its own layout (`clean_model`) and shares only the bracket rule, `drop_instructions`.
See DESIGN.md#model-documents.
"""

import itertools
import re
import urllib.parse
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import PurePosixPath

from lineage.collect import CollectContext, CollectReport
from lineage.collect_base import (
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
from lineage.extract import ExtractError, pdf_text
from lineage.modeldocs import Source
from lineage.modeltext import ModelTextError, docx_text, drop_instructions

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
# can be named for dispute resolution, so the word alone is not enough.
_PASSED_OVER = re.compile(r"\bdraft\b|^\s*(?:NCOIL\s+)?resolution\b", re.IGNORECASE)
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

# "Section 1. Title, Scope and Purposes"; some older models number in roman numerals.
_HEADING = re.compile(r"^\s*\[?(?:Section|Sec\.)\s+(?:\d+[A-Za-z]?|[IVXL]+)\.", re.IGNORECASE)
_NOTE = re.compile(r"^\s*(?:drafting|legislative)\s+notes?\b", re.IGNORECASE)
# The start of a subsection or list item, which ends a drafting note: "B. ", "(2) ", "(a) ".
_ITEM = re.compile(r"^\s*(?:\(?(?:[A-Za-z]|\d+|[ivx]+)\)|[A-Z]\.|\d+\.)\s")
_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$")
# The organization's name alone on a line is a running header or footer, not law.
_RUNNING = re.compile(r"^\W*national council of insurance legislators(?:\s*\(ncoil\))?\W*$", re.I)
# NCOIL's own word for an instruction to the drafter: "[citation to specific statute]".
_CITATION = re.compile(r"[ \t]*\[\s*citation\b[^\[\]]*\]", re.IGNORECASE)


class NcoilTextError(Exception):
    pass


def clean_model(raw: str) -> str:
    """An NCOIL model as matching reads it: its sections, without front matter, the table of
    contents, page numbers, drafting notes or bracketed instructions to the drafter."""
    # Page numbers and running headers are dropped outright, so a note running over a page
    # break stays one note.
    lines = [
        line.rstrip()
        for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if not (_PAGE_NUMBER.match(line) or _RUNNING.match(line))
    ]
    headings = [i for i, line in enumerate(lines) if _HEADING.match(line)]
    if not headings:
        raise NcoilTextError("no section heading: is this the model's text?")
    # A table of contents is a run of headings with nothing but blank lines between them; the
    # run's last heading is the first section's own, followed by its text.
    toc = {
        i
        for i, j in itertools.pairwise(headings)
        if not any(line.strip() for line in lines[i + 1 : j])
    }
    kept: list[str] = []
    in_note = False
    for i in range(headings[0], len(lines)):
        line = lines[i]
        if i in toc:
            continue
        if _HEADING.match(line):
            in_note = False
        elif in_note and (not line.strip() or _ITEM.match(line)):
            # A note is one paragraph: it ends at a blank line or the next lettered subsection.
            in_note = False
        elif _NOTE.match(line):
            in_note = True
        if not in_note:
            kept.append(line)
    text = drop_instructions(_CITATION.sub("", "\n".join(kept)))
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
