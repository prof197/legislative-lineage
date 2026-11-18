"""The Goldwater Institute's model legislation, collected from its state policy handbook.

The institute keeps no catalog of model bills. Its current compilation is the handbook page,
"The Freedom Agenda: Goldwater's State Policy Handbook", one section per policy, most ending in a
"View model legislation" link to a PDF in the site's uploads (read 2026-11-15: 16 links, 15 of
them on its own site, and one section without a link). That page is the library: every link on
it whose words, or whose short label around it, name model legislation is a document, and the
rest of the page (the policy write-ups, the newsletter form, the handbook's own PDF, a report)
is not collected. Older proposals published
inside policy papers elsewhere on the site are papers, not the library, and are left out.

One link on the page points off the site, to a state legislature's printed bill (Arizona's,
2022). That is a state's bill, not a text the institute publishes, and the corpus reads bills
from Open States, so a link to another host is not a document of this source.

A document is named by its file: "Permit-Freedom-Act-2026.pdf" is `goldwater/permit-freedom-act`,
the year dropped so the next edition is a new version of the same document rather than a new
one. Its title is the one the document gives itself, the line above its enacting text, and the
link's qualifier where a section links two ("eminent domain", "regulatory takings"). The
institute dates neither the page's links nor the bills on them; a document that prints a full
date above its text is given it, and the upload folder's month, which is the site's and not the
institute's dating, is not used.

Cleaning (`clean_text`) keeps what a state would enact: from the first enacting line ("AN ACT
relating to ...", "Section 1.", "Be it enacted"), so a title page or an FAQ set before the bill
goes; the package's "Proposal 2: ..." labels between its bills go; page numbers go; a drafting
note or commentary goes up to the next blank line, section or numbered provision; bracketed
instructions go as `lineage.modeltext` drops them, and with them this library's "[appropriate
section of the state code]", which names a citation for the drafter to supply. Placeholders a
state fills in as written, "[STATE]" and "[20XX]", stay.

Fetching is on (`SITE_FETCH`): read on 2026-11-15, robots.txt disallows nothing for any agent,
the terms ask no visitor to accept them and say nothing of automated reading, and the texts are
kept locally and never shown (DESIGN.md#lanes, Source lanes). See DESIGN.md#model-documents.
"""

import re
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path

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
from lineage.modeldocs import Source
from lineage.modeltext import ModelTextError, docx_text, drop_instructions
from lineage.text.extract import ExtractError, pdf_text

SITE = "https://www.goldwaterinstitute.org"
HOST = "www.goldwaterinstitute.org"
LIBRARY = f"{SITE}/the-freedom-agenda-goldwater-state-policy-handbook/"
# On: the terms allow it, as read on 2026-11-15 (see the module docstring).
SITE_FETCH = True

# The descriptor's sources, read 2026-11-15:
# - kind: https://www.goldwaterinstitute.org/about/ (a public policy research and litigation
#   organization, founded in 1988, based in Phoenix, Arizona); it describes no membership.
# - IRS: ProPublica's Nonprofit Explorer lists the institute under its legal name, Barry
#   Goldwater Institute for Public Policy Research, EIN 86-0597661, Phoenix, AZ, designated a
#   501(c)(3), exempt since February 1989, filing Form 990 (latest: fiscal year 2024):
#   https://projects.propublica.org/nonprofits/organizations/860597661
# - terms: https://www.goldwaterinstitute.org/terms-and-conditions/ (no date on the page).
SOURCE = Source(
    key="goldwater",
    organization="Goldwater Institute (Barry Goldwater Institute for Public Policy Research)",
    # Type: goldwaterinstitute.org/about/. IRS: EIN 86-0597661 in ProPublica's Nonprofit Explorer.
    descriptor="nonprofit public policy research and litigation organization based in Phoenix, "
    "Arizona; files with the IRS as a 501(c)(3)",
    library_url=LIBRARY,
    terms_url=f"{SITE}/terms-and-conditions/",
    terms_note="All rights reserved; visitors may print or reproduce discrete portions for "
    "commentary or personal non-commercial use, crediting the institute and without altering "
    "the material or its meaning; telling the institute of a reprint is optional. Nothing on "
    "automated access, and no acceptance is asked of a visitor.",
    terms_read_on=date(2026, 11, 15),
)

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MEDIA = {".pdf": "application/pdf", ".docx": DOCX}
_HOSTS = {HOST, HOST.removeprefix("www.")}
_FILES = {".pdf", ".docx", ".doc"}
# Elements whose words give a link its context, and the most words such a text has to be read
# as the link's label: a sentence that mentions model legislation beside a download is not one.
_BLOCKS = {"p", "li", "div", "td", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"}
_LABEL_WORDS = 12
_MODEL_LINK = re.compile(r"\bmodel\s+(?:legislation|bills?|policy|policies)\b", re.IGNORECASE)
# What a link's text says beyond "view model legislation": the qualifier of one of two links.
_LINK_WORDS = re.compile(r"\b(?:view|read|download|the|model|legislation|bills?|polic(?:y|ies))\b")
# "Permit-Freedom-Act-2026", "Freedom-From-Indoctrination-Act-Model-Policy": the edition's year
# and the label go, so a new edition is a new version of the same document.
_STEM_NOISE = re.compile(r"(?:[-_ ](?:model[-_ ]polic(?:y|ies)|final|web|\d{4}))+$", re.IGNORECASE)

# The first line of what a state would enact.
_ACT_START = re.compile(
    r"^\s*(?:an\s+act\b(?:\s+(?:relating|concerning|to|providing|establishing|amending)\b|\s*$)"
    r"|(?:section|sec\.?)\s+1\b"
    r"|proposal\s+1\s*[:.]"
    r"|be\s+it\s+enacted\b)",
    re.IGNORECASE,
)
# The package's labels between its bills: "Proposal 2: End Mandatory Diversity Training".
_PROPOSAL = re.compile(r"^\s*proposal\s+\d+\s*[:.]", re.IGNORECASE)
# Where a note ends: a section heading or the next numbered provision, "(B)", "(1)", "2.". A
# PDF's text layer often loses the blank line after a note, which would carry it into the law.
_PROVISION = re.compile(
    r"^\s*(?:(?:section|sec\.?)\s+\d+|\([A-Za-z0-9]{1,4}\)\s|[A-Za-z0-9]{1,3}\.\s)", re.IGNORECASE
)
# A paragraph of commentary, to the next blank line or section heading.
_NOTE = re.compile(
    r"^\s*(?:drafting|legislative|explanatory|editor'?s)\s+notes?\b"
    r"|^\s*(?:note|comment|commentary|explanation)s?\s*[:.\u2013\u2014-]"
    r"|^\s*commentary\s*$",
    re.IGNORECASE,
)
# A page's number or running label on a line of its own.
_PAGE = re.compile(r"^\s*(?:-\s*\d{1,3}\s*-|\d{1,3}|page\s+\d+(?:\s+of\s+\d+)?)\s*$", re.IGNORECASE)
_DRAFTER = re.compile(r"[ \t]*\[\s*appropriate\b[^\[\]]*\]", re.IGNORECASE)
_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),\s+(\d{4})\b"
)
_DATE_LINE_WORDS = 6


class GoldwaterTextError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Listed:
    """One model-legislation link on the handbook page."""

    url: str  # absolute
    text: str  # the link's own words
    context: str = ""  # the words of the paragraph it sits in, when that is short enough


@dataclass(slots=True)
class _Block:
    parts: list[str]
    links: list[int]  # indexes into the parser's links


class _Library(HTMLParser):
    """Links with their own words and their paragraph's: a label can set the words "model
    legislation" around a link rather than in it ("VIEW <a>EMINENT DOMAIN</a> MODEL
    LEGISLATION")."""

    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.links: list[Listed] = []
        self._blocks: list[_Block] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCKS:
            self._blocks.append(_Block([], []))
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and not href.lower().startswith(("javascript:", "#", "mailto:")):
                self._href = urllib.parse.urljoin(self.base, href)
                self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCKS and self._blocks:
            block = self._blocks.pop()
            text = " ".join("".join(block.parts).split())
            context = text if len(text.split()) <= _LABEL_WORDS else ""
            for i in block.links:
                self.links[i] = Listed(self.links[i].url, self.links[i].text, context)
            if self._blocks:
                self._blocks[-1].parts.append(f" {text} ")
        elif tag == "a" and self._href is not None:
            self.links.append(Listed(self._href, " ".join("".join(self._text).split())))
            if self._blocks:
                self._blocks[-1].links.append(len(self.links) - 1)
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._blocks:
            self._blocks[-1].parts.append(data)
        if self._href is not None:
            self._text.append(data)


def _is_model_link(link: Listed) -> bool:
    if _MODEL_LINK.search(link.text):
        return True
    is_file = Path(urllib.parse.urlsplit(link.url).path).suffix.lower() in _FILES
    return is_file and _MODEL_LINK.search(link.context) is not None


def listed(page: Fetched) -> list[Listed]:
    """The page's links to model legislation on the institute's own site, once each, in order.
    A link to another host is a state's bill, not this source's document (module docstring)."""
    parser = _Library(page.url)
    parser.feed(page.body.decode("utf-8", errors="replace"))
    parser.close()
    seen: set[str] = set()
    out: list[Listed] = []
    for link in parser.links:
        host = (urllib.parse.urlsplit(link.url).hostname or "").lower()
        if _is_model_link(link) and host in _HOSTS and link.url not in seen:
            seen.add(link.url)
            out.append(link)
    return out


def file_name(url: str) -> str:
    return urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)


def qualifier(link_text: str) -> str:
    """ "VIEW EMINENT DOMAIN MODEL LEGISLATION" -> "eminent domain"; "" for a plain link."""
    return " ".join(_LINK_WORDS.sub(" ", link_text.lower()).replace(".", " ").split())


def doc_slug(link: Listed) -> str:
    """The document's slug, from its file name without the edition's year."""
    stem = _STEM_NOISE.sub("", Path(file_name(link.url)).stem)
    return slugify(stem)


def act_start(lines: list[str]) -> int | None:
    return next((i for i, line in enumerate(lines) if _ACT_START.match(line)), None)


def clean_text(raw: str) -> str:
    """A document's text as matching reads it: the enacted text from its first enacting line,
    without the package's labels, page numbers, commentary or instructions to the drafter."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = act_start(lines)
    if start is None:
        raise GoldwaterTextError("no enacting text (no AN ACT, Section 1 or Be it enacted line)")
    kept: list[str] = []
    dropping = False
    for line in lines[start:]:
        if _PAGE.match(line):
            continue
        if _PROPOSAL.match(line):
            dropping = False
            continue
        if not line.strip() or _PROVISION.match(line):
            dropping = False
        elif _NOTE.match(line):
            dropping = True
        if not dropping:
            kept.append(line.rstrip())
    text = drop_instructions(_DRAFTER.sub("", "\n".join(kept)))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise GoldwaterTextError("no enacting text left after cleaning")
    return text + "\n"


def front_matter(raw: str) -> list[str]:
    """The non-empty lines before the enacting text: a title page, an FAQ."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = act_start(lines)
    return [line.strip() for line in lines[: start or 0] if line.strip()]


def own_title(raw: str) -> str | None:
    """The title the document gives itself: the line just above its enacting text, when it reads
    as a title (short, not a question, not a date or a label)."""
    before = front_matter(raw)
    if not before:
        return None
    line = before[-1]
    if (
        len(line.split()) > 14
        or line.endswith(("?", ".", ":", ";", ","))
        or _DATE.search(line)
        or _PAGE.match(line)
        or re.search(r"model\s+(?:legislation|policy)|goldwater", line, re.IGNORECASE)
    ):
        return None
    return line


def own_date(raw: str) -> date | None:
    """A full date printed on a line of its own above the enacting text ("Revised September 2,
    2026"), as the institute would date a document; a date in an FAQ's sentence is not one."""
    for line in front_matter(raw):
        if len(line.split()) <= _DATE_LINE_WORDS and (found := _DATE.search(line)) is not None:
            month, day, year = found.groups()
            try:
                return datetime.strptime(f"{month} {day} {year}", "%B %d %Y").date()
            except ValueError:
                return None
    return None


def title_for(link: Listed, raw_text: str) -> str:
    """The document's own title, else its file's name; with the link's qualifier, if any."""
    base = own_title(raw_text) or " ".join(
        _STEM_NOISE.sub("", Path(file_name(link.url)).stem).replace("_", " ").split("-")
    )
    base = " ".join(base.split())
    extra = qualifier(link.text)
    if extra and extra not in base.lower():
        return f"{base} ({extra.title()})"
    return base


def file_text(data: bytes, media_type: str) -> str:
    return docx_text(data) if media_type == DOCX else pdf_text(data)


def _fetch_step(c: Collection, doc_url: str, url: str) -> Fetched | None:
    """One request; a refusal or failure settles the document it was for."""
    if c.permission(url) == "blocked":
        c.block(doc_url)
        return None
    try:
        return c.get(url)
    except CollectFetchError as e:
        c.fail(doc_url, f"{url}: {e}" if url != doc_url else str(e))
        return None


def _collect_doc(c: Collection, link: Listed, slug: str) -> None:
    media_type = _MEDIA.get(Path(file_name(link.url)).suffix.lower())
    if media_type is None:
        c.fail(link.url, f"{file_name(link.url)}: neither PDF nor Word (.docx)")
        return
    if (robots := c.check(link.url)) is None:
        return
    if (got := _fetch_step(c, link.url, link.url)) is None:
        return
    try:
        extracted = file_text(got.body, media_type)
        text = clean_text(extracted)
    except (ExtractError, ModelTextError, GoldwaterTextError) as e:
        c.fail(link.url, f"{file_name(link.url)}: {e}")
        return
    doc = provenance(
        SOURCE,
        slug=slug,
        title=title_for(link, extracted),
        url=link.url,
        raw=got.body,
        media_type=media_type,
        text=text,
        robots=robots,
        collected_on=c.context.on,
        published=own_date(extracted),
    )
    c.keep(doc, raw=got.body, text=text)


def slugs(links: list[Listed]) -> list[str]:
    """Each link's slug; two files with one name are told apart by their link's qualifier."""
    base = [doc_slug(link) for link in links]
    out: list[str] = []
    for link, slug in zip(links, base, strict=True):
        if base.count(slug) > 1 and qualifier(link.text):
            slug = f"{slug}-{slugify(qualifier(link.text))}"
        n, candidate = 2, slug
        while candidate in out:
            candidate, n = f"{slug}-{n}", n + 1
        out.append(candidate)
    return out


def collect_from(
    context: CollectContext,
    *,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    site: bool = SITE_FETCH,
) -> CollectReport:
    """Every model-legislation link on the handbook page, each fetched, cleaned and stored. With
    `site` off nothing is asked, not even robots.txt, and the page is reported failed."""
    c = Collection(SOURCE, context, fetch=fetch, robots_reader=robots_reader)
    if not site:
        c.fail(LIBRARY, "fetching from the site is off (SITE_FETCH)")
        return c.report()
    if (page := _fetch_step(c, LIBRARY, LIBRARY)) is None:
        return c.report()
    links = listed(page)
    if not links:
        c.fail(LIBRARY, "no model legislation links on the page: has its layout changed?")
        return c.report()
    for link, slug in zip(links, slugs(links), strict=True):
        _collect_doc(c, link, slug)
    return c.report()


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network.
    return collect_from(context, fetch=fetch_url, robots_reader=read_robots, site=SITE_FETCH)
