"""Public Citizen's model state legislation, collected from its site's model legislation page.

The library is one page on citizen.org, "Model Artificial Intelligence Legislation for State
Lawmakers", which lists each model state law under a heading of its own, linked as "Model
Legislation" beside an issue explainer and a state tracker; only the model links are documents.
Each model is an article on the same site whose text is the page itself, set inline in HTML, so
the page's bytes are what is kept and its text is read from them: the article's title, the date
above or below it where the organization gives one, and the model's text from its first section,
lettered subsection or "Model Language" label to the page's "Topics" block. A drafting
recommendation before the model, an "Important Note" for lawmakers after it, share buttons and
download links are not the model's text and are dropped; bracketed instructions to the drafter
go as `lineage.modeltext` drops them, and bracketed alternatives stay. A "Model Legislation" link
off the site, such as to a shared document, is reported failed, not followed.

Fetching is on (`SITE_FETCH`): read on 2026-11-15, the site's Terms of Use page has no text, its
footer licenses its text for non-commercial use with attribution, robots.txt allows every path,
and no page asks a visitor to accept terms. robots.txt asks for a 10-second crawl delay before its
first User-agent line, where Python's parser does not read it, so the collector waits that long
between its requests itself. See DESIGN.md#model-documents.
"""

import re
import urllib.parse
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser

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
from lineage.modeltext import drop_instructions

SITE = "https://www.citizen.org"
LIBRARY = f"{SITE}/article/model-artificial-intelligence-legislation-for-state-lawmakers/"
# The library's index pages, read in order. One today; a second topic's page would join it here.
LIBRARY_PAGES = (LIBRARY,)
# On: see the module docstring. Tests pass `site=` either way.
SITE_FETCH = True
# robots.txt's "Crawl-delay: 10" stands before its "User-agent: *" group, so
# `lineage.robots` reads no delay; the collector honors it anyway (read 2026-11-15).
CRAWL_DELAY_SECONDS = 10.0
HOSTS = frozenset({"www.citizen.org", "citizen.org"})

# The descriptor's sources, read 2026-11-15:
# - kind: https://www.citizen.org/about/ (a nonprofit organization founded in 1971, in two parts:
#   Public Citizen, Inc., for lobbying and organizing, and Public Citizen Foundation, for
#   research and public education) and https://www.citizen.org/donate/pc-v-pc-foundation/
#   (the website is shared by the two; a contribution to Public Citizen, Inc. makes a member).
# - IRS: Public Citizen, Inc., EIN 23-7104508, files as a 501(c)(4):
#   https://projects.propublica.org/nonprofits/organizations/237104508
#   Public Citizen Foundation, Inc., EIN 52-1263996, files as a 501(c)(3):
#   https://projects.propublica.org/nonprofits/organizations/521263996
#   Their FY2024 returns, as the site posts them (https://www.citizen.org/about/annual-report/):
#   https://www.citizen.org/wp-content/uploads/2024_0930_PUBLIC-CITIZEN_TAX-RETURN_PUBLIC-DISCLOSURE.pdf
#   https://www.citizen.org/wp-content/uploads/2024_0930_PUBLIC-CITIZEN-FOUNDATION_TAX-RETURN_PUBLIC-DISCLOSURE.pdf
# - which entity: no model page names one. The library offers state lawmakers help "to pass this
#   legislation", which the about page assigns to Public Citizen, Inc., whose return describes
#   programs that champion legislation and lobby; the Foundation's describes public education.
# - terms: https://www.citizen.org/terms-of-use/ has a heading and no text; every page's footer
#   permits non-commercial use of the text with attribution under CC BY-NC-ND 3.0 US, the
#   license below; https://www.citizen.org/robots.txt disallows nothing.
SOURCE = Source(
    key="public-citizen",
    organization="Public Citizen (Public Citizen, Inc.)",
    descriptor="nonprofit membership organization, 501(c)(4) (Public Citizen, Inc., EIN "
    "23-7104508), which by its own account lobbies and organizes; it shares its website with "
    "its affiliate Public Citizen Foundation, Inc., a 501(c)(3) (EIN 52-1263996) for research "
    "and public education",
    members="People who contribute to Public Citizen, Inc. become its members",
    members_source="https://www.citizen.org/donate/pc-v-pc-foundation/",
    library_url=LIBRARY,
    terms_url="https://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    terms_note="The site's Terms of Use page has no text; its footer licenses Public Citizen's "
    "text for non-commercial use with attribution under Creative Commons BY-NC-ND 3.0 US, which "
    "bars sharing altered copies. robots.txt allows every path and asks a 10-second crawl delay. "
    "No page asks a visitor to accept terms.",
    terms_read_on=date(2026, 11, 15),
)

MEDIA_TYPE = "text/html"

# Text a browser shows in its own block: the boundary between two of the model's lines.
_BLOCKS = frozenset({
    "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "figure",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p", "pre", "section",
    "table", "td", "th", "tr", "ul",
})  # fmt: skip
# Never the article's text: menus, sidebars, the footer, scripts, forms and buttons.
_SKIP = frozenset({
    "aside", "button", "footer", "form", "iframe", "nav", "noscript", "script", "select", "style",
    "svg",
})  # fmt: skip
_SHARE_HOSTS = frozenset({
    "facebook.com", "www.facebook.com", "twitter.com", "x.com", "bsky.app", "threads.net",
    "www.threads.net", "linkedin.com", "www.linkedin.com",
})  # fmt: skip
# The heading that ends the article: its tags, the related articles, the follow and donate
# blocks, the copyright line.
_END = re.compile(
    r"^(?:topics|you might be interested in|stay updated(?: on public citizen)?|support our work"
    r"|copyright\b.*)\s*:?$",
    re.IGNORECASE,
)
# A share widget's own label, which is not a link.
_SHARE_LABEL = re.compile(r"^share(?:\s+this(?:\s+page|\s+article)?)?\s*:?$", re.IGNORECASE)
# The label some pages set above the model, after drafting recommendations.
_MODEL_LABEL = re.compile(r"^model\s+(?:language|legislation|bill(?:\s+text)?|text)\s*:?$", re.I)
# A section heading: "SECTION 1.", "Sec. 2", "Section 3", "I.", "[[SEC. 1.".
_SECTION = re.compile(
    r"^\[*(?:(?i:section|sec\b\.?)\s*[0-9]+[A-Za-z]?\b|(?i:article|part)\s+[0-9IVXLC]+\b"
    r"|[IVXLC]+\.\s)"
)
# Where a model's text may begin: a section heading, a lettered or numbered subsection, a title
# or enacting clause.
_LAW_START = re.compile(
    _SECTION.pattern + r"|^\(\s*(?:[a-z]|[0-9]+)\s*\)|^(?i:an act\b|be it enacted\b)|^§"
)
# Advice to lawmakers set after a model or between its sections.
_NOTE = re.compile(
    r"^(?:important\s+note|note\s+(?:for|to)\s+(?:lawmakers|legislators|drafters)"
    r"|drafting\s+notes?|drafter'?s'?\s+notes?|commentary)\b",
    re.IGNORECASE,
)
_MONTH_NAMES = (
    ("january", "jan"),
    ("february", "feb"),
    ("march", "mar"),
    ("april", "apr"),
    ("may",),
    ("june", "jun"),
    ("july", "jul"),
    ("august", "aug"),
    ("september", "sept", "sep"),
    ("october", "oct"),
    ("november", "nov"),
    ("december", "dec"),
)
_MONTHS = {name: month for month, names in enumerate(_MONTH_NAMES, start=1) for name in names}
_DATE = re.compile(r"^([A-Za-z]+)\.?\s+([0-9]{1,2}),\s+([0-9]{4})$")
# How far from the title the page's own date may stand, in short blocks such as share buttons.
_DATE_REACH = 8
_SHORT_WORDS = 4


class PageError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Block:
    text: str
    title: bool  # inside the page's h1
    link_only: bool  # every word of it is a link's text: a share button, a download link


@dataclass(frozen=True, slots=True)
class Link:
    href: str  # absolute, without its fragment
    text: str


@dataclass(frozen=True, slots=True)
class Page:
    title: str
    published: date | None
    text: str  # cleaned, as matching reads it


class _Reader(HTMLParser):
    """A page's text as blocks, in reading order, and its links."""

    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.blocks: list[Block] = []
        self.links: list[Link] = []
        self._skip = 0
        self._h1 = 0
        self._parts: list[str] = []
        self._linked: list[bool] = []
        self._href: str | None = None
        self._share = False
        self._link_text: list[str] = []

    def _flush(self) -> None:
        text = " ".join("".join(self._parts).split())
        if text:
            pairs = zip(self._parts, self._linked, strict=True)
            linked = "".join(part for part, inside in pairs if inside)
            self.blocks.append(Block(text, self._h1 > 0, len(linked.split()) == len(text.split())))
        self._parts, self._linked = [], []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
            return
        if tag in _BLOCKS:
            self._flush()
        if tag == "h1":
            self._h1 += 1
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            absolute = urllib.parse.urldefrag(urllib.parse.urljoin(self.base, href)).url
            parts = urllib.parse.urlsplit(absolute)
            self._share = parts.scheme in {"mailto", "sms"} or (
                (parts.hostname or "") in _SHARE_HOSTS
            )
            scheme_ok = parts.scheme in {"http", "https"} and not href.startswith("#")
            self._href = absolute if scheme_ok else None
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag in _BLOCKS:
            # Before the h1 count drops, so the title's block is marked as the title.
            self._flush()
        if tag == "h1":
            self._h1 = max(0, self._h1 - 1)
        elif tag == "a":
            if self._href is not None and self._skip == 0:
                self.links.append(Link(self._href, " ".join("".join(self._link_text).split())))
            self._href, self._share = None, False

    def handle_data(self, data: str) -> None:
        if self._skip or self._share:
            return
        in_link = self._href is not None
        if in_link:
            self._link_text.append(data)
        self._parts.append(data)
        self._linked.append(in_link)

    def close(self) -> None:
        super().close()
        self._flush()


def read(page: bytes, base: str = SITE) -> _Reader:
    reader = _Reader(base)
    reader.feed(page.decode("utf-8", errors="replace"))
    reader.close()
    return reader


def model_links(index: Fetched) -> list[str]:
    """The documents a library page lists: its "Model Legislation" links, in order, once each."""
    links = read(index.body, index.url).links
    found = [link.href for link in links if _MODEL_LABEL.match(link.text)]
    return list(dict.fromkeys(found))


def doc_slug(url: str) -> str:
    """ "/article/public-citizen-model-state-law-protecting-x/" -> "protecting-x"."""
    segment = urllib.parse.urlsplit(url).path.rstrip("/").rpartition("/")[2]
    slug = slugify(segment)
    for prefix in ("public-citizen-", "model-state-law-"):
        slug = slug.removeprefix(prefix)
    return slug


def _date(text: str) -> date | None:
    m = _DATE.match(text)
    if m is None or (month := _MONTHS.get(m.group(1).lower())) is None:
        return None
    try:
        return date(int(m.group(3)), month, int(m.group(2)))
    except ValueError:
        return None


def _published(blocks: list[Block], title_at: int) -> date | None:
    """The date the page sets beside its title, across only short blocks such as share buttons;
    the related articles' dates further down are theirs, not the model's."""
    before = range(title_at - 1, max(-1, title_at - 1 - _DATE_REACH), -1)
    after = range(title_at + 1, min(len(blocks), title_at + 1 + _DATE_REACH))
    for steps in (before, after):
        for i in steps:
            if (found := _date(blocks[i].text)) is not None:
                return found
            if len(blocks[i].text.split()) > _SHORT_WORDS:
                break
    return None


def _model_lines(body: list[Block]) -> list[str]:
    """The model's own lines from the article's body: from its label or first section to its
    end, without notes to lawmakers, share buttons or download links."""
    label = next((i for i, b in enumerate(body) if _MODEL_LABEL.match(b.text)), None)
    if label is not None:
        start = label + 1
    else:
        start = next((i for i, b in enumerate(body) if _LAW_START.match(b.text)), None)
        if start is None:
            raise PageError("no model text: no section, subsection or model label on the page")
    lines: list[str] = []
    in_note = False
    for block in body[start:]:
        if _NOTE.match(block.text):
            in_note = True
        elif in_note and _SECTION.match(block.text):
            in_note = False
        if not in_note and not block.link_only and not _SHARE_LABEL.match(block.text):
            lines.append(block.text)
    return lines


def read_page(raw: bytes, url: str = SITE) -> Page:
    """A model's article as collected: its title, its date where the page gives one, and the
    model's text as matching reads it."""
    blocks = read(raw, url).blocks
    title_at = next((i for i, b in enumerate(blocks) if b.title), None)
    if title_at is None:
        raise PageError("no title (h1) on the page")
    title_end = title_at
    while title_end + 1 < len(blocks) and blocks[title_end + 1].title:
        title_end += 1
    title = " ".join(b.text for b in blocks[title_at : title_end + 1])
    end = next(
        (i for i in range(title_end + 1, len(blocks)) if _END.match(blocks[i].text)), len(blocks)
    )
    lines = _model_lines(blocks[title_end + 1 : end])
    text = re.sub(r"\n{3,}", "\n\n", drop_instructions("\n".join(lines))).strip()
    if not text:
        raise PageError("no model text after the title")
    return Page(title=title, published=_published(blocks, title_at), text=text + "\n")


def _on_site(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == "https" and (parts.hostname or "").lower() in HOSTS


def _index(c: Collection, url: str) -> list[str]:
    """One library page's documents; a page it may not or could not read is settled here."""
    if c.permission(url) == "blocked":
        c.block(url)
        return []
    try:
        page = c.get(url)
    except CollectFetchError as e:
        c.fail(url, f"the library page: {e}")
        return []
    found = model_links(page)
    if not found:
        c.fail(url, "the library page lists no model legislation")
    return found


def _collect_doc(c: Collection, url: str, slugs: set[str]) -> None:
    """One document, settled one way: kept, unchanged, blocked or failed."""
    if not _on_site(url):
        c.fail(url, "off the library's site; this collector reads citizen.org pages only")
        return
    slug = doc_slug(url)
    if slug in slugs:
        c.fail(url, f"another document already has the slug {slug}")
        return
    slugs.add(slug)
    if (robots := c.check(url)) is None:
        return
    try:
        got = c.get(url)
    except CollectFetchError as e:
        c.fail(url, str(e))
        return
    if "html" not in got.content_type.lower():
        c.fail(url, f"not an HTML page: {got.content_type or 'no Content-Type'}")
        return
    try:
        page = read_page(got.body, url)
    except PageError as e:
        c.fail(url, str(e))
        return
    doc = provenance(
        SOURCE,
        slug=slug,
        title=page.title,
        url=url,
        raw=got.body,
        media_type=MEDIA_TYPE,
        text=page.text,
        robots=robots,
        collected_on=c.context.on,
        published=page.published,
    )
    c.keep(doc, raw=got.body, text=page.text)


def collect_from(
    context: CollectContext,
    *,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    site: bool = SITE_FETCH,
    library: tuple[str, ...] = LIBRARY_PAGES,
) -> CollectReport:
    """Every model the `library` pages list, each fetched at least the site's crawl delay after
    the last request. With `site` off nothing is asked, and the report says so."""

    def paced(url: str, interval: float) -> Fetched:
        return fetch(url, max(interval, CRAWL_DELAY_SECONDS))

    c = Collection(SOURCE, context, fetch=paced, robots_reader=robots_reader)
    if not site:
        for url in library:
            c.fail(url, "fetching from the site is off (SITE_FETCH)")
        return c.report()
    seen: set[str] = set()
    slugs: set[str] = set()
    for page in library:
        for url in _index(c, page):
            # A model two library pages list is one document.
            if url not in seen:
                seen.add(url)
                _collect_doc(c, url, slugs)
    return c.report()


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network.
    return collect_from(context, fetch=fetch_url, robots_reader=read_robots, site=SITE_FETCH)
