"""The Ethics and Public Policy Center's model state legislation.

What is collected: the documents EPPC presents as model state legislation, listed in
`DOCUMENTS` as read on 2026-11-15 from its public "Model Legislation" library, and its
assisted reproductive technology package, which the library does not list. A document another
organization publishes with EPPC names it as a co-publisher, as the document or its page names
it: the 2023 social media bill, written with the Institute for Family Studies (which carries it
too, and is its own source, `lineage.collectors.ifs`), and the 2025 age verification bill,
EPPC's with the Center for Renewing America. The library's article on four legislative
proposals is left out: its bill texts are another publisher's, on another host.

A document is its publication page: the page names the organizations and the date, and links
the model text, a PDF or Word file, or a page of its own. The link is the one whose text names
the model legislation, never a summary or fact sheet, and only on the organization's own hosts.
Where the link answers with a redirect, or there is no page, `Listed.file` records the file's
own URL, so the host that serves it is the one whose robots.txt is read; EPPC's files under
eppc.org/wp-content/uploads/ are served from media.eppc.org and are asked for there.

Cleaning (`clean_model_text`) keeps the bill's parts and sections and drops what no state
enacts: the cover, author lines and preamble before the first heading, the running page footers,
program banners and page numbers, footnotes, the capitalized line that introduces an
alternative section, overviews, commentary and drafting notes, the author biographies after the
last section, and bracketed instructions to the drafter (`lineage.modeltext.drop_instructions`);
brackets a state enacts as written stay. The Institute for Family Studies' collector reads its
bill the same way.

Fetching is on (`SITE_FETCH`): read on 2026-11-15, EPPC publishes no terms of use and nothing
asks a visitor to accept terms, and its robots.txt lets this project read the documents' paths.
It signals that its content is not for training AI; it is read here to measure the wording bills
share with it, kept local and never redistributed. See DESIGN.md#model-documents.
"""

import re
import urllib.parse
from collections.abc import Sequence
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
)
from lineage.extract import ExtractError, extract
from lineage.modeldocs import Source
from lineage.modeltext import ModelTextError, docx_text, drop_instructions

EPPC = "https://eppc.org"
MEDIA = "https://media.eppc.org"
LIBRARY = f"{EPPC}/category/model-legislation/"
# On, from the terms as read (see the module docstring); tests pass `site=` either way.
SITE_FETCH = True

# The descriptor's sources, read 2026-11-15:
# - IRS: ProPublica's Nonprofit Explorer, Ethics And Public Policy Center Inc, EIN 52-1162185,
#   501(c)(3), exempt since 1980, latest Form 990 for the year ending June 2025:
#   https://projects.propublica.org/nonprofits/organizations/521162185
# - kind: https://eppc.org/about/ (an institute in Washington, D.C.; no membership).
# Terms, read 2026-11-15: eppc.org's footer links no terms, and /terms-of-use and
# /privacy-policy answer 404. robots.txt: https://eppc.org/robots.txt and
# https://media.eppc.org/robots.txt allow every path to any crawler not named, name
# AI-training crawlers to bar, and carry a Content-Signal of search yes, AI training no, use as
# reference.
SOURCE = Source(
    key="eppc",
    organization="Ethics and Public Policy Center",
    # Type: eppc.org/about/. IRS: EIN 52-1162185 in ProPublica's Nonprofit Explorer (above).
    descriptor="nonprofit corporation exempt under IRS section 501(c)(3); a public policy "
    "institute in Washington, D.C.; no members",
    library_url=LIBRARY,
    terms_url=None,
    terms_note="EPPC publishes no terms of use: its site links none. Pages carry a copyright "
    "notice and nothing asks a visitor to accept terms. Its robots.txt lets any crawler not "
    "named read the documents, bars named AI-training crawlers, and signals that its content "
    "may be used for search and reference, not to train AI. The texts are read for matching, "
    "kept local and never redistributed.",
    terms_read_on=date(2026, 11, 15),
)


@dataclass(frozen=True, slots=True)
class Listed:
    """A document as its organization lists it."""

    slug: str
    title: str  # the document's own title
    page: str  # its publication page, recorded as the document's URL
    published: date | None  # as the organization dates it
    # Other organizations the document or its page names as publishing it, as named there.
    co_publishers: tuple[str, ...] = ()
    # The model text's own URL, where the page's link redirects to it or there is no page.
    file: str | None = None


# EPPC's model state legislation, read 2026-11-15.
DOCUMENTS: tuple[Listed, ...] = (
    # Written with IFS's executive director and carried by IFS as its "Model Legislation for
    # Age Verification of Social Media" brief, whose download buttons are scripts.
    Listed(
        slug="age-verification-of-social-media",
        title="Model Legislation for Age Verification of Social Media",
        page=f"{EPPC}/publication/protecting-kids-online-model-legislation/",
        published=date(2023, 10, 17),
        co_publishers=("Institute for Family Studies",),
    ),
    # Its page calls it a model "from the Ethics and Public Policy Center and Center for
    # Renewing America", and the PDF's authors are one from each.
    Listed(
        slug="age-verification-for-websites-containing-obscene-and-indecent-material",
        title="Age Verification for Websites Containing Obscene and Indecent Material",
        page=f"{EPPC}/publication/model-legislation-age-verification-for-websites-containing"
        "-obscene-and-indecent-material/",
        published=date(2025, 1, 29),
        co_publishers=("Center for Renewing America",),
    ),
    # "View Model Legislation" links eppc.org/the-safe-chat-act/, which answers with a 301 to
    # this PDF.
    Listed(
        slug="safe-chat-act",
        title="The SAFE CHAT Act",
        page=f"{EPPC}/publication/safeguarding-children-from-ai-companion-chatbots-model"
        "-legislation/",
        published=date(2025, 10, 29),
        file=f"{MEDIA}/2025/10/The-SAFE-CHAT-Act.pdf",
    ),
    Listed(
        slug="books-act",
        title="The BOOKS (Books Optimize Our Kids' Schools) Act",
        page=f"{EPPC}/publication/the-books-act/",
        published=date(2026, 3, 2),
    ),
    # Nine bills in one PDF, with no publication page found and no date of its own; the upload
    # path says November 2024.
    Listed(
        slug="state-assisted-reproductive-technology-model-legislation",
        title="State Assisted Reproductive Technology Model Legislation",
        page=f"{EPPC}/wp-content/uploads/2024/11/State-Assisted-Reproductive-Technology-Model"
        "-Legislation.pdf",
        published=None,
        file=f"{MEDIA}/2024/11/State-Assisted-Reproductive-Technology-Model-Legislation.pdf",
    ),
)

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF = "application/pdf"
_FILES = {".docx": DOCX, ".pdf": PDF}
# Only the organization's own hosts: a link elsewhere is another publisher's, under terms not
# read for this source.
HOSTS = frozenset({"eppc.org", "www.eppc.org", "media.eppc.org"})
_UPLOADS = re.compile(r"^https?://(?:www\.)?eppc\.org/wp-content/uploads/", re.IGNORECASE)
_MODEL = re.compile(r"\bmodel\s+(?:legislation|bill|act)\b", re.IGNORECASE)
_PDF_BUTTON = re.compile(r"^(?:view\s+|download\s+)?(?:pdf|word)$", re.IGNORECASE)
_NOT_MODEL = re.compile(r"summary|fact[\s_-]*sheet|brief|one[\s_-]*pager", re.IGNORECASE)
# Listings and tags link "Model Legislation" too; they are never the text.
_LISTING = re.compile(r"^/(?:category|tag|author|authors-scholars|publication|news)/", re.I)


@dataclass(frozen=True, slots=True)
class Link:
    href: str  # absolute
    text: str


class _Links(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.links: list[Link] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href and not href.lower().startswith(("javascript:", "#", "mailto:")):
                self._href = urllib.parse.urljoin(self.base, href)
                self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append(Link(self._href, " ".join("".join(self._text).split())))
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)


def links(page: Fetched) -> list[Link]:
    parser = _Links(page.url)
    parser.feed(page.body.decode("utf-8", errors="replace"))
    parser.close()
    return parser.links


def _suffix(url: str) -> str:
    return PurePosixPath(urllib.parse.urlsplit(url).path).suffix.lower()


def model_text_link(page: Fetched, hosts: frozenset[str] = HOSTS) -> str | None:
    """The link to the model text on a publication page: one whose text names the model
    legislation first, then a bare "PDF" button; a file before a page, Word before PDF. Never a
    summary, fact sheet or brief, a listing, or a link off the organization's `hosts`."""
    here = urllib.parse.urlsplit(page.url)._replace(fragment="").geturl()
    ranked: list[tuple[int, int, int, str]] = []
    for order, link in enumerate(links(page)):
        parts = urllib.parse.urlsplit(link.href)
        name = PurePosixPath(parts.path).name
        if (
            (parts.hostname or "").lower() not in hosts
            or parts._replace(fragment="").geturl() == here
            or _LISTING.search(parts.path)
            or _NOT_MODEL.search(link.text)
            or _NOT_MODEL.search(name)
        ):
            continue
        suffix = _suffix(link.href)
        if _MODEL.search(link.text):
            tier = 0
        elif _PDF_BUTTON.match(link.text) and suffix in _FILES:
            tier = 1
        else:
            continue
        kind = list(_FILES).index(suffix) if suffix in _FILES else len(_FILES)
        ranked.append((tier, kind, order, link.href))
    return min(ranked)[3] if ranked else None


def on_media_host(url: str) -> str:
    """EPPC's upload URL as the host that serves it."""
    return _UPLOADS.sub(f"{MEDIA}/", url)


# ---- cleaning -------------------------------------------------------------------------------

_SPELLED = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)
# "Part 1. General Requirements", "Section 101. Definitions.", "SECTION 1. TITLE.",
# "SECTION A", "Section One: Findings", and an optional one in brackets. A lettered section is
# a capital letter standing alone, never a word such as "Section a person".
_HEADING = re.compile(
    r"^\s*\[?\s*(?:(?:part|article)\s+(?:\d+|[ivxlc]+)\b"
    rf"|(?:section|sec\.)\s+(?:\d+|(?-i:[A-Z])(?![\w'\u2019])|(?:{_SPELLED})\b))",
    re.IGNORECASE,
)
_CONTENTS_ENTRY = re.compile(r"(?:\t|\.{3,}|\s{2,})\s*\d+\s*$")
_ORGS = r"(?:ethics\s+and\s+public\s+policy\s+center|institute\s+for\s+family\s+studies)"
# The site a footer names after the organization: "♦ www.eppc.org".
_SITE = r"(?:\s*[^\w\s]?\s*(?:www\.)?(?:eppc|ifstudies)\.org)?"
# EPPC's programs, whose names head a document's pages.
_PROGRAMS = (
    r"(?:bioethics,?\s+)?technology,?\s+and\s+human\s+flourishing"
    r"|education\s+and\s+american\s+ideals"
)
# A page's running footer ("5 ethics and public policy center ♦ www.eppc.org"), its running
# header, or a bare page number.
_FURNITURE = re.compile(
    rf"^\s*(?:\d{{1,3}}\s+{_ORGS}{_SITE}|{_ORGS}{_SITE}\s+\d{{1,3}}|\d{{1,3}}|{_PROGRAMS})\s*$",
    re.IGNORECASE,
)
# A footnote opens with its number and a sentence, and runs to the page's footer.
_FOOTNOTE = re.compile(r"^\s*(\d{1,2})\s+[A-Z][a-z]")
# A footnote's mark at the end of a line, "... years.1", or on a line of its own.
_FOOTNOTE_MARK = re.compile(r"(?<=[.;:,\"”])(\d{1,2})$")
_BARE_NUMBER = re.compile(r"^\s*(\d{1,2})\s*$")
# An item's number left alone once the drafter's instruction it held is dropped: "3.".
_BARE_ITEM = re.compile(r"^\s*\(?(?:\d{1,3}|[A-Za-z]|[ivxlc]{2,5})[.)]\s*$")
# The capitalized line that introduces an alternative: "OR ANOTHER MORE LIMITED OPTION ...".
_ALTERNATIVE = re.compile(r"^\s*(?:OR|ALTERNATIVE(?:LY)?|OPTION)\b[^a-z]*$")
# Commentary, notes and overviews run from their heading to the next part or section: a
# heading on its own line, or "Drafting note: ...", never a sentence that opens with the word.
_COMMENTARY = re.compile(
    r"^\s*(?:commentary|comments?|overview|(?:drafting|drafter'?s|explanatory|legislative)"
    r"\s+notes?)\s*(?::.*)?$",
    re.IGNORECASE,
)
# What follows the last section: the authors, endorsements, notes, the copyright line, or a
# biography set without a heading ("A. Name is a senior fellow ...").
_BACK_MATTER = re.compile(
    r"^\s*(?:about\s+the\s+authors?\b|endnotes\b|endorse(?:ments?|d\s+by)\b|"
    r"acknowledge?ments\b|copyright\s+(?:©|\(c\)|\d{4})|©)"
    r"|^\s*(?!(?:the|this|that|each|any|such|no|every|a|an)\s)"
    r"(?-i:[A-Z][a-z'\u2019-]*\.?(?:\s+[A-Z]\.)*(?:\s+[A-Z][a-z'\u2019-]+){1,2})\s+is\s+"
    r"(?:(?:a|an|the|currently)\s+)?(?:senior|contributing|professor|fellow|executive|director|"
    r"president|research|policy|visiting|distinguished)\b",
    re.IGNORECASE,
)


def is_heading(line: str) -> bool:
    return _HEADING.match(line) is not None and _CONTENTS_ENTRY.search(line) is None


def clean_model_text(raw: str) -> str:
    """A model bill's text as matching reads it: its parts and sections, without the cover,
    page furniture, footnotes, alternative markers, commentary, back matter or instructions to
    the drafter."""
    lines = drop_instructions(raw.replace("\r\n", "\n").replace("\r", "\n")).split("\n")
    start = next((i for i, line in enumerate(lines) if is_heading(line)), None)
    if start is None:
        raise ModelTextError("no part or section heading: is this the model's text?")
    kept: list[str] = []
    marks: set[str] = set()  # footnote numbers marked so far; a footnote must have been marked
    commentary = footnote = False
    for i, line in enumerate(lines[start:], start=start):
        if bare := _BARE_NUMBER.match(line):
            marks.add(bare[1])
        if is_heading(line):
            commentary = footnote = False
        elif _BACK_MATTER.match(line):
            break
        elif _FURNITURE.match(line):
            footnote = False
            continue
        elif _COMMENTARY.match(line):
            commentary = True
        elif (
            (note := _FOOTNOTE.match(line))
            and note[1] in marks
            and _footer_comes_first(lines, i + 1)
        ):
            footnote = True
        if (
            commentary
            or footnote
            or _ALTERNATIVE.match(line)
            or _CONTENTS_ENTRY.search(line)
            or _BARE_ITEM.match(line)
        ):
            continue
        line = line.strip()
        if mark := _FOOTNOTE_MARK.search(line):
            marks.add(mark[1])
            line = line[: mark.start()]
        if line:
            kept.append(line)
    return "\n".join(kept) + "\n"


def _footer_comes_first(lines: Sequence[str], start: int) -> bool:
    """Whether a page footer comes before the next heading: a footnote sits at a page's foot."""
    for line in lines[start:]:
        if _FURNITURE.match(line):
            return True
        if is_heading(line) or _BACK_MATTER.match(line):
            return False
    return False


def media_type(url: str, content_type: str) -> str | None:
    """A PDF, Word file or page, by the URL's suffix, else by the Content-Type."""
    if (suffix := _suffix(url)) in _FILES:
        return _FILES[suffix]
    media = content_type.partition(";")[0].strip().lower()
    if media in {PDF, DOCX}:
        return media
    return "text/html" if media in {"text/html", "application/xhtml+xml"} else None


def document_text(got: Fetched, media: str) -> str:
    """The text of a fetched model: Word through lineage.modeltext, PDF and HTML through
    lineage.extract, as the text layer reads them."""
    if media == DOCX:
        return docx_text(got.body)
    return extract(got.body, got.content_type if media == "text/html" else media)


# ---- collecting -----------------------------------------------------------------------------


def _fetch_step(c: Collection, doc_url: str, url: str) -> Fetched | None:
    """One request on the way to a document's text; a refusal or failure settles it."""
    permission = c.permission(url)
    if permission == "blocked":
        c.block(url)
        return None
    try:
        return c.get(url)
    except CollectFetchError as e:
        c.fail(doc_url, f"{url}: {e}")
        return None


def _model_text_url(c: Collection, listed: Listed, hosts: frozenset[str]) -> str | None:
    """The URL of a document's model text: its recorded file, or its page's link."""
    if listed.file is not None:
        return listed.file
    if (page := _fetch_step(c, listed.page, listed.page)) is None:
        return None
    if (link := model_text_link(page, hosts)) is None:
        c.fail(listed.page, "no link to the model legislation on its page")
        return None
    return on_media_host(link)


def _collect_document(c: Collection, listed: Listed, hosts: frozenset[str]) -> None:
    if (url := _model_text_url(c, listed, hosts)) is None:
        return
    if (robots := c.check(url)) is None:
        return
    if (got := _fetch_step(c, listed.page, url)) is None:
        return
    if (media := media_type(url, got.content_type)) is None:
        c.fail(listed.page, f"{url}: not a PDF, Word file or page ({got.content_type})")
        return
    try:
        text = clean_model_text(document_text(got, media))
    except (ModelTextError, ExtractError) as e:
        c.fail(listed.page, f"{url}: {e}")
        return
    doc = provenance(
        c.source,
        slug=listed.slug,
        title=listed.title,
        url=listed.page,
        raw=got.body,
        media_type=media,
        text=text,
        robots=robots,
        collected_on=c.context.on,
        published=listed.published,
        co_publishers=listed.co_publishers,
    )
    c.keep(doc, raw=got.body, text=text)


def collect_documents(
    source: Source,
    context: CollectContext,
    documents: Sequence[Listed],
    *,
    hosts: frozenset[str],
    fetch: Fetch,
    robots_reader: ReadRobots,
    site: bool,
) -> CollectReport:
    """Each listed document of `source` from its publication page or file; with `site` off,
    each is reported failed and nothing is asked."""
    c = Collection(source, context, fetch=fetch, robots_reader=robots_reader)
    for listed in sorted(documents, key=lambda d: d.slug):
        if site:
            _collect_document(c, listed, hosts)
        else:
            c.fail(listed.page, "fetching from the site is off (SITE_FETCH)")
    return c.report()


def collect_from(
    context: CollectContext,
    *,
    documents: Sequence[Listed] = DOCUMENTS,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    site: bool = SITE_FETCH,
) -> CollectReport:
    return collect_documents(
        SOURCE,
        context,
        documents,
        hosts=HOSTS,
        fetch=fetch,
        robots_reader=robots_reader,
        site=site,
    )


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network.
    return collect_from(
        context,
        documents=DOCUMENTS,
        fetch=fetch_url,
        robots_reader=read_robots,
        site=SITE_FETCH,
    )
