"""The model state legislation the Ethics and Public Policy Center publishes with the Institute
for Family Studies, as one source.

What is collected: the documents both organizations present as model state legislation, listed
in `DOCUMENTS` as read on 2026-11-15. EPPC keeps a public library of model legislation (its
"Model Legislation" category), and IFS carries the same joint bill among its research briefs;
one joint document stands in both, and it is collected once, from EPPC's page. Each
organization also publishes model bills on its own (EPPC's library holds several, IFS hosts one
on its technology initiative's page); those are not this source's and are not collected here.
Policy briefs, reports, summaries and fact sheets are not model legislation and are left out.

A document is its publication page: the page names the organizations and the date, and links
the model text, a PDF or Word file, or a page of its own. The link is the one whose text names
the model legislation, never a summary or fact sheet, and only on the organizations' own hosts.
EPPC's files under eppc.org/wp-content/uploads/ answer with a permanent redirect to
media.eppc.org; the file is asked for there, so that host's robots.txt is the one read.

Cleaning (`clean_model_text`) keeps the bill's parts and sections and drops what no state
enacts: the cover and author lines before the first heading, the running page footers and page
numbers, footnotes, the capitalized line that introduces an alternative section, commentary
and drafting notes, the author biographies after the last section, and bracketed instructions
to the drafter (`lineage.modeltext.drop_instructions`); brackets a state enacts as written stay.

Fetching is on (`SITE_FETCH`): read on 2026-11-15, neither organization publishes terms of use
and nothing asks a visitor to accept terms, and both sites' robots.txt let this project read
the documents' paths. EPPC's robots.txt signals that its content is not for training AI; it is
read here to measure the wording bills share with it, kept local and never redistributed. See
DESIGN.md#model-documents.
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
IFS = "https://ifstudies.org"
LIBRARY = f"{EPPC}/category/model-legislation/"
# On, from the terms as read (see the module docstring); tests pass `site=` either way.
SITE_FETCH = True

# The descriptor's sources, read 2026-11-15:
# - EPPC, IRS: ProPublica's Nonprofit Explorer, Ethics And Public Policy Center Inc, EIN
#   52-1162185, 501(c)(3), exempt since 1980, latest Form 990 for the year ending June 2025:
#   https://projects.propublica.org/nonprofits/organizations/521162185
#   kind: https://eppc.org/about/ (an institute in Washington, D.C.; no membership).
# - IFS, IRS: ProPublica's Nonprofit Explorer, The Institute For Family Studies, EIN
#   27-0950140, 501(c)(3), exempt since 2010, latest Form 990 for the year ending December 2025:
#   https://projects.propublica.org/nonprofits/organizations/270950140
#   kind: https://ifstudies.org/about-us (research and public engagement; no membership).
# Terms, read 2026-11-15: eppc.org's footer links no terms, and /terms-of-use and
# /privacy-policy answer 404; ifstudies.org links https://ifstudies.org/privacy-policy and
# https://ifstudies.org/ai-submission-policy, neither about using its content. robots.txt:
# https://eppc.org/robots.txt and https://media.eppc.org/robots.txt allow every path to any
# crawler not named, name AI-training crawlers to bar, and carry a Content-Signal of search
# yes, AI training no, use as reference; https://ifstudies.org/robots.txt disallows one report
# directory to all.
SOURCE = Source(
    key="eppc-ifs",
    organization="Ethics and Public Policy Center, with the Institute for Family Studies",
    # Type: each organization's about page. IRS: each one's filings in ProPublica's Nonprofit
    # Explorer, EINs 52-1162185 and 27-0950140 (URLs above).
    descriptor="two nonprofit corporations, each exempt under IRS section 501(c)(3): the "
    "Ethics and Public Policy Center, a public policy institute in Washington, D.C., and the "
    "Institute for Family Studies, a research and public education organization in "
    "Charlottesville, Virginia; neither has members",
    library_url=LIBRARY,
    terms_url=None,
    terms_note="Neither organization publishes terms of use: eppc.org links none and "
    "ifstudies.org links only a privacy policy and a policy for submissions. Pages carry a "
    "copyright notice and nothing asks a visitor to accept terms. Both sites' robots.txt let "
    "any crawler not named read the documents; EPPC's bars named AI-training crawlers and "
    "signals that its content may be used for search and reference, not to train AI. The "
    "texts are read for matching, kept local and never redistributed.",
    terms_read_on=date(2026, 11, 15),
)


@dataclass(frozen=True, slots=True)
class Listed:
    """A document as the organizations list it."""

    slug: str
    title: str  # the document's own title
    page: str  # its publication page, recorded as the document's URL
    published: date | None  # as the organizations date it


# Both organizations' model state legislation, read 2026-11-15. The joint social media bill,
# by authors from both, stands in EPPC's library and as IFS's "Model Legislation for Age
# Verification of Social Media" brief
# (https://ifstudies.org/report-brief/model-legislation-for-age-verification-of-social-media),
# whose download buttons are scripts; EPPC's page links the file.
DOCUMENTS: tuple[Listed, ...] = (
    Listed(
        slug="age-verification-of-social-media",
        title="Model Legislation for Age Verification of Social Media",
        page=f"{EPPC}/publication/protecting-kids-online-model-legislation/",
        published=date(2023, 10, 17),
    ),
)

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_FILES = {".docx": DOCX, ".pdf": "application/pdf"}
# Only the organizations' own hosts: a link elsewhere is another publisher's, under terms not
# read for this source.
HOSTS = frozenset(
    {"eppc.org", "www.eppc.org", "media.eppc.org", "ifstudies.org", "www.ifstudies.org"}
)
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


def model_text_link(page: Fetched) -> str | None:
    """The link to the model text on a publication page: one whose text names the model
    legislation first, then a bare "PDF" button; a file before a page, Word before PDF. Never a
    summary, fact sheet or brief, a listing, or a link off the organizations' hosts."""
    here = urllib.parse.urlsplit(page.url)._replace(fragment="").geturl()
    ranked: list[tuple[int, int, int, str]] = []
    for order, link in enumerate(links(page)):
        parts = urllib.parse.urlsplit(link.href)
        name = PurePosixPath(parts.path).name
        if (
            (parts.hostname or "").lower() not in HOSTS
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

# "Part 1. General Requirements", "Section 101. Definitions.", "SECTION 1. TITLE.", and an
# optional one in brackets.
_HEADING = re.compile(
    r"^\s*\[?\s*(?:(?:part|article)\s+(?:\d+|[ivxlc]+)\b|(?:section|sec\.)\s+\d+)", re.IGNORECASE
)
_CONTENTS_ENTRY = re.compile(r"(?:\t|\.{3,}|\s{2,})\s*\d+\s*$")
_ORGS = r"(?:ethics\s+and\s+public\s+policy\s+center|institute\s+for\s+family\s+studies)"
# The site a footer names after the organization: "♦ www.eppc.org".
_SITE = r"(?:\s*[^\w\s]?\s*(?:www\.)?(?:eppc|ifstudies)\.org)?"
# A page's running footer ("5 ethics and public policy center ♦ www.eppc.org"), its running
# header, or a bare page number.
_FURNITURE = re.compile(
    rf"^\s*(?:\d{{1,3}}\s+{_ORGS}{_SITE}|{_ORGS}{_SITE}\s+\d{{1,3}}|\d{{1,3}}"
    r"|technology\s+and\s+human\s+flourishing)\s*$",
    re.IGNORECASE,
)
# A footnote opens with its number and a sentence, and runs to the page's footer.
_FOOTNOTE = re.compile(r"^\s*(\d{1,2})\s+[A-Z][a-z]")
# A footnote's mark at the end of a line, "... years.1", or on a line of its own.
_FOOTNOTE_MARK = re.compile(r"(?<=[.;:,\"”])(\d{1,2})$")
_BARE_NUMBER = re.compile(r"^\s*(\d{1,2})\s*$")
# The capitalized line that introduces an alternative: "OR ANOTHER MORE LIMITED OPTION ...".
_ALTERNATIVE = re.compile(r"^\s*(?:OR|ALTERNATIVE(?:LY)?|OPTION)\b[^a-z]*$")
# Commentary and notes run from their heading to the next part or section.
# A heading on its own line, or "Drafting note: ...", never a sentence that opens with the word.
_COMMENTARY = re.compile(
    r"^\s*(?:commentary|comments?|(?:drafting|drafter'?s|explanatory|legislative)\s+notes?)"
    r"\s*(?::.*)?$",
    re.IGNORECASE,
)
# What follows the last section: the authors, endorsements, notes, the copyright line.
_BACK_MATTER = re.compile(
    r"^\s*(?:about\s+the\s+authors?\b|endnotes\b|endorse(?:ments?|d\s+by)\b|"
    r"acknowledge?ments\b|copyright\s+(?:©|\(c\)|\d{4})|©)",
    re.IGNORECASE,
)


def _is_heading(line: str) -> bool:
    return _HEADING.match(line) is not None and _CONTENTS_ENTRY.search(line) is None


def clean_model_text(raw: str) -> str:
    """A model bill's text as matching reads it: its parts and sections, without the cover,
    page furniture, footnotes, alternative markers, commentary, back matter or instructions to
    the drafter."""
    lines = drop_instructions(raw.replace("\r\n", "\n").replace("\r", "\n")).split("\n")
    start = next((i for i, line in enumerate(lines) if _is_heading(line)), None)
    if start is None:
        raise ModelTextError("no part or section heading: is this the model's text?")
    kept: list[str] = []
    marks: set[str] = set()  # footnote numbers marked so far; a footnote must have been marked
    commentary = footnote = False
    for i, line in enumerate(lines[start:], start=start):
        if bare := _BARE_NUMBER.match(line):
            marks.add(bare[1])
        if _is_heading(line):
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
        if commentary or footnote or _ALTERNATIVE.match(line) or _CONTENTS_ENTRY.search(line):
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
        if _is_heading(line) or _BACK_MATTER.match(line):
            return False
    return False


def media_type(url: str, content_type: str) -> str | None:
    if (suffix := _suffix(url)) in _FILES:
        return _FILES[suffix]
    media = content_type.partition(";")[0].strip().lower()
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


def _collect_document(c: Collection, listed: Listed) -> None:
    if (page := _fetch_step(c, listed.page, listed.page)) is None:
        return
    if (link := model_text_link(page)) is None:
        c.fail(listed.page, "no link to the model legislation on its page")
        return
    url = on_media_host(link)
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
        SOURCE,
        slug=listed.slug,
        title=listed.title,
        url=listed.page,
        raw=got.body,
        media_type=media,
        text=text,
        robots=robots,
        collected_on=c.context.on,
        published=listed.published,
    )
    c.keep(doc, raw=got.body, text=text)


def collect_from(
    context: CollectContext,
    *,
    documents: Sequence[Listed] = DOCUMENTS,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    site: bool = SITE_FETCH,
) -> CollectReport:
    """Each listed document from its publication page; with `site` off, each is reported failed
    and nothing is asked."""
    c = Collection(SOURCE, context, fetch=fetch, robots_reader=robots_reader)
    for listed in sorted(documents, key=lambda d: d.slug):
        if site:
            _collect_document(c, listed)
        else:
            c.fail(listed.page, "fetching from the site is off (SITE_FETCH)")
    return c.report()


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network.
    return collect_from(
        context,
        documents=DOCUMENTS,
        fetch=fetch_url,
        robots_reader=read_robots,
        site=SITE_FETCH,
    )
