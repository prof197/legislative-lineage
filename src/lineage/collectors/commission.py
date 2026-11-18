"""The Uniform Law Commission's acts, the first library of model legislation collected.

The acts already on disk come first: `lineage.acts` knows the development and held-out acts the
commission eval reads from `data/acts/`, and the two reserve acts origin recall's sealed key
reads, each saved by hand from the commission's download as plain text. They are collected as
they are, without a request beyond robots.txt, so a collected document and the eval's query come
from the same file. The rest come from the commission's site:
the current-acts catalog, one page per initial letter, then each act's page, its "Final Act"
library document, and the file attached to it, the edition without comments where there is one.

Which of the rest are collected: the acts the commission's committed enactment lists show
enacted in 2023-2026, the corpus years (`lineage.ulc`). Those lists load only in a browser, so
they are hand copies, and an act missing from them is not collected; `wanted=None` collects the
whole catalog.

Fetching from the site is off (`SITE_FETCH`) until the commission's terms are ruled on for this
use: read on 2026-11-15, its document pages appear to ask a visitor to accept a copyright
notice before a download, which a request for the file's URL would pass by. The fetch path is
built and tested against recorded responses (tests/fixtures/collect/). See
DESIGN.md#model-documents.
"""

import re
import urllib.parse
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from lineage.acts import (
    ACT_NAMES,
    ACTS,
    ACTS_DIR,
    HELD_OUT_ACTS,
    HELD_OUT_DIR,
    ORIGIN_HELD_OUT_ACTS,
    ActTextError,
)
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
from lineage.modeltext import ModelTextError, clean_act, docx_text
from lineage.text.extract import ExtractError, pdf_text
from lineage.ulc import ENACTMENTS, HELD_OUT_ENACTMENTS, load_enactments

SITE = "https://www.uniformlaws.org"
CATALOG = f"{SITE}/acts/catalog/current"
# Off until the terms are ruled on (see the module docstring); tests pass `site=True`.
SITE_FETCH = False
CORPUS_YEARS = (2023, 2026)

# The descriptor's sources, read 2026-11-15:
# - kind: https://www.uniformlaws.org/aboutulc/overview ("a non-profit unincorporated
#   association, comprised of state commissions on uniform laws from each state") and
#   https://www.uniformlaws.org/aboutulc/faq (funded mainly by state appropriations).
# - IRS: ProPublica's Nonprofit Explorer lists no filer under the commission's names; its
#   affiliate the Uniform Law Foundation (EIN 36-3761561) files as a 501(c)(3):
#   https://projects.propublica.org/nonprofits/organizations/363761561
SOURCE = Source(
    key="commission",
    organization="Uniform Law Commission (National Conference of Commissioners on Uniform "
    "State Laws)",
    # Type: uniformlaws.org/aboutulc/overview. IRS: no exemption filing under its own names in
    # ProPublica's Nonprofit Explorer; the affiliate is EIN 36-3761561 there.
    descriptor="nonprofit unincorporated association of the states' commissions on uniform "
    "laws; no IRS exemption filing of its own found, while its affiliate, the Uniform Law "
    "Foundation, files as a 501(c)(3)",
    members="Commissioners are lawyers appointed by their state governments; each jurisdiction "
    "decides how they are appointed",
    members_source="https://www.uniformlaws.org/aboutulc/overview",
    library_url=CATALOG,
    terms_url=f"{SITE}/termsofuse",
    terms_note="Using the site counts as agreeing to its terms, with nothing to accept first. "
    "They allow viewing, copying, printing and distributing its content, unmodified, for "
    "informational, noncommercial purposes with the copyright notice or other attribution, and "
    "modifying it as part of a legislative process with prominent acknowledgement of the "
    "commission. Nothing on automated access. Its document pages appear to ask a visitor to "
    "accept a copyright notice before a download, so fetching from the site stays off; the acts "
    "matched are copies saved by hand.",
    terms_read_on=date(2026, 11, 17),
)

# Each on-disk act's page on the commission's site, from its catalog (read 2026-11-15).
ACT_KEYS = {
    "public-expression-protection": "4f486460-199c-49d7-9fac-05570be1e7b1",
    "special-deposits": "36c7493b-d503-40e7-a74c-018a850ef519",
    "electronic-estate-planning-documents": "f911ff58-34ae-47a1-8928-68d5396a72b1",
    "assignment-for-benefit-of-creditors": "b7e5e644-b4b2-44eb-acbc-019859883add",
    "easement-relocation": "ec690784-90d6-42c3-99ea-1e13a49c8540",
    "mortgage-modification": "5c724df8-d5a1-4bc8-be3b-0191302a6a8d",
    "directed-trust": "ca4d8a5a-55d7-4c43-b494-5f8858885dd8",
    # The reserve acts, origin recall's sealed key: from each act's Final Act document link
    # (read 2026-11-16).
    "health-care-decisions": "3df274d6-776b-4780-8e4e-018a850ef44e",
    "antitrust-pre-merger-notification": "6bf5d101-d698-4c72-b7c1-0191302a6a95",
}

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MEDIA = {".docx": DOCX, ".pdf": "application/pdf"}
_LETTER_PAGE = re.compile(r"/acts/catalog/current/[a-z]+/?$", re.IGNORECASE)
_ACT_PAGE = re.compile(
    r"/(?:committees/community-home\?CommunityKey=[0-9a-f-]+|communities/[a-z0-9-]+/?)$",
    re.IGNORECASE,
)
_FILE_LINK = re.compile(r"/HigherLogic/System/DownloadDocumentFile\.ashx\?", re.IGNORECASE)
# "MAPA_Final_NC_2018.docx", "..._Final Act_No Comments_2022.docx": the edition without the
# prefatory note and comments, which the eval's acts already are.
_NO_COMMENTS = re.compile(r"(?:^|[_ .-])(?:NC|No[ _-]?Comments)(?:[_ .-]|$)", re.IGNORECASE)


def act_page(key: str) -> str:
    return f"{SITE}/committees/community-home?CommunityKey={key}"


def act_slug(name: str) -> str:
    """ "Health-Care Decisions Act" -> "health-care-decisions", as lineage.acts names acts."""
    return slugify(name).removesuffix("-act")


@dataclass(frozen=True, slots=True)
class Link:
    href: str  # absolute
    text: str
    in_heading: bool  # inside an h1-h6, where the catalog names its acts


class _Links(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.links: list[Link] = []
        self._heading = 0
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading += 1
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and not href.lower().startswith(("javascript:", "#", "mailto:")):
                self._href = urllib.parse.urljoin(self.base, href)
                self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading = max(0, self._heading - 1)
        elif tag == "a" and self._href is not None:
            text = " ".join("".join(self._text).split())
            self.links.append(Link(self._href, text, self._heading > 0))
            self._href = None

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)


def links(page: Fetched) -> list[Link]:
    parser = _Links(page.url)
    parser.feed(page.body.decode("utf-8", errors="replace"))
    parser.close()
    return parser.links


def letter_pages(catalog: Fetched) -> list[str]:
    """The catalog's pages by initial letter, which list every act without paging."""
    return sorted({link.href for link in links(catalog) if _LETTER_PAGE.search(link.href)})


def catalog_acts(page: Fetched) -> dict[str, tuple[str, str]]:
    """The acts a catalog page lists: slug -> (name, the act's page)."""
    return {
        act_slug(link.text): (link.text, link.href)
        for link in links(page)
        if link.in_heading and link.text and _ACT_PAGE.search(link.href)
    }


def final_act_document(page: Fetched) -> str | None:
    """The act page's "Final Act" library document."""
    return next(
        (
            link.href
            for link in links(page)
            if "/viewdocument/" in link.href and link.text.lower().startswith("final act")
        ),
        None,
    )


def chosen_file(page: Fetched) -> tuple[str, str] | None:
    """The file to collect from a library document's attachments, as (url, file name): the
    edition without comments first, then Word before PDF, whose text layer reads less cleanly."""
    files = [
        (link.href, link.text)
        for link in links(page)
        if _FILE_LINK.search(link.href) and Path(link.text).suffix.lower() in _MEDIA
    ]
    if not files:
        return None

    def rank(file: tuple[str, str]) -> tuple[bool, int]:
        name = Path(file[1])
        return (_NO_COMMENTS.search(name.stem) is None, list(_MEDIA).index(name.suffix.lower()))

    return min(files, key=rank)


def file_text(data: bytes, media_type: str) -> str:
    return docx_text(data) if media_type == DOCX else pdf_text(data)


def enacted_in_corpus_years(
    paths: tuple[Path, ...] = (ENACTMENTS, HELD_OUT_ENACTMENTS),
) -> set[str]:
    """Acts the committed enactment lists show enacted in a corpus year."""
    first, last = CORPUS_YEARS
    return {
        row.act
        for path in paths
        for row in load_enactments(path).rows
        if row.stratum is not None and first <= row.year <= last
    }


def _collect_disk(c: Collection, acts_dir: Path, held_out_dir: Path, site: bool) -> set[str]:
    """The acts on disk; returns those settled here."""
    settled: set[str] = set()
    for act, directory in [(a, acts_dir) for a in ACTS] + [
        (a, held_out_dir) for a in HELD_OUT_ACTS + ORIGIN_HELD_OUT_ACTS
    ]:
        url = act_page(ACT_KEYS[act])
        path = directory / f"{act}.txt"
        if not path.exists():
            if not site:
                settled.add(act)
                c.fail(url, f"{act}.txt is not on disk, and fetching from the site is off")
            continue
        settled.add(act)
        if (robots := c.check(url)) is None:
            continue
        raw = path.read_bytes()
        try:
            text = clean_act(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, ActTextError) as e:
            c.fail(url, f"{act}.txt: {e}")
            continue
        doc = provenance(
            SOURCE,
            slug=act,
            title=f"{ACT_NAMES[act]} Act",
            url=url,
            raw=raw,
            media_type="text/plain",
            text=text,
            robots=robots,
            collected_on=c.context.on,
        )
        c.keep(doc, raw=raw, text=text)
    return settled


def _fetch_step(c: Collection, act_url: str, url: str) -> Fetched | None:
    """One request on the way to an act's file; a refusal or failure settles the act."""
    permission = c.permission(url)
    if permission == "blocked":
        c.block(url)
        return None
    try:
        return c.get(url)
    except CollectFetchError as e:
        c.fail(act_url, f"{url}: {e}")
        return None


def _collect_act(c: Collection, slug: str, name: str, url: str) -> None:
    if (page := _fetch_step(c, url, url)) is None:
        return
    if (document := final_act_document(page)) is None:
        c.fail(url, "no Final Act document on the act's page")
        return
    if (library := _fetch_step(c, url, document)) is None:
        return
    if (chosen := chosen_file(library)) is None:
        c.fail(url, f"{document}: no Word or PDF file attached")
        return
    file_url, file_name = chosen
    if (robots := c.check(file_url)) is None:
        return
    if (got := _fetch_step(c, url, file_url)) is None:
        return
    media_type = _MEDIA[Path(file_name).suffix.lower()]
    try:
        text = clean_act(file_text(got.body, media_type))
    except (ModelTextError, ExtractError, ActTextError) as e:
        c.fail(url, f"{file_name}: {e}")
        return
    doc = provenance(
        SOURCE,
        slug=slug,
        title=name,
        url=url,
        raw=got.body,
        media_type=media_type,
        text=text,
        robots=robots,
        collected_on=c.context.on,
    )
    c.keep(doc, raw=got.body, text=text)


def _collect_site(c: Collection, wanted: set[str] | None, settled: set[str]) -> None:
    if (catalog := _fetch_step(c, CATALOG, CATALOG)) is None:
        return
    listed: dict[str, tuple[str, str]] = {}
    for letter in letter_pages(catalog):
        if (page := _fetch_step(c, letter, letter)) is not None:
            listed |= catalog_acts(page)
    for slug in sorted(listed):
        if slug in settled or (wanted is not None and slug not in wanted):
            continue
        name, url = listed[slug]
        _collect_act(c, slug, name, url)
    for slug in sorted((wanted or set()) - settled - set(listed)):
        c.fail(CATALOG, f"{slug}: enacted in the corpus years but not in the catalog")


def collect_from(
    context: CollectContext,
    *,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    acts_dir: Path = ACTS_DIR,
    held_out_dir: Path = HELD_OUT_DIR,
    site: bool = SITE_FETCH,
    wanted: set[str] | None = None,
) -> CollectReport:
    """The acts on disk, then, with `site`, the `wanted` acts from the site (None: every act in
    the catalog)."""
    c = Collection(SOURCE, context, fetch=fetch, robots_reader=robots_reader)
    settled = _collect_disk(c, acts_dir, held_out_dir, site)
    if site:
        _collect_site(c, wanted, settled)
    return c.report()


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network and the disk.
    return collect_from(
        context,
        fetch=fetch_url,
        robots_reader=read_robots,
        acts_dir=ACTS_DIR,
        held_out_dir=HELD_OUT_DIR,
        site=SITE_FETCH,
        wanted=enacted_in_corpus_years() if SITE_FETCH else None,
    )
