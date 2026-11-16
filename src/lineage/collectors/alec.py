"""The American Legislative Exchange Council's model policy library.

ALEC publishes its model policies on its site, one WordPress page each under /model-policy/,
listed in the site's sitemaps. A page says what the document is ("Type:" Model Policy, Model
Resolution, Policy Statement, Statement of Principle or Model Ordinance), whether it is a
"Draft" or "Final", and its dates ("Date Introduced:", "Date Finalized:", "Date Amended:"). Those
fields are not in the site's REST API, so the page is read for them; the policy's text is read
from the API, whose `content` field holds the policy body alone, without the page's summary,
sidebar and sign-up form. An amendment is posted as a link to a Word or PDF file, which is read
instead.

Which documents are collected: the final model policies and model resolutions, the library's
model state legislation. Policy statements and statements of principle state the organization's
positions, and a model ordinance is written for local government, so neither is collected, and
neither is counted as found. Drafts are not collected either: a draft page says it is posted
before its task force has considered it and is taken down if not adopted, so it is not yet a
model the organization presents, and a later run collects it once it is final.

Fetching is on (`SITE_FETCH`). Read on 2026-11-15, the site's terms of use say that using the
site is agreeing to them, with nothing to accept first, and they forbid copying or reproducing
any of its material; robots.txt allows every path. A text is kept only on the machine that runs
the matching, is never shown or published, and a page gives only the share of wording and a
link, so collecting it was ruled acceptable (2026-11-15), and the sources page states the terms
as read. See DESIGN.md#model-documents.
"""

import html
import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import cast
from xml.etree import ElementTree

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
from lineage.extract import ExtractError, html_text, pdf_text
from lineage.modeldocs import Source
from lineage.modeltext import ModelTextError, docx_text, drop_instructions

SITE = "https://alec.org"
LIBRARY = f"{SITE}/model-policy/"
SITEMAP_INDEX = f"{SITE}/sitemap_index.xml"
# On: local copies for matching were ruled acceptable under the terms (see the module docstring).
SITE_FETCH = True
# The library's model state legislation, by the page's "Type:" and "Status:" (see the docstring).
TYPES = frozenset({"model policy", "model resolution"})
STATUSES = frozenset({"final"})

# The descriptor's sources, read 2026-11-15:
# - IRS: ProPublica's Nonprofit Explorer, "American Legislative Exchange Council", EIN 52-0140979,
#   Arlington, VA, "Designated as a 501(c)(3)", tax exemption issued March 1977, Form 990 filed
#   through FY2024: https://projects.propublica.org/nonprofits/organizations/520140979
# - kind and members: https://alec.org/about/ (a "voluntary membership organization of state
#   legislators" whose members are "nearly a quarter" of them, with private-sector members) and
#   https://alec.org/membership/ (legislative, private-sector, local-official and alumni
#   memberships).
# - terms: https://alec.org/terms-of-use/ (not linked from the site's footer, which links only
#   its privacy policy, https://alec.org/privacy-policy/, silent on reuse); robots.txt
#   (https://alec.org/robots.txt) disallows nothing for any user agent.
SOURCE = Source(
    key="alec",
    organization="American Legislative Exchange Council (ALEC)",
    descriptor="nonprofit membership organization of state legislators and private-sector "
    "members; exempt under section 501(c)(3) since 1977 (EIN 52-0140979)",
    members="State legislators, nearly a quarter of the country's by the organization's count, "
    "and private-sector members",
    members_source="https://alec.org/about/",
    library_url=LIBRARY,
    terms_url=f"{SITE}/terms-of-use/",
    terms_note="Using the site counts as agreeing to the terms, with no click-through. They "
    "forbid modifying, copying, reproducing or distributing any material from the site and give "
    "no exception; they say nothing of automated access. robots.txt disallows nothing.",
    terms_read_on=date(2026, 11, 15),
)

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MEDIA = {".docx": DOCX, ".pdf": "application/pdf"}
_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
_POLICY_SITEMAP = re.compile(r"/model-policy-sitemap\d*\.xml$")
_POLICY_PAGE = re.compile(r"^https://alec\.org/model-policy/(?!page/)([^/?#]+)/?$")
_FIELDS = "id,slug,link,title,content"

# A numbered section's heading, "Section 1. {Title}" or "SECTION 2 Definitions".
_SECTION = re.compile(r"^\s*\[?(?:section|sec\.)\s+\d+[a-z]?\b", re.IGNORECASE)
# Where a text without numbered sections begins: its enacting or resolving clause.
_OPENING = re.compile(
    r"^\s*(?:be\s+it\s+(?:enacted|resolved)|whereas\b|now,?\s+therefore|resolved\b)", re.I
)
# The board's and task forces' approval lines that close a policy: "Approved by ALEC Board of
# Directors in 1995", "Reapproved by ALEC Board of Directors on January 28, 2013".
_APPROVAL = re.compile(
    r"^\s*\(?\s*(?:final\s+version\s+)?(?:re-?)?(?:approved|adopted|amended|affirmed|reaffirmed)"
    r"\b.{0,200}\b(?:ALEC|American Legislative Exchange Council|Board\s+of\s+Directors|Task"
    r"\s+Force)\b.{0,200}$",
    re.IGNORECASE,
)
# A drafting, legislative or editor's note set as its own paragraph.
_NOTE_LINE = re.compile(
    r"^\s*[\[{(]?\s*(?:(?:drafting|legislative|editor'?s?)\s+notes?\b|notes?\s*:)", re.I
)
# Labels the page sets before the body: "Model Policy", "Model Resolution", "Summary".
_LABEL = re.compile(r"^\s*(?:model\s+(?:policy|resolution)|summary)\s*:?\s*$", re.IGNORECASE)
# The commission's instruction words (lineage.modeltext), in braces as ALEC also writes them,
# and "reference", which ALEC uses: "[reference any state compulsory insurance law]". Braced
# captions such as "{Title}" or "{Definitions}" are enacted headings and stay.
_INSTRUCTION = re.compile(
    r"[ \t]*\{\s*(?:(?:cite|insert|specify|identify|describe|enter|reference)\b|name\s+of\b)"
    r"[^{}]*\}"
    r"|[ \t]*\[\s*reference\b[^\[\]]*\]",
    re.IGNORECASE,
)
_INLINE_NOTE = re.compile(
    r"[ \t]*\{\s*(?:drafting\s+note|legislative\s+note|note|comment)s?\s*:[^{}]*\}", re.I
)
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y")


def clean_policy(text: str) -> str:
    """A policy's text as matching reads it: from its first numbered section, or, without one,
    its enacting or resolving clause, to its end, without labels, notes, approval lines or
    instructions to the drafter. Enacted brackets and braced captions stay as written."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [
        line
        for line in lines
        if line and not (_LABEL.match(line) or _NOTE_LINE.match(line) or _APPROVAL.match(line))
    ]
    start = next((i for i, line in enumerate(lines) if _SECTION.match(line)), None)
    if start is None:
        start = next((i for i, line in enumerate(lines) if _OPENING.match(line)), None)
    if start is None:
        raise ModelTextError("no numbered section, enacting clause or resolving clause")
    body = _INLINE_NOTE.sub("", "\n".join(lines[start:]))
    while (dropped := _INSTRUCTION.sub("", drop_instructions(body))) != body:
        body = dropped
    kept = [line.strip() for line in body.split("\n")]
    return "\n".join(line for line in kept if line) + "\n"


@dataclass(frozen=True, slots=True)
class PolicyPage:
    """What a policy's page says about it."""

    kind: str | None  # "Type:", e.g. "Model Policy"
    status: str | None  # "Draft" or "Final"
    published: date | None  # the latest of "Date Amended:" and "Date Finalized:"

    @property
    def wanted(self) -> bool:
        return (self.kind or "").lower() in TYPES and (self.status or "").lower() in STATUSES


def _label(lines: list[str], label: str) -> str | None:
    """The value after `label:`, on its line or the next."""
    pattern = re.compile(rf"^{re.escape(label)}\s*:\s*(.*)$", re.IGNORECASE)
    for i, line in enumerate(lines):
        if match := pattern.match(line):
            value = match.group(1).strip()
            if value:
                return value
            return lines[i + 1].strip() if i + 1 < len(lines) else None
    return None


def _date(value: str | None) -> date | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime((value or "").strip(), fmt).date()
        except ValueError:
            continue
    return None


def policy_page(page: Fetched) -> PolicyPage:
    lines = html_text(page.body, None).split("\n")
    # The page's "Policy Status" block holds the fields; read from it where it is found.
    lines = lines[next((i for i, x in enumerate(lines) if x.lower() == "policy status"), 0) :]
    dates = [d for d in (_date(_label(lines, f"Date {k}")) for k in ("Amended", "Finalized")) if d]
    return PolicyPage(
        kind=_label(lines, "Type"),
        status=_label(lines, "Status"),
        published=max(dates, default=None),
    )


def _locs(sitemap: Fetched, tag: str) -> list[str]:
    """The <loc> of each <url> or <sitemap> in a sitemap."""
    try:
        root = ElementTree.fromstring(sitemap.body)
    except ElementTree.ParseError as e:
        raise CollectFetchError(f"unreadable sitemap: {e}") from e
    return [
        (loc.text or "").strip()
        for entry in root.iter(f"{_SITEMAP_NS}{tag}")
        if (loc := entry.find(f"{_SITEMAP_NS}loc")) is not None
    ]


def policy_sitemaps(index: Fetched) -> list[str]:
    return [loc for loc in _locs(index, "sitemap") if _POLICY_SITEMAP.search(loc)]


def policy_pages(sitemap: Fetched) -> list[str]:
    """The policy pages a sitemap lists; the library's own page and its pages of cards are not
    policies."""
    return [loc for loc in _locs(sitemap, "url") if _POLICY_PAGE.match(loc)]


def api_url(slug: str) -> str:
    query = urllib.parse.urlencode({"slug": slug, "_fields": _FIELDS})
    return f"{SITE}/wp-json/wp/v2/model-policy?{query}"


@dataclass(frozen=True, slots=True)
class ApiPolicy:
    title: str
    content: str  # the body's HTML


def api_policy(response: Fetched, slug: str) -> ApiPolicy:
    try:
        items = json.loads(response.body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise CollectFetchError(f"unreadable API response: {e}") from e
    if not isinstance(items, list):
        raise CollectFetchError("the API did not answer with a list of policies")
    for item in cast(list[object], items):
        if isinstance(item, dict):
            policy = cast(dict[str, object], item)
            if policy.get("slug") == slug:
                return ApiPolicy(
                    title=_rendered(policy.get("title")), content=_rendered(policy.get("content"))
                )
    raise CollectFetchError(f"the API lists no policy {slug}")


def _rendered(field: object) -> str:
    if isinstance(field, dict):
        value = cast(dict[str, object], field).get("rendered")
        if isinstance(value, str):
            return value
    return ""


class _Attachments(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.files: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href") if tag == "a" else None
        if href and PurePosixPath(urllib.parse.urlsplit(href).path).suffix.lower() in _MEDIA:
            self.files.append(urllib.parse.urljoin(LIBRARY, href))


def attachment(content: str) -> str | None:
    """The Word or PDF file a body links to, as an amendment's does."""
    parser = _Attachments()
    parser.feed(content)
    parser.close()
    return parser.files[0] if parser.files else None


def _title(rendered: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", rendered)).split())


def _has_law(text: str) -> bool:
    return any(_SECTION.match(line) or _OPENING.match(line) for line in text.split("\n"))


def _fetch_step(c: Collection, page_url: str, url: str) -> Fetched | None:
    """One request on the way to a policy's text; a refusal or failure settles the policy."""
    if c.permission(url) == "blocked":
        c.block(url)
        return None
    try:
        return c.get(url)
    except CollectFetchError as e:
        c.fail(page_url, f"{url}: {e}")
        return None


def _collect_policy(
    c: Collection, url: str, *, types: frozenset[str], statuses: frozenset[str]
) -> None:
    if c.permission(url) == "blocked":
        c.block(url)  # its type is unknown, so it is counted as found
        return
    try:
        page = policy_page(c.get(url))
    except (CollectFetchError, ExtractError) as e:
        c.fail(url, str(e))
        return
    if page.kind is None or page.status is None:
        c.fail(url, "no Type or Status on the policy's page")
        return
    if page.kind.lower() not in types or page.status.lower() not in statuses:
        return
    if (match := _POLICY_PAGE.match(url)) is None:
        c.fail(url, "not a policy page")
        return
    slug = urllib.parse.unquote(match.group(1))
    api = api_url(slug)
    if (robots := c.check(api)) is None:
        return
    if (response := _fetch_step(c, url, api)) is None:
        return
    try:
        policy = api_policy(response, slug)
        raw, media_type = response.body, "application/json"
        body = html_text(policy.content.encode(), "utf-8") if policy.content.strip() else ""
        if not _has_law(body) and (file_url := attachment(policy.content)) is not None:
            if (robots := c.check(file_url)) is None:
                return
            if (got := _fetch_step(c, url, file_url)) is None:
                return
            raw = got.body
            media_type = _MEDIA[PurePosixPath(urllib.parse.urlsplit(file_url).path).suffix.lower()]
            body = docx_text(raw) if media_type == DOCX else pdf_text(raw)
        text = clean_policy(body)
    except (CollectFetchError, ExtractError, ModelTextError) as e:
        c.fail(url, str(e))
        return
    doc = provenance(
        SOURCE,
        slug=slugify(slug),
        title=_title(policy.title) or slug,
        url=url,
        raw=raw,
        media_type=media_type,
        text=text,
        robots=robots,
        collected_on=c.context.on,
        published=page.published,
    )
    c.keep(doc, raw=raw, text=text)


def _collect_site(c: Collection, types: frozenset[str], statuses: frozenset[str]) -> None:
    if (index := _fetch_step(c, SITEMAP_INDEX, SITEMAP_INDEX)) is None:
        return
    try:
        sitemaps = policy_sitemaps(index)
    except CollectFetchError as e:
        c.fail(SITEMAP_INDEX, str(e))
        return
    if not sitemaps:
        c.fail(SITEMAP_INDEX, "no model policy sitemap in the sitemap index")
        return
    urls: set[str] = set()
    for sitemap in sitemaps:
        if (got := _fetch_step(c, sitemap, sitemap)) is None:
            continue
        try:
            urls |= {u if u.endswith("/") else u + "/" for u in policy_pages(got)}
        except CollectFetchError as e:
            c.fail(sitemap, str(e))
    for url in sorted(urls):
        _collect_policy(c, url, types=types, statuses=statuses)


def collect_from(
    context: CollectContext,
    *,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    site: bool = SITE_FETCH,
    types: frozenset[str] = TYPES,
    statuses: frozenset[str] = STATUSES,
) -> CollectReport:
    """With `site`, every policy the sitemaps list whose page gives one of `types` and
    `statuses`; without it, nothing is asked and the library is reported as not read."""
    c = Collection(SOURCE, context, fetch=fetch, robots_reader=robots_reader)
    if site:
        _collect_site(c, types, statuses)
    else:
        c.fail(LIBRARY, "fetching from the site is off until its terms are ruled on for this use")
    return c.report()


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network.
    return collect_from(context, fetch=fetch_url, robots_reader=read_robots, site=SITE_FETCH)
