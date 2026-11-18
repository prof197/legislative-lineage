"""The States Project's policy library: model state bills set out as text on its policy pages.

The library (statesproject.org/policywork/policy-library/) shows a selection of its policies;
the site's sitemap lists every policy page, so both are read and their policy pages joined.
Each policy page sets out its model bill inline, as HTML, under a "MODEL POLICY" heading and
before "Related Policies": an act title, then "Section 1 (Title):", a purpose or summary
section, and the provisions. Some pages give the provisions as numbered approaches, each a
sentence on what it does, its "Sample language:" and an "Examples include:" line of links to
state bills. The pages give no date of their own, so no document is dated.

Matching reads the sample language, not the page's asides: a summary section, an approach's
description, the "Sample language:" label and the examples line are dropped, as are bracketed
instructions to the drafter (`lineage.modeltext.drop_instructions`). A heading such as
"Section 1 (Title):" is written "Section 1. Title" so that `lineage.sections` splits at it.

Fetching is on (`SITE_FETCH`): read on 2026-11-15, the policy pages' terms say nothing on
automated reading, bind a visitor by use alone, with nothing to accept first, and robots.txt
disallows no path. See DESIGN.md#model-documents.
"""

import re
import urllib.parse
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
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
)
from lineage.modeldocs import Source
from lineage.modeltext import drop_instructions

SITE = "https://statesproject.org"
LIBRARY = f"{SITE}/policywork/policy-library/"
SITEMAP_INDEX = f"{SITE}/sitemap_index.xml"
TERMS = f"{SITE}/policywork/terms-of-use/"
# On: the terms are silent on automated reading and ask nothing first (module docstring).
SITE_FETCH = True

# The descriptor's sources, read 2026-11-15:
# - kind: the policy pages' footer says The States Project's policy work is conducted by
#   Future Now Action, a 501(c)(4) issue advocacy organization
#   (https://statesproject.org/policywork/); https://statesproject.org/about-us/ calls The
#   States Project a joint initiative of Future Now Action and a political action committee.
# - IRS: Future Now Action, EIN 82-2390410, Washington, DC, 501(c)(4), exempt since October
#   2020, Form 990 filed for 2018-2025:
#   https://projects.propublica.org/nonprofits/organizations/822390410
SOURCE = Source(
    key="states-project",
    organization="The States Project (policy work conducted by Future Now Action)",
    # IRS: ProPublica's Nonprofit Explorer, EIN 82-2390410; type: the policy pages' footer.
    descriptor="social welfare organization, 501(c)(4): Future Now Action (EIN 82-2390410), "
    "which calls itself an issue advocacy organization and conducts The States Project's "
    "policy work",
    library_url=LIBRARY,
    terms_url=TERMS,
    terms_note="Using the site binds a visitor to its terms, with nothing to accept first. They "
    "forbid posting unlawful or infringing content and distributing unauthorized copies of "
    "copyrighted material through the site, grant no license to reuse its content, and say "
    "nothing about automated reading. Last updated 2023-07-10. robots.txt disallows no path.",
    terms_read_on=date(2026, 11, 15),
)

MEDIA_TYPE = "text/html"
_POLICY_PAGE = re.compile(
    r"^https?://(?:www\.)?statesproject\.org/policy/(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)/?$"
)
# Yoast names the sitemap of a post type "<type>-sitemap.xml", then "<type>-sitemap2.xml".
_POLICY_SITEMAP = re.compile(r"/policy-sitemap\d*\.xml$")
_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

_BLOCKS = frozenset(
    {"address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
     "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main",
     "nav", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul"}
)  # fmt: skip
_SKIPPED = frozenset({"head", "script", "style", "noscript", "template", "svg", "iframe"})
_VOID = frozenset({"br", "hr", "img", "input", "meta", "link", "source", "wbr"})

_START = re.compile(r"^model\s+policy:?$", re.IGNORECASE)
_END = re.compile(r"^related\s+polic(?:y|ies):?$", re.IGNORECASE)
_SECTION = re.compile(r"^section\s+\d+\b", re.IGNORECASE)
# "Section 1 (Title):", "Section 3 (Provisions):" -> "Section 1. Title"
_LABELED_SECTION = re.compile(r"^(Section\s+\d+)\s*\(\s*([^()]+?)\s*\)\s*:?\s*", re.IGNORECASE)
_SUMMARY = re.compile(r"^section\s+\d+\s*\(\s*summary\s*\)", re.IGNORECASE)
_MARKER = r"(?:\(?[a-z0-9]{1,4}[.)]\s*)?"
# One or more: a page can give the label twice over, "Sample language: Sample language: ...".
_LANGUAGE = re.compile(
    rf"^(?:{_MARKER}(?:sample|suggested|model)\s+language\s*:\s*)+", re.IGNORECASE
)
_EXAMPLES = re.compile(rf"^{_MARKER}examples?\s+(?:include|includes|of)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Policy:
    title: str
    text: str  # cleaned, as matching reads it


class _Lines(HTMLParser):
    """A page's text, one line per block element; script, style and the head are not text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.h1: list[str] = []
        self.links: list[str] = []
        self._buffer: list[str] = []
        self._skip = 0
        self._in_h1 = False

    def _flush(self) -> None:
        line = " ".join("".join(self._buffer).split())
        self._buffer = []
        if line:
            self.lines.append(line)
            if self._in_h1:
                self.h1.append(line)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "a" and (href := dict(attrs).get("href")):
            self.links.append(href)
        if tag in _BLOCKS:
            self._flush()
        if tag == "h1":
            self._in_h1 = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._skip and tag in _BLOCKS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip or tag in _VOID:
            return
        if tag in _BLOCKS:
            self._flush()
        if tag == "h1":
            self._in_h1 = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def _parse(page: Fetched) -> _Lines:
    parser = _Lines()
    parser.feed(page.body.decode("utf-8", errors="replace"))
    parser.close()
    return parser


def policy_url(href: str, base: str = SITE) -> str | None:
    """A policy page's canonical URL, or None for any other link."""
    absolute = urllib.parse.urljoin(base, href.strip())
    parts = urllib.parse.urlsplit(absolute)
    match = _POLICY_PAGE.match(urllib.parse.urlunsplit(parts._replace(query="", fragment="")))
    return f"{SITE}/policy/{match['slug']}/" if match else None


def _locs(sitemap: Fetched) -> list[str]:
    try:
        root = ElementTree.fromstring(sitemap.body)
    except ElementTree.ParseError as e:
        raise CollectFetchError(f"unreadable sitemap: {e}") from e
    return [(loc.text or "").strip() for loc in root.iter(f"{_SITEMAP_NS}loc")]


def policy_sitemaps(index: Fetched) -> list[str]:
    """The sitemaps of the policy post type that the sitemap index lists."""
    return sorted(
        {loc for loc in _locs(index) if _POLICY_SITEMAP.search(urllib.parse.urlsplit(loc).path)}
    )


def sitemap_policies(sitemap: Fetched) -> set[str]:
    return {url for loc in _locs(sitemap) if (url := policy_url(loc)) is not None}


def library_policies(page: Fetched) -> set[str]:
    """The policy pages the library page links to."""
    return {url for href in _parse(page).links if (url := policy_url(href, page.url)) is not None}


def _heading(line: str) -> str:
    return _LABELED_SECTION.sub(lambda m: f"{m[1]}. {m[2].rstrip(':').strip()}\n", line).rstrip()


def clean_policy(lines: list[str]) -> str:
    """The model bill's lines as matching reads them: the sample language, without the page's
    summary, approach descriptions, labels, examples or instructions to the drafter."""
    kept: list[str] = []
    in_summary = False
    for i, line in enumerate(lines):
        if _SECTION.match(line):
            in_summary = _SUMMARY.match(line) is not None
        if in_summary or _EXAMPLES.match(line):
            continue
        following = lines[i + 1] if i + 1 < len(lines) else ""
        if _LANGUAGE.match(following) and not _SECTION.match(line):
            continue  # an approach's description, which its sample language follows
        if (rest := _LANGUAGE.sub("", line, count=1)) != line:
            if rest:
                kept.append(rest)
            continue
        kept.append(_heading(line))
    text = drop_instructions("\n".join(kept))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def model_policy(page: Fetched) -> Policy | None:
    """The model bill a policy page sets out, or None when it sets out none."""
    parsed = _parse(page)
    lines = parsed.lines
    starts = [i for i, line in enumerate(lines) if _START.match(line)]
    if not starts:
        return None
    end = next((i for i, line in enumerate(lines) if i > starts[0] and _END.match(line)), None)
    if end is None:
        return None
    # A heading repeated in a layout's copy for another screen width: the last before the end.
    start = max(i for i in starts if i < end)
    body = lines[start + 1 : end]
    if not body:
        return None
    if _SECTION.match(body[0]):
        title = parsed.h1[0] if parsed.h1 else ""
    else:
        title, body = body[0].rstrip("."), body[1:]
    text = clean_policy(body)
    if not title or not text:
        return None
    return Policy(title, text + "\n")


def _fetch_index(c: Collection, url: str) -> Fetched | None:
    """A page that lists documents; a refusal or failure is reported against it."""
    if c.check(url) is None:
        return None
    try:
        return c.get(url)
    except CollectFetchError as e:
        c.fail(url, str(e))
        return None


def find_policies(c: Collection) -> list[str]:
    """Every policy page the sitemap or the library lists."""
    found: set[str] = set()
    sitemaps: list[str] = []
    if (index := _fetch_index(c, SITEMAP_INDEX)) is not None:
        try:
            sitemaps = policy_sitemaps(index)
        except CollectFetchError as e:
            c.fail(SITEMAP_INDEX, str(e))
        else:
            if not sitemaps:
                c.fail(SITEMAP_INDEX, "no policy sitemap in the sitemap index")
        for url in sitemaps:
            if (sitemap := _fetch_index(c, url)) is not None:
                try:
                    found |= sitemap_policies(sitemap)
                except CollectFetchError as e:
                    c.fail(url, str(e))
    if (library := _fetch_index(c, LIBRARY)) is not None:
        found |= library_policies(library)
    return sorted(found)


def _collect_policy(c: Collection, url: str) -> None:
    if (robots := c.check(url)) is None:
        return
    try:
        page = c.get(url)
    except CollectFetchError as e:
        c.fail(url, str(e))
        return
    if (policy := model_policy(page)) is None:
        c.fail(url, "no model policy text on the page")
        return
    match = _POLICY_PAGE.match(url)
    assert match is not None  # find_policies keeps policy pages only
    doc = provenance(
        SOURCE,
        slug=match["slug"],
        title=policy.title,
        url=url,
        raw=page.body,
        media_type=MEDIA_TYPE,
        text=policy.text,
        robots=robots,
        collected_on=c.context.on,
    )
    c.keep(doc, raw=page.body, text=policy.text)


def collect_from(
    context: CollectContext,
    *,
    fetch: Fetch = fetch_url,
    robots_reader: ReadRobots = read_robots,
    site: bool = SITE_FETCH,
) -> CollectReport:
    """Every policy page's model bill; with `site` off, nothing is asked and nothing found."""
    c = Collection(SOURCE, context, fetch=fetch, robots_reader=robots_reader)
    if site:
        for url in find_policies(c):
            _collect_policy(c, url)
    return c.report()


def collect(context: CollectContext) -> CollectReport:
    # Module names are looked up per call, so a test can stand in for the network.
    return collect_from(context, fetch=fetch_url, robots_reader=read_robots, site=SITE_FETCH)
