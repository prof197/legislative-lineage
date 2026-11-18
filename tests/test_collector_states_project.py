"""The States Project's collector (DESIGN.md#model-documents): policy pages found through the
sitemap and the library, each model bill read from its page and stored with provenance, against
recorded responses (tests/fixtures/collect/states-project/), never the network. Every page there
is synthetic; no text from the site is committed.

The real-sample tests read the collected texts from the local data directory (LINEAGE_DATA) and
are skipped until integration has collected them; they assert counts only, never print text."""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.collect import CollectContext, CollectReport, collectors
from lineage.collect_base import CollectFetchError, Fetched
from lineage.collectors import states_project
from lineage.collectors.states_project import (
    LIBRARY,
    SITEMAP_INDEX,
    SOURCE,
    clean_policy,
    collect_from,
    model_policy,
    policy_url,
)
from lineage.core.robots import Robots, parse_robots
from lineage.modeldocs import (
    MODELDOCS_DIR,
    ModelDoc,
    ModelDocError,
    load_text,
    raw_path,
    read_docs,
    read_versions,
    sha256,
)
from lineage.sections import split_sections

FIXTURES = Path(__file__).parent / "fixtures" / "collect" / "states-project"
DAY = date(2026, 11, 15)
SITE = "https://statesproject.org"
LANTERN = f"{SITE}/policy/lantern-registry/"
HARBOR = f"{SITE}/policy/harbor-lights/"
PEWTER = f"{SITE}/policy/pewter-lamp/"
KETTLE = f"{SITE}/policy/kettle-tools/"
WITHHELD = f"{SITE}/policy/withheld-wick/"
SNUFFED = f"{SITE}/policy/snuffed-candle/"

LANTERN_TEXT = (
    "Section 1. Title\n"
    "This act shall be known as the Synthetic Lantern Registry Act.\n"
    "Section 2. Purpose\n"
    "This act keeps an invented list of lanterns current.\n"
    "Section 3. Provisions\n"
    "a) (Definitions)\n"
    "i) “Lantern” means a portable lamp kept under.\n"
    "ii) “Registrar” means the [Secretary of State].\n"
    "b) (Registration)\n"
    "i) A keeper shall register each lantern within [X DAYS] of acquiring it.\n"
    "ii) The registrar shall publish the list each year.\n"
    "c) The provisions of this act are severable.\n"
)
HARBOR_TEXT = (
    "Section 1. Title\n"
    "This act shall be known as the Synthetic Harbor Lights Act.\n"
    "Section 3. Approaches\n"
    "i) The [STATE] harbor master shall review each beacon every [X MONTHS].\n"
    "ii) A beacon found dark shall be relit within days.\n"
    "The harbor master shall train each keeper before the season opens.\n"
    "The harbor master shall keep a record of each lamp.\n"
    "A record shall be kept for [X YEARS].\n"
)


@dataclass
class RecordedSite:
    """The site as the fixture manifest records it; every request is logged."""

    robots_body: bytes = (FIXTURES / "robots.txt").read_bytes()
    responses: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            url: entry
            for url, entry in json.loads((FIXTURES / "site.json").read_text()).items()
            if url != "_note"
        }
    )
    overrides: dict[str, bytes] = field(default_factory=dict[str, bytes])
    requests: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    robots_reads: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def fetch(self, url: str, interval: float) -> Fetched:
        self.requests.append((url, interval))
        if url in self.overrides:
            return Fetched(url, "text/html; charset=UTF-8", self.overrides[url])
        entry = self.responses.get(url)
        if entry is None:
            raise CollectFetchError("HTTP 404")
        return Fetched(url, entry["content_type"], (FIXTURES / entry["file"]).read_bytes())

    def read_robots(self, scheme: str, host: str) -> Robots:
        self.robots_reads.append((scheme, host))
        url = f"{scheme}://{host}/robots.txt"
        return parse_robots(url, 200, self.robots_body, "2026-11-15T00:00:00+00:00")


def _run(
    tmp_path: Path, site: RecordedSite, *, on: date = DAY, on_site: bool = True
) -> CollectReport:
    return collect_from(
        CollectContext(root=tmp_path / "modeldocs", on=on),
        fetch=site.fetch,
        robots_reader=site.read_robots,
        site=on_site,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _page(name: str, url: str = LANTERN) -> Fetched:
    return Fetched(url, "text/html", (FIXTURES / name).read_bytes())


def test_every_policy_found_is_collected_blocked_or_failed_with_provenance(
    tmp_path: Path,
) -> None:
    site = RecordedSite()
    report = _run(tmp_path, site)

    # Five from the sitemap, and one only the library lists.
    assert (report.found, report.collected, report.unchanged) == (6, 3, 0)
    assert report.blocked == [WITHHELD]
    assert {f.url: f.reason for f in report.failed} == {
        KETTLE: "no model policy text on the page",
        SNUFFED: "HTTP 404",
    }
    asked = [url for url, _ in site.requests]
    # robots.txt first: the page it disallows is never asked for.
    assert WITHHELD not in asked
    assert asked[:3] == [SITEMAP_INDEX, f"{SITE}/policy-sitemap.xml", LIBRARY]
    # The index's other sitemaps are not policy pages' and are never read.
    assert not any("page-sitemap" in url or "policy_category" in url for url in asked)
    assert all(interval >= 1.0 for _, interval in site.requests)
    assert site.robots_reads == [("https", "statesproject.org")]

    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}
    assert sorted(docs) == [
        "states-project/harbor-lights",
        "states-project/lantern-registry",
        "states-project/pewter-lamp",
    ]
    doc = docs["states-project/lantern-registry"]
    raw = (FIXTURES / "lantern-registry.html").read_bytes()
    assert (doc.title, doc.url) == ("Lantern Registry Act", LANTERN)
    assert (doc.robots, doc.leaked, doc.media_type) == ("allowed", False, "text/html")
    # The pages give no date of their own.
    assert (doc.published, doc.collected_on) == (None, DAY)
    assert doc.raw_sha256 == sha256(raw)
    assert raw_path(doc, root).read_bytes() == raw
    assert load_text(doc, root) == LANTERN_TEXT
    assert load_text(docs["states-project/harbor-lights"], root) == HARBOR_TEXT
    # A page without an act title line is named by its heading; the library's link to it
    # carried a query and a fragment, which the canonical URL drops.
    pewter = docs["states-project/pewter-lamp"]
    assert (pewter.title, pewter.url) == ("Polish the Pewter Lamps", PEWTER)


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, RecordedSite())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, RecordedSite(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert (again.collected, again.unchanged) == (0, first.collected)
    assert (again.blocked, again.failed) == (first.blocked, first.failed)


def test_a_revised_page_becomes_a_new_version(tmp_path: Path) -> None:
    _run(tmp_path, RecordedSite())
    site = RecordedSite()
    revised = (FIXTURES / "lantern-registry.html").read_text().replace("each year", "each month")
    site.overrides[LANTERN] = revised.encode()
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (1, 2)
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "states-project/lantern-registry"
    ]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [
        (DAY, date(2026, 12, 1)),
        (date(2026, 12, 1), None),
    ]


def test_a_page_whose_markup_changed_but_not_its_text_is_unchanged(tmp_path: Path) -> None:
    _run(tmp_path, RecordedSite())
    site = RecordedSite()
    page = (FIXTURES / "lantern-registry.html").read_text()
    site.overrides[LANTERN] = page.replace("<main>", '<main class="rebuilt">').encode()
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (0, 3)


def test_a_failed_index_is_reported_and_the_library_still_read(tmp_path: Path) -> None:
    site = RecordedSite()
    del site.responses[SITEMAP_INDEX]
    report = _run(tmp_path, site)

    assert [(f.url, f.reason) for f in report.failed] == [(SITEMAP_INDEX, "HTTP 404")]
    assert (report.found, report.collected) == (3, 2)


def test_an_index_without_a_policy_sitemap_is_reported(tmp_path: Path) -> None:
    site = RecordedSite()
    index = (FIXTURES / "sitemap_index.xml").read_text()
    site.overrides[SITEMAP_INDEX] = index.replace(
        "/policy-sitemap.xml", "/post-sitemap.xml"
    ).encode()
    report = _run(tmp_path, site)

    assert (SITEMAP_INDEX, "no policy sitemap in the sitemap index") in [
        (f.url, f.reason) for f in report.failed
    ]
    assert report.collected == 2


def test_a_robots_txt_that_disallows_the_site_blocks_every_index(tmp_path: Path) -> None:
    site = RecordedSite(robots_body=b"User-agent: *\nDisallow: /\n")
    report = _run(tmp_path, site)

    assert report.blocked == sorted([SITEMAP_INDEX, LIBRARY])
    assert (report.collected, report.failed, site.requests) == (0, [], [])


def test_with_the_site_off_nothing_is_asked(tmp_path: Path) -> None:
    site = RecordedSite()
    report = _run(tmp_path, site, on_site=False)

    assert report.found == 0
    assert (site.requests, site.robots_reads) == ([], [])


def test_cleaning_keeps_the_sections_and_drops_the_pages_asides() -> None:
    lantern = model_policy(_page("lantern-registry.html"))
    harbor = model_policy(_page("harbor-lights.html", HARBOR))
    assert lantern is not None
    assert harbor is not None
    assert lantern.text == LANTERN_TEXT
    # The summary section, each approach's description, the language labels, the examples
    # line and the instruction to the drafter go; enacted placeholders stay as written.
    assert harbor.text == HARBOR_TEXT
    for gone in (
        "summary",
        "Beacon Review",
        "Lamp Records",
        "Sample language",
        "Examples",
        "specify",
    ):
        assert gone not in harbor.text
    # Only the model bill is read: nothing before its heading, after "Related Policies", or
    # in a script that names both.
    for outside in ("FAQ", "Stakeholders", "VIEW POLICY", "Copyright", "MODEL POLICY"):
        assert outside not in lantern.text
    assert model_policy(_page("kettle-tools.html", KETTLE)) is None


def test_the_cleaned_headings_are_where_sections_split() -> None:
    split = split_sections(LANTERN_TEXT)
    assert split.by_heading
    assert len(split.sections) == 3


@pytest.mark.parametrize(
    ("lines", "cleaned"),
    [
        (["Section 1 (Title):", "This act shall be known as the Act."],
         "Section 1. Title\nThis act shall be known as the Act."),
        (["Section 1 (Title): This act shall be known as the Act."],
         "Section 1. Title\nThis act shall be known as the Act."),
        (["Section 2 (Summary): A summary.", "More summary.", "Section 3 (Provisions):", "a) Law."],
         "Section 3. Provisions\na) Law."),
        (["1) (Label) A description.", "Sample language:", "i) Law."], "i) Law."),
        (["a) Model language: Law."], "Law."),
        (["Sample language: Sample Language: Law."], "Law."),
        (["Example includes: a link"], ""),
        (["i) Filed with [insert office] by [DATE]."], "i) Filed with by [DATE]."),
    ],
)  # fmt: skip
def test_clean_policy(lines: list[str], cleaned: str) -> None:
    assert clean_policy(lines) == cleaned


@pytest.mark.parametrize(
    ("href", "url"),
    [
        ("/policy/right-to-lamps/", f"{SITE}/policy/right-to-lamps/"),
        ("https://www.statesproject.org/policy/right-to-lamps", f"{SITE}/policy/right-to-lamps/"),
        ("/policy/right-to-lamps/?ref=card#policy", f"{SITE}/policy/right-to-lamps/"),
        ("/policy_category/lamps/", None),
        ("/policy/", None),
        ("https://example.org/policy/right-to-lamps/", None),
        ("/policy/right-to-lamps/feed/", None),
    ],
)
def test_policy_url(href: str, url: str | None) -> None:
    assert policy_url(href) == url


def test_the_source_entry() -> None:
    assert collectors()["states-project"].source == SOURCE
    assert not SOURCE.leaked
    # Not a membership organization: no members line.
    assert (SOURCE.members, SOURCE.members_source) == (None, None)
    assert SOURCE.library_url == LIBRARY
    assert SOURCE.terms_url == f"{SITE}/policywork/terms-of-use/"
    assert SOURCE.terms_read_on == DAY
    assert "501(c)(4)" in SOURCE.descriptor


def test_the_command_collects_through_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = RecordedSite()
    monkeypatch.setattr(states_project, "fetch_url", site.fetch)
    monkeypatch.setattr(states_project, "read_robots", site.read_robots)
    monkeypatch.setattr(states_project, "SITE_FETCH", True)
    args = ["collect-models", "states-project", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "6 found: 3 collected, 0 unchanged, 1 blocked by robots.txt, 2 failed" in first.output
    report = json.loads((tmp_path / "reports" / "states-project.json").read_text())
    assert str(tmp_path) not in json.dumps(report)
    again = CliRunner().invoke(app, args)
    assert "6 found: 0 collected, 3 unchanged" in again.output


# Real samples: the pages integration collected, under the data directory, cleaned as this
# collector cleans them now, so a fix to the cleaning is tested before the next collection.
def _cleaned(doc: ModelDoc) -> str:
    page = Fetched(doc.url, doc.media_type, raw_path(doc, MODELDOCS_DIR).read_bytes())
    policy = model_policy(page)
    return "" if policy is None else policy.text


def _collected() -> list[tuple[str, str]]:
    try:
        return [(d.id, _cleaned(d)) for d in read_docs(SOURCE, MODELDOCS_DIR)]
    except (OSError, ValueError, ModelDocError):
        return []


_REAL = _collected()
_ASIDE = re.compile(
    r"^(?:\(?[a-z0-9]{1,4}[.)]\s*)?(?:sample|suggested|model)\s+language\s*:"
    r"|^(?:\(?[a-z0-9]{1,4}[.)]\s*)?examples?\s+include"
    r"|^model\s+policy$|^related\s+policies$|\[\s*(?:insert|specify|cite)\b",
    re.IGNORECASE | re.MULTILINE,
)


@pytest.mark.skipif(not _REAL, reason="no states-project texts under the data directory")
def test_real_texts_keep_their_sections_and_none_of_the_asides() -> None:
    unsplit = [doc_id for doc_id, text in _REAL if not split_sections(text).by_heading]
    with_asides = [doc_id for doc_id, text in _REAL if _ASIDE.search(text)]
    assert (unsplit, with_asides) == ([], [])


@pytest.mark.skipif(not _REAL, reason="no states-project texts under the data directory")
def test_real_texts_are_bills_not_pages() -> None:
    # Each opens at its first section and is a bill's length, not a page's: a page's navigation
    # or footer reaching the text would show as a word count far past any model bill's.
    words = [len(text.split()) for _, text in _REAL]
    assert all(text.lower().startswith("section 1") for _, text in _REAL)
    assert min(words) >= 20
    assert max(words) < 20_000
