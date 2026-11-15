"""Public Citizen's collector (DESIGN.md#model-documents): its library page's models, each kept
with provenance, against recorded responses (tests/fixtures/collect/public-citizen/), never the
network. Every page there is synthetic; no Public Citizen text is committed.

The real-sample tests read the collected pages from the local data directory (LINEAGE_DATA) and
are skipped until integration has collected them; they assert counts and patterns only, never
print a model's text."""

import json
import re
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.collect import CollectContext, CollectReport, collectors
from lineage.collect_base import Fetched
from lineage.collectors import public_citizen
from lineage.collectors.public_citizen import (
    LIBRARY,
    MEDIA_TYPE,
    SOURCE,
    PageError,
    collect_from,
    doc_slug,
    read_page,
)
from lineage.modeldocs import (
    MODELDOCS_DIR,
    ModelDoc,
    load_text,
    raw_path,
    read_docs,
    read_versions,
    sha256,
)
from lineage.robots import Robots, parse_robots
from tests.recorded_site import RecordedSite

FIXTURES = Path(__file__).parent / "fixtures" / "collect" / "public-citizen"
DAY = date(2026, 11, 15)
ARTICLE = "https://www.citizen.org/article"
LABELING = f"{ARTICLE}/public-citizen-model-state-law-lantern-labeling/"
MOTHS = f"{ARTICLE}/model-state-law-protecting-moths-from-synthetic-lanterns/"
SMOKE = f"{ARTICLE}/public-citizen-model-state-law-regulating-synthetic-lantern-smoke/"
RIGHT_TO_KNOW = f"{ARTICLE}/public-citizen-model-state-law-a-keepers-right-to-know/"
WITHHELD = f"{ARTICLE}/withheld-model-state-law-synthetic/"
MISSING = f"{ARTICLE}/model-state-law-gone-missing/"
NO_TEXT = f"{ARTICLE}/model-state-law-without-text/"
ELSEWHERE = "https://docs.google.com/document/d/synthetic-model/edit"

LABELING_TEXT = (
    "SEC 1 \u2013 DEFINITIONS\n"
    "(a) “Lantern” means a portable lamp, as used in this section.\n"
    "(b) “Keeper” means a person who sells a lantern.\n"
    "SEC 2 \u2013 LABELING\n"
    "(a) A keeper [shall][may] label each lantern with its fuel.\n"
    "(b) The may enforce this section.\n"
)


def recorded() -> RecordedSite:
    responses = json.loads((FIXTURES / "site.json").read_text())
    responses.pop("_note")
    return RecordedSite(responses, (FIXTURES / "robots.txt").read_bytes())


def _run(
    tmp_path: Path, site: RecordedSite, *, on: date = DAY, fetching: bool = True
) -> CollectReport:
    return collect_from(
        CollectContext(root=tmp_path / "modeldocs", on=on),
        fetch=site.fetch,
        robots_reader=site.read_robots,
        site=fetching,
    )


def _docs(tmp_path: Path) -> dict[str, ModelDoc]:
    return {d.id: d for d in read_docs(SOURCE, tmp_path / "modeldocs")}


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_library_page_s_models_are_collected_with_provenance(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    assert (report.source, report.run_on) == ("public-citizen", DAY)
    assert (report.found, report.collected, report.unchanged) == (8, 4, 0)
    docs = _docs(tmp_path)
    assert sorted(docs) == [
        "public-citizen/a-keepers-right-to-know",
        "public-citizen/lantern-labeling",
        "public-citizen/protecting-moths-from-synthetic-lanterns",
        "public-citizen/regulating-synthetic-lantern-smoke",
    ]
    doc = docs["public-citizen/lantern-labeling"]
    page = site.responses[LABELING]["body"].encode()
    assert doc.title == "Synthetic Model State Law on Lantern Labeling"
    assert doc.url == LABELING
    assert doc.published == date(2025, 3, 3)
    assert (doc.robots, doc.leaked, doc.media_type, doc.collected_on) == (
        "allowed",
        False,
        MEDIA_TYPE,
        DAY,
    )
    # The page's bytes as fetched are kept; the text is the model's alone.
    assert doc.raw_sha256 == sha256(page)
    assert raw_path(doc, tmp_path / "modeldocs").read_bytes() == page
    assert load_text(doc, tmp_path / "modeldocs") == LABELING_TEXT


def test_each_model_is_dated_as_its_page_dates_it_or_not_at_all(tmp_path: Path) -> None:
    _run(tmp_path, recorded())
    published = {k.partition("/")[2]: d.published for k, d in _docs(tmp_path).items()}

    # Above the title, below it past the share buttons, or none: the related articles' dates
    # (2019, 2024) and the menu's (2026) are never the model's.
    assert published == {
        "lantern-labeling": date(2025, 3, 3),
        "protecting-moths-from-synthetic-lanterns": date(2025, 10, 9),
        "regulating-synthetic-lantern-smoke": None,
        "a-keepers-right-to-know": date(2024, 11, 18),
    }


def test_only_model_links_are_documents_and_robots_and_the_crawl_delay_are_honored(
    tmp_path: Path,
) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    asked = [url for url, _ in site.requests]
    assert asked[0] == LIBRARY
    # The library page links the first model twice; it is asked for once. Explainers and
    # trackers are not documents, and the disallowed page is never asked for.
    assert asked.count(LABELING) == 1
    assert not any("tracker" in url or "synthetic-info" in url for url in asked)
    assert WITHHELD not in asked
    assert ELSEWHERE not in asked
    assert report.blocked == [WITHHELD]
    # robots.txt's Crawl-delay stands before its User-agent line; it is honored all the same.
    assert all(interval >= 10.0 for _, interval in site.requests)
    assert site.robots_reads == [("https", "www.citizen.org")]


def test_a_failed_fetch_or_a_page_without_a_model_is_reported_failed(tmp_path: Path) -> None:
    report = _run(tmp_path, recorded())

    failed = {f.url: f.reason for f in report.failed}
    assert sorted(failed) == sorted([ELSEWHERE, MISSING, NO_TEXT])
    assert failed[MISSING] == "HTTP 404"
    assert "off the library's site" in failed[ELSEWHERE]
    assert failed[NO_TEXT].startswith("no model text")


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert (again.found, again.collected, again.unchanged) == (8, 0, first.collected)
    assert (again.blocked, again.failed) == (first.blocked, first.failed)


def test_a_revised_model_becomes_a_new_version(tmp_path: Path) -> None:
    _run(tmp_path, recorded())
    site = recorded()
    entry = site.responses[LABELING]
    entry["body"] = entry["body"].replace("its fuel", "its fuel and wick")
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (1, 3)
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "public-citizen/lantern-labeling"
    ]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [
        (DAY, date(2026, 12, 1)),
        (date(2026, 12, 1), None),
    ]


def test_a_response_that_is_not_a_page_fails_its_model_only(tmp_path: Path) -> None:
    site = recorded()
    site.overrides[MOTHS] = b"%PDF-1.4"
    report = _run(tmp_path, site)

    assert report.collected == 3
    assert {f.url: f.reason for f in report.failed}[MOTHS].startswith("not an HTML page")


def _unreadable(scheme: str, host: str) -> Robots:
    return parse_robots(f"{scheme}://{host}/robots.txt", 503, b"", "2026-11-15T00:00:00+00:00")


def test_an_unreadable_robots_txt_fails_the_library_and_nothing_is_fetched(
    tmp_path: Path,
) -> None:
    site = recorded()
    report = collect_from(
        CollectContext(root=tmp_path / "modeldocs", on=DAY),
        fetch=site.fetch,
        robots_reader=_unreadable,
    )

    assert (report.found, report.collected) == (1, 0)
    assert report.failed[0].url == LIBRARY
    assert "robots.txt could not be read" in report.failed[0].reason
    assert site.requests == []


def test_with_fetching_off_nothing_is_asked_and_the_report_says_why(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site, fetching=False)

    assert (report.found, report.collected) == (1, 0)
    assert report.failed[0].reason == "fetching from the site is off (SITE_FETCH)"
    assert (site.requests, site.robots_reads) == ([], [])


# ---- cleaning -------------------------------------------------------------------------------


def _page(site: RecordedSite, url: str) -> str:
    return read_page(site.responses[url]["body"].encode(), url).text


def test_cleaning_keeps_sections_and_drops_the_note_for_lawmakers() -> None:
    assert _page(recorded(), MOTHS) == (
        "SECTION 1. DEFINITIONS.\n"
        "(1) MOTH.—The term “moth” means an invented insect.\n"
        "SECTION 2. PROTECTION.\n"
        "(1) IN GENERAL. — No lantern may be lit near a moth.\n"
        "SECTION 3. SEVERABILITY.\n"
        "If any part of this invented act is held invalid, the rest stands.\n"
    )


def test_cleaning_starts_at_the_model_label_after_drafting_recommendations() -> None:
    # A recommendation that begins "Section 1 of ..." is advice, not the model's first section.
    assert _page(recorded(), SMOKE) == (
        "SEC. 1. Short title.\n"
        "This Act may be cited as the “[[Synthetic Smoke] Act]”.\n"
        "SEC. 2: Limits.\n"
        "(a) Definitions.—In this section, smoke means invented soot.\n"
    )


def test_cleaning_reads_roman_sections_and_drops_download_links() -> None:
    assert _page(recorded(), RIGHT_TO_KNOW) == (
        "I. It is an unfair practice to sell an unlabeled synthetic lantern.\n"
        "II. (a) The attorney general may seek an injunction.\n"
        "(b) A civil penalty of up to $1,000 applies.\n"
    )


def test_cleaning_drops_the_framing_line_share_buttons_and_instructions() -> None:
    # The line before the first section, the second share widget and "[insert ...]" go;
    # the bracketed alternatives a state enacts stay.
    assert _page(recorded(), LABELING) == LABELING_TEXT


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("<p>SEC. 1. A section without a title.</p>", "no title"),
        ("<h1>A title</h1><p>Only prose, and no section.</p>", "no model text"),
        ("<h1>A title</h1><p>Model Language</p><h4>Topics</h4>", "no model text"),
    ],
)
def test_a_page_without_a_model_is_an_error(body: str, reason: str) -> None:
    with pytest.raises(PageError, match=reason):
        read_page(body.encode())


@pytest.mark.parametrize(
    ("url", "slug"),
    [
        (LABELING, "lantern-labeling"),
        (MOTHS, "protecting-moths-from-synthetic-lanterns"),
        (f"{ARTICLE}/some-other-model/", "some-other-model"),
    ],
)
def test_a_document_s_slug_comes_from_its_url(url: str, slug: str) -> None:
    assert doc_slug(url) == slug


# ---- the source and the command -------------------------------------------------------------


def test_the_source_is_registered_and_described_without_labels() -> None:
    assert collectors()["public-citizen"].source == SOURCE
    assert not SOURCE.leaked
    assert "501(c)(4)" in SOURCE.descriptor
    assert SOURCE.members is not None
    assert SOURCE.terms_read_on == DAY
    assert SOURCE.library_url == LIBRARY


def test_the_command_collects_through_the_module_s_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = recorded()
    monkeypatch.setattr(public_citizen, "read_robots", site.read_robots)
    monkeypatch.setattr(public_citizen, "fetch_url", site.fetch)
    monkeypatch.setattr(public_citizen, "SITE_FETCH", True)
    args = ["collect-models", "public-citizen", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "8 found: 4 collected, 0 unchanged, 1 blocked by robots.txt, 3 failed" in first.output
    report = json.loads((tmp_path / "reports" / "public-citizen.json").read_text())
    # The report names no local path: it is committed.
    assert str(tmp_path) not in json.dumps(report)
    again = CliRunner().invoke(app, args)
    assert "8 found: 0 collected, 4 unchanged, 1 blocked by robots.txt, 3 failed" in again.output


def test_the_command_with_fetching_off_asks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = recorded()

    def no_fetch(url: str, interval: float) -> Fetched:
        raise AssertionError(f"fetched {url} with fetching off")

    monkeypatch.setattr(public_citizen, "read_robots", site.read_robots)
    monkeypatch.setattr(public_citizen, "fetch_url", no_fetch)
    monkeypatch.setattr(public_citizen, "SITE_FETCH", False)
    args = ["collect-models", "public-citizen", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "1 found: 0 collected, 0 unchanged, 0 blocked by robots.txt, 1 failed" in result.output
    assert site.robots_reads == []


# ---- real samples, local only ---------------------------------------------------------------

_LEFTOVER = re.compile(
    r"^(?:topics|you might be interested in|stay updated|support our work|copyright\b"
    r"|important note|note for lawmakers|share|facebook|twitter|bluesky|threads|download pdf)\b",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_OR_SUBSECTION = re.compile(
    r"^\[*(?:(?i:section|sec\b\.?)\s*[0-9]+|[IVXLC]+\.\s|\(\s*(?:[a-z]|[0-9]+)\s*\))", re.MULTILINE
)
_INSTRUCTION_LEFT = re.compile(r"\[\s*(?:cite|insert|specify)\b", re.IGNORECASE)


def _collected() -> list[ModelDoc]:
    if not (MODELDOCS_DIR / SOURCE.key).exists():
        return []
    return read_docs(SOURCE, MODELDOCS_DIR)


@pytest.mark.skipif(not _collected(), reason="no public-citizen documents under the data directory")
@pytest.mark.parametrize("doc", _collected(), ids=lambda d: d.id.partition("/")[2][:32])
def test_a_collected_model_reads_as_law_without_the_page_around_it(doc: ModelDoc) -> None:
    raw = raw_path(doc, MODELDOCS_DIR).read_bytes()
    text = load_text(doc, MODELDOCS_DIR)
    # Counts and flags only, so a failure never prints the model's text.
    reread = read_page(raw, doc.url).text == text
    leftovers = len(_LEFTOVER.findall(text))
    sections = len(_SECTION_OR_SUBSECTION.findall(text))
    starts_as_law = _SECTION_OR_SUBSECTION.match(text) is not None
    instructions_left = len(_INSTRUCTION_LEFT.findall(text))
    words, page_words = len(text.split()), len(read_page(raw, doc.url).title.split())

    assert reread
    assert leftovers == 0
    assert starts_as_law
    assert sections >= 2
    assert instructions_left == 0
    # A model is longer than its title and shorter than a page of navigation would make it.
    assert words > 10 * page_words
    assert doc.media_type == MEDIA_TYPE


@pytest.mark.skipif(not _collected(), reason="no public-citizen documents under the data directory")
def test_the_collected_library_holds_its_listed_models() -> None:
    # Read on 2026-11-15, the library page listed seven models, each an article on citizen.org.
    docs = _collected()
    assert len(docs) >= 7
    assert all(d.url.startswith(ARTICLE) for d in docs)
    assert len({d.text_sha256 for d in docs}) == len(docs)
