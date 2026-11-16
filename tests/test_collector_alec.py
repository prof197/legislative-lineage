"""ALEC's collector (DESIGN.md#model-documents): the model policy library found through the
sitemaps, each policy's type and status read from its page, its text from the site's API or an
attached file, against synthetic responses (tests/fixtures/collect/alec/), never the network.
Every text here is invented; no ALEC text is committed.

The real-sample tests read collected texts from the local data directory (LINEAGE_DATA) and are
skipped until integration has collected them; they assert counts only, never print a text."""

import json
import re
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.collect import CollectContext, CollectReport, collectors
from lineage.collect_base import Fetched
from lineage.collectors import alec
from lineage.collectors.alec import (
    DOCX,
    LIBRARY,
    SOURCE,
    ApiPolicy,
    api_policy,
    attachment,
    clean_policy,
    collect_from,
)
from lineage.extract import html_text, pdf_text
from lineage.modeldocs import (
    MODELDOCS_DIR,
    load_text,
    raw_path,
    read_docs,
    read_versions,
    sha256,
)
from lineage.modeltext import ModelTextError, docx_text
from lineage.robots import Robots, parse_robots
from tests.recorded_site import COLLECT, RecordedSite, docx_bytes

DAY = date(2026, 11, 15)
SITE = "https://alec.org"
FIXTURES = COLLECT / "alec"
AMENDMENT = FIXTURES / "synthetic_amendment.txt"
FILE = f"{SITE}/wp-content/uploads/2026/01/Synthetic-Lantern-Amendments.docx"
WITHHELD = f"{SITE}/model-policy/withheld-lamp-act/"

LANTERN = (
    "Section 1. {Title}\n"
    "This Act may be cited as the Synthetic Lantern Registry Act.\n"
    "Section 2. {Definitions}\n"
    "(A) “Lantern” means a portable lamp kept under.\n"
    "(B) “Registry” means the list kept by the [secretary of state].\n"
    "Section 3. {Registration}\n"
    "A keeper shall register each lantern within [60] days with.\n"
    "A fee of twelve dollars applies.\n"
)


def policy(slug: str) -> str:
    return f"{SITE}/model-policy/{slug}/"


def api(slug: str) -> str:
    return (
        f"{SITE}/wp-json/wp/v2/model-policy?slug={slug}&_fields=id%2Cslug%2Clink%2Ctitle%2Ccontent"
    )


def recorded() -> RecordedSite:
    """The fixture's responses; a `json` entry is a body written as JSON."""
    raw: dict[str, dict[str, object]] = json.loads((FIXTURES / "site.json").read_text())
    raw.pop("_note")
    responses = {
        url: {k: (json.dumps(v) if k == "json" else str(v)) for k, v in entry.items()}
        for url, entry in raw.items()
    }
    for entry in responses.values():
        if "json" in entry:
            entry["body"] = entry.pop("json")
    return RecordedSite(responses, (FIXTURES / "robots.txt").read_bytes())


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


def test_the_library_is_collected_with_provenance_and_every_policy_found_accounted_for(
    tmp_path: Path,
) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    # Seven policies of the kinds collected; the statement, the ordinance and the draft are not
    # model state legislation, so they are neither collected nor counted.
    assert (report.found, report.collected, report.unchanged) == (7, 3, 0)
    assert report.blocked == [WITHHELD]
    failed = {f.url: f.reason for f in report.failed}
    assert failed == {
        policy("hollow-lamp-act"): "the API lists no policy hollow-lamp-act",
        policy("missing-lamp-act"): "HTTP 404",
        policy("unlabeled-lamp-act"): "no Type or Status on the policy's page",
    }

    asked = [url for url, _ in site.requests]
    # robots.txt first: the disallowed page is never asked for. A policy outside the kinds
    # collected costs its page only; the archive's own page and its pages of cards are skipped.
    assert WITHHELD not in asked
    assert policy("statement-of-principles-on-lanterns") in asked
    assert not any(
        slug in url and "wp-json" in url
        for url in asked
        for slug in ("statement-of", "draft-lamp", "lamp-post")
    )
    assert f"{SITE}/model-policy/" not in asked
    assert f"{SITE}/model-policy/page/2/" not in asked
    assert f"{SITE}/post-sitemap.xml" not in asked
    assert all(interval >= 1.0 for _, interval in site.requests)
    assert site.robots_reads == [("https", "alec.org")]

    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}
    assert sorted(docs) == [
        "alec/amendments-to-lantern-registry-act",
        "alec/lantern-registry-act",
        "alec/resolution-in-support-of-lamp-week",
    ]
    lantern = docs["alec/lantern-registry-act"]
    assert (lantern.title, lantern.url) == ("Lantern Registry Act", policy("lantern-registry-act"))
    assert (lantern.robots, lantern.leaked, lantern.media_type) == (
        "allowed",
        False,
        "application/json",
    )
    # The organization's own date: the latest it gives, the amendment's over the finalizing.
    assert (lantern.published, lantern.collected_on) == (date(2018, 9, 18), DAY)
    body = site.fetch(api("lantern-registry-act"), 1.0).body
    assert lantern.raw_sha256 == sha256(body)
    assert raw_path(lantern, root).read_bytes() == body
    assert load_text(lantern, root) == LANTERN

    resolution = docs["alec/resolution-in-support-of-lamp-week"]
    # The API title is HTML: entities decoded, a no-break space read as a space.
    assert resolution.title == "Resolution in Support of Lamp Week \u2013 Invented"
    assert resolution.published == date(2018, 9, 18)
    assert load_text(resolution, root).split("\n")[0].startswith("WHEREAS, invented lamps")

    amendment = docs["alec/amendments-to-lantern-registry-act"]
    assert (amendment.media_type, amendment.published) == (DOCX, date(2026, 3, 3))
    assert amendment.raw_sha256 == sha256(docx_bytes(AMENDMENT.read_text()))
    assert load_text(amendment, root) == (
        "Section 3. {Registration}\nA keeper shall register each lantern within [90] days with.\n"
    )
    assert FILE in asked


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert again.collected == 0
    assert again.unchanged == first.collected
    assert (again.found, again.blocked, again.failed) == (first.found, first.blocked, first.failed)


def test_a_revised_text_becomes_a_new_version(tmp_path: Path) -> None:
    _run(tmp_path, recorded())
    site = recorded()
    site.overrides[FILE] = docx_bytes(AMENDMENT.read_text().replace("[90]", "[120]"))
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (1, 2)
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "alec/amendments-to-lantern-registry-act"
    ]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [
        (DAY, date(2026, 12, 1)),
        (date(2026, 12, 1), None),
    ]


def test_an_unreadable_file_fails_its_policy_only(tmp_path: Path) -> None:
    site = recorded()
    site.overrides[FILE] = b"not a Word file"
    report = _run(tmp_path, site)

    assert (report.found, report.collected) == (7, 2)
    reason = {f.url: f.reason for f in report.failed}[policy("amendments-to-lantern-registry-act")]
    assert "unreadable Word document" in reason


def test_a_host_whose_robots_txt_could_not_be_read_fails_and_nothing_is_fetched(
    tmp_path: Path,
) -> None:
    site = recorded()

    def overloaded(scheme: str, host: str) -> Robots:
        url = f"{scheme}://{host}/robots.txt"
        return parse_robots(url, 503, b"", "2026-11-15T00:00:00+00:00")

    report = collect_from(
        CollectContext(root=tmp_path / "modeldocs", on=DAY),
        fetch=site.fetch,
        robots_reader=overloaded,
        site=True,
    )

    assert site.requests == []
    assert (report.found, report.collected) == (1, 0)
    assert "robots.txt could not be read" in report.failed[0].reason


def test_with_fetching_off_nothing_is_asked_and_the_library_is_reported_unread(
    tmp_path: Path,
) -> None:
    site = recorded()
    report = _run(tmp_path, site, on_site=False)

    assert (site.requests, site.robots_reads) == ([], [])
    assert (report.found, report.collected) == (1, 0)
    assert report.failed[0].url == LIBRARY
    assert "fetching from the site is off" in report.failed[0].reason


def test_the_source_entry() -> None:
    assert collectors()["alec"].source == SOURCE
    assert alec.SITE_FETCH is True
    assert SOURCE.library_url == "https://alec.org/model-policy/"
    assert SOURCE.members_source == "https://alec.org/about/"
    assert SOURCE.terms_url == "https://alec.org/terms-of-use/"
    assert SOURCE.terms_read_on == date(2026, 11, 15)
    assert "501(c)(3)" in SOURCE.descriptor
    assert not SOURCE.leaked


def test_the_command_reports_the_library_unread_without_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_fetch(url: str, interval: float) -> Fetched:
        raise AssertionError(f"fetched {url} with the site off")

    monkeypatch.setattr(alec, "fetch_url", no_fetch)
    monkeypatch.setattr(alec, "read_robots", recorded().read_robots)
    monkeypatch.setattr(alec, "SITE_FETCH", False)
    args = ["collect-models", "alec", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "1 found: 0 collected, 0 unchanged, 0 blocked by robots.txt, 1 failed" in result.output
    report = json.loads((tmp_path / "reports" / "alec.json").read_text())
    assert str(tmp_path) not in json.dumps(report)


# ---- cleaning -------------------------------------------------------------------------------


def test_a_policy_keeps_its_sections_and_loses_labels_notes_approvals_and_instructions() -> None:
    raw = json.loads((FIXTURES / "site.json").read_text())[api("lantern-registry-act")]["json"]
    body = html_text(raw[0]["content"]["rendered"].encode(), "utf-8")
    assert clean_policy(body) == LANTERN
    assert clean_policy(LANTERN) == LANTERN


def test_a_resolution_keeps_its_clauses_from_the_first_whereas() -> None:
    given = (
        "Model Resolution\nSummary\nAn invented summary.\n"
        "WHEREAS, lamps are invented; and\n"
        "NOW, THEREFORE, BE IT RESOLVED, that [state] supports them.\n"
        "BE IT FURTHER RESOLVED, that copies go to [insert recipients].\n"
        "Adopted by the Invented Task Force on May 1, 2020.\n"
    )
    assert clean_policy(given) == (
        "WHEREAS, lamps are invented; and\n"
        "NOW, THEREFORE, BE IT RESOLVED, that [state] supports them.\n"
        "BE IT FURTHER RESOLVED, that copies go to.\n"
    )


@pytest.mark.parametrize(
    ("given", "cleaned"),
    [
        ("under {insert citation} and", "under and"),
        ("kept by {name of agency}.", "kept by."),
        ("under [reference any state lamp code].", "under."),
        ("a fee. {Drafting Note: optional.}", "a fee."),
        ("filed with [cite the code] here", "filed with here"),
    ],
)
def test_instructions_to_the_drafter_go(given: str, cleaned: str) -> None:
    assert clean_policy(f"Section 1. {{Title}}\n{given}\n") == f"Section 1. {{Title}}\n{cleaned}\n"


@pytest.mark.parametrize(
    "line",
    [
        "Approved by ALEC Board of Directors in 1995.",
        "Reapproved by ALEC Board of Directors on January 28, 2013.",
        "(Adopted by the Invented Task Force at the Annual Meeting, July 2015.)",
        "Drafting Note: A state may rename the registry.",
        "{Drafting note: a state may waive it.}",
        "Note: an invented note.",
    ],
)
def test_notes_and_approval_lines_go(line: str) -> None:
    assert clean_policy(f"Section 1. {{Title}}\nA rule.\n{line}\n") == (
        "Section 1. {Title}\nA rule.\n"
    )


@pytest.mark.parametrize(
    "kept",
    [
        "Section 2. {Definitions}",
        "The [secretary of state] keeps it within [60] days.",
        "(C) Approved by the board of directors of the corporation.",
        "The court [shall][may] dismiss it.",
    ],
)
def test_enacted_text_stays_as_written(kept: str) -> None:
    assert clean_policy(f"Section 1. {{Title}}\n{kept}\n") == f"Section 1. {{Title}}\n{kept}\n"


def test_a_text_with_no_section_or_clause_is_refused() -> None:
    with pytest.raises(ModelTextError, match="no numbered section"):
        clean_policy("(See Amendments Here)\n")


def test_an_amendment_is_read_from_the_file_it_links() -> None:
    assert attachment('<p><a href="/wp-content/uploads/a.docx">(See Amendments Here)</a></p>') == (
        f"{SITE}/wp-content/uploads/a.docx"
    )
    assert attachment("<p><a href='/model-policy/other/'>other</a></p>") is None
    response = Fetched(api("x"), "application/json", b'[{"slug": "y"}]')
    with pytest.raises(Exception, match="lists no policy x"):
        api_policy(response, "x")
    got = api_policy(
        Fetched(api("x"), "", b'[{"slug": "x", "title": {"rendered": "T"}, "content": {}}]'), "x"
    )
    assert got == ApiPolicy(title="T", content="")


# ---- real samples, local only ---------------------------------------------------------------

_INSTRUCTION_LEFT = re.compile(r"[\[{]\s*(?:cite|insert|specify|reference)\b", re.IGNORECASE)
_LAW = re.compile(
    r"^\s*\[?(?:section|sec\.)\s+\d+|^\s*(?:whereas|be\s+it|now,?\s+therefore|resolved)\b", re.I
)


def _collected() -> list[Path]:
    docs = read_docs(SOURCE, MODELDOCS_DIR)
    return [raw_path(d) for d in docs if raw_path(d).exists()]


@pytest.mark.skipif(not _collected(), reason="no ALEC policies under the data directory")
def test_every_collected_policy_rereads_to_its_stored_text_and_keeps_its_law() -> None:
    # Counts only, so a failure never prints a policy's text.
    rereads = heads = instructions = notes = approvals = 0
    docs = read_docs(SOURCE, MODELDOCS_DIR)
    for doc in docs:
        raw = raw_path(doc).read_bytes()
        if doc.media_type == "application/json":
            content = json.loads(raw)[0]["content"]["rendered"]
            body = html_text(content.encode(), "utf-8")
        elif doc.media_type == DOCX:
            body = docx_text(raw)
        else:
            body = pdf_text(raw)
        stored = load_text(doc)
        rereads += clean_policy(body) == stored and clean_policy(stored) == stored
        heads += bool(_LAW.match(stored.split("\n")[0]))
        instructions += len(_INSTRUCTION_LEFT.findall(stored))
        notes += sum(1 for line in stored.split("\n") if alec._NOTE_LINE.match(line))  # pyright: ignore[reportPrivateUsage]
        approvals += sum(1 for line in stored.split("\n") if alec._APPROVAL.match(line))  # pyright: ignore[reportPrivateUsage]
    assert rereads == len(docs)
    assert heads == len(docs)
    assert (instructions, notes, approvals) == (0, 0, 0)


@pytest.mark.skipif(not _collected(), reason="no ALEC policies under the data directory")
def test_cleaning_keeps_most_of_each_collected_policys_words() -> None:
    low = 0
    docs = read_docs(SOURCE, MODELDOCS_DIR)
    for doc in docs:
        if doc.media_type != "application/json":
            continue
        content = json.loads(raw_path(doc).read_bytes())[0]["content"]["rendered"]
        before = len(html_text(content.encode(), "utf-8").split())
        kept = len(load_text(doc).split())
        # Labels, approval lines and instructions are a few words; losing more is losing law.
        low += kept < 0.8 * before
    assert low <= 0.05 * len(docs)
