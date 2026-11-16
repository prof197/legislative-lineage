"""The ifs collector (DESIGN.md#model-documents): each listed model bill from its page, with
provenance, against recorded responses (tests/fixtures/collect/ifs/), never the network. It reads
pages and cleans texts through the eppc collector's functions, tested there; this tests the
source, its host and the outcomes. Every page and text in the fixtures is synthetic.

The real-sample tests read the documents collected under the data directory (LINEAGE_DATA) and
are skipped until integration has collected them; they assert counts only, never print text."""

import json
import re
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.collect import CollectContext, CollectReport, collectors
from lineage.collect_base import Fetched
from lineage.collectors import ifs
from lineage.collectors.eppc import DOCX, Listed, clean_model_text, document_text, is_heading
from lineage.collectors.ifs import DOCUMENTS, IFS, LIBRARY, SITE_FETCH, SOURCE, collect_from
from lineage.modeldocs import load_text, raw_path, read_docs, read_versions
from tests.recorded_site import COLLECT, RecordedSite, docx_bytes

FIXTURES = COLLECT / "ifs"
DAY = date(2026, 11, 15)

PEWTER = Listed(
    "synthetic-pewter",
    "Synthetic Pewter Accountability Act",
    f"{IFS}/family-first-technology-initiative/synthetic-pewter",
    None,
)
TIN = Listed(
    "synthetic-tin", "Synthetic Tin Act", f"{IFS}/report-brief/synthetic-tin-model-bill", None
)
MAP = Listed("synthetic-map", "Synthetic Map", f"{IFS}/reports/synthetic-map/model-bill", None)
MISSING = Listed("synthetic-missing", "Synthetic Missing", f"{IFS}/report-brief/missing", None)
LISTED = (PEWTER, TIN, MAP, MISSING)
PEWTER_FILE = f"{IFS}/ifs-admin/resources/synthetic-pewter-act.pdf"
TIN_FILE = f"{IFS}/ifs-admin/resources/synthetic-tin-act.docx"

CLEANED = (
    "SECTION 1. TITLE.\n"
    "This bill may be cited as the Synthetic Pewter Accountability Act.\n"
    "SECTION 2. DEFINITIONS.\n"
    '"Pewter" means an alloy sold in this state under.\n'
    "SECTION 3. ENFORCEMENT.\n"
    "The attorney general may enforce this act.\n"
)


def recorded() -> RecordedSite:
    responses = json.loads((FIXTURES / "site.json").read_text())
    responses.pop("_note")
    return RecordedSite(responses, (FIXTURES / "robots.txt").read_bytes())


def _run(
    tmp_path: Path, site: RecordedSite, *, on: date = DAY, on_site: bool = True
) -> CollectReport:
    return collect_from(
        CollectContext(root=tmp_path / "modeldocs", on=on),
        documents=LISTED,
        fetch=site.fetch,
        robots_reader=site.read_robots,
        site=on_site,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_bills_are_found_collected_and_stored_with_provenance(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    assert (report.found, report.collected, report.unchanged) == (4, 2, 0)
    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}
    assert sorted(docs) == ["ifs/synthetic-pewter", "ifs/synthetic-tin"]
    pewter = docs["ifs/synthetic-pewter"]
    assert (pewter.title, pewter.url, pewter.published, pewter.co_publishers) == (
        PEWTER.title,
        PEWTER.page,
        None,
        (),
    )
    assert (pewter.media_type, pewter.robots, pewter.leaked) == (
        "application/pdf",
        "allowed",
        False,
    )
    assert raw_path(pewter, root).read_bytes() == site.fetch(PEWTER_FILE, 1.0).body
    assert load_text(pewter, root).split() == CLEANED.split()
    tin = docs["ifs/synthetic-tin"]
    assert tin.media_type == DOCX
    assert raw_path(tin, root).read_bytes() == docx_bytes(
        (FIXTURES / "synthetic_word_bill.txt").read_text()
    )
    assert load_text(tin, root) == CLEANED

    asked = [url for url, _ in site.requests]
    # The model bill, never the fact sheet; IFS's host only, paced.
    assert {PEWTER_FILE, TIN_FILE} <= set(asked)
    assert not any("fact-sheet" in url for url in asked)
    assert all(interval >= 1.0 for _, interval in site.requests)
    assert site.robots_reads == [("https", "ifstudies.org")]


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert (again.found, again.collected, again.unchanged) == (4, 0, 2)
    assert (again.blocked, again.failed) == (first.blocked, first.failed)
    assert [v.replaced_on for v in read_versions(SOURCE, tmp_path / "modeldocs")] == [None, None]


def test_a_blocked_page_and_a_failed_fetch_are_reported(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    assert report.blocked == [MAP.page]
    assert MAP.page not in [url for url, _ in site.requests]
    assert [(f.url, f.reason) for f in report.failed] == [
        (MISSING.page, f"{MISSING.page}: HTTP 404")
    ]


def test_with_fetching_off_each_bill_is_reported_and_nothing_asked(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site, on_site=False)

    assert (report.found, len(report.failed)) == (4, 4)
    assert (site.requests, site.robots_reads) == ([], [])


def test_a_link_to_another_organization_is_not_this_source(tmp_path: Path) -> None:
    site = recorded()
    site.responses[PEWTER.page] = {
        "content_type": "text/html",
        "body": '<a href="https://eppc.org/x-model-bill.pdf">Download the Model Bill</a>',
    }
    report = _run(tmp_path, site)

    assert (PEWTER.page, "no link to the model legislation on its page") in [
        (f.url, f.reason) for f in report.failed
    ]


def test_the_source_and_what_it_collects() -> None:
    assert collectors()["ifs"].source == SOURCE
    assert (SOURCE.key, SOURCE.organization) == ("ifs", "Institute for Family Studies")
    assert not SOURCE.leaked
    assert (SOURCE.members, SOURCE.members_source) == (None, None)
    assert "exempt under IRS section 501(c)(3)" in SOURCE.descriptor
    assert SOURCE.library_url == LIBRARY
    assert SOURCE.terms_read_on == DAY
    assert SITE_FETCH
    # The joint 2023 bill is eppc's, crediting IFS; it is not collected twice.
    assert [(d.slug, d.page, d.co_publishers) for d in DOCUMENTS] == [
        ("app-store-accountability-act", LIBRARY, ())
    ]


def test_the_command_collects_through_the_recorded_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = recorded()
    monkeypatch.setattr(ifs, "DOCUMENTS", LISTED)
    monkeypatch.setattr(ifs, "fetch_url", site.fetch)
    monkeypatch.setattr(ifs, "read_robots", site.read_robots)
    args = [
        "collect-models",
        "ifs",
        "--modeldocs",
        str(tmp_path / "modeldocs"),
        "--reports",
        str(tmp_path / "reports"),
        "--min-free-gib",
        "0",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "4 found: 2 collected, 0 unchanged, 1 blocked by robots.txt, 1 failed" in first.output
    report = json.loads((tmp_path / "reports" / "ifs.json").read_text())
    assert str(tmp_path) not in json.dumps(report)
    again = CliRunner().invoke(app, args)
    assert "4 found: 0 collected, 2 unchanged" in again.output


# ---- real samples, local only ---------------------------------------------------------------

_INSTRUCTION_LEFT = re.compile(r"\[\s*(?:cite|insert|specify|enter)\b", re.IGNORECASE)


def _collected() -> list[str]:
    return [d.id for d in read_docs(SOURCE) if raw_path(d).exists()]


@pytest.mark.skipif(not _collected(), reason="no ifs documents under the data directory")
@pytest.mark.parametrize("doc_id", _collected())
def test_a_collected_bill_keeps_its_sections_and_drops_the_rest(doc_id: str) -> None:
    doc = next(d for d in read_docs(SOURCE) if d.id == doc_id)
    raw = raw_path(doc).read_bytes()
    extracted = document_text(Fetched(doc.url, doc.media_type, raw), doc.media_type)
    cleaned = clean_model_text(extracted)

    # Counts only, so a failure never prints the bill's text.
    stored_is_current = load_text(doc) == cleaned
    headings = sum(map(is_heading, cleaned.split("\n")))
    instructions_left = len(_INSTRUCTION_LEFT.findall(cleaned))
    kept, before = len(cleaned.split()), len(extracted.split())

    assert stored_is_current
    assert headings > 0
    assert instructions_left == 0
    # The App Store Accountability Act as a reader downloads it (2026-11-15) has only its title
    # before the first section and a few words of instructions: 99% of its words are kept.
    assert 0.9 * before <= kept <= before
    assert clean_model_text(cleaned) == cleaned
