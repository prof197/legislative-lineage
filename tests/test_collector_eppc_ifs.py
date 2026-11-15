"""The eppc-ifs collector (DESIGN.md#model-documents): each listed document from its publication
page, with provenance, against recorded responses (tests/fixtures/collect/eppc-ifs/), never the
network. Every page and text in the fixtures is synthetic; no model text is committed.

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
from lineage.collectors import eppc_ifs
from lineage.collectors.eppc_ifs import (
    DOCUMENTS,
    DOCX,
    EPPC,
    IFS,
    MEDIA,
    SITE_FETCH,
    SOURCE,
    Listed,
    clean_model_text,
    collect_from,
    document_text,
    model_text_link,
    on_media_host,
)
from lineage.modeldocs import load_text, raw_path, read_docs, read_versions, sha256
from lineage.modeltext import ModelTextError
from tests.recorded_site import COLLECT, RecordedSite, docx_bytes

FIXTURES = COLLECT / "eppc-ifs"
SYNTHETIC_BILL = FIXTURES / "synthetic_bill.txt"
SYNTHETIC_WORD_BILL = FIXTURES / "synthetic_word_bill.txt"
DAY = date(2026, 11, 15)

LANTERN = Listed(
    "synthetic-lantern",
    "Model Legislation for a Synthetic Lantern Registry",
    f"{EPPC}/publication/synthetic-lantern-model-legislation/",
    date(2026, 1, 2),
)
HARBOR = Listed(
    "synthetic-harbor",
    "The Synthetic Harbor Act",
    f"{EPPC}/publication/synthetic-harbor-model-legislation/",
    date(2026, 2, 3),
)
PEWTER = Listed(
    "synthetic-pewter",
    "Synthetic Pewter Accountability Act",
    f"{IFS}/report-brief/synthetic-pewter-model-bill",
    None,
)
PUBLICATION = f"{EPPC}/publication"
WITHHELD = Listed(
    "synthetic-withheld",
    "Synthetic Withheld",
    f"{PUBLICATION}/synthetic-withheld-model-legislation/",
    None,
)
MISSING = Listed(
    "synthetic-missing",
    "Synthetic Missing",
    f"{PUBLICATION}/synthetic-missing-model-legislation/",
    None,
)
UNLINKED = Listed(
    "synthetic-unlinked", "Synthetic Unlinked", f"{PUBLICATION}/synthetic-unlinked/", None
)
LISTED = (LANTERN, HARBOR, PEWTER, WITHHELD, MISSING, UNLINKED)
LANTERN_FILE = f"{MEDIA}/2026/01/synthetic-lantern-bill_web.pdf"
PEWTER_FILE = f"{IFS}/ifs-admin/resources/synthetic-pewter-act.docx"
WITHHELD_FILE = f"{MEDIA}/withheld/synthetic-withheld-bill.pdf"

LANTERN_CLEANED = (
    "Part 1. General Requirements\n"
    "Section 101. Definitions.\n"
    '(1) "Lantern" means a portable lamp sold in [State].\n'
    '(2) "Registrar" means the office named under.\n'
    "Section 102. Registration of lanterns.\n"
    "(1) Beginning, a seller shall register each lantern.\n"
    "(2) A seller shall keep the record for [two] years.\n"
    "Section 102. Notice of sale.\n"
    "A seller shall give the registrar notice of each sale.\n"
    "Part 2. Enforcement\n"
    "Section 201. Enforcement by the attorney general.\n"
    "The attorney general may bring an action under this [act].\n"
    "Section 202. Severability.\n"
    "If any provision of this [act] is held invalid, the rest stands.\n"
    "Effective date.\n"
    "This [act] takes effect on.\n"
)
HARBOR_CLEANED = (
    "Section 1. Short title.\n"
    "This [act] may be cited as the Synthetic Harbor Act.\n"
    "Section 2. Harbors.\n"
    "A harbor keeper shall light the harbor.\n"
)
PEWTER_CLEANED = (
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


def test_the_documents_are_found_collected_and_stored_with_provenance(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    assert (report.found, report.collected, report.unchanged) == (6, 3, 0)
    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}
    assert sorted(docs) == [
        "eppc-ifs/synthetic-harbor",
        "eppc-ifs/synthetic-lantern",
        "eppc-ifs/synthetic-pewter",
    ]

    lantern = docs["eppc-ifs/synthetic-lantern"]
    assert (lantern.title, lantern.url, lantern.published) == (
        LANTERN.title,
        LANTERN.page,
        date(2026, 1, 2),
    )
    assert (lantern.media_type, lantern.robots, lantern.leaked) == (
        "application/pdf",
        "allowed",
        False,
    )
    assert lantern.collected_on == DAY
    assert sha256(raw_path(lantern, root).read_bytes()) == lantern.raw_sha256
    assert raw_path(lantern, root).read_bytes() == site.fetch(LANTERN_FILE, 1.0).body
    assert load_text(lantern, root).split() == LANTERN_CLEANED.split()

    harbor = docs["eppc-ifs/synthetic-harbor"]
    assert (harbor.media_type, harbor.url) == ("text/html", HARBOR.page)
    assert load_text(harbor, root) == HARBOR_CLEANED

    pewter = docs["eppc-ifs/synthetic-pewter"]
    assert (pewter.media_type, pewter.published) == (DOCX, None)
    assert raw_path(pewter, root).read_bytes() == docx_bytes(SYNTHETIC_WORD_BILL.read_text())
    assert load_text(pewter, root) == PEWTER_CLEANED

    assert [(v.doc_id, v.first_seen, v.replaced_on) for v in read_versions(SOURCE, root)] == [
        ("eppc-ifs/synthetic-harbor", DAY, None),
        ("eppc-ifs/synthetic-lantern", DAY, None),
        ("eppc-ifs/synthetic-pewter", DAY, None),
    ]


def test_only_the_model_text_is_asked_for_and_each_host_paced(tmp_path: Path) -> None:
    site = recorded()
    _run(tmp_path, site)

    asked = [url for url, _ in site.requests]
    # EPPC's upload URL is asked for at the host that serves it; summaries, briefs, fact
    # sheets, listings and another publisher's file never are.
    assert LANTERN_FILE in asked
    assert not any("/wp-content/uploads/" in url for url in asked)
    assert not any(
        re.search(r"[Ss]ummary|Policy-Brief|fact-sheet|/category/|example\.org", u) for u in asked
    )
    assert PEWTER_FILE in asked
    assert all(interval >= 1.0 for _, interval in site.requests)
    # robots.txt once per host, each host's before any of its pages.
    assert sorted(site.robots_reads) == [
        ("https", "eppc.org"),
        ("https", "ifstudies.org"),
        ("https", "media.eppc.org"),
    ]


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert (again.found, again.collected, again.unchanged) == (6, 0, 3)
    assert (again.blocked, again.failed) == (first.blocked, first.failed)


def test_a_revised_text_becomes_a_new_version(tmp_path: Path) -> None:
    _run(tmp_path, recorded())
    site = recorded()
    revised = SYNTHETIC_WORD_BILL.read_text().replace("Pewter", "Tin")
    site.overrides[PEWTER_FILE] = docx_bytes(revised)
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (1, 2)
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "eppc-ifs/synthetic-pewter"
    ]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [
        (DAY, date(2026, 12, 1)),
        (date(2026, 12, 1), None),
    ]


def test_a_url_robots_txt_disallows_is_reported_blocked_and_never_asked(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    assert report.blocked == [WITHHELD_FILE]
    assert WITHHELD_FILE not in [url for url, _ in site.requests]
    stored = {d.id for d in read_docs(SOURCE, tmp_path / "modeldocs")}
    assert "eppc-ifs/synthetic-withheld" not in stored


def test_a_failed_fetch_or_a_page_without_the_text_is_reported_failed(tmp_path: Path) -> None:
    site = recorded()
    site.overrides[PEWTER_FILE] = b"not a Word file"
    report = _run(tmp_path, site)

    failed = {f.url: f.reason for f in report.failed}
    assert sorted(failed) == sorted([MISSING.page, PEWTER.page, UNLINKED.page])
    assert failed[MISSING.page] == f"{MISSING.page}: HTTP 404"
    assert failed[UNLINKED.page] == "no link to the model legislation on its page"
    assert "unreadable Word document" in failed[PEWTER.page]
    assert (report.found, report.collected) == (6, 2)


def test_with_fetching_off_each_document_is_reported_and_nothing_asked(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site, on_site=False)

    assert (report.found, report.collected, len(report.failed)) == (6, 0, 6)
    assert all("fetching from the site is off" in f.reason for f in report.failed)
    assert (site.requests, site.robots_reads) == ([], [])


# ---- the model text's link and its cleaning -------------------------------------------------


def _page(url: str, links: str) -> Fetched:
    return Fetched(url, "text/html", f"<html><body>{links}</body></html>".encode())


def test_the_link_naming_the_model_legislation_wins_over_summaries_and_tags() -> None:
    page = _page(
        f"{EPPC}/publication/x/",
        '<a href="/wp-content/uploads/x-summary_web.pdf">PDF</a>'
        '<a href="/category/model-legislation/">Model Legislation</a>'
        '<a href="https://example.org/model.pdf">Model Legislation</a>'
        '<a href="/the-x-act/">View Model Legislation</a>'
        '<a href="/wp-content/uploads/x-bill.pdf">Model Legislation</a>'
        '<a href="/wp-content/uploads/x-bill-summary.pdf">Summary</a>',
    )
    assert model_text_link(page) == f"{EPPC}/wp-content/uploads/x-bill.pdf"


def test_a_bare_pdf_button_is_the_text_when_no_link_names_it() -> None:
    page = _page(
        f"{EPPC}/publication/books/",
        '<a href="https://media.eppc.org/2026/08/1-26-X-Act.pdf">View PDF</a>'
        '<a href="/wp-content/uploads/report.pdf">report</a>',
    )
    assert model_text_link(page) == f"{MEDIA}/2026/08/1-26-X-Act.pdf"
    assert model_text_link(_page(f"{EPPC}/p/", '<a href="/x.pdf">Executive Summary</a>')) is None


def test_eppc_uploads_are_asked_for_at_the_media_host() -> None:
    assert (
        on_media_host(f"{EPPC}/wp-content/uploads/2023/12/a_web.pdf")
        == f"{MEDIA}/2023/12/a_web.pdf"
    )
    assert on_media_host(f"{IFS}/ifs-admin/resources/a.pdf") == f"{IFS}/ifs-admin/resources/a.pdf"


def test_cleaning_keeps_the_sections_and_drops_commentary_notes_and_furniture() -> None:
    assert clean_model_text(SYNTHETIC_BILL.read_text()) == LANTERN_CLEANED
    assert clean_model_text(LANTERN_CLEANED) == LANTERN_CLEANED


@pytest.mark.parametrize(
    ("given", "cleaned"),
    [
        # A sentence that opens with "Comments" is law, not a commentary heading.
        ("Section 1. Records.\nComments received shall be kept.\n", None),
        # A wrapped line that opens with a number is law unless a mark named that footnote.
        (
            "Section 1. Time.\nwithin\n30 Days of notice.\n4 ethics and public policy center\n",
            "Section 1. Time.\nwithin\n30 Days of notice.\n",
        ),
        (
            "Section 1. A.\nText.\nDrafting note: pick a court.\nMore note.\nSection 2. B.\nLaw.\n",
            "Section 1. A.\nText.\nSection 2. B.\nLaw.\n",
        ),
        ("Section 1. A.\nLaw.\nCopyright © 2026 Example\nContact\n", "Section 1. A.\nLaw.\n"),
        # The footer as EPPC sets it, and a running header.
        (
            "Section 1. A.\nLaw\n5 ethics and public policy center ♦ www.eppc.org\n"
            "Technology and Human Flourishing\nmore law.\n",
            "Section 1. A.\nLaw\nmore law.\n",
        ),
    ],
)
def test_cleaning_cases(given: str, cleaned: str | None) -> None:
    assert clean_model_text(given) == (cleaned or given)


def test_a_text_without_a_section_is_not_a_model() -> None:
    with pytest.raises(ModelTextError, match="no part or section heading"):
        clean_model_text("An invented policy brief.\nIt proposes ideas.\n")


# ---- the source and the command -------------------------------------------------------------


def test_the_source_and_what_it_collects() -> None:
    assert collectors()["eppc-ifs"].source == SOURCE
    assert SOURCE.key == "eppc-ifs"
    assert not SOURCE.leaked
    assert (SOURCE.members, SOURCE.members_source) == (None, None)
    assert "each exempt under IRS section 501(c)(3)" in SOURCE.descriptor
    assert SOURCE.terms_read_on == DAY
    assert SITE_FETCH
    # The joint document, from EPPC's page; no duplicate of it from IFS's.
    assert [(d.slug, d.page, d.published) for d in DOCUMENTS] == [
        (
            "age-verification-of-social-media",
            f"{EPPC}/publication/protecting-kids-online-model-legislation/",
            date(2023, 10, 17),
        )
    ]


def test_the_command_collects_through_the_recorded_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = recorded()
    monkeypatch.setattr(eppc_ifs, "DOCUMENTS", LISTED)
    monkeypatch.setattr(eppc_ifs, "fetch_url", site.fetch)
    monkeypatch.setattr(eppc_ifs, "read_robots", site.read_robots)
    args = ["collect-models", "eppc-ifs", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "6 found: 3 collected, 0 unchanged, 1 blocked by robots.txt, 2 failed" in first.output
    report = json.loads((tmp_path / "reports" / "eppc-ifs.json").read_text())
    assert str(tmp_path) not in json.dumps(report)
    again = CliRunner().invoke(app, args)
    assert "6 found: 0 collected, 3 unchanged" in again.output


# ---- real samples, local only ---------------------------------------------------------------

_INSTRUCTION_LEFT = re.compile(r"\[\s*(?:cite|insert|specify|enter)\b", re.IGNORECASE)
_HEADING = re.compile(r"^\s*\[?\s*(?:part|article|section|sec\.)\s+[0-9ivxlc]+", re.I | re.M)
_FURNITURE_LEFT = re.compile(
    r"^\s*\d+\s+(?:ethics and public policy center|institute for family studies)", re.I | re.M
)
_BACK_MATTER_LEFT = re.compile(r"^\s*about the authors?\b", re.I | re.M)


def _collected() -> list[tuple[str, Path]]:
    return [(d.id, raw_path(d)) for d in read_docs(SOURCE) if raw_path(d).exists()]


@pytest.mark.skipif(not _collected(), reason="no eppc-ifs documents under the data directory")
@pytest.mark.parametrize("doc_id", [i for i, _ in _collected()])
def test_a_collected_model_keeps_its_sections_and_drops_the_rest(doc_id: str) -> None:
    doc = next(d for d in read_docs(SOURCE) if d.id == doc_id)
    raw = raw_path(doc).read_bytes()
    extracted = document_text(Fetched(doc.url, doc.media_type, raw), doc.media_type)
    cleaned = clean_model_text(extracted)

    # Counts only, so a failure never prints the model's text.
    stored_is_current = load_text(doc) == cleaned
    headings = len(_HEADING.findall(cleaned))
    instructions_left = len(_INSTRUCTION_LEFT.findall(cleaned))
    furniture_left = len(_FURNITURE_LEFT.findall(cleaned))
    back_matter_left = len(_BACK_MATTER_LEFT.findall(cleaned))
    kept, before = len(cleaned.split()), len(extracted.split())
    settled = clean_model_text(cleaned) == cleaned

    assert stored_is_current
    assert headings > 0
    assert (instructions_left, furniture_left, back_matter_left) == (0, 0, 0)
    # The cover, footers, footnotes and biographies are a small share of a bill's words.
    assert 0.6 * before <= kept <= before
    assert settled
