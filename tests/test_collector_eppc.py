"""The eppc collector (DESIGN.md#model-documents): each listed document from its publication page
or its file, with provenance and co-publishers, against recorded responses
(tests/fixtures/collect/eppc/), never the network. Every page and text in the fixtures is
synthetic; no model text is committed.

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
from lineage.collectors import eppc
from lineage.collectors.eppc import (
    DOCUMENTS,
    EPPC,
    MEDIA,
    SITE_FETCH,
    SOURCE,
    Listed,
    clean_model_text,
    collect_from,
    document_text,
    is_heading,
    model_text_link,
    on_media_host,
)
from lineage.modeldocs import load_text, raw_path, read_docs, read_versions, sha256
from lineage.modeltext import ModelTextError
from tests.recorded_site import COLLECT, RecordedSite, pdf_bytes

FIXTURES = COLLECT / "eppc"
DAY = date(2026, 11, 15)
PUBLICATION = f"{EPPC}/publication"

LANTERN = Listed(
    "synthetic-lantern",
    "Model Legislation for a Synthetic Lantern Registry",
    f"{PUBLICATION}/synthetic-lantern-model-legislation/",
    date(2026, 1, 2),
    co_publishers=("Example Family Institute",),
)
HARBOR = Listed(
    "synthetic-harbor",
    "The Synthetic Harbor Act",
    f"{PUBLICATION}/synthetic-harbor-model-legislation/",
    date(2026, 2, 3),
)
SHELF = Listed(
    "synthetic-shelf", "The Synthetic Shelf Act", f"{PUBLICATION}/the-synthetic-shelf-act/", None
)
PACKAGE_PAGE = f"{EPPC}/wp-content/uploads/2026/04/Synthetic-Harbor-Package-Model-Legislation.pdf"
PACKAGE = Listed(
    "synthetic-harbor-package",
    "Synthetic Harbor Package Model Legislation",
    PACKAGE_PAGE,
    None,
    file=on_media_host(PACKAGE_PAGE),
)
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
LISTED = (LANTERN, HARBOR, SHELF, PACKAGE, WITHHELD, MISSING, UNLINKED)
LANTERN_FILE = f"{MEDIA}/2026/01/synthetic-lantern-bill_web.pdf"
SHELF_FILE = f"{MEDIA}/2026/08/1-26-Synthetic-Shelf-Act.pdf"
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
SHELF_CLEANED = (
    "SECTION A\n"
    "(1) Each school in [state] shall keep a shelf of printed books.\n"
    "(2) By [date specified] the board shall report on the shelves.\n"
    "SECTION B\n"
    "The [Chief State School Officer] may adopt rules under this act.\n"
    "DEFINITIONS\n"
    '1) "Shelf" means a fixed rack for books.\n'
)
PACKAGE_CLEANED = (
    "Section One: Findings\n"
    "The [State Body] finds that harbors need lights.\n"
    "Section Two: Prohibition Section\n"
    "Section Three: Reporting\n"
    "A. Each harbor shall report its lights each year.\n"
    "Lighthouse Keeper Protection Act\n"
    "Section One: Definitions\n"
    '- Keeper.--The term "keeper" means a person who tends a light.\n'
    "Section Two: Severability Clause\n"
    "If any provision of this Act is held invalid, the rest stands.\n"
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

    assert (report.found, report.collected, report.unchanged) == (7, 4, 0)
    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}
    assert sorted(docs) == [
        "eppc/synthetic-harbor",
        "eppc/synthetic-harbor-package",
        "eppc/synthetic-lantern",
        "eppc/synthetic-shelf",
    ]

    lantern = docs["eppc/synthetic-lantern"]
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
    assert lantern.co_publishers == ("Example Family Institute",)
    assert lantern.collected_on == DAY
    assert sha256(raw_path(lantern, root).read_bytes()) == lantern.raw_sha256
    assert raw_path(lantern, root).read_bytes() == site.fetch(LANTERN_FILE, 1.0).body
    assert load_text(lantern, root).split() == LANTERN_CLEANED.split()

    harbor = docs["eppc/synthetic-harbor"]
    assert (harbor.media_type, harbor.url, harbor.co_publishers) == ("text/html", HARBOR.page, ())
    assert load_text(harbor, root) == HARBOR_CLEANED

    shelf = docs["eppc/synthetic-shelf"]
    assert (shelf.media_type, shelf.published) == ("application/pdf", None)
    assert load_text(shelf, root).split() == SHELF_CLEANED.split()

    package = docs["eppc/synthetic-harbor-package"]
    assert (package.url, package.media_type) == (PACKAGE_PAGE, "application/pdf")
    assert load_text(package, root).split() == PACKAGE_CLEANED.split()

    assert [(v.doc_id, v.first_seen, v.replaced_on) for v in read_versions(SOURCE, root)] == [
        ("eppc/synthetic-harbor", DAY, None),
        ("eppc/synthetic-harbor-package", DAY, None),
        ("eppc/synthetic-lantern", DAY, None),
        ("eppc/synthetic-shelf", DAY, None),
    ]


def test_only_the_model_text_is_asked_for_and_each_host_paced(tmp_path: Path) -> None:
    site = recorded()
    _run(tmp_path, site)

    asked = [url for url, _ in site.requests]
    # EPPC's upload URLs are asked for at the host that serves them; a recorded file is asked
    # for without its page; summaries, briefs, listings and another publisher's file never are.
    assert {LANTERN_FILE, SHELF_FILE, on_media_host(PACKAGE_PAGE)} <= set(asked)
    assert not any("/wp-content/uploads/" in url for url in asked)
    never = re.compile(r"[Ss]ummary|Policy-Brief|/category/|example\.org")
    assert not any(never.search(url) for url in asked)
    assert asked.count(SHELF_FILE) == 1
    assert all(interval >= 1.0 for _, interval in site.requests)
    assert sorted(site.robots_reads) == [("https", "eppc.org"), ("https", "media.eppc.org")]


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert (again.found, again.collected, again.unchanged) == (7, 0, 4)
    assert (again.blocked, again.failed) == (first.blocked, first.failed)


def test_a_revised_text_becomes_a_new_version(tmp_path: Path) -> None:
    _run(tmp_path, recorded())
    site = recorded()
    revised = (FIXTURES / "synthetic_lettered_bill.txt").read_text().replace("shelf", "rack")
    site.overrides[SHELF_FILE] = pdf_bytes(revised)
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (1, 3)
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "eppc/synthetic-shelf"
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
    assert "eppc/synthetic-withheld" not in stored


def test_a_failed_fetch_or_a_page_without_the_text_is_reported_failed(tmp_path: Path) -> None:
    site = recorded()
    site.overrides[SHELF_FILE] = b"not a PDF"
    report = _run(tmp_path, site)

    failed = {f.url: f.reason for f in report.failed}
    assert sorted(failed) == sorted([MISSING.page, SHELF.page, UNLINKED.page])
    assert failed[MISSING.page] == f"{MISSING.page}: HTTP 404"
    assert failed[UNLINKED.page] == "no link to the model legislation on its page"
    assert failed[SHELF.page].startswith(f"{SHELF_FILE}: unreadable PDF")
    assert (report.found, report.collected) == (7, 3)


def test_with_fetching_off_each_document_is_reported_and_nothing_asked(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site, on_site=False)

    assert (report.found, report.collected, len(report.failed)) == (7, 0, 7)
    assert all("fetching from the site is off" in f.reason for f in report.failed)
    assert (site.requests, site.robots_reads) == ([], [])


# ---- the model text's link and its cleaning -------------------------------------------------


def _page(url: str, links: str) -> Fetched:
    return Fetched(url, "text/html", f"<html><body>{links}</body></html>".encode())


def test_the_link_naming_the_model_legislation_wins_over_summaries_and_tags() -> None:
    page = _page(
        f"{PUBLICATION}/x/",
        '<a href="/wp-content/uploads/x-summary_web.pdf">PDF</a>'
        '<a href="/category/model-legislation/">Model Legislation</a>'
        '<a href="https://example.org/model.pdf">Model Legislation</a>'
        '<a href="https://ifstudies.org/model.pdf">Model Legislation</a>'
        '<a href="/the-x-act/">View Model Legislation</a>'
        '<a href="/wp-content/uploads/x-bill.pdf">Model Legislation</a>'
        '<a href="/wp-content/uploads/x-bill.docx">Model Legislation (Word)</a>'
        '<a href="/wp-content/uploads/x-bill-summary.pdf">Summary</a>',
    )
    # Word before PDF before a page; another organization's host is not this source's.
    assert model_text_link(page) == f"{EPPC}/wp-content/uploads/x-bill.docx"


def test_a_bare_pdf_button_is_the_text_when_no_link_names_it() -> None:
    page = _page(
        f"{PUBLICATION}/books/",
        '<a href="https://media.eppc.org/2026/08/1-26-X-Act.pdf">View PDF</a>'
        '<a href="/wp-content/uploads/report.pdf">report</a>',
    )
    assert model_text_link(page) == f"{MEDIA}/2026/08/1-26-X-Act.pdf"
    assert model_text_link(_page(f"{EPPC}/p/", '<a href="/x.pdf">Executive Summary</a>')) is None


def test_eppc_uploads_are_asked_for_at_the_media_host() -> None:
    assert on_media_host(f"{EPPC}/wp-content/uploads/2023/12/a_web.pdf") == (
        f"{MEDIA}/2023/12/a_web.pdf"
    )
    other = "https://ifstudies.org/ifs-admin/resources/a.pdf"
    assert on_media_host(other) == other


def test_cleaning_keeps_the_sections_and_drops_commentary_notes_and_furniture() -> None:
    assert clean_model_text((FIXTURES / "synthetic_bill.txt").read_text()) == LANTERN_CLEANED
    assert clean_model_text(LANTERN_CLEANED) == LANTERN_CLEANED


def test_cleaning_reads_lettered_and_spelled_sections() -> None:
    # A preamble before the first section, a program banner, footers and biographies without
    # a heading go; so do an overview and an item left empty by a dropped instruction.
    lettered = (FIXTURES / "synthetic_lettered_bill.txt").read_text()
    assert clean_model_text(lettered) == SHELF_CLEANED
    assert clean_model_text((FIXTURES / "synthetic_package.txt").read_text()) == PACKAGE_CLEANED


@pytest.mark.parametrize(
    ("line", "heading"),
    [
        ("Section 101. Definitions.", True),
        ("SECTION A", True),
        ("Section B.", True),
        ("Section One: Findings", True),
        ("[Section 4. Optional Fees", True),
        ("Part 2. Enforcement", True),
        ("Section a person who", False),
        ("Section Agreements", False),
        ("Section 1. Short Title\t1", False),  # a contents entry
        ("Sections of this act", False),
    ],
)
def test_what_reads_as_a_heading(line: str, heading: bool) -> None:
    assert is_heading(line) is heading


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
        # An office's sentence is law, not a biography.
        ("Section 1. A.\nThe Secretary is the director of the fund.\n", None),
    ],
)
def test_cleaning_cases(given: str, cleaned: str | None) -> None:
    assert clean_model_text(given) == (cleaned or given)


def test_a_text_without_a_section_is_not_a_model() -> None:
    with pytest.raises(ModelTextError, match="no part or section heading"):
        clean_model_text("An invented policy brief.\nIt proposes ideas.\n")


# ---- the source and the command -------------------------------------------------------------


def test_the_source_and_what_it_collects() -> None:
    assert collectors()["eppc"].source == SOURCE
    assert (SOURCE.key, SOURCE.organization) == ("eppc", "Ethics and Public Policy Center")
    assert not SOURCE.leaked
    assert (SOURCE.members, SOURCE.members_source) == (None, None)
    assert "exempt under IRS section 501(c)(3)" in SOURCE.descriptor
    assert SOURCE.library_url == f"{EPPC}/category/model-legislation/"
    assert SOURCE.terms_read_on == DAY
    assert SITE_FETCH
    assert {d.slug: d.co_publishers for d in DOCUMENTS} == {
        "age-verification-of-social-media": ("Institute for Family Studies",),
        "age-verification-for-websites-containing-obscene-and-indecent-material": (
            "Center for Renewing America",
        ),
        "safe-chat-act": (),
        "books-act": (),
        "state-assisted-reproductive-technology-model-legislation": (),
    }
    # A recorded file is on the host that serves it, never behind a redirect.
    assert all(d.file is None or d.file.startswith(f"{MEDIA}/") for d in DOCUMENTS)
    assert all(d.page.startswith(f"{EPPC}/") for d in DOCUMENTS)


def test_the_command_collects_through_the_recorded_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = recorded()
    monkeypatch.setattr(eppc, "DOCUMENTS", LISTED)
    monkeypatch.setattr(eppc, "fetch_url", site.fetch)
    monkeypatch.setattr(eppc, "read_robots", site.read_robots)
    args = ["collect-models", "eppc", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "7 found: 4 collected, 0 unchanged, 1 blocked by robots.txt, 2 failed" in first.output
    report = json.loads((tmp_path / "reports" / "eppc.json").read_text())
    assert str(tmp_path) not in json.dumps(report)
    again = CliRunner().invoke(app, args)
    assert "7 found: 0 collected, 4 unchanged" in again.output


# ---- real samples, local only ---------------------------------------------------------------

_INSTRUCTION_LEFT = re.compile(r"\[\s*(?:cite|insert|specify|enter)\b", re.IGNORECASE)
_FURNITURE_LEFT = re.compile(
    r"^\s*\d+\s+(?:ethics and public policy center|institute for family studies)", re.I | re.M
)
_BACK_MATTER_LEFT = re.compile(r"^\s*(?:about the authors?|endorsements)\b", re.I | re.M)


def _collected() -> list[str]:
    return [d.id for d in read_docs(SOURCE) if raw_path(d).exists()]


@pytest.mark.skipif(not _collected(), reason="no eppc documents under the data directory")
@pytest.mark.parametrize("doc_id", _collected())
def test_a_collected_model_keeps_its_sections_and_drops_the_rest(doc_id: str) -> None:
    doc = next(d for d in read_docs(SOURCE) if d.id == doc_id)
    raw = raw_path(doc).read_bytes()
    extracted = document_text(Fetched(doc.url, doc.media_type, raw), doc.media_type)
    cleaned = clean_model_text(extracted)

    # Counts only, so a failure never prints the model's text.
    stored_is_current = load_text(doc) == cleaned
    headings = sum(map(is_heading, cleaned.split("\n")))
    instructions_left = len(_INSTRUCTION_LEFT.findall(cleaned))
    furniture_left = len(_FURNITURE_LEFT.findall(cleaned))
    back_matter_left = len(_BACK_MATTER_LEFT.findall(cleaned))
    kept, before = len(cleaned.split()), len(extracted.split())
    settled = clean_model_text(cleaned) == cleaned

    assert stored_is_current
    assert headings > 0
    assert (instructions_left, furniture_left, back_matter_left) == (0, 0, 0)
    # The cover, preamble, footers, footnotes and biographies are what goes: on the bills as a
    # reader downloads them (2026-11-15), 55% of words kept (the BOOKS Act's preamble of
    # quotations and its biographies) to 99%.
    assert 0.5 * before <= kept <= before
    assert settled
