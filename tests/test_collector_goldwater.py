"""The Goldwater Institute's collector (DESIGN.md#model-documents): the handbook page's model
legislation links, each fetched, cleaned and stored with provenance, against recorded responses
(tests/fixtures/collect/goldwater/), never the network. Every text here is synthetic; no
institute text is committed.

The real-sample tests read the documents collected under the data directory (LINEAGE_DATA) and
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
from lineage.collectors import goldwater
from lineage.collectors.goldwater import (
    DOCX,
    LIBRARY,
    SOURCE,
    GoldwaterTextError,
    Listed,
    clean_text,
    collect_from,
    doc_slug,
    file_text,
    listed,
    own_date,
    own_title,
    qualifier,
    title_for,
)
from lineage.core.robots import Robots, parse_robots
from lineage.modeldocs import (
    MODELDOCS_DIR,
    load_text,
    raw_path,
    read_docs,
    read_versions,
    sha256,
)
from tests.recorded_site import docx_bytes, pdf_bytes

FIXTURES = Path(__file__).parent / "fixtures" / "collect" / "goldwater"
DAY = date(2026, 11, 15)
UPLOADS = "https://www.goldwaterinstitute.org/wp-content/uploads"
LANTERN = f"{UPLOADS}/2026/11/Lantern_Permit_Act-2026.pdf"
TIDE = f"{UPLOADS}/2026/11/Harbor_Fairness_Act-2026.pdf"
MOORING = f"{UPLOADS}/2026/11/Harbor_Fairness_Act-2025.docx"
FAQ = f"{UPLOADS}/2025/09/3.0-FAQ-and-Bill.pdf"
PACKAGE = f"{UPLOADS}/2026/11/Lamp_Model_Package-2026.pdf"
WITHHELD = f"{UPLOADS}/withheld/Withheld_Lamp_Act-2026.pdf"
LOST = f"{UPLOADS}/2026/11/Lost_Lamp_Act-2026.pdf"

LANTERN_CLEANED = (
    "AN ACT relating to fair permits for lanterns.\n"
    "\n"
    "Section 1. Definitions\n"
    '(A) "Agency" means any office of [STATE] that issues lantern permits.\n'
    '(B) "Permit" means a written leave to light a lantern under.\n'
    "\n"
    "Section 2. Permit Timelines\n"
    "(A) Within ten (10) business days an agency shall decide each application.\n"
    "\n"
    "(B) An application not decided in time is deemed approved for Fiscal Year [20XX].\n"
    "\n"
    "Section 3. Review\n"
    "A person aggrieved by a violation of this Act may bring an action.\n"
)


@dataclass
class Site:
    """The site as site.json records it; every request is logged."""

    responses: dict[str, dict[str, str]]
    robots_body: bytes
    requests: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    robots_reads: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    overrides: dict[str, bytes] = field(default_factory=dict[str, bytes])

    def fetch(self, url: str, interval: float) -> Fetched:
        self.requests.append((url, interval))
        if url in self.overrides:
            return Fetched(url, "application/octet-stream", self.overrides[url])
        entry = self.responses.get(url)
        if entry is None:
            raise CollectFetchError("HTTP 404")
        if "body" in entry:
            return Fetched(url, entry["content_type"], entry["body"].encode())
        text = (FIXTURES / entry["file"]).read_text()
        body = {"docx": docx_bytes, "pdf": pdf_bytes}[entry["build"]](text)
        return Fetched(url, entry["content_type"], body)

    def read_robots(self, scheme: str, host: str) -> Robots:
        self.robots_reads.append((scheme, host))
        url = f"{scheme}://{host}/robots.txt"
        return parse_robots(url, 200, self.robots_body, "2026-11-15T00:00:00+00:00")


def recorded() -> Site:
    responses = json.loads((FIXTURES / "site.json").read_text())
    responses.pop("_note")
    return Site(responses, (FIXTURES / "robots.txt").read_bytes())


def _run(tmp_path: Path, site: Site, *, on: date = DAY, on_site: bool = True) -> CollectReport:
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


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---- collecting ------------------------------------------------------------------------------


def test_every_model_bill_the_page_links_is_found_and_settled(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site)

    assert (report.found, report.collected, report.unchanged) == (7, 5, 0)
    assert report.blocked == [WITHHELD]
    assert [(f.url, f.reason) for f in report.failed] == [(LOST, "HTTP 404")]

    asked = [url for url, _ in site.requests]
    # The page, then each document once (the package's two links are one file). robots.txt
    # first: the disallowed file is never asked for. The handbook, a study in a footnote, the
    # nav and a legislature's bill on another host are not documents.
    assert asked == [LIBRARY, LANTERN, TIDE, MOORING, FAQ, PACKAGE, LOST]
    assert site.robots_reads == [("https", "www.goldwaterinstitute.org")]
    assert all(interval >= 1.0 for _, interval in site.requests)


def test_each_document_is_stored_with_provenance(tmp_path: Path) -> None:
    site = recorded()
    _run(tmp_path, site)
    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}

    assert sorted(docs) == [
        "goldwater/3-0-faq-and-bill",
        "goldwater/harbor-fairness-act-mooring",
        "goldwater/harbor-fairness-act-tide",
        "goldwater/lamp-model-package",
        "goldwater/lantern-permit-act",
    ]
    titles = {d.id.partition("/")[2]: (d.title, d.media_type, d.published) for d in docs.values()}
    assert titles == {
        "3-0-faq-and-bill": ("The Lamp Access for Everyone Act", "application/pdf", None),
        "harbor-fairness-act-mooring": ("Harbor Fairness Act (Mooring)", DOCX, None),
        "harbor-fairness-act-tide": ("Harbor Fairness Act (Tide)", "application/pdf", None),
        "lamp-model-package": ("Lamp Model Package", "application/pdf", None),
        "lantern-permit-act": ("Lantern Permit Act", "application/pdf", date(2026, 9, 2)),
    }
    lantern = docs["goldwater/lantern-permit-act"]
    fetched = site.fetch(LANTERN, 1.0).body
    assert lantern.url == LANTERN
    assert (lantern.robots, lantern.leaked, lantern.collected_on) == ("allowed", False, DAY)
    assert lantern.raw_sha256 == sha256(fetched)
    assert raw_path(lantern, root).read_bytes() == fetched
    # The PDF's text layer is read as lineage.extract reads it; words, not layout, are compared.
    assert load_text(lantern, root).split() == LANTERN_CLEANED.split()

    mooring = docs["goldwater/harbor-fairness-act-mooring"]
    assert load_text(mooring, root) == clean_text(_fixture("harbor_mooring.txt"))
    faq = load_text(docs["goldwater/3-0-faq-and-bill"], root)
    assert faq.startswith("Section 1.")
    assert "?" not in faq


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert (again.found, again.collected, again.unchanged) == (7, 0, first.collected)
    assert (again.blocked, again.failed) == (first.blocked, first.failed)


def test_a_new_edition_of_a_file_becomes_a_new_version(tmp_path: Path) -> None:
    _run(tmp_path, recorded())
    site = recorded()
    site.overrides[MOORING] = docx_bytes(_fixture("harbor_mooring.txt").replace("worth", "value"))
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (1, 4)
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "goldwater/harbor-fairness-act-mooring"
    ]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [
        (DAY, date(2026, 12, 1)),
        (date(2026, 12, 1), None),
    ]


def test_an_unreadable_file_or_one_without_a_bill_fails_that_document_only(
    tmp_path: Path,
) -> None:
    site = recorded()
    site.overrides[MOORING] = b"not a zip"
    site.overrides[TIDE] = docx_bytes("Why lanterns?\nBecause.")  # a .pdf name, Word bytes
    site.overrides[FAQ] = pdf_bytes("Lamp Access Act FAQ\nWhy is it needed?\nBecause.")
    report = _run(tmp_path, site)

    assert (report.found, report.collected) == (7, 2)
    reasons = {f.url: f.reason for f in report.failed}
    assert "unreadable Word document" in reasons[MOORING]
    assert "unreadable PDF" in reasons[TIDE]
    assert "no enacting text" in reasons[FAQ]
    assert reasons[LOST] == "HTTP 404"


def test_a_page_robots_txt_disallows_is_reported_blocked_and_never_fetched(
    tmp_path: Path,
) -> None:
    site = recorded()
    site.robots_body = b"User-agent: *\nDisallow: /the-freedom-agenda\n"
    report = _run(tmp_path, site)

    assert (report.found, report.blocked, report.failed) == (1, [LIBRARY], [])
    assert site.requests == []


def test_a_failed_page_is_reported_failed(tmp_path: Path) -> None:
    site = recorded()
    del site.responses[LIBRARY]
    report = _run(tmp_path, site)

    assert [(f.url, f.reason) for f in report.failed] == [(LIBRARY, "HTTP 404")]
    assert report.found == 1


def test_a_page_whose_layout_lost_its_links_is_reported_not_passed_over(tmp_path: Path) -> None:
    site = recorded()
    site.overrides[LIBRARY] = b"<html><body><p>Redesigned.</p></body></html>"
    report = _run(tmp_path, site)

    assert report.found == 1
    assert "no model legislation links" in report.failed[0].reason


def test_with_fetching_off_nothing_is_asked(tmp_path: Path) -> None:
    site = recorded()
    report = _run(tmp_path, site, on_site=False)

    assert (site.requests, site.robots_reads) == ([], [])
    assert [f.url for f in report.failed] == [LIBRARY]


def test_the_command_collects_and_writes_a_report_without_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = recorded()
    monkeypatch.setattr(goldwater, "fetch_url", site.fetch)
    monkeypatch.setattr(goldwater, "read_robots", site.read_robots)
    monkeypatch.setattr(goldwater, "SITE_FETCH", True)
    args = ["collect-models", "goldwater", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "7 found: 5 collected, 0 unchanged, 1 blocked by robots.txt, 1 failed" in first.output
    report = json.loads((tmp_path / "reports" / "goldwater.json").read_text())
    assert str(tmp_path) not in json.dumps(report)
    again = CliRunner().invoke(app, args)
    assert "7 found: 0 collected, 5 unchanged" in again.output


# ---- the page --------------------------------------------------------------------------------


def _page(body: str) -> Fetched:
    return Fetched(LIBRARY, "text/html", f"<html><body>{body}</body></html>".encode())


def test_a_link_counts_by_its_own_words_or_its_short_label() -> None:
    page = _page(
        '<p><a href="/wp-content/uploads/a.pdf">View model legislation</a></p>'
        '<p>VIEW <a href="/wp-content/uploads/b.pdf">EMINENT DOMAIN</a> MODEL LEGISLATION</p>'
        "<p>This long sentence says our handbook provides model legislation for states to use, "
        'with research. <a href="/wp-content/uploads/handbook.pdf">Download</a></p>'
        '<p>VIEW MODEL LEGISLATION <a href="/policy-reports/">reports</a></p>'
        '<p><a href="https://goldwaterinstitute.org/wp-content/uploads/c.pdf">MODEL BILL</a></p>'
        '<p><a href="https://www.legislature.example.gov/x.pdf">View model legislation</a></p>'
    )
    assert [(link.url.rpartition("/")[2], link.text) for link in listed(page)] == [
        ("a.pdf", "View model legislation"),
        ("b.pdf", "EMINENT DOMAIN"),
        ("c.pdf", "MODEL BILL"),
    ]


@pytest.mark.parametrize(
    ("link_text", "expected"),
    [
        ("VIEW MODEL LEGISLATION.", ""),
        ("VIEW EMINENT DOMAIN MODEL LEGISLATION", "eminent domain"),
        ("REGULATORY TAKINGS", "regulatory takings"),
        ("Read the model bill", ""),
    ],
)
def test_a_link_qualifies_its_document_only_with_words_beyond_its_label(
    link_text: str, expected: str
) -> None:
    assert qualifier(link_text) == expected


@pytest.mark.parametrize(
    ("file", "slug"),
    [
        ("Permit-Freedom-Act-2026.pdf", "permit-freedom-act"),
        ("Home-Based_Business_Fairness_Act-2026.pdf", "home-based-business-fairness-act"),
        ("Freedom-From-Indoctrination-Act-Model-Policy.pdf", "freedom-from-indoctrination-act"),
        ("Lamp%20Act%20FINAL.docx", "lamp-act"),
        ("2.0-FAQ-and-Bill.pdf", "2-0-faq-and-bill"),
    ],
)
def test_a_document_is_named_by_its_file_without_the_edition(file: str, slug: str) -> None:
    assert doc_slug(Listed(f"{UPLOADS}/2026/11/{file}", "VIEW MODEL LEGISLATION")) == slug


def test_a_title_is_the_documents_own_else_its_files() -> None:
    plain = Listed(LANTERN, "VIEW MODEL LEGISLATION.")
    assert title_for(plain, _fixture("lantern_permit.txt")) == "Lantern Permit Act"
    package = Listed(PACKAGE, "VIEW MODEL LEGISLATION.")
    assert title_for(package, _fixture("lamp_package.txt")) == "Lamp Model Package"
    two = Listed(TIDE, "VIEW TIDE MODEL LEGISLATION")
    assert title_for(two, _fixture("harbor_tide.txt")) == "Harbor Fairness Act (Tide)"


# ---- cleaning --------------------------------------------------------------------------------


def test_a_bill_keeps_its_sections_and_loses_notes_commentary_and_instructions() -> None:
    assert clean_text(_fixture("lantern_permit.txt")) == LANTERN_CLEANED


def test_an_faq_before_the_bill_goes() -> None:
    assert clean_text(_fixture("lamp_faq_and_bill.txt")) == (
        "Section 1.\n"
        '(1) As used in this act, "lamp" means an invented light.\n'
        "Section 2.\n"
        "(1) A keeper may lend a lamp to any qualified person.\n"
    )


def test_a_package_keeps_each_bill_and_drops_its_proposal_labels() -> None:
    cleaned = clean_text(_fixture("lamp_package.txt"))
    assert "Proposal" not in cleaned
    assert cleaned.count("AN ACT relating to") == 2
    # Placeholders a state fills in as written stay.
    assert "[STATE]" in cleaned
    assert "[Lantern Office at Example Hall A.]" in cleaned


@pytest.mark.parametrize(
    "name", ["lantern_permit.txt", "lamp_faq_and_bill.txt", "lamp_package.txt", "harbor_tide.txt"]
)
def test_cleaning_settles(name: str) -> None:
    once = clean_text(_fixture(name))
    assert clean_text(once) == once


def test_a_text_without_a_bill_is_refused() -> None:
    with pytest.raises(GoldwaterTextError, match="no enacting text"):
        clean_text("Lamp Act FAQ\nWhy?\nBecause.\n")


def test_a_full_date_above_the_bill_is_the_documents_date() -> None:
    assert own_date(_fixture("lantern_permit.txt")) == date(2026, 9, 2)
    assert own_date(_fixture("lamp_package.txt")) is None
    # A date inside the bill is the law's, and one in an FAQ's sentence the story's.
    assert own_date("Lamp Act\nAN ACT relating to lamps.\nOn January 1, 2027, it applies.") is None
    assert own_date("Since May 1, 2022, one state has lit lamps.\nSection 1.\nA lamp.") is None
    assert own_title("Lamp Act\nRevised May 1, 2026\nAN ACT relating to lamps.") is None


# ---- the source ------------------------------------------------------------------------------


def test_the_source_entry_is_complete_and_carries_no_political_label() -> None:
    assert collectors()["goldwater"].source == SOURCE
    assert SOURCE.library_url == LIBRARY
    assert "501(c)(3)" in SOURCE.descriptor
    assert SOURCE.terms_url == "https://www.goldwaterinstitute.org/terms-and-conditions/"
    assert SOURCE.terms_read_on == DAY
    assert (SOURCE.members, SOURCE.members_source, SOURCE.leaked) == (None, None, False)
    words = f"{SOURCE.organization} {SOURCE.descriptor} {SOURCE.terms_note}".lower()
    labels = ["conservative", "liberal", "libertarian", "progressive", "free-market", "right-"]
    assert not [label for label in labels if label in words]


# ---- real samples, local only ----------------------------------------------------------------

_INSTRUCTION_LEFT = re.compile(r"\[\s*(?:cite|insert|specify|appropriate)\b", re.IGNORECASE)
_NOTE_LINE = re.compile(
    r"^\s*(?:(?:drafting|legislative)\s+notes?\b|commentary\b|proposal\s+\d+\s*:)",
    re.IGNORECASE | re.MULTILINE,
)


def _collected() -> list[str]:
    return [d.id for d in read_docs(SOURCE, MODELDOCS_DIR)]


@pytest.mark.skipif(not _collected(), reason="no goldwater documents under the data directory")
@pytest.mark.parametrize("doc_id", _collected())
def test_a_collected_document_reads_as_a_bill(doc_id: str) -> None:
    doc = next(d for d in read_docs(SOURCE, MODELDOCS_DIR) if d.id == doc_id)
    text = load_text(doc, MODELDOCS_DIR)
    extracted = file_text(raw_path(doc, MODELDOCS_DIR).read_bytes(), doc.media_type)
    lines = [line for line in text.split("\n") if line.strip()]

    # Counts and flags only, so a failure never prints the document's text.
    reproduced = clean_text(extracted) == text
    settled = clean_text(text) == text
    starts_as_a_bill = goldwater.act_start(lines[:1]) == 0
    questions = sum(1 for line in lines if line.rstrip().endswith("?"))
    instructions_left = len(_INSTRUCTION_LEFT.findall(text))
    note_lines = len(_NOTE_LINE.findall(text))
    kept, before = len(text.split()), len(extracted.split())

    assert reproduced
    assert settled
    assert starts_as_a_bill
    assert questions == 0
    assert instructions_left == 0
    assert note_lines == 0
    assert 100 <= kept <= before
