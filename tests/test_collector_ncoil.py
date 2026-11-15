"""NCOIL's collector (DESIGN.md#model-documents): the model laws library paged through, each
model's file from its document page, against recorded responses (tests/fixtures/collect/ncoil/),
never the network. Every page, title and model text there is synthetic.

The real-sample tests read the collected texts from the local data directory (LINEAGE_DATA) and
are skipped until integration has collected them; they assert counts only, never print text."""

import itertools
import json
import re
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.collect import CollectContext, CollectReport, collectors
from lineage.collect_base import Fetched
from lineage.collectors import ncoil
from lineage.collectors.ncoil import (
    DOCX,
    LIBRARY,
    SOURCE,
    NcoilTextError,
    adopted_on,
    clean_model,
    collect_from,
    file_text,
    library_entries,
    model_file,
    model_name,
)
from lineage.extract import pdf_text
from lineage.modeldocs import (
    MODELDOCS_DIR,
    load_text,
    raw_path,
    read_docs,
    read_versions,
    sha256,
)
from lineage.modeltext import docx_text
from tests.recorded_site import COLLECT, RecordedSite, docx_bytes, pdf_bytes

DAY = date(2026, 11, 15)
SITE = "https://ncoil.org"
EN = "\u2013"  # the en dash NCOIL's titles set before their adoption history
FIXTURES = COLLECT / "ncoil"
SYNTHETIC_MODEL = FIXTURES / "synthetic_model.txt"
UPLOADS = f"{SITE}/wp-content/uploads"
LANTERN = f"{SITE}/resource/synthetic-lantern-coverage-model-act-adopted-1-2-26/"
PEWTER = f"{SITE}/resource/ncoil-pewter-lamp-model-act/"
WICK = f"{SITE}/resource/ncoil-wick-trimming-model-act/"
TALLOW = f"{SITE}/resource/ncoil-tallow-candle-model-act/"
TORCH = (
    f"{SITE}/resource/torch-model-act-originally-adopted-11-18-12-readopted-7-16-22-amended-on-"
    "7-19-25/"
)
WITHHELD = f"{SITE}/resource/withheld-harbor-lights-model-act/"
WICK_FILE = f"{UPLOADS}/2026/02/NCOIL-Wick-Trimming-Model-November-2025.pdf"

CLEANED = (
    "Section 1. Title and Purpose\n"
    "A. This Act shall be known as the Synthetic Lantern Coverage Act.\n"
    "\n"
    "Section 2. Definitions\n"
    "As used in this Act:\n"
    'A. "Commissioner" means the commissioner of insurance of this state.\n'
    'B. "Lantern" means a portable lamp insured under.\n'
    "(1) A lantern kept under is covered.\n"
    "(2) A lantern kept in [this state] is covered.\n"
    "\n"
    "Section 3. Lantern Coverage\n"
    "An insurer [shall][may] cover a lantern for [60] days.\n"
    "\n"
    "The commissioner may adopt rules to carry out this section.\n"
    "\n"
    "Section 4. Severability\n"
    "If any provision of this Act is held invalid, the remainder is not affected.\n"
)


def recorded_site() -> RecordedSite:
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


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _page(url: str, site: RecordedSite) -> Fetched:
    return Fetched(url, "text/html", site.responses[url]["body"].encode())


# ---- the collector, against the recorded site --------------------------------------------------


def test_every_model_in_the_library_is_found_collected_and_stored_with_provenance(
    tmp_path: Path,
) -> None:
    site = recorded_site()
    report = _run(tmp_path, site)

    # Six models listed over two pages (the draft passed over, the repeat counted once): three
    # collected, one blocked by robots.txt, one without a file and one whose file is missing.
    assert (report.source, report.run_on) == ("ncoil", DAY)
    assert (report.found, report.collected, report.unchanged) == (6, 3, 0)
    assert report.blocked == [WITHHELD]
    failed = {f.url: f.reason for f in report.failed}
    assert failed == {
        TALLOW: "no PDF or Word file on the document's page",
        TORCH: f"{UPLOADS}/2025/08/Torch-Model-Amended-7-19-25.pdf: HTTP 404",
    }

    root = tmp_path / "modeldocs"
    docs = {d.id: d for d in read_docs(SOURCE, root)}
    assert sorted(docs) == [
        "ncoil/ncoil-pewter-lamp-model-act",
        "ncoil/ncoil-synthetic-lantern-coverage-model-act",
        "ncoil/ncoil-wick-trimming-model-act",
    ]
    lantern = docs["ncoil/ncoil-synthetic-lantern-coverage-model-act"]
    assert lantern.title == "NCOIL Synthetic Lantern Coverage Model Act"
    assert lantern.url == LANTERN
    assert (lantern.media_type, lantern.robots) == ("application/pdf", "allowed")
    assert not lantern.leaked
    assert (lantern.collected_on, lantern.published) == (DAY, date(2026, 1, 2))
    raw = pdf_bytes(SYNTHETIC_MODEL.read_text())
    assert lantern.raw_sha256 == sha256(raw)
    assert raw_path(lantern, root).read_bytes() == raw
    assert load_text(lantern, root) == clean_model(pdf_text(raw))

    pewter = docs["ncoil/ncoil-pewter-lamp-model-act"]
    assert (pewter.media_type, pewter.url) == (DOCX, PEWTER)
    # A month and year in the title takes its day from the listing date in that month.
    assert pewter.published == date(2025, 11, 15)
    assert load_text(pewter, root) == CLEANED
    # The latest readoption's month is not the listing date's, so no day is guessed.
    assert docs["ncoil/ncoil-wick-trimming-model-act"].published is None


def test_robots_txt_is_read_once_and_a_disallowed_page_is_never_asked_for(
    tmp_path: Path,
) -> None:
    site = recorded_site()
    _run(tmp_path, site)

    asked = [url for url, _ in site.requests]
    assert site.robots_reads == [("https", "ncoil.org")]
    assert WITHHELD not in asked
    assert not any("beacon" in url for url in asked)  # the draft is passed over
    # Paging stops at the page that lists nothing new: past its end the library repeats page 1.
    assert [u for u in asked if u.startswith(LIBRARY)] == [
        LIBRARY,
        f"{LIBRARY}?_paged=2",
        f"{LIBRARY}?_paged=3",
    ]
    assert asked.count(LANTERN) == 1
    assert all(interval >= 1.0 for _, interval in site.requests)


def test_a_rerun_changes_nothing(tmp_path: Path) -> None:
    first = _run(tmp_path, recorded_site())
    before = _tree(tmp_path / "modeldocs")
    again = _run(tmp_path, recorded_site(), on=date(2026, 11, 16))

    assert _tree(tmp_path / "modeldocs") == before
    assert (again.found, again.collected, again.unchanged) == (6, 0, first.collected)
    assert (again.blocked, again.failed) == (first.blocked, first.failed)


def test_a_readopted_text_becomes_a_new_version_of_the_same_model(tmp_path: Path) -> None:
    _run(tmp_path, recorded_site())
    site = recorded_site()
    site.overrides[WICK_FILE] = pdf_bytes(SYNTHETIC_MODEL.read_text().replace("Lantern", "Torch"))
    report = _run(tmp_path, site, on=date(2026, 12, 1))

    assert (report.collected, report.unchanged) == (1, 2)
    versions = [
        v
        for v in read_versions(SOURCE, tmp_path / "modeldocs")
        if v.doc_id == "ncoil/ncoil-wick-trimming-model-act"
    ]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [
        (DAY, date(2026, 12, 1)),
        (date(2026, 12, 1), None),
    ]


def test_a_failed_library_page_is_reported_and_paging_stops(tmp_path: Path) -> None:
    site = recorded_site()
    del site.responses[f"{LIBRARY}?_paged=2"]
    report = _run(tmp_path, site)

    assert (report.found, report.collected) == (4, 2)
    assert report.blocked == [WITHHELD]
    assert [(f.url, f.reason) for f in report.failed] == [
        (f"{LIBRARY}?_paged=2", f"{LIBRARY}?_paged=2: HTTP 404")
    ]
    assert f"{LIBRARY}?_paged=3" not in [url for url, _ in site.requests]


def test_an_unreadable_file_fails_its_model_only(tmp_path: Path) -> None:
    site = recorded_site()
    site.overrides[WICK_FILE] = b"not a pdf"
    report = _run(tmp_path, site)

    assert report.collected == 2
    assert any(f.url == WICK and "unreadable PDF" in f.reason for f in report.failed)


def test_two_entries_naming_the_same_model_are_not_merged(tmp_path: Path) -> None:
    site = recorded_site()
    page = site.responses[f"{LIBRARY}?_paged=2"]["body"]
    site.overrides[f"{LIBRARY}?_paged=2"] = page.replace(
        "NCOIL Wick Trimming Model Act &#8211;", "NCOIL Pewter Lamp Model Act &#8211;"
    ).encode()
    report = _run(tmp_path, site)

    assert any(
        f.url == WICK and f.reason == f"the same model name as {PEWTER}" for f in report.failed
    )
    assert report.found == 6


def test_with_fetching_off_nothing_is_asked(tmp_path: Path) -> None:
    site = recorded_site()
    report = _run(tmp_path, site, fetching=False)

    assert (report.found, report.collected) == (1, 0)
    assert report.failed[0].url == LIBRARY
    assert site.requests == []
    assert site.robots_reads == []


def test_the_command_collects_and_writes_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = recorded_site()
    monkeypatch.setattr(ncoil, "fetch_url", site.fetch)
    monkeypatch.setattr(ncoil, "read_robots", site.read_robots)
    args = ["collect-models", "ncoil", "--modeldocs", str(tmp_path / "modeldocs"),
            "--reports", str(tmp_path / "reports"), "--min-free-gib", "0"]  # fmt: skip
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "6 found: 3 collected, 0 unchanged, 1 blocked by robots.txt, 2 failed" in first.output
    report = json.loads((tmp_path / "reports" / "ncoil.json").read_text())
    # The report names no local path: it is committed.
    assert str(tmp_path) not in json.dumps(report)
    again = CliRunner().invoke(app, args)
    assert "6 found: 0 collected, 3 unchanged" in again.output


def test_the_source_entry() -> None:
    assert collectors()["ncoil"].source == SOURCE
    assert ncoil.SITE_FETCH
    assert not SOURCE.leaked
    assert SOURCE.leak_note is None
    assert SOURCE.members
    assert SOURCE.members_source == "https://ncoil.org/about/"
    assert SOURCE.terms_read_on == date(2026, 11, 15)
    assert "501(c)(4)" in SOURCE.descriptor
    assert "PENDING" not in SOURCE.terms_note


# ---- the library's pages and titles --------------------------------------------------------------


def test_a_library_page_lists_its_models_once_each_with_their_listing_dates() -> None:
    site = recorded_site()
    entries = library_entries(_page(LIBRARY, site))

    assert [(e.url, e.listed_on) for e in entries] == [
        (LANTERN, date(2026, 1, 1)),
        (PEWTER, date(2025, 11, 15)),
        (WITHHELD, date(2024, 7, 20)),
    ]
    # The card's image link has no text; the title link's is kept.
    assert entries[0].title == f"NCOIL Synthetic Lantern Coverage Model Act {EN} Adopted 1/2/26"


def test_a_document_page_gives_its_file_and_not_a_related_document() -> None:
    site = recorded_site()
    assert model_file(_page(LANTERN, site)) == (
        f"{UPLOADS}/2026/01/Synthetic-Lantern-Model-Adopted-1-2-26.pdf"
    )
    assert model_file(_page(PEWTER, site)) == f"{UPLOADS}/2025/12/NCOIL-Pewter-Lamp-Model.docx"
    assert model_file(_page(TALLOW, site)) is None


@pytest.mark.parametrize(
    ("title", "listed"),
    [
        (f"Beacon Model Act {EN} Draft as of 10/21/2026", False),
        ("Resolution in Support of Lantern Safety", False),
        ("NCOIL Resolution on Wicks", False),
        (f"Lantern Dispute Resolution Model Act {EN} Adopted 1/2/26", True),
    ],
)
def test_drafts_and_resolutions_are_passed_over_and_a_model_on_resolution_is_not(
    title: str, listed: bool
) -> None:
    body = f'<h3><a href="/resource/x/">{title}</a></h3><p>January 1, 2026</p>'
    page = Fetched(LIBRARY, "text/html", body.encode())
    assert bool(library_entries(page)) is listed


@pytest.mark.parametrize(
    ("title", "name"),
    [
        (f"NCOIL Lamp Model Act {EN} Adopted 4-27-25", "NCOIL Lamp Model Act"),
        ("NCOIL Lamp Act; Adopted November, 2021; Readopted temporarily", "NCOIL Lamp Act"),
        (f"Wick Model Act {EN} Originally Adopted 11/17/00; Amended 7/16/04", "Wick Model Act"),
        (
            "Model Wick Act (Compromise Model) Supported 2/27/04; Amended 7/16/22",
            "Model Wick Act (Compromise Model)",
        ),
        ("Candle Model Act - Adopted 7/13/19; Re-adopted 3/6/22", "Candle Model Act"),
        ("Amendments to the Wick Model Act", "Amendments to the Wick Model Act"),
    ],
)
def test_a_models_name_drops_its_adoption_history(title: str, name: str) -> None:
    assert model_name(title) == name


@pytest.mark.parametrize(
    ("title", "listed_on", "adopted"),
    [
        (f"Lamp Model Act {EN} Adopted 4-27-25", None, date(2025, 4, 27)),
        (
            f"Wick Act {EN} Originally Adopted 11/17/00; Readopted 3/9/14, 7/20/24; Readopted with "
            "amendments 11/24/24",
            None,
            date(2024, 11, 24),
        ),
        (f"Fuel Act {EN} Adopted July, 1995; amended 2/26/98", None, date(1998, 2, 26)),
        (f"Lamp Act {EN} Adopted November, 2025", date(2025, 11, 15), date(2025, 11, 15)),
        (f"Lamp Act {EN} Adopted March, 2020; Re-adopted November, 2025", date(2020, 3, 8), None),
        (
            "Lamp Act; Adopted November, 2021; Readopted temporarily until November 2026 Meeting",
            date(2021, 11, 20),
            date(2021, 11, 20),
        ),
        (
            f"Lamp Act {EN} Adopted 2015; Last readopted July 2026 (temporarily until the November "
            "2026 meeting)",
            date(2015, 7, 15),
            None,
        ),
        (f"Lamp Act {EN} Adopted 13/40/25", None, None),
        ("Lamp Act", date(2025, 1, 1), None),
    ],
)
def test_a_models_date_is_its_latest_adoption(
    title: str, listed_on: date | None, adopted: date | None
) -> None:
    assert adopted_on(title, listed_on) == adopted


# ---- cleaning ------------------------------------------------------------------------------------


def test_a_model_keeps_its_sections_and_loses_front_matter_contents_and_notes() -> None:
    assert clean_model(SYNTHETIC_MODEL.read_text()) == CLEANED


def test_a_word_file_reads_as_its_text_does_and_a_pdf_keeps_every_section() -> None:
    text = SYNTHETIC_MODEL.read_text()
    assert clean_model(docx_text(docx_bytes(text))) == CLEANED
    from_pdf = clean_model(file_text(pdf_bytes(text), "application/pdf"))
    # A PDF's text layer has no blank lines, so a note runs to the next numbered item or
    # section; every section and every numbered item is kept.
    assert [line for line in from_pdf.split("\n") if re.match(r"Section|[A-Z]\.|\(\d\)", line)] == [
        line for line in CLEANED.split("\n") if re.match(r"Section|[A-Z]\.|\(\d\)", line)
    ]
    assert "Drafting Note" not in from_pdf
    assert "Table of Contents" not in from_pdf


def test_a_drafting_note_ends_at_the_next_subsection_not_the_next_section() -> None:
    raw = (
        "Section 1. Definitions\n"
        "A. One.\n"
        "Drafting Note: an aside\n"
        "that wraps.\n"
        "B. Two.\n"
        "(1) Three.\n"
        "Legislative Note: another.\n"
        "(a) Four.\n"
    )
    assert clean_model(raw) == "Section 1. Definitions\nA. One.\nB. Two.\n(1) Three.\n(a) Four.\n"


def test_a_text_without_sections_fails() -> None:
    with pytest.raises(NcoilTextError, match="no section heading"):
        clean_model("A memo about a model act.\n")


# ---- real samples, local only --------------------------------------------------------------------

_HEADING_LINE = re.compile(r"^\s*\[?(?:Section|Sec\.)\s+(?:\d+[A-Za-z]?|[IVXL]+)\.", re.I | re.M)
_INSTRUCTION_LEFT = re.compile(r"\[\s*(?:cite|citation|insert|specify)\b", re.IGNORECASE)
_NOTE_LINE = re.compile(r"^\s*(?:legislative|drafting)\s+notes?\b", re.IGNORECASE | re.MULTILINE)


def _collected() -> list[str]:
    try:
        return [d.id for d in read_docs(SOURCE, MODELDOCS_DIR)]
    except (OSError, ValueError):
        return []


@pytest.mark.skipif(not _collected(), reason="no NCOIL models collected under the data directory")
@pytest.mark.parametrize("doc_id", _collected(), ids=lambda i: i.partition("/")[2][:40])
def test_a_collected_model_keeps_its_sections_and_no_note(doc_id: str) -> None:
    doc = next(d for d in read_docs(SOURCE, MODELDOCS_DIR) if d.id == doc_id)
    raw = raw_path(doc, MODELDOCS_DIR).read_bytes()
    read = file_text(raw, doc.media_type)
    text = load_text(doc, MODELDOCS_DIR)

    # Counts only, so a failure never prints the model's text.
    lines = [line for line in text.split("\n") if line.strip()]
    headings = sum(1 for line in lines if _HEADING_LINE.match(line))
    adjacent_headings = sum(
        1 for a, b in itertools.pairwise(lines)
        if _HEADING_LINE.match(a) and _HEADING_LINE.match(b)
    )  # fmt: skip
    instructions_left = len(_INSTRUCTION_LEFT.findall(text))
    note_lines = len(_NOTE_LINE.findall(text))
    contents = text.lower().count("table of contents")
    kept, before = len(text.split()), len(read.split())

    assert clean_model(read) == text
    assert clean_model(text) == text
    assert headings > 0
    assert adjacent_headings == 0  # no table of contents left
    assert instructions_left == 0
    assert note_lines == 0
    assert contents == 0
    # Front matter, contents and notes are a share of a model; losing most of it is losing law.
    assert 0.6 * before <= kept <= before
