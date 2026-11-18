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
from lineage.modeldocs import (
    MODELDOCS_DIR,
    load_text,
    raw_path,
    read_docs,
    read_versions,
    sha256,
)
from lineage.modeltext import docx_text
from lineage.text.extract import pdf_text
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
    "\n"  # a page break, its number and running header gone
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
        (f"Model Agreement Between Lamp Agencies {EN} Supported 7/22/06", False),
        (f"Lantern Dispute Resolution Model Act {EN} Adopted 1/2/26", True),
        (f"Model Act on Lamp Agreements {EN} Adopted 1/2/26", True),
    ],
)
def test_drafts_resolutions_and_agreements_are_passed_over_and_models_naming_them_are_not(
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


def test_a_text_without_any_provision_fails() -> None:
    with pytest.raises(NcoilTextError, match="no heading or numbered provision"):
        clean_model("A memo about a model act.\n")


PARTS_MODEL = FIXTURES / "synthetic_parts_model.txt"
PARTS_CLEANED = (
    "A. Definitions *\n"
    "\n"
    '"Lamp network" means a list of lamp keepers under contract.\n'
    '"Lessee" means a person that leases a lamp network.\n'
    "\n"
    "B. Lamp Network Contracting Act\n"
    "Section I. Leasing\n"
    "A. A keeper may lease a lamp network only under.\n"
    "B. A lessee shall give notice of each lease.\n"
    "\n"
    "Section IV. Notice\n"
    "A lessee shall post notice of each lease.\n"
    "\n"
    "C. Effective Date\n"
    "This Act takes effect on.\n"
)
UNNUMBERED_MODEL = FIXTURES / "synthetic_unnumbered_model.txt"
UNNUMBERED_CLEANED = (
    "(A) As used in this section:\n"
    '"Remote lamplighter" means a lamplighter who lights a lamp from a distance.\n'
    "(B) A remote lamplighter shall keep a record of each lamp lit.\n"
    "(C) This section applies to lamps lit after its effective date.\n"
)


@pytest.mark.parametrize(
    ("path", "cleaned"),
    [(PARTS_MODEL, PARTS_CLEANED), (UNNUMBERED_MODEL, UNNUMBERED_CLEANED)],
    ids=["lettered-parts", "unnumbered"],
)
def test_a_model_in_parts_or_without_headings_keeps_its_provisions(
    path: Path, cleaned: str
) -> None:
    # Lettered parts under a "Contents:" list, sections numbered I, II within a part, a
    # footnote, a bracketed note, a section whose only text is a note, a placeholder section;
    # or no headings at all, an "AN ACT" title before "(A)". Read as text, Word and PDF.
    text = path.read_text()
    assert clean_model(text) == cleaned
    assert clean_model(cleaned) == cleaned
    assert clean_model(docx_text(docx_bytes(text))) == cleaned
    assert clean_model(file_text(pdf_bytes(text), "application/pdf")).split() == cleaned.split()


@pytest.mark.parametrize(
    "heading",
    [
        "Section 1. Short Title",
        "Section 1: Short Title",
        "Section 1 Short Title",
        "SECTION 1. SHORT TITLE",
        "Section I. Short Title",
        "Section __. Short Title",
        "Chapter 1. Short Title",
        f"TITLE I {EN} SHORT TITLE",
        "I. SHORT TITLE",
    ],
)
def test_every_heading_form_opens_the_text(heading: str) -> None:
    raw = f"A Synthetic Model Act\n*Adopted on an invented day.\n\n{heading}\nA lamp is a lamp.\n"
    assert clean_model(raw) == f"{heading}\nA lamp is a lamp.\n"


def test_a_heading_carrying_its_text_opens_the_text() -> None:
    raw = f"A Synthetic Model Act\n\nSection 1 {EN} The commissioner shall keep a lamp.\n"
    assert clean_model(raw) == f"Section 1 {EN} The commissioner shall keep a lamp.\n"


@pytest.mark.parametrize(
    "line",
    [
        "Section 3 may recover its costs.",
        "Section C.2 of this Act applies.",
        "Sections 2 or 3 of this Act.",
        "Title 18 of the federal code.",
    ],
)
def test_a_sentence_naming_a_section_is_not_a_heading(line: str) -> None:
    raw = f"Section 1. Scope\nA. One.\n{line}\nB. Two.\n"
    assert clean_model(raw) == raw


def test_a_contents_list_numbered_apart_from_the_text_is_dropped() -> None:
    # The list says 1 and 2 where the text says I and II, shortens a title, wraps an entry and
    # gives page numbers; the list's first entry is found again by its number.
    raw = (
        "Table of Contents        Page Numbers\n"
        "Section 1. Title        (1)\n"
        "Section 2. Lamp Keepers and the Lamps They\n"
        "Keep        (1-2)\n"
        "\n"
        "Section I. Short Title\n"
        "This Act is the Lamp Act.\n"
        "Section II. Lamp Keepers and the Lamps They Keep\n"
        "A keeper keeps lamps.\n"
    )
    assert clean_model(raw) == raw[raw.index("Section I.") :]


def test_a_contents_list_split_by_a_note_is_dropped() -> None:
    raw = (
        "A Synthetic Model Act\n"
        "Section 1. Definitions         (1)\n"
        "Section 2. Keepers         (2)\n"
        "\n"
        "[Drafting Note: an invented note\n"
        "on the whole Act.]\n"
        "\n"
        "Section 1. Definitions\n"
        "A. One.\n"
        "Section 2. Keepers\n"
        "A. Two.\n"
    )
    assert clean_model(raw) == "Section 1. Definitions\nA. One.\nSection 2. Keepers\nA. Two.\n"


def test_a_section_left_to_the_states_goes_and_one_with_text_stays() -> None:
    raw = (
        "Section 1. Scope\n"
        "A. One.\n"
        "Section 2. Penalties\n"
        "Drafting Note: a state may write its own.\n"
        "Section 3. {Repealer clause}\n"
        "Section 4 - This Act takes effect upon enactment by the state.\n"
        "Drafting Note: an invented note.\n"
        "Section 5. {Records}\n"
        "A keeper keeps records.\n"
    )
    assert clean_model(raw) == (
        "Section 1. Scope\nA. One.\n"
        "Section 4 - This Act takes effect upon enactment by the state.\n"
        "Section 5. {Records}\nA keeper keeps records.\n"
    )


def test_a_note_wrapping_onto_a_sections_name_stays_one_note() -> None:
    raw = (
        "Section 5. Rules\n"
        "A. The department shall adopt rules.\n"
        "Drafting Note: rules should include enforcement for\n"
        "Section 6.\n"
        "\n"
        "Section 7. Effective Date\n"
        "This Act takes effect at once.\n"
    )
    assert clean_model(raw) == (
        "Section 5. Rules\nA. The department shall adopt rules.\n\n"
        "Section 7. Effective Date\nThis Act takes effect at once.\n"
    )


def test_footnotes_go_and_a_page_break_ends_one() -> None:
    raw = (
        "Section 1. Scope\nA. One.*\n*1 Based on an invented statute.\n4\napplies to every lamp.\n"
    )
    assert clean_model(raw) == "Section 1. Scope\nA. One.*\n\napplies to every lamp.\n"


# ---- real samples, local only --------------------------------------------------------------------

# A section heading in any of the forms the models use (see the module); "Section 3 may" is not.
_HEADING_LINE = re.compile(
    r"^\s*\[?(?i:section|sec\.?)\s+(?P<number>\d+[A-Za-z]?|[IVXL]+|[A-Z]|_+)"
    r"(?:\s*[.:](?!\d)|\s+[\u2013\u2014-]|\s+(?=[A-Z(\[\u201c\"])|\s*$)"
)
_PROVISION = re.compile(r"^\s*(?:\(?(?:[A-Za-z]|\d{1,3})\)|[A-Z]\.|\d{1,3}\.)\s")
_INSTRUCTION_LEFT = re.compile(r"\[\s*(?:cite|citation|insert|specify)\b", re.IGNORECASE)
_NOTE_LINE = re.compile(r"^\s*(?:legislative|drafting)\s+notes?\b", re.IGNORECASE | re.MULTILINE)
_ROMAN = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
}


def _number(line: str) -> str | None:
    m = _HEADING_LINE.match(line)
    if m is None:
        return None
    number = m.group("number")
    return str(_ROMAN.get(number, number))


def _carries_text(line: str) -> bool:
    """A heading with its section's one sentence on the same line."""
    m = _HEADING_LINE.match(line)
    rest = line[m.end() :].strip() if m else ""
    return len(rest.split()) >= 5 and rest.endswith((".", ";", ":"))


def _collected() -> list[str]:
    try:
        return [d.id for d in read_docs(SOURCE, MODELDOCS_DIR)]
    except (OSError, ValueError):
        return []


def _read(doc_id: str) -> tuple[str, str]:
    """A collected model's stored raw file as read, and its stored text."""
    doc = next(d for d in read_docs(SOURCE, MODELDOCS_DIR) if d.id == doc_id)
    raw = raw_path(doc, MODELDOCS_DIR).read_bytes()
    return file_text(raw, doc.media_type), load_text(doc, MODELDOCS_DIR)


@pytest.mark.skipif(not _collected(), reason="no NCOIL models collected under the data directory")
@pytest.mark.parametrize("doc_id", _collected(), ids=lambda i: i.partition("/")[2][:40])
def test_a_collected_models_stored_text_is_its_cleaning(doc_id: str) -> None:
    read, stored = _read(doc_id)
    same = clean_model(read) == stored  # a bool, so a failure never prints the model's text
    assert same


@pytest.mark.skipif(not _collected(), reason="no NCOIL models collected under the data directory")
@pytest.mark.parametrize("doc_id", _collected(), ids=lambda i: i.partition("/")[2][:40])
def test_a_collected_model_keeps_its_sections_and_no_note(doc_id: str) -> None:
    read, _ = _read(doc_id)
    text = clean_model(read)

    # Counts only, so a failure never prints the model's text.
    lines = [line for line in text.split("\n") if line.strip()]
    raw_lines = [line for line in read.split("\n") if line.strip()]
    headings = sum(1 for line in lines if _HEADING_LINE.match(line))
    raw_headings = sum(1 for line in raw_lines if _HEADING_LINE.match(line))
    provisions = sum(1 for line in lines if _HEADING_LINE.match(line) or _PROVISION.match(line))
    first_kept = next((_number(line) for line in lines if _HEADING_LINE.match(line)), None)
    first_raw = next((_number(line) for line in raw_lines if _HEADING_LINE.match(line)), None)
    adjacent_headings = sum(
        1 for a, b in itertools.pairwise(lines)
        if _HEADING_LINE.match(a) and not _carries_text(a) and _HEADING_LINE.match(b)
    )  # fmt: skip
    instructions_left = len(_INSTRUCTION_LEFT.findall(text))
    note_lines = len(_NOTE_LINE.findall(text))
    contents = text.lower().count("table of contents")
    # The words from where the text's first line first appears in the raw, its contents list
    # included: the title, adoption and sponsor lines before it are not law.
    first = " ".join(lines[0].split())
    at = next((i for i, line in enumerate(raw_lines) if " ".join(line.split()) == first), 0)
    kept, before = len(text.split()), sum(len(line.split()) for line in raw_lines[at:])

    assert clean_model(text) == text
    assert provisions > 0
    assert headings > 0 or raw_headings == 0
    # The text opens at the model's first section, not at one after it.
    assert first_kept == first_raw
    assert adjacent_headings == 0  # no table of contents left
    assert instructions_left == 0
    assert note_lines == 0
    assert contents == 0
    # Notes and footnotes are a share of a model's text; losing most of it is losing law.
    assert 0.6 * before <= kept <= before
