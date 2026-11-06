"""Which version of a bill the text layer fetches: the latest text of the bill itself."""

import pytest

from lineage.versions import Choice, Link, Version, choose_version, is_bill_text

PDF = "application/pdf"
HTML = "text/html"


def _v(note: str, day: str, *media: str) -> Version:
    slug = note.replace(" ", "-")
    links = tuple(Link(f"https://leg.example/{slug}.{m.split('/')[1]}", m) for m in media)
    return Version(note, day, links)


@pytest.mark.parametrize(
    "note",
    [
        "Introduced",
        "Enrolled",
        "Chaptered",
        "05/20/25 - Amended Assembly",  # California: the bill as amended, not an amendment
        "First Engrossment with Senate Amendments",  # North Dakota
        "Amendments in Context",  # New Mexico: the bill with its amendments marked
        "Senate Committee Report",  # Texas: the bill as the committee reported it
        "First - Budget Bill (Fiscal Year 2026)",  # Maryland names the bill in the note
        "Public Act No. 25-61",
        "Slip Law",
        "LD 870, HP 551",
        "First - Maryland Uniform Disposition of Abandoned Property Act - Revisions",
        "Documents and Disposition",  # Maine: the bill as printed
        "As Introducedopens in a new window",  # Vermont
    ],
)
def test_bill_versions_are_bill_text(note: str) -> None:
    assert is_bill_text(note)


@pytest.mark.parametrize(
    "note",
    [
        "Senate Amendment 1",
        "House Amendment A01234",
        "Amendment H-8211",
        "Adopted Amendments",
        "Sen Floor Amend 1 AM 28 0412",
        "HA 1 to HB 5",
        "Senate Amendments to Introduced",
        "Racial Equity Impact Note",
        "Fiscal Note",
        "Senate Transmittal Letter 2",
        "Final Disposition",
        "Act Summary",
        "Act Summaryopens in a new window",
        "Senate Amendments to A-Engrossed",  # Oregon
        "Judiciary AM1234",  # Nebraska
        "Letter Explaining Corrected Copy of Enrolled Act",
        "Amended Excerpts",
        "Status in Committee",
        "Statement",
        "Motion to Adopt - Adopted Roll Call 112",
        "Local Ad",
    ],
)
def test_amendments_and_other_documents_are_not(note: str) -> None:
    assert not is_bill_text(note)


def test_the_latest_bill_version_is_chosen_over_a_later_amendment() -> None:
    versions = [
        _v("Introduced", "2025-01-10", PDF, HTML),
        _v("Enrolled", "2025-04-02", PDF),
        _v("Senate Amendment 1", "2025-04-05", PDF),
    ]

    choice = choose_version(versions)

    assert choice == Choice(
        note="Enrolled",
        date="2025-04-02",
        links=(Link("https://leg.example/Enrolled.pdf", PDF),),
        passed_over=1,
    )


def test_versions_on_the_same_day_keep_the_export_order() -> None:
    versions = [_v("Engrossed", "2025-03-01", PDF), _v("Enrolled", "2025-03-01", PDF)]

    choice = choose_version(versions)

    assert choice is not None
    assert choice.note == "Enrolled"


def test_an_undated_version_sorts_by_its_place_in_the_export() -> None:
    versions = [_v("Introduced", "", PDF), _v("Engrossed", "", PDF)]

    choice = choose_version(versions)

    assert choice is not None
    assert choice.note == "Engrossed"


def test_only_supported_formats_are_kept_html_first() -> None:
    word = "application/msword"
    versions = [_v("Printer's No. 1466", "2025-05-01", word, PDF, HTML)]

    choice = choose_version(versions)

    assert choice is not None
    assert [link.media_type for link in choice.links] == [HTML, PDF]


def test_a_record_with_no_bill_version_has_no_choice() -> None:
    assert choose_version([_v("Fiscal Note", "2025-02-01", PDF)]) is None
    assert choose_version([]) is None


def test_a_latest_version_in_an_unsupported_format_is_not_replaced_by_an_older_one() -> None:
    versions = [_v("Introduced", "2025-01-10", PDF), _v("Enrolled", "2025-04-02", "text/xml")]

    choice = choose_version(versions)

    assert choice is not None
    assert (choice.note, choice.links) == ("Enrolled", ())


def test_a_scanned_signed_copy_gives_way_to_the_text_beside_it() -> None:
    # Colorado's "Signed Act" is an image of the signed bill; "Final Act" is its text.
    versions = [
        _v("Final Act", "2024-05-01", PDF),
        _v("Signed Act", "2024-05-20", PDF),
    ]

    choice = choose_version(versions)

    assert choice is not None
    assert choice.note == "Final Act"


def test_a_scanned_copy_is_still_taken_when_it_is_the_only_bill_version() -> None:
    choice = choose_version([_v("Signed Act", "2024-05-20", PDF)])

    assert choice is not None
    assert choice.note == "Signed Act"
