"""The commission's enactment lists: reading the snapshot and finding each enactment's bills.

The bill records here are invented. Each case is one the real lists raise: a cell naming
companion bills, a bill number reused in a later session, a year that is the signing year of a
bill passed the December before, and states that number bills in their own ways.
"""

from datetime import date
from pathlib import Path

import pytest

from lineage.ulc import (
    ULC_STATES,
    BillRecord,
    UlcError,
    canonical_number,
    find_bills,
    load_enactments,
)

SNAPSHOT = """\
# ULC Bill List, read 2099-01-02
act,jurisdiction,year,bill_number,status
widget-registration,Utah,2098,HB 12/SB 7,Enacted
widget-registration,Oregon,2097,SB 3,Enacted - Substantially Similiar
widget-registration,District of Columbia,2098,26-0001,Enacted
widget-registration,Idaho,2098,S 1001,Introduced
"""


def _write(tmp_path: Path, text: str = SNAPSHOT) -> Path:
    path = tmp_path / "ulc.csv"
    path.write_text(text)
    return path


def test_reads_the_snapshot_with_its_read_date(tmp_path: Path) -> None:
    snapshot = load_enactments(_write(tmp_path))

    assert snapshot.read_note == "ULC Bill List, read 2099-01-02"
    assert len(snapshot.rows) == 4
    utah = snapshot.rows[0]
    assert (utah.state, utah.year, utah.bill_numbers) == ("UT", 2098, ("HB 12", "SB 7"))


def test_statuses_are_matched_by_prefix_so_the_lists_own_spelling_is_kept(tmp_path: Path) -> None:
    rows = load_enactments(_write(tmp_path)).rows

    assert [r.stratum for r in rows] == ["enacted", "substantially_similar", "enacted", None]
    assert rows[1].status == "Enacted - Substantially Similiar"


def test_the_district_of_columbia_is_listed_but_has_no_state(tmp_path: Path) -> None:
    dc = load_enactments(_write(tmp_path)).rows[2]

    assert dc.state is None
    assert "District of Columbia" in ULC_STATES


def test_an_unknown_jurisdiction_or_status_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UlcError, match="Atlantis"):
        load_enactments(
            _write(tmp_path, SNAPSHOT + "widget-registration,Atlantis,2098,HB 1,Enacted\n")
        )
    with pytest.raises(UlcError, match="Vetoed"):
        load_enactments(_write(tmp_path, SNAPSHOT + "widget-registration,Utah,2098,HB 1,Vetoed\n"))


@pytest.mark.parametrize(
    ("listed", "stored"),
    [
        ("HB 1232", "HB 24-1232"),  # Colorado prefixes the session's year
        ("HB 1005", "HB 23B-1005"),  # and marks a special session
        ("H 40", "HB 40"),  # North Carolina's list drops the B
        ("S 1001", "S 1001"),
        ("SF 47", "SF 47"),
        ("LB 231", "LB 231"),
        ("S.B. 80", "SB 80"),
        ("HB 0029", "HB 29"),
    ],
)
def test_bill_numbers_compare_in_one_form(listed: str, stored: str) -> None:
    assert canonical_number(listed) == canonical_number(stored)


def test_different_bills_stay_different() -> None:
    assert canonical_number("HB 40") != canonical_number("HR 40")
    assert canonical_number("SB 40") != canonical_number("SJR 40")
    assert canonical_number("HF 47") != canonical_number("SF 47")


def _bill(
    session: str,
    identifier: str,
    status: str = "passed",
    *,
    on: date | None = None,
    chambers: int = 2,
    state: str = "UT",
) -> BillRecord:
    return BillRecord(
        id=f"{state}-{session}-{identifier}",
        state=state,
        session=session,
        identifier=identifier,
        status=status,
        status_date=on,
        passage_chambers=chambers,
    )


YEARS = {
    ("UT", "2097"): (2097, 2097),
    ("UT", "2098"): (2098, 2098),
    ("UT", "2099"): (2099, 2099),
    ("UT", "97-98"): (2097, 2098),
    ("UT", "99-00"): (2099, 2100),
}


def test_every_listed_bill_of_the_session_year_is_a_target(tmp_path: Path) -> None:
    utah = load_enactments(_write(tmp_path)).rows[0]
    bills = [
        _bill("2098", "HB 12", on=date(2098, 3, 1)),
        _bill("2098", "SB 7", "died", chambers=0),
        _bill("2097", "HB 12", on=date(2097, 3, 1)),  # the number, reused a year earlier
        _bill("2098", "HB 13"),
    ]

    found = find_bills(utah, bills, YEARS)

    assert found.rule == "in_year"
    assert [b.id for b in found.bills] == ["UT-2098-HB 12", "UT-2098-SB 7"]


def test_a_bill_passed_in_december_and_signed_in_january_is_found_in_the_prior_session(
    tmp_path: Path,
) -> None:
    # The list's year is the signing year; the session that passed the bill had ended.
    utah = load_enactments(_write(tmp_path)).rows[0]
    bills = [
        _bill("2098", "HB 12", "introduced", chambers=0),  # the number, reused for another bill
        _bill("2097", "HB 12", "died", on=date(2097, 12, 31)),  # passed both chambers, no signing
    ]

    found = find_bills(utah, bills, YEARS)

    assert found.rule == "prior_session"
    assert [b.id for b in found.bills] == ["UT-2097-HB 12"]


def test_a_bill_passed_after_the_listed_year_is_not_the_enactment(tmp_path: Path) -> None:
    # A session running from 2097 to 2098 and the next one, begun in late 2098, both contain
    # 2098; the later session's bill of the same number passed in 2099.
    row = load_enactments(_write(tmp_path)).rows[0]
    years = {("UT", "97-98"): (2097, 2098), ("UT", "98-99"): (2098, 2099)}
    bills = [
        _bill("97-98", "HB 12", "died", chambers=1),
        _bill("98-99", "HB 12", on=date(2099, 4, 1)),
    ]

    found = find_bills(row, bills, years)

    assert found.rule == "listed_only"
    assert [b.id for b in found.bills] == ["UT-97-98-HB 12"]


def test_no_bill_in_the_session_years_finds_nothing(tmp_path: Path) -> None:
    row = load_enactments(_write(tmp_path)).rows[0]

    found = find_bills(row, [_bill("99-00", "HB 12")], YEARS)

    assert found.bills == ()


def test_a_unicameral_legislature_passes_a_bill_in_one_chamber(tmp_path: Path) -> None:
    header = "act,jurisdiction,year,bill_number,status\n"
    row = load_enactments(
        _write(tmp_path, header + "widget-registration,Nebraska,2098,LB 5,Enacted\n")
    )
    bills = [
        _bill("2098", "LB 5", "introduced", chambers=0, state="NE"),
        _bill("2097", "LB 5", "died", chambers=1, state="NE"),
    ]
    years = {("NE", "2098"): (2098, 2098), ("NE", "2097"): (2097, 2097)}

    found = find_bills(row.rows[0], bills, years)

    assert found.rule == "prior_session"
    assert [b.id for b in found.bills] == ["NE-2097-LB 5"]
