"""The resolved answer key: each commission list row, and the corpus bill it resolved to."""

import json
from datetime import date
from pathlib import Path

from typer.testing import CliRunner

from lineage.answer_key import ANSWER_KEY, HELD_OUT_KEY, KeyBill, answer_key_rows, read_answer_key
from lineage.cli import app
from lineage.commission import RESULTS
from lineage.ulc import BillRecord, development_bills, load_enactments
from tests.conftest import COMMISSION

SNAPSHOT = """\
# ULC Bill List, read 2099-01-02
act,jurisdiction,year,bill_number,status
widget-registration,Utah,2098,HB 12/SB 7,Enacted
widget-registration,Idaho,2098,S 1001,Introduced
widget-registration,District of Columbia,2098,26-0001,Enacted
widget-registration,Oregon,2090,SB 3,Enacted
widget-registration,Ohio,2098,SB 9,Enacted
"""
YEARS = {("UT", "2098"): (2098, 2098), ("OH", "2098"): (2098, 2098)}
BILLS = [
    BillRecord("ut-hb12", "UT", "2098", "HB 12", "passed", date(2098, 3, 20), 2),
    BillRecord("ut-sb7", "UT", "2098", "SB 7", "died", None, 0),
]
DETAILS = {
    "ut-hb12": KeyBill("Widget Registration Act", date(2098, 3, 1)),
    "ut-sb7": KeyBill("Widget Registration Act", None),
}


def _rows(tmp_path: Path) -> list[dict[str, str]]:
    path = tmp_path / "ulc.csv"
    path.write_text(SNAPSHOT)
    return answer_key_rows(load_enactments(path), BILLS, DETAILS, YEARS, first_year=2095)


def test_each_named_bill_is_a_row_with_the_listed_number_it_matched(tmp_path: Path) -> None:
    utah = [r for r in _rows(tmp_path) if r["jurisdiction"] == "Utah"]

    assert [(r["matched_number"], r["bill_id"], r["rule"]) for r in utah] == [
        ("HB 12", "ut-hb12", "in_year"),
        ("SB 7", "ut-sb7", "in_year"),
    ]
    assert utah[0]["title"] == "Widget Registration Act"
    assert utah[0]["last_passage_on"] == "2098-03-01"
    assert utah[1]["last_passage_on"] == ""
    assert all(r["counted"] == "yes" for r in utah)


def test_every_list_row_appears_even_when_it_is_not_counted(tmp_path: Path) -> None:
    rows = _rows(tmp_path)

    counted = {r["jurisdiction"]: r["counted"] for r in rows}
    assert counted["Idaho"] == "no: listed as introduced"
    assert counted["District of Columbia"].startswith("no: District of Columbia")
    assert counted["Oregon"].startswith("no: listed year 2090")
    # Counted, but no corpus bill carries the number: one row, with no bill.
    assert counted["Ohio"] == "yes"
    ohio = next(r for r in rows if r["jurisdiction"] == "Ohio")
    assert (ohio["rule"], ohio["bill_id"]) == ("", "")


def test_a_row_naming_a_bill_of_the_development_set_is_not_counted(tmp_path: Path) -> None:
    development = tmp_path / "development.csv"
    development.write_text(
        "act,jurisdiction,year,bill_number,status\nspecial-deposits,Ohio,2098,S 9,Enacted\n"
    )
    path = tmp_path / "ulc.csv"
    path.write_text(SNAPSHOT)

    rows = answer_key_rows(
        load_enactments(path), BILLS, DETAILS, YEARS, first_year=2095,
        development=development_bills(load_enactments(development)),
    )  # fmt: skip

    counted = {r["jurisdiction"]: r["counted"] for r in rows}
    assert counted["Ohio"] == "no: names a bill of the development set"
    assert counted["Utah"] == "yes"


def test_a_row_of_a_reserve_act_is_listed_but_not_counted(tmp_path: Path) -> None:
    path = tmp_path / "ulc.csv"
    path.write_text(SNAPSHOT + "spare-act,Utah,2098,HB 12,Enacted\n")

    rows = answer_key_rows(
        load_enactments(path), BILLS, DETAILS, YEARS, first_year=2095,
        acts=frozenset({"widget-registration"}),
    )  # fmt: skip

    counted = {(r["act"], r["jurisdiction"]): r["counted"] for r in rows}
    assert counted[("spare-act", "Utah")] == "no: a reserve act, not selected"
    assert counted[("widget-registration", "Utah")] == "yes"


def test_the_held_out_key_drops_rows_that_share_a_bill_with_the_development_set(
    commission_corpus: Path, tmp_path: Path
) -> None:
    enactments, out = tmp_path / "held_out.csv", tmp_path / "answer_key_heldout.csv"
    # Ohio's SB 237 enacted the Public Expression Protection Act, in the development set.
    enactments.write_text(
        "# ULC Bill List, read 2099-01-02\n"
        "act,jurisdiction,year,bill_number,status\n"
        "directed-trust,Ohio,2025,SB 237,Enacted\n"
    )

    result = CliRunner().invoke(
        app,
        [
            "eval", "answer-key", "--held-out", "--tables", str(commission_corpus),
            "--enactments", str(enactments), "--sessions", str(COMMISSION / "sessions.csv"),
            "--out", str(out),
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    _, rows = read_answer_key(out)
    assert [(r["act"], r["counted"]) for r in rows] == [
        ("directed-trust", "no: names a bill of the development set")
    ]


def test_the_command_writes_the_key_from_the_fixture_corpus(
    commission_corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "answer_key.csv"

    result = CliRunner().invoke(
        app,
        [
            "eval", "answer-key", "--tables", str(commission_corpus),
            "--sessions", str(COMMISSION / "sessions.csv"), "--out", str(out),
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    note, rows = read_answer_key(out)
    assert note.startswith("uv run lineage eval answer-key")
    listed = load_enactments().rows
    assert {(r["act"], r["jurisdiction"], r["year"]) for r in rows} == {
        (e.act, e.jurisdiction, str(e.year)) for e in listed
    }
    # Ohio's act passed in December 2024 and is listed under 2025.
    ohio = next(r for r in rows if r["jurisdiction"] == "Ohio" and r["bill_id"])
    assert (ohio["rule"], ohio["session"], ohio["identifier"]) == ("prior_session", "135", "SB 237")


def test_the_committed_key_and_the_committed_results_name_the_same_bills() -> None:
    _, rows = read_answer_key(ANSWER_KEY)
    key = {
        (r["act"], r["year"], r["bill_id"]) for r in rows if r["counted"] == "yes" and r["bill_id"]
    }
    results = json.loads(RESULTS.read_text())["commission"]["enactments"]
    named = {(e["act"], str(e["year"]), b["id"]) for e in results for b in e["bills"]}

    assert key == named


def test_no_held_out_bill_was_studied_in_the_development_set() -> None:
    # The list rule drops rows by state, year and number; this checks the resolved bills, which
    # would also catch a bill listed under a different year in each set.
    _, development = read_answer_key(ANSWER_KEY)
    _, held_out = read_answer_key(HELD_OUT_KEY)
    studied = {r["bill_id"] for r in development if r["bill_id"]}
    counted = {r["bill_id"] for r in held_out if r["counted"] == "yes" and r["bill_id"]}

    assert counted
    assert not counted & studied
