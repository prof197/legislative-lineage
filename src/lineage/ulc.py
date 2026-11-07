"""The Uniform Law Commission's enactment lists, and the corpus bills each enactment names.

The lists load only in a browser, so `evals/ulc_enactments.csv` is a hand copy, dated on its
first line. A row's cell can name several bills, usually companions in each chamber, and its
year can be the signing year rather than the session's. So an enactment's bills are every bill
the cell names in a session whose years contain the listed year. When none of those shows
passage, a bill the legislature passed the year before is looked for: one passed in December
and signed in January is listed under the signing year. See DESIGN.md#commission-recall.
"""

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

ENACTMENTS = Path("evals/ulc_enactments.csv")
HELD_OUT_ENACTMENTS = Path("evals/ulc_enactments_heldout.csv")

ULC_STATES: dict[str, str | None] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    # The corpus is the 50 states, so the District's enactments are listed but never matched.
    "District of Columbia": None,
}  # fmt: skip
_UNICAMERAL = frozenset({"NE"})

Stratum = Literal["enacted", "substantially_similar"]
# Longest first: "Enacted - Substantially Similar" also starts with "Enacted". The lists'
# own spelling ("Similiar") is kept in the file, so statuses are matched by prefix.
_STRATA: tuple[tuple[str, Stratum | None], ...] = (
    ("Enacted - Substantially Simil", "substantially_similar"),
    ("Enacted", "enacted"),
    ("Introduced", None),
)

# Colorado writes the session's year into the number ("HB 24-1232", "HB 23B-1005").
_YEAR_PREFIX = re.compile(r"\s+\d{2}[A-Z]?-")
_NUMBER = re.compile(r"^([A-Z]+)\s*0*(\d+[A-Z]?)$")
# North Carolina's list writes "H 40" for the bill Open States stores as "HB 40".
_BILL_PREFIX = {"HB": "H", "SB": "S", "AB": "A"}


class UlcError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Enactment:
    """One row of a commission list: a jurisdiction's bill or bills for one act."""

    act: str
    jurisdiction: str
    state: str | None  # None for the District of Columbia
    year: int
    bill_cell: str
    bill_numbers: tuple[str, ...]
    status: str
    stratum: Stratum | None  # None for a bill the list shows only as introduced


@dataclass(frozen=True, slots=True)
class Snapshot:
    read_note: str
    rows: list[Enactment]


@dataclass(frozen=True, slots=True)
class BillRecord:
    """What matching needs to know about one corpus bill."""

    id: str
    state: str
    session: str
    identifier: str
    status: str | None
    status_date: date | None
    passage_chambers: int  # the chambers with a recorded passage action


@dataclass(frozen=True, slots=True)
class Found:
    rule: Literal["in_year", "prior_session", "listed_only"]
    bills: tuple[BillRecord, ...]


def canonical_number(number: str) -> str:
    """A bill number in one form for comparison: "HB 24-1232", "H 40", "HB 0040" -> "H1232"..."""
    text = _YEAR_PREFIX.sub(" ", number.upper().replace(".", "")).strip()
    m = _NUMBER.match(text)
    if m is None:
        return text.replace(" ", "")
    return _BILL_PREFIX.get(m[1], m[1]) + m[2]


def _stratum(status: str, line: int) -> Stratum | None:
    for prefix, stratum in _STRATA:
        if status.startswith(prefix):
            return stratum
    raise UlcError(f"line {line}: unknown status {status!r}")


def load_enactments(path: Path = ENACTMENTS) -> Snapshot:
    lines = path.read_text(encoding="utf-8").splitlines()
    note = lines[0].removeprefix("#").strip() if lines and lines[0].startswith("#") else ""
    body = [line for line in lines if not line.startswith("#")]
    rows: list[Enactment] = []
    for line, r in enumerate(csv.DictReader(body), start=2 + bool(note)):
        if r["jurisdiction"] not in ULC_STATES:
            raise UlcError(f"line {line}: unknown jurisdiction {r['jurisdiction']!r}")
        rows.append(
            Enactment(
                act=r["act"],
                jurisdiction=r["jurisdiction"],
                state=ULC_STATES[r["jurisdiction"]],
                year=int(r["year"]),
                bill_cell=r["bill_number"],
                bill_numbers=tuple(n.strip() for n in r["bill_number"].split("/") if n.strip()),
                status=r["status"],
                stratum=_stratum(r["status"], line),
            )
        )
    return Snapshot(note, rows)


NO_BILLS: frozenset[tuple[str, int, str]] = frozenset()


def development_bills(snapshot: Snapshot) -> frozenset[tuple[str, int, str]]:
    """(state, listed year, bill number) for every bill the development lists name."""
    return frozenset(
        (r.state, r.year, canonical_number(n))
        for r in snapshot.rows
        if r.state is not None
        for n in r.bill_numbers
    )


def _passed(bill: BillRecord) -> bool:
    """The legislature passed it: a passed status, or passage recorded in every chamber."""
    chambers = 1 if bill.state in _UNICAMERAL else 2
    return bill.status == "passed" or bill.passage_chambers >= chambers


def find_bills(
    enactment: Enactment,
    bills: list[BillRecord],
    session_years: dict[tuple[str, str], tuple[int, int]],
) -> Found:
    """The corpus bills an enactment names, and which rule found them."""
    wanted = {canonical_number(n) for n in enactment.bill_numbers}
    year = enactment.year
    named = [
        b
        for b in bills
        if b.state == enactment.state
        and canonical_number(b.identifier) in wanted
        and (b.state, b.session) in session_years
    ]

    def passed_later(b: BillRecord) -> bool:
        # A passed bill is dated by its passage or signing; one dated after the listed year
        # is a later bill that reused the number.
        return b.status == "passed" and b.status_date is not None and b.status_date.year > year

    in_year = [
        b
        for b in named
        if session_years[b.state, b.session][0] <= year <= session_years[b.state, b.session][1]
        and not passed_later(b)
    ]
    if any(_passed(b) for b in in_year):
        return Found("in_year", tuple(in_year))
    prior = [b for b in named if session_years[b.state, b.session][1] == year - 1 and _passed(b)]
    if prior:
        return Found("prior_session", tuple(prior))
    return Found("listed_only", tuple(in_year))
