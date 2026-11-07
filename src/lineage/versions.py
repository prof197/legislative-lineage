"""Which version of a bill the text layer fetches: the latest text of the bill itself.

Open States lists every document a legislature posts for a bill among its versions: the bill as
introduced, engrossed and enrolled, but also floor amendments, fiscal and impact notes,
transmittal letters and dispositions. Louisiana's stored text is often the latest of these, a
one-page amendment instead of the bill. The rule keeps the versions whose note names the bill's
own text and takes the latest. Most exports leave version dates empty and list versions out of
order, so "latest" is read from the note first: its legislative stage (introduced, then amended
or substituted, then passed a chamber, then enrolled or enacted), then a date (the version's,
or one written in the note), then the note's number (Pennsylvania's printer's number, Utah's
substitute number), then the export's order. For a passed bill that is the enrolled or chaptered
text. See DESIGN.md#dr-6.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Extractors exist for these, in order of preference: HTML keeps a document's structure, a PDF's
# text layer has to be reassembled from positioned glyphs.
SUPPORTED_FORMATS = ("text/html", "application/pdf")

# An amendment names itself ("Senate Amendment 1", "Amendment H-8211", "HA 1 to HB 5", Nebraska's
# "AM123"). The bill as amended does not: "Amended Assembly", "as amended by the House", "First
# Engrossment with Senate Amendments", New Mexico's "Amendments in Context".
_AMENDMENT = re.compile(r"\bamend(?:ments?)?\b|\bam\d+\b|^[hs][cf]?a \d+ to\b", re.IGNORECASE)
_AMENDED_BILL = re.compile(r"\bengross\w* with\b|\bin.context\b", re.IGNORECASE)
# Documents about a bill that are not its text. Maryland's notes carry the bill's title after
# its reading ("First - Budget Bill (Fiscal Year 2026)"), so only that reading is examined.
_OTHER_DOCUMENT = re.compile(
    r"\b(?:fiscal|impact|actuarial|equity)(?: \w+)? (?:note|statement|estimate|analysis)\b"
    r"|\bletter\b|\bfinal disposition\b|\b(?:act|bill) summary\b|^statement$|\bexcerpts?\b"
    r"|\broll call\b|^motion\b|^local ad\b|\bveto message\b|^status\b",
    re.IGNORECASE,
)
_READING_THEN_TITLE = re.compile(
    r"^(?:first|second|third|enrolled|engrossed|chapter)\s+-\s", re.IGNORECASE
)
# Colorado's "Signed Act" is a scanned image of the signed bill, with no text to extract. It
# yields to the latest other bill version, and is taken only when it is the sole one.
_SCANNED = re.compile(r"^signed act\b", re.IGNORECASE)
# Notes that are bill text in one state and not in another. Utah's committee reports are the
# committee's letter, and its "Comparison to Sub #2" a redline between substitutes; Texas's
# committee reports are the bill as reported.
_OTHER_DOCUMENT_IN: dict[str, re.Pattern[str]] = {
    "UT": re.compile(r"\bcommittee report\b|^comparison\b", re.IGNORECASE),
}
# A note's legislative stage, latest first. Mississippi's "Current version" is the bill as sent to
# the governor. A note that names no stage (Massachusetts's "Bill Text", a bare "HB 12") ranks
# between introduced and passed.
_STAGES = (
    (
        4,
        re.compile(
            r"\benroll|\bchapter|\bchap\d|\bsigned\b|\bpublic act\b|\bslip law\b"
            r"|\bfinal\b|\bact\b(?: no\.)? ?\d|\b(?:hb|sb|hr|sr|hj|sj)\s?\d+ act\b"
            r"|\bpassed legislature\b|^[hs][bjr]?\d+er$|\b(?:current|latest) version\b",
            re.IGNORECASE,
        ),
    ),
    (
        3,
        re.compile(
            r"\bengross|\bpassed\b|\bperfected\b|\badopted by\b|\bthird reading\b",
            re.IGNORECASE,
        ),
    ),
    (1, re.compile(r"\bintroduc|\bfiled\b|\boffered\b|^proposed bill$|\boriginal$", re.IGNORECASE)),
)
_UNSTAGED = 2
_NOTE_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b")
_NUMBER = re.compile(r"\d+")
# New York marks each amended print with a letter after the bill number: S1234, S1234A, S1234B.
_PRINT_LETTER = re.compile(r"^[a-z]\d+([a-z]?)$", re.IGNORECASE)
# Vermont's export keeps a link's screen-reader hint in the note.
_LINK_HINT = re.compile(r"\s*opens in a new window", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Link:
    url: str
    media_type: str


@dataclass(frozen=True, slots=True)
class Version:
    note: str
    date: str  # as exported: a plain date, a timestamp, or empty
    links: tuple[Link, ...]


@dataclass(frozen=True, slots=True)
class Choice:
    note: str
    date: str
    links: tuple[Link, ...]  # the supported formats, in order of preference; empty if none
    passed_over: int  # versions skipped as not the bill's text


def is_bill_text(note: str, state: str = "") -> bool:
    note = _LINK_HINT.sub("", note)
    if _READING_THEN_TITLE.match(note):
        return True
    local = _OTHER_DOCUMENT_IN.get(state)
    if _OTHER_DOCUMENT.search(note) or (local is not None and local.search(note)):
        return False
    return not (_AMENDMENT.search(note) and not _AMENDED_BILL.search(note))


def stage(note: str) -> int:
    """The legislative stage a version note names: 1 introduced to 4 enrolled or enacted."""
    return next((rank for rank, pattern in _STAGES if pattern.search(note)), _UNSTAGED)


def _date(version: Version) -> str:
    """The version's date as YYYY-MM-DD, or a date written in its note, or empty."""
    if version.date:
        return version.date[:10]
    match = _NOTE_DATE.search(version.note)
    if match is None:
        return ""
    month, day, year = match.groups()
    return f"{year if len(year) == 4 else '20' + year}-{int(month):02}-{int(day):02}"


def _number(note: str) -> tuple[int, str]:
    """The note's last number outside a date, then New York's print letter."""
    numbers = _NUMBER.findall(_NOTE_DATE.sub("", note))
    letter = _PRINT_LETTER.match(note.strip())
    return (int(numbers[-1]) if numbers else -1, letter.group(1).lower() if letter else "")


def _order(item: tuple[int, Version]) -> tuple[int, str, tuple[int, str], int]:
    i, version = item
    note = _LINK_HINT.sub("", version.note)
    return (stage(note), _date(version), _number(note), i)


def choose_version(versions: Sequence[Version], state: str = "") -> Choice | None:
    """The latest version that is the bill's own text, or None if no version is."""
    bill = [(i, v) for i, v in enumerate(versions) if is_bill_text(v.note, state)]
    if not bill:
        return None
    texts = [b for b in bill if not _SCANNED.match(_LINK_HINT.sub("", b[1].note))] or bill
    _, latest = max(texts, key=_order)
    links = tuple(
        link for kind in SUPPORTED_FORMATS for link in latest.links if link.media_type == kind
    )
    return Choice(latest.note, latest.date, links, passed_over=len(versions) - len(bill))
