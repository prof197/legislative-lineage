"""Which version of a bill the text layer fetches: the latest text of the bill itself.

Open States lists every document a legislature posts for a bill among its versions: the bill as
introduced, engrossed and enrolled, but also floor amendments, fiscal and impact notes,
transmittal letters and dispositions. Louisiana's stored text is often the latest of these, a
one-page amendment instead of the bill. The rule keeps the versions whose note names the bill's
own text and takes the latest by date, the export's order breaking ties. For a passed bill that
is the enrolled or chaptered text. See DESIGN.md#dr-6.
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


def is_bill_text(note: str) -> bool:
    note = _LINK_HINT.sub("", note)
    if _READING_THEN_TITLE.match(note):
        return True
    if _OTHER_DOCUMENT.search(note):
        return False
    return not (_AMENDMENT.search(note) and not _AMENDED_BILL.search(note))


def choose_version(versions: Sequence[Version]) -> Choice | None:
    """The latest version that is the bill's own text, or None if no version is."""
    bill = [(v.date[:10], i, v) for i, v in enumerate(versions) if is_bill_text(v.note)]
    if not bill:
        return None
    texts = [b for b in bill if not _SCANNED.match(_LINK_HINT.sub("", b[2].note))] or bill
    *_, latest = max(texts, key=lambda t: (t[0], t[1]))
    links = tuple(
        link for kind in SUPPORTED_FORMATS for link in latest.links if link.media_type == kind
    )
    return Choice(latest.note, latest.date, links, passed_over=len(versions) - len(bill))
