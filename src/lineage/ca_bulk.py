"""California's bill text from the Legislature's bulk download.

leginfo.legislature.ca.gov answers this project's request for its robots.txt with 403, which
the fetcher reads as a bar, and the project does not change its User-Agent to get around that.
The Legislature also publishes every bill version as data at downloads.leginfo.legislature.ca.gov,
one zip per biennium (its "pubinfo" files), which California law places in the public domain.
Each zip holds BILL_VERSION_TBL.dat, a tab-separated table with one row per version, and one
.lob file per version holding the bill in CAML, the Legislative Counsel's XML.

One biennium at a time: the zip is downloaded after a free-space check against its declared
size, the versions wanted are read straight from it, and it is deleted once their text is
stored. The store has the fetcher's layout, one JSON line per record. See DESIGN.md#dr-6.
"""

import hashlib
import json
import re
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lineage.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.extract import ExtractError, html_text
from lineage.fetch import download
from lineage.text_fetch import Fetched

HOST = "downloads.leginfo.legislature.ca.gov"
VERSIONS_TABLE = "BILL_VERSION_TBL.dat"
# BILL_VERSION_TBL's columns, in the order capublic.sql (pubinfo_load.zip) creates them.
_COLUMNS = (
    "bill_version_id", "bill_id", "version_num", "bill_version_action_date",
    "bill_version_action", "request_num", "subject", "vote_required", "appropriation",
    "fiscal_committee", "local_program", "substantive_changes", "urgency", "taxlevy", "bill_xml",
    "active_flg", "trans_uid", "trans_update",
)  # fmt: skip
# Open States' "20252026 Special Session 1" is the bulk data's session 2025-2026, special 1.
_SESSION = re.compile(r"^(\d{8})(?: Special Session (\d+))?$")
_IDENTIFIER = re.compile(r"^([A-Z]+) ?(\d+)$")
# The measure's metadata: authors, title, the Legislative Counsel's digest, vote keys. The
# digest summarizes the bill in the counsel's words, so only the measure itself is kept.
_DESCRIPTION = re.compile(rb"<caml:Description\b.*?</caml:Description>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class CaVersion:
    bill_id: str  # "202520260AB76": session years, special session number, type, number
    version_num: int  # 99 as introduced, counting down with each version
    action_date: str
    action: str  # "Introduced", "Amended Assembly", "Enrolled", "Chaptered", ...
    lob: str  # the member of the zip that holds this version's CAML
    active: bool


def bill_id(session: str, identifier: str) -> str | None:
    """The bulk data's bill id for an Open States session and identifier, if it maps."""
    s, i = _SESSION.match(session), _IDENTIFIER.match(identifier.replace("\u00a0", " ").strip())
    if s is None or i is None:
        return None
    years, special = s.groups()
    kind, number = i.groups()
    return f"{years}{special or 0}{kind}{int(number)}"


def zip_year(session: str) -> int | None:
    s = _SESSION.match(session)
    return int(s.group(1)[:4]) if s else None


def _field(raw: bytes) -> str | None:
    value = raw.decode("utf-8", errors="replace")
    if value == "NULL":
        return None
    return value[1:-1] if value.startswith("`") and value.endswith("`") else value


def parse_versions(lines: Iterable[bytes]) -> Iterator[CaVersion]:
    """BILL_VERSION_TBL rows: tab-separated, text fields in backticks, NULL for none."""
    for line in lines:
        fields = line.rstrip(b"\r\n").split(b"\t")
        if len(fields) != len(_COLUMNS):
            continue
        row = dict(zip(_COLUMNS, (_field(f) for f in fields), strict=True))
        if not row["bill_id"] or not row["bill_xml"]:
            continue
        yield CaVersion(
            bill_id=row["bill_id"],
            version_num=int(row["version_num"] or 0),
            action_date=(row["bill_version_action_date"] or "")[:10],
            action=row["bill_version_action"] or "",
            lob=row["bill_xml"],
            active=row["active_flg"] == "Y",
        )


def latest_versions(versions: Iterable[CaVersion], wanted: set[str]) -> dict[str, CaVersion]:
    """Each wanted bill's latest active version: by action date, then the lower version number,
    which the Legislative Counsel assigns counting down."""
    latest: dict[str, CaVersion] = {}
    for v in versions:
        if v.bill_id not in wanted or not v.active:
            continue
        current = latest.get(v.bill_id)
        if current is None or (v.action_date, -v.version_num) > (
            current.action_date,
            -current.version_num,
        ):
            latest[v.bill_id] = v
    return latest


def caml_text(xml: bytes) -> str:
    """The measure's text from its CAML, without the metadata and digest."""
    text = html_text(_DESCRIPTION.sub(b"", xml), "utf-8")
    if not text:
        raise ExtractError("no text in the CAML")
    return text


def _stored_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["id"] for line in f}


def process_zip(
    zip_path: Path,
    url: str,
    wanted: dict[str, str],
    store: Path,
    *,
    floor_bytes: int,
    check: Callable[[str], bool] | None = None,
) -> dict[str, int]:
    """Store the latest version's text for each wanted record. `wanted` maps a bulk bill id to
    the Open States record id; text that fails `check` (the usable-text bar) is marked
    not_usable, as the fetcher marks it."""
    counts = {"ok": 0, "not_usable": 0, "extract_error": 0, "not_in_bulk": 0}
    with zipfile.ZipFile(zip_path) as z:
        with z.open(VERSIONS_TABLE) as f:
            latest = latest_versions(parse_versions(f), set(wanted))
        with store.open("a") as out:
            for bid, record_id in sorted(wanted.items()):
                at = datetime.now(UTC).isoformat(timespec="seconds")
                version = latest.get(bid)
                if version is None:
                    row = Fetched(
                        record_id, url, url, "xml", "", "", "not_in_bulk", None, None, 0, None,
                        None, 0.0, at,
                    )  # fmt: skip
                    out.write(json.dumps(asdict(row)) + "\n")
                    counts["not_in_bulk"] += 1
                    continue
                member = f"{url}#{version.lob}"
                try:
                    text = caml_text(z.read(version.lob))
                    outcome = "ok" if check is None or check(text) else "not_usable"
                    error = None
                except (ExtractError, KeyError) as e:
                    text, outcome, error = None, "extract_error", str(e)
                digest = hashlib.sha256(text.encode()).hexdigest() if text is not None else None
                row = Fetched(
                    record_id, member, member, "xml", version.action, version.action_date,
                    outcome, None, error, z.getinfo(version.lob).file_size, text, digest, 0.0,
                    at,
                )  # fmt: skip
                line = json.dumps(asdict(row), ensure_ascii=False) + "\n"
                require_free_space(store.parent, needed=len(line.encode()), floor=floor_bytes)
                out.write(line)
                counts[outcome] += 1
    return counts


def fetch_california(
    records: Iterable[tuple[str, str, str]],
    store_dir: Path,
    raw_dir: Path,
    *,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    check: Callable[[str], bool] | None = None,
) -> dict[int, dict[str, int]]:
    """Text for California records (id, session, identifier) from the bulk download, one
    biennium at a time; return the outcome counts per biennium."""
    store_dir.mkdir(parents=True, exist_ok=True)
    store = store_dir / f"{HOST}.jsonl"
    done = _stored_ids(store)
    by_year: dict[int, dict[str, str]] = {}
    unmapped = 0
    for record_id, session, identifier in records:
        bid, year = bill_id(session, identifier), zip_year(session)
        if bid is None or year is None:
            unmapped += 1
            continue
        if record_id not in done:
            by_year.setdefault(year, {})[bid] = record_id
    results: dict[int, dict[str, int]] = {}
    for year, wanted in sorted(by_year.items()):
        url = f"https://{HOST}/pubinfo_{year}.zip"
        zip_path = raw_dir / f"pubinfo_{year}.zip"
        try:
            download(url, zip_path, floor=floor_bytes)
            results[year] = process_zip(
                zip_path, url, wanted, store, floor_bytes=floor_bytes, check=check
            )
        finally:
            zip_path.unlink(missing_ok=True)
        require_free_space(store_dir, needed=0, floor=floor_bytes)
    if unmapped:
        results[0] = {"unmapped": unmapped}
    return results
