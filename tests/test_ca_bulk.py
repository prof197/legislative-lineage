"""California's bulk download: the version table, the latest version, CAML text."""

import json
import zipfile
from pathlib import Path

from lineage.ca_bulk import bill_id, caml_text, latest_versions, parse_versions, process_zip

ROWS = [
    b"`20250AB7699INT`\t`202520260AB76`\t99\t2024-12-16 00:00:00\t`Introduced`\tNULL\t"
    b"`Surplus land.`\tNULL\tNULL\t`No`\tNULL\tNULL\tNULL\tNULL\tBILL_VERSION_TBL_1.lob\t`Y`\t"
    b"`LEG_ESI`\t2024-12-16 10:00:00\n",
    b"`20250AB7698AMD`\t`202520260AB76`\t98\t2025-03-10 00:00:00\t`Amended Assembly`\tNULL\t"
    b"`Surplus land.`\tNULL\tNULL\t`No`\tNULL\tNULL\tNULL\tNULL\tBILL_VERSION_TBL_2.lob\t`Y`\t"
    b"`LEG_ESI`\t2025-03-10 10:00:00\n",
    b"`20250AB7697AMD`\t`202520260AB76`\t97\t2025-03-10 00:00:00\t`Amended Senate`\tNULL\t"
    b"`Surplus land.`\tNULL\tNULL\t`No`\tNULL\tNULL\tNULL\tNULL\tBILL_VERSION_TBL_3.lob\t`N`\t"
    b"`LEG_ESI`\t2025-03-10 11:00:00\n",
]
CAML = (
    b'<?xml version="1.0" encoding="UTF-8"?><caml:MeasureDoc>'
    b"<caml:Description><caml:Title>An act relating to surplus land.</caml:Title>"
    b"<caml:DigestText><p>Existing law requires a digest.</p></caml:DigestText></caml:Description>"
    b'<caml:Bill id="bill"><caml:Content><p>SECTION 1. Section 54221 is amended to read:</p>'
    b"<p>The agency shall <strike>not</strike> dispose of land.</p></caml:Content></caml:Bill>"
    b"</caml:MeasureDoc>"
)


def test_open_states_sessions_and_identifiers_map_to_bulk_bill_ids() -> None:
    assert bill_id("20252026", "AB 76") == "202520260AB76"
    assert bill_id("20232024 Special Session 1", "SBX1 2") is None  # no such identifier form
    assert bill_id("20232024 Special Session 1", "AB 2") == "202320241AB2"


def test_the_latest_active_version_is_the_newest_by_date_then_version_number() -> None:
    latest = latest_versions(parse_versions(ROWS), {"202520260AB76"})

    # Version 97 is newer but inactive (withdrawn), so version 98 stands.
    assert (latest["202520260AB76"].version_num, latest["202520260AB76"].action) == (
        98,
        "Amended Assembly",
    )


def test_caml_text_keeps_the_measure_and_drops_the_digest_and_struck_text() -> None:
    assert caml_text(CAML).splitlines() == [
        "SECTION 1. Section 54221 is amended to read:",
        "The agency shall dispose of land.",
    ]


def test_a_biennium_zip_yields_one_stored_row_per_record(tmp_path: Path) -> None:
    zip_path = tmp_path / "pubinfo_2025.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("BILL_VERSION_TBL.dat", b"".join(ROWS))
        for i in (1, 2, 3):
            z.writestr(f"BILL_VERSION_TBL_{i}.lob", CAML)
    store = tmp_path / "store.jsonl"

    counts = process_zip(
        zip_path, "https://bulk.example/pubinfo_2025.zip",
        {"202520260AB76": "ocd-bill/1", "202520260AB77": "ocd-bill/2"}, store, floor_bytes=0,
    )  # fmt: skip

    assert counts == {"ok": 1, "not_usable": 0, "extract_error": 0, "not_in_bulk": 1}
    row, absent = (json.loads(line) for line in store.read_text().splitlines())
    assert (row["id"], row["note"], row["date"]) == ("ocd-bill/1", "Amended Assembly", "2025-03-10")
    assert row["url"].endswith("#BILL_VERSION_TBL_2.lob")
    # Stored too, so a rerun does not download the biennium again for a bill it lacks.
    assert (absent["id"], absent["outcome"], absent["text"]) == ("ocd-bill/2", "not_in_bulk", None)
