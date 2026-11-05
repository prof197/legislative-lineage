from pathlib import Path

import duckdb
import pytest

from lineage.disk import InsufficientDiskSpaceError
from lineage.ingest import ingest
from lineage.sectioning import SectioningError, build_sections
from lineage.snapshot import scan, write_manifest
from tests.conftest import SESSIONS


@pytest.fixture
def tables(raw_dir: Path, tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    write_manifest(scan(raw_dir), manifest)
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)
    return out


def _sections(tables: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW sections AS SELECT * FROM read_parquet('{tables / 'sections' / '*.parquet'}')"
    )
    con.execute(
        f"CREATE VIEW bills AS SELECT * FROM read_parquet('{tables / 'bills' / '*.parquet'}')"
    )
    return con


def test_splits_each_bill_at_its_headings(tables: Path) -> None:
    result = build_sections(tables, workers=2, floor_bytes=0)

    rows = (
        _sections(tables)
        .execute(
            "SELECT ordinal, heading, by_heading, starts_with(normalized, 'section') "
            "FROM sections JOIN bills ON bill_id = id WHERE identifier = 'SB 238' ORDER BY ordinal"
        )
        .fetchall()
    )

    # The bill quotes the act it amends, which opens with "SECTION 3.; that is not a heading.
    assert rows == [
        (0, None, True, False),
        (1, "SECTION 1.", True, True),
        (2, "SECTION 2.", True, True),
    ]
    assert (result.bills, result.with_text, result.by_heading) == (5, 3, 1)


def test_bills_without_text_have_no_sections(tables: Path) -> None:
    build_sections(tables, workers=1, floor_bytes=0)

    states = _sections(tables).execute(
        "SELECT DISTINCT state FROM sections JOIN bills ON bill_id = id ORDER BY state"
    )

    # AK's text is blank and HI's is missing.
    assert states.fetchall() == [("AL",), ("GA",)]


def test_rerun_replaces_the_previous_table(tables: Path) -> None:
    first = build_sections(tables, workers=1, floor_bytes=0)
    (tables / "sections" / "stale.parquet").write_bytes(b"")

    second = build_sections(tables, workers=1, floor_bytes=0)

    assert second.sections == first.sections
    assert not (tables / "sections" / "stale.parquet").exists()


def test_refuses_to_run_below_the_disk_floor(tables: Path) -> None:
    with pytest.raises(InsufficientDiskSpaceError):
        build_sections(tables, workers=1, floor_bytes=1 << 60)
    assert not (tables / "sections").exists()


def test_needs_the_bills_table(tmp_path: Path) -> None:
    with pytest.raises(SectioningError, match="lineage ingest"):
        build_sections(tmp_path, workers=1, floor_bytes=0)
