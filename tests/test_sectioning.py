from pathlib import Path

import duckdb
import pytest

from lineage import queries
from lineage.disk import InsufficientDiskSpaceError
from lineage.ingest import ingest
from lineage.sectioning import SectioningError, build_sections
from lineage.snapshot import scan, write_manifest
from lineage.text_merge import merge_texts
from tests.conftest import SESSIONS


@pytest.fixture
def tables(raw_dir: Path, tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    write_manifest(scan(raw_dir), manifest)
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)
    merge_texts(out, exported_only=True, floor_bytes=0)
    return out


def _sections(tables: Path) -> duckdb.DuckDBPyConnection:
    con = queries.connect_quiet()
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
            "SELECT ordinal, heading, by_heading, starts_with(normalized, 'section'), "
            "typeof(boilerplate), boilerplate "
            "FROM sections JOIN bills ON bill_id = id WHERE identifier = 'SB 238' ORDER BY ordinal"
        )
        .fetchall()
    )

    # The bill quotes the act it amends, which opens with "SECTION 3.; that is not a heading.
    # The preamble holds the enacting clause but also the title, so nothing is boilerplate.
    assert rows == [
        (0, None, True, False, "VARCHAR", None),
        (1, "SECTION 1.", True, True, "VARCHAR", None),
        (2, "SECTION 2.", True, True, "VARCHAR", None),
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


def test_needs_the_texts_table(tmp_path: Path) -> None:
    with pytest.raises(SectioningError, match="lineage text merge"):
        build_sections(tmp_path, workers=1, floor_bytes=0)


def test_each_file_is_split_by_its_own_state_rules(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    (tables / "bills").mkdir(parents=True)
    (tables / "texts").mkdir()
    text = "Section A. Enacted as follows:\n474.540. The first.\n474.542. The second.\n"
    with queries.connect_quiet() as con:
        for state in ("MO", "KS"):
            con.execute(
                f"COPY (SELECT '{state}-1' AS id, '{state}' AS state) "
                f"TO '{tables / 'bills' / f'{state}_2025.parquet'}' (FORMAT parquet)"
            )
            con.execute(
                f"COPY (SELECT '{state}-1' AS id, $text AS text) "
                f"TO '{tables / 'texts' / f'{state}_2025.parquet'}' (FORMAT parquet)",
                {"text": text},
            )

    build_sections(tables, workers=1, floor_bytes=0)

    headings = (
        _sections(tables)
        .execute(
            "SELECT bill_id, list(heading ORDER BY ordinal) FROM sections GROUP BY 1 ORDER BY 1"
        )
        .fetchall()
    )
    # Missouri's RSMo sections are headings in Missouri's bill only.
    assert headings == [("KS-1", [None]), ("MO-1", ["Section A.", "474.540.", "474.542."])]
