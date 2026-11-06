import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from lineage import queries
from lineage.bands import SWEEP_CAPS, load_band_index
from lineage.disk import InsufficientDiskSpaceError
from lineage.index import build_index
from lineage.lsh import BANDS, BUCKET_CAP, build_band_index
from lineage.signatures import load_signatures

COPIED = "a person who knowingly sells a catalytic converter without a receipt commits a felony"
SECTIONS = {
    # bill: (state, normalized text of its one section)
    "b1": ("AL", COPIED),
    "b2": ("AK", COPIED),
    "b3": ("AL", COPIED),
    "b4": ("AK", "the department of revenue shall publish the rates for each fiscal year"),
}


@pytest.fixture
def tables(tmp_path: Path) -> Path:
    tables = tmp_path / "tables"
    (tables / "sections").mkdir(parents=True)
    (tables / "bills").mkdir()
    ids = sorted(SECTIONS)
    texts = [SECTIONS[b][1] for b in ids]
    n = len(ids)
    with duckdb.connect() as con:
        queries.write_sections(
            con,
            parquet_path=tables / "sections" / "s.parquet",
            columns=queries.SectionColumns(
                ids, [1] * n, [None] * n, texts, texts, [True] * n, [None] * n
            ),
        )
        queries.write_eligibility(
            con,
            parquet_path=tables / "eligible.parquet",
            ids=ids,
            eligible=[True] * n,
            rules=[1] * n,
        )
        queries.write_stoplist(
            con, parquet_path=tables / "stoplist.parquet", hashes=[], sections=[]
        )
        # Only the columns the band report reads.
        con.execute(
            "COPY (SELECT unnest($ids) AS id, unnest($states) AS state) TO "
            f"'{tables / 'bills' / 'b.parquet'}' (FORMAT parquet)",
            {"ids": ids, "states": [SECTIONS[b][0] for b in ids]},
        )
    return tables


def test_writes_one_report_for_signatures_and_bands(tables: Path, tmp_path: Path) -> None:
    report = tmp_path / "index.json"

    doc = build_index(tables, workers=1, floor_bytes=0, report_path=report)

    assert json.loads(report.read_text()) == doc
    assert doc["command"] == "uv run lineage index"
    assert doc["sections"]["signed"] == 4
    lsh = doc["lsh"]
    assert (lsh["bands"], lsh["rows"], lsh["bucket_cap"]) == (32, 4, BUCKET_CAP)
    assert lsh["index_bytes"] == 4 * BANDS * 12
    assert [entry["cap"] for entry in lsh["cap_sweep"]] == list(SWEEP_CAPS)


def test_identical_sections_are_candidates_counted_by_state(tables: Path, tmp_path: Path) -> None:
    doc = build_index(tables, workers=1, floor_bytes=0, report_path=tmp_path / "index.json")

    # b1, b2 and b3 share every band: (b1, b2) and (b2, b3) cross states, (b1, b3) does not.
    assert doc["lsh"]["candidate_pairs"] == {"all": 3, "cross_state": 2, "same_state": 1}
    assert doc["lsh"]["largest_bucket"] == 3
    assert doc["lsh"]["capped"] == {"buckets": 0, "section_memberships": 0, "pairs": 0}


def test_the_stored_band_index_memory_maps_and_matches_the_signatures(
    tables: Path, tmp_path: Path
) -> None:
    build_index(tables, workers=1, floor_bytes=0, report_path=tmp_path / "index.json")

    stored = load_band_index(tables)
    expected = build_band_index(load_signatures(tables).values)

    assert isinstance(stored.keys, np.memmap)
    assert np.array_equal(stored.keys, expected.keys)
    assert np.array_equal(stored.rows, expected.rows)


def test_refuses_to_fill_the_disk(tables: Path, tmp_path: Path) -> None:
    with pytest.raises(InsufficientDiskSpaceError):
        build_index(tables, workers=1, floor_bytes=1 << 62, report_path=tmp_path / "index.json")

    assert not (tables / "bands").exists()
    assert not (tmp_path / "index.json").exists()
