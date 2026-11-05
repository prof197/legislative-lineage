import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from lineage import queries
from lineage.disk import InsufficientDiskSpaceError
from lineage.minhash import NUM_PERM, permutations, signatures
from lineage.shingles import section_shingles
from lineage.signatures import build_index, load_signatures
from lineage.stoplist import StoplistError, apply_stoplist

STOPLISTED = "is hereby amended to read as follows"
TEXTS = {
    # (bill, ordinal): normalized text
    ("b1", 1): "the department of revenue shall publish the rates each year",
    ("b1", 2): "the department of revenue shall publish the rates each month",
    ("b2", 1): "too short to shingle",
    ("b2", 2): STOPLISTED,
    ("b3", 1): "a person who violates this section commits a misdemeanor",
    # A boilerplate section and a section of an excluded document type: neither is signed.
    ("b3", 2): "this act takes effect on july first of next year",
    ("r1", 1): "commending the state champions for a remarkable season",
}
EXCLUDED_BILL = "r1"
BOILERPLATE = ("b3", 2)


@pytest.fixture
def tables(tmp_path: Path) -> Path:
    tables = tmp_path / "tables"
    (tables / "sections").mkdir(parents=True)
    keys = sorted(TEXTS)
    # Two sections files, as the real table has one or more per archive.
    halves = {"a.parquet": keys[:3], "b.parquet": keys[3:]}
    with duckdb.connect() as con:
        for name, part in halves.items():
            columns = queries.SectionColumns(
                bill_id=[b for b, _ in part],
                ordinal=[o for _, o in part],
                heading=[None] * len(part),
                text=[TEXTS[k] for k in part],
                normalized=[TEXTS[k] for k in part],
                by_heading=[True] * len(part),
                boilerplate=["effective date" if k == BOILERPLATE else None for k in part],
            )
            queries.write_sections(con, parquet_path=tables / "sections" / name, columns=columns)
        bills = sorted({b for b, _ in keys})
        queries.write_eligibility(
            con,
            parquet_path=tables / "eligible.parquet",
            ids=bills,
            eligible=[b != EXCLUDED_BILL for b in bills],
            rules=[1] * len(bills),
        )
        stoplist = section_shingles([STOPLISTED])[0]
        queries.write_stoplist(
            con,
            parquet_path=tables / "stoplist.parquet",
            hashes=stoplist.tolist(),
            sections=[100] * stoplist.size,
        )
    return tables


def test_signs_every_matchable_section_that_has_a_shingle(tables: Path, tmp_path: Path) -> None:
    build_index(tables, workers=1, floor_bytes=0, report_path=tmp_path / "index.json")

    table = load_signatures(tables)

    keys = list(zip(table.bill_ids, table.ordinals.tolist(), strict=True))
    assert keys == [("b1", 1), ("b1", 2), ("b3", 1)]
    assert table.values.shape == (3, NUM_PERM)
    assert table.values.dtype == np.uint32
    for row, key in zip(table.values, keys, strict=True):
        (expected,) = signatures(section_shingles([TEXTS[key]]), permutations())
        assert np.array_equal(row, expected)


def test_the_stoplist_is_applied_before_signing(tables: Path, tmp_path: Path) -> None:
    text = TEXTS[("b1", 1)] + " " + STOPLISTED
    (tables / "sections" / "b.parquet").unlink()
    with duckdb.connect() as con:
        columns = queries.SectionColumns(["b9"], [1], [None], [text], [text], [True], [None])
        queries.write_sections(con, parquet_path=tables / "sections" / "b.parquet", columns=columns)
        queries.write_eligibility(
            con,
            parquet_path=tables / "eligible.parquet",
            ids=["b1", "b9"],
            eligible=[True, True],
            rules=[1, 1],
        )
        stoplist = np.array(
            queries.read_stoplist(con, parquet_path=tables / "stoplist.parquet"), dtype=np.uint32
        )

    build_index(tables, workers=1, floor_bytes=0, report_path=tmp_path / "index.json")

    table = load_signatures(tables)
    (kept,) = apply_stoplist(section_shingles([text]), stoplist)
    assert kept.size < section_shingles([text])[0].size
    assert np.array_equal(table.values[-1], signatures([kept], permutations())[0])


def test_reports_what_was_signed_and_why_the_rest_was_not(tables: Path, tmp_path: Path) -> None:
    report = tmp_path / "index.json"

    doc = build_index(tables, workers=2, floor_bytes=0, report_path=report)

    assert json.loads(report.read_text()) == doc
    assert doc["command"] == "uv run lineage index"
    assert doc["minhash"] == {"num_perm": NUM_PERM, "seed": 42, "prime": "2^61 - 1"}
    assert doc["sections"] == {
        "matchable": 5,
        "signed": 3,
        "too_short": 1,
        "all_stoplisted": 1,
    }
    # The two b1 sections have 6 shingles each and b3's has 5.
    assert doc["shingle_set_size"] == {"median": 6, "p90": 6, "p99": 6, "max": 6}
    assert doc["signature_bytes"] == 3 * NUM_PERM * 4


def test_each_signature_file_memory_maps(tables: Path, tmp_path: Path) -> None:
    build_index(tables, workers=1, floor_bytes=0, report_path=tmp_path / "index.json")

    files = sorted((tables / "signatures").glob("*.npy"))
    mapped = np.concatenate([np.load(f, mmap_mode="r") for f in files])

    assert np.array_equal(mapped, load_signatures(tables).values)


def test_refuses_without_a_stoplist(tables: Path, tmp_path: Path) -> None:
    (tables / "stoplist.parquet").unlink()

    with pytest.raises(StoplistError, match="lineage boilerplate"):
        build_index(tables, workers=1, floor_bytes=0, report_path=tmp_path / "index.json")


def test_refuses_to_fill_the_disk_and_leaves_no_partial_output(
    tables: Path, tmp_path: Path
) -> None:
    with pytest.raises(InsufficientDiskSpaceError):
        build_index(tables, workers=1, floor_bytes=1 << 62, report_path=tmp_path / "index.json")

    assert sorted(p.name for p in tables.iterdir()) == [
        "eligible.parquet",
        "sections",
        "stoplist.parquet",
    ]
