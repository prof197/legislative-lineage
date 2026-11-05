import json
from pathlib import Path

import duckdb
import numpy as np
import pytest

from lineage import queries
from lineage.shingles import shingle_hashes
from lineage.stoplist import build_stoplist, document_frequencies, over_threshold

ENACTING = "be it enacted by the general assembly"


def test_document_frequency_counts_sections_not_occurrences() -> None:
    sets = [
        np.array([1, 2, 3], dtype=np.uint32),
        np.array([2, 3], dtype=np.uint32),
        np.array([3], dtype=np.uint32),
    ]

    hashes, counts = document_frequencies(sets)

    assert hashes.tolist() == [1, 2, 3]
    assert counts.tolist() == [1, 2, 3]


def test_the_threshold_is_a_strict_share_of_sections() -> None:
    hashes = np.array([10, 20, 30], dtype=np.uint32)
    counts = np.array([5, 10, 11], dtype=np.int64)

    # 10% of 100 sections is 10; only a shingle in more than 10 sections is stoplisted.
    assert over_threshold(hashes, counts, sections=100, max_share=0.1).tolist() == [30]


def _write_sections(tables: Path, texts: list[str]) -> None:
    (tables / "sections").mkdir(parents=True)
    columns = queries.SectionColumns(
        bill_id=[f"b{i}" for i in range(len(texts))],
        ordinal=[1] * len(texts),
        heading=[None] * len(texts),
        text=texts,
        normalized=texts,
        by_heading=[False] * len(texts),
        boilerplate=["enacting clause" if t == ENACTING else None for t in texts],
    )
    with duckdb.connect() as con:
        queries.write_sections(con, parquet_path=tables / "sections" / "s.parquet", columns=columns)


@pytest.fixture
def tables(tmp_path: Path) -> Path:
    tables = tmp_path / "tables"
    texts = [
        ENACTING,
        ENACTING + " that the department shall act",
        ENACTING + " that the board shall meet",
        ENACTING + " that the fund is created",
        "the department shall publish a report",
        "the department shall publish a list",
        *(f"unrelated text number {i} about water rights" for i in range(4)),
    ]
    _write_sections(tables, texts)
    return tables


def test_builds_the_stoplist_and_a_report(tables: Path, tmp_path: Path) -> None:
    report = tmp_path / "boilerplate.json"

    result = build_stoplist(
        tables, max_share=0.3, workers=1, report_path=report, manifest_sha256="abc"
    )

    stoplist = duckdb.connect().execute(
        f"SELECT hash FROM read_parquet('{tables / 'stoplist.parquet'}') ORDER BY hash"
    )
    enacting = sorted(set(shingle_hashes([ENACTING])[0].tolist()))
    # The three shingles of the enacting clause occur in 4 of 10 sections, over 30%.
    assert [h for (h,) in stoplist.fetchall()] == enacting
    assert (result.sections, result.stoplisted) == (10, 3)

    doc = json.loads(report.read_text())
    assert doc["command"] == "uv run lineage boilerplate"
    assert doc["manifest_sha256"] == "abc"
    assert doc["sections"] == 10
    assert doc["max_share"] == 0.3
    assert doc["boilerplate_sections"] == {"enacting clause": 1}
    assert doc["top_shingles"][0] == {
        "shingle": "be it enacted by the",
        "sections": 4,
        "share": 0.4,
    }
    assert {row["max_share"] for row in doc["by_threshold"]} >= {0.3}
