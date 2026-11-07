import json
from pathlib import Path

import numpy as np
import pytest

from lineage import queries
from lineage.shingles import shingle_hashes
from lineage.stoplist import (
    StoplistError,
    apply_stoplist,
    build_stoplist,
    document_frequencies,
    load_stoplist,
    over_threshold,
)

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


def test_apply_stoplist_drops_listed_hashes_from_each_set() -> None:
    sets = [np.array([1, 5, 9], dtype=np.uint32), np.array([5], dtype=np.uint32)]

    kept = apply_stoplist(sets, np.array([5, 7], dtype=np.uint32))

    assert [k.tolist() for k in kept] == [[1, 9], []]


def _write_sections(tables: Path, texts: list[str], eligible: list[bool] | None = None) -> None:
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
    with queries.connect_quiet() as con:
        queries.write_sections(con, parquet_path=tables / "sections" / "s.parquet", columns=columns)
        queries.write_eligibility(
            con,
            parquet_path=tables / "eligible.parquet",
            ids=columns.bill_id,
            eligible=eligible if eligible is not None else [True] * len(texts),
            rules=[1] * len(texts),
        )


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
        tables,
        max_share=0.3,
        workers=1,
        report_path=report,
        manifest_sha256="abc",
        rules_sha256="def",
    )

    shared = sorted(set(shingle_hashes([ENACTING + " that the"])[0].tolist()))
    # The section that is only an enacting clause is flagged boilerplate and not counted. The
    # other three open with the clause and "that the", so their five shared shingles occur in
    # 3 of 9 matchable sections, over 30%.
    assert load_stoplist(tables).tolist() == shared
    assert (result.sections, result.stoplisted) == (9, 5)

    doc = json.loads(report.read_text())
    assert doc["command"] == "uv run lineage boilerplate"
    assert doc["manifest_sha256"] == "abc"
    assert doc["rules_sha256"] == "def"
    assert doc["sections"] == 9
    assert doc["max_share"] == 0.3
    assert doc["boilerplate_sections"] == {"enacting clause": 1}
    assert doc["top_shingles"][0] == {
        "shingle": "be it enacted by the",
        "sections": 3,
        "share": 0.333333,
    }
    assert {row["max_share"] for row in doc["by_threshold"]} >= {0.3}


def test_sections_of_excluded_document_types_are_not_counted(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    texts = [
        ENACTING,
        *(f"{ENACTING} that the {thing}" for thing in ("department shall act", "board shall meet")),
        "the department shall publish a report",
        "the department shall publish a list",
        *(f"unrelated text number {i} about water rights" for i in range(3)),
    ]
    # The two enacted-clause sections belong to bills of an excluded type.
    _write_sections(tables, texts, eligible=[True, False, False, True, True, True, True, True])

    result = build_stoplist(
        tables,
        max_share=0.3,
        workers=1,
        report_path=tmp_path / "r.json",
        manifest_sha256="abc",
        rules_sha256="def",
    )

    # Matchable: the two "publish" sections and three unrelated ones. Only the shingle the
    # publish sections share is in more than 30% of them.
    assert result.sections == 5
    assert (
        load_stoplist(tables).tolist()
        == shingle_hashes(["the department shall publish a"])[0].tolist()
    )


def test_refuses_to_build_without_the_eligibility_table(tables: Path, tmp_path: Path) -> None:
    (tables / "eligible.parquet").unlink()

    with pytest.raises(StoplistError, match="lineage eligible"):
        build_stoplist(
            tables,
            max_share=0.3,
            workers=1,
            report_path=tmp_path / "r.json",
            manifest_sha256="abc",
            rules_sha256="def",
        )
