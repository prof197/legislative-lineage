"""The queries the semantic-stage sizing reads: matchable sections and an act's related bills."""

from pathlib import Path

import pytest

from lineage import queries

# id: (state, session, identifier, title, sections as (text, boilerplate), eligible, family)
BILLS = {
    "b1": ("OH", "135", "SB 237", "Public expression", [("one two three", None)], True, "f1"),
    "b2": ("OH", "135", "HB 9", "Companion", [("four five", None)], True, "f1"),
    "b3": ("TX", "88", "SB 1", "An Anti-SLAPP act", [("six", None)], True, None),
    "b4": ("TX", "88", "SB 2", "Highways", [("seven eight", None), ("Sec. 9. x", "e")], True, None),
    "b5": ("TX", "88", "HR 3", "Commemorating", [("nine", None)], False, None),
}


@pytest.fixture
def tables(tmp_path: Path) -> Path:
    rows = [
        (bill, ordinal, text, kind)
        for bill, (*_, sections, _, _) in sorted(BILLS.items())
        for ordinal, (text, kind) in enumerate(sections, 1)
    ]
    ids = [r[0] for r in rows]
    n = len(rows)
    with queries.connect_quiet() as con:
        queries.write_sections(
            con,
            parquet_path=tmp_path / "s.parquet",
            columns=queries.SectionColumns(
                ids, [r[1] for r in rows], [None] * n, [r[2] for r in rows],
                [r[2] for r in rows], [True] * n, [r[3] for r in rows],
            ),
        )  # fmt: skip
        bills = sorted(BILLS)
        queries.write_eligibility(
            con, parquet_path=tmp_path / "eligible.parquet", ids=bills,
            eligible=[BILLS[b][5] for b in bills], rules=[1] * len(bills),
        )  # fmt: skip
        in_family = [b for b in bills if BILLS[b][6]]
        queries.write_families(
            con, parquet_path=tmp_path / "families.parquet",
            bill_ids=in_family, families=[str(BILLS[b][6]) for b in in_family],
        )  # fmt: skip
        con.execute(
            "COPY (SELECT unnest($ids) AS id, unnest($st) AS state, unnest($se) AS session, "
            f"unnest($ide) AS identifier, unnest($ti) AS title) TO '{tmp_path / 'b.parquet'}' "
            "(FORMAT parquet)",
            {
                "ids": bills, "st": [BILLS[b][0] for b in bills],
                "se": [BILLS[b][1] for b in bills], "ide": [BILLS[b][2] for b in bills],
                "ti": [BILLS[b][3] for b in bills],
            },
        )  # fmt: skip
    return tmp_path


def test_only_eligible_non_boilerplate_sections_count(tables: Path) -> None:
    glob, eligible = str(tables / "s.parquet"), tables / "eligible.parquet"
    with queries.connect_quiet() as con:
        words = queries.section_word_counts(con, sections_glob=glob, eligible_path=eligible)
        ids = queries.matchable_bill_ids(con, sections_glob=glob, eligible_path=eligible)
        texts = queries.matchable_section_texts(
            con, sections_glob=glob, eligible_path=eligible, ids=["b4", "b5"]
        )

    assert words == [3, 2, 1, 2]
    assert ids == ["b1", "b2", "b3", "b4"]
    assert texts == [("b4", 1, "seven eight")]


def test_related_bills_are_the_named_ones_the_titled_ones_and_their_families(
    tables: Path,
) -> None:
    with queries.connect_quiet() as con:
        got = queries.related_bill_ids(
            con, bills_glob=str(tables / "b.parquet"), families_path=tables / "families.parquet",
            listed=[("OH", "135", "SB 237")], title_regex="slapp",
        )  # fmt: skip

    assert got == ["b1", "b2", "b3"]
