"""The texts table: fetched text where the text layer kept some, else the exported text."""

import json
from pathlib import Path

from lineage import queries


def _bills(path: Path) -> None:
    with queries.connect_quiet() as con:
        con.execute(
            "COPY (SELECT * FROM (VALUES "
            "('b1', 'Exported text one.', 'https://leg.example/b1.htm', "
            "  [{'note': 'Introduced', 'date': '', 'links': "
            "    [{'url': 'https://leg.example/b1.htm', 'media_type': 'text/html'}]}]), "
            "('b2', NULL, NULL, []), "
            "('b3', NULL, NULL, [])"
            ") t(id, text, text_url, versions)) TO '" + str(path) + "' (FORMAT parquet)"
        )


def test_fetched_text_replaces_the_exported_text_and_names_its_source(tmp_path: Path) -> None:
    bills, fetched, out = tmp_path / "b.parquet", tmp_path / "f.jsonl", tmp_path / "t.parquet"
    _bills(bills)
    row = {
        "id": "b2", "text": "Fetched text.", "source": "legislature",
        "url": "https://leg.example/b2.pdf", "version": "Enrolled", "date": None,
    }  # fmt: skip
    fetched.write_text(json.dumps(row) + "\n")

    with queries.connect_quiet() as con:
        queries.write_texts(con, bills_path=bills, fetched_path=fetched, out_path=out)
        rows = con.execute(
            "SELECT id, text, text_source, text_url, text_version FROM read_parquet($p) "
            "ORDER BY id",
            {"p": str(out)},
        ).fetchall()

    assert rows == [
        ("b1", "Exported text one.", "openstates", "https://leg.example/b1.htm", "Introduced"),
        ("b2", "Fetched text.", "legislature", "https://leg.example/b2.pdf", "Enrolled"),
        ("b3", None, None, None, None),
    ]


def test_the_recall_eval_judges_the_merged_text_not_the_exported_text(tmp_path: Path) -> None:
    bills, texts = tmp_path / "bills.parquet", tmp_path / "texts.parquet"
    eligible, families = tmp_path / "eligible.parquet", tmp_path / "families.parquet"
    with queries.connect_quiet() as con:
        con.execute(
            "COPY (SELECT * FROM (VALUES "
            "('b1', 'NJ', '2025', 'A 1', 'passed', DATE '2025-06-30', "
            "  [{'classification': ['passage'], 'body': 'Assembly'}], NULL)"
            ") t(id, state, session, identifier, status, status_date, actions, text)) "
            f"TO '{bills}' (FORMAT parquet)"
        )
        con.execute(
            "COPY (SELECT 'b1' AS id, 'Section 1. Fetched text.' AS text, "
            f"'legislature' AS text_source) TO '{texts}' (FORMAT parquet)"
        )
        con.execute(f"COPY (SELECT 'b1' AS id, true AS eligible) TO '{eligible}' (FORMAT parquet)")
        con.execute(
            f"COPY (SELECT 'b1' AS bill_id, 'f1' AS family) TO '{families}' (FORMAT parquet)"
        )
        [bill] = queries.enactment_bills(
            con, bills_glob=str(bills), texts_glob=str(texts), eligible_path=eligible,
            families_path=families, states=["NJ"], opening="^Amend", chars=100,
            substitute="strike out everything",
        )  # fmt: skip

    assert (bill.text_chars, bill.amendment_only, bill.passage_chambers) == (24, False, 1)
