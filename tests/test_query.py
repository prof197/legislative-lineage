"""`lineage query` end to end: the fixture archives through every pipeline command."""

from pathlib import Path

from typer.testing import CliRunner

from lineage import queries
from lineage.cli import app
from lineage.query import query_text

runner = CliRunner()


def _run(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def _raw_text(tables: Path, identifier: str) -> str:
    with queries.connect_quiet() as con:
        row = con.execute(
            "SELECT text FROM read_parquet($p) WHERE identifier = $i",
            {"p": str(tables / "bills" / "*.parquet"), "i": identifier},
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_a_bill_finds_itself_with_every_section_matched(corpus: Path) -> None:
    result = query_text(_raw_text(corpus, "SB 238"), corpus)

    assert result.signed > 1
    top = result.matches[0]
    assert (top.state, top.identifier) == ("GA", "SB 238")
    assert top.sections_matched == result.signed
    assert top.share == 1.0
    assert top.best_estimate == 1.0


def test_an_edited_copy_still_finds_its_source(corpus: Path) -> None:
    source = _raw_text(corpus, "SB 238")
    # Change one word in every fifty, as a state adapting a model bill might.
    words = source.split(" ")
    edited = " ".join("Wyoming" if i % 50 == 25 else w for i, w in enumerate(words))

    result = query_text(edited, corpus)

    top = result.matches[0]
    assert (top.state, top.identifier) == ("GA", "SB 238")
    assert top.share >= 0.5
    assert top.best_estimate < 1.0


def test_unrelated_text_finds_nothing(corpus: Path) -> None:
    text = "Section 1. " + " ".join(f"word{i}" for i in range(300))

    result = query_text(text, corpus)

    assert result.signed == 1
    assert result.matches == []


def test_the_command_prints_the_relatives(corpus: Path, tmp_path: Path) -> None:
    bill = tmp_path / "bill.txt"
    bill.write_text(_raw_text(corpus, "SB 238"))

    output = _run("query", str(bill), "--tables", str(corpus))

    assert "SB 238" in output
    assert "100%" in output


def test_the_command_says_when_nothing_matches(corpus: Path, tmp_path: Path) -> None:
    bill = tmp_path / "bill.txt"
    bill.write_text("Section 1. " + " ".join(f"word{i}" for i in range(300)))

    output = _run("query", str(bill), "--tables", str(corpus))

    assert "No near-duplicates" in output


def test_the_command_fails_cleanly_without_an_index(tmp_path: Path) -> None:
    bill = tmp_path / "bill.txt"
    bill.write_text("Section 1. Some text.")

    result = runner.invoke(app, ["query", str(bill), "--tables", str(tmp_path / "none")])

    assert result.exit_code == 1
    assert "lineage index" in result.output
