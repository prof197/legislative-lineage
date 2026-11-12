"""`lineage query` end to end: the fixture archives through every pipeline command."""

from pathlib import Path

from typer.testing import CliRunner

from lineage import queries
from lineage.cli import app
from lineage.query import matchable_sections, query_text

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


def test_the_share_is_of_the_query_words_in_matched_sections(corpus: Path) -> None:
    # A short unrelated section added to a bill lowers its copy's share by its words, not by
    # one section's worth: a provision inside a larger bill is scored by how much of it matched.
    unrelated = "Section 99. " + " ".join(f"word{i}" for i in range(40))
    query = _raw_text(corpus, "SB 238") + "\n\n" + unrelated

    result = query_text(query, corpus)

    _, matchable = matchable_sections(query)
    words = [len(normalized.split()) for _, normalized in matchable]
    top = result.matches[0]
    assert (top.state, top.identifier) == ("GA", "SB 238")
    assert top.sections_matched == len(words) - 1
    assert top.share == (sum(words) - words[-1]) / sum(words)
    assert top.share != (len(words) - 1) / len(words)


def test_unrelated_text_finds_nothing(corpus: Path) -> None:
    text = "Section 1. " + " ".join(f"word{i}" for i in range(300))

    result = query_text(text, corpus)

    assert result.signed == 1
    assert result.matches == []


def test_the_command_prints_the_relatives(corpus: Path, tmp_path: Path) -> None:
    bill = tmp_path / "bill.txt"
    bill.write_text(_raw_text(corpus, "SB 238"))

    output = _run("query", str(bill), "--no-semantic", "--tables", str(corpus))

    assert "SB 238" in output
    assert "100%" in output


def test_the_command_says_when_nothing_matches(corpus: Path, tmp_path: Path) -> None:
    bill = tmp_path / "bill.txt"
    bill.write_text("Section 1. " + " ".join(f"word{i}" for i in range(300)))

    output = _run("query", str(bill), "--no-semantic", "--tables", str(corpus))

    assert "No relatives found" in output


def test_the_command_fails_cleanly_without_an_index(tmp_path: Path) -> None:
    bill = tmp_path / "bill.txt"
    bill.write_text("Section 1. Some text.")

    result = runner.invoke(app, ["query", str(bill), "--tables", str(tmp_path / "none")])

    assert result.exit_code == 1
    assert "lineage index" in result.output


def test_a_query_times_each_stage(corpus: Path) -> None:
    result = query_text(_raw_text(corpus, "SB 238"), corpus)

    assert set(result.seconds) == {"near_duplicate", "scoring"}
    assert all(s >= 0 for s in result.seconds.values())
