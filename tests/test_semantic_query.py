"""The semantic stage in a query: matches by meaning, merged with near-duplicates by stage.

The model is replaced by a bag of words that ignores a marker suffix (tests.fakes), so a copy
whose words carry the marker reads the same to it while its shingles, and so the near-duplicate
stage, no longer agree."""

from pathlib import Path

import pytest

from lineage import queries
from lineage.query import query_text, regrouped
from lineage.semantic import CORROBORATING, THRESHOLD, SemanticSearch, search_sections
from tests.fakes import MARK, bag_of_words_search


@pytest.fixture(scope="module")
def search(corpus: Path) -> SemanticSearch:
    return bag_of_words_search(corpus)


def _raw_text(tables: Path, identifier: str) -> str:
    with queries.connect_quiet() as con:
        row = con.execute(
            "SELECT text FROM read_parquet($p) WHERE identifier = $i",
            {"p": str(tables / "bills" / "*.parquet"), "i": identifier},
        ).fetchone()
    assert row is not None
    return str(row[0])


def _marked(text: str) -> str:
    # Every other long lowercase word carries the marker; headings are left as they are.
    marked: list[str] = []
    for i, word in enumerate(text.split(" ")):
        long_word = word.isalpha() and word.islower() and len(word) > 3
        marked.append(word + MARK if long_word and i % 2 else word)
    return " ".join(marked)


def test_a_bill_is_found_by_both_stages(corpus: Path, search: SemanticSearch) -> None:
    result = query_text(_raw_text(corpus, "SB 238"), corpus, semantic=search)

    top = result.matches[0]
    assert (top.state, top.identifier) == ("GA", "SB 238")
    assert top.share == 1.0
    assert top.sections_by_stage == {
        "near_duplicate": result.signed,
        "semantic": result.searched,
    }
    assert top.best_cosine == pytest.approx(1.0, abs=1e-4)


def test_a_reworded_copy_is_found_by_meaning_alone(corpus: Path, search: SemanticSearch) -> None:
    text = _marked(_raw_text(corpus, "SB 238"))

    near = query_text(text, corpus)
    both = query_text(text, corpus, semantic=search)

    assert all((m.state, m.identifier) != ("GA", "SB 238") for m in near.matches)
    top = both.matches[0]
    assert (top.state, top.identifier) == ("GA", "SB 238")
    assert set(top.sections_by_stage) == {"semantic"}
    assert top.best_estimate == 0.0


def test_unrelated_text_finds_nothing_by_meaning(search: SemanticSearch) -> None:
    text = " ".join(f"word{i}" for i in range(300))

    assert search_sections([text], search) == []


def test_without_the_semantic_stage_the_share_counts_signed_sections(corpus: Path) -> None:
    result = query_text(_raw_text(corpus, "SB 238"), corpus)

    assert result.searched == result.signed
    assert set(result.matches[0].sections_by_stage) == {"near_duplicate"}


def test_a_stricter_threshold_regroups_the_same_hits_and_only_drops(
    corpus: Path, search: SemanticSearch
) -> None:
    text = _marked(_raw_text(corpus, "SB 238"))
    result = query_text(text, corpus, semantic=search)
    cosines = sorted({h[3] for h in result.hits.by_meaning}) if result.hits else []

    assert cosines
    assert regrouped(result, corpus, THRESHOLD) == result.matches
    above = regrouped(result, corpus, cosines[-1] + 1e-6)
    assert all("semantic" not in m.sections_by_stage for m in above)
    families = {m.family for m in result.matches}
    for t in cosines:
        if t >= THRESHOLD:
            assert {m.family for m in regrouped(result, corpus, t)} <= families


def test_a_query_keeps_its_hits_by_meaning_down_to_the_cosine_that_corroborates(
    corpus: Path, search: SemanticSearch
) -> None:
    # The state boilerplate flag's exemption reads them, whatever threshold the query lists at.
    text = _marked(_raw_text(corpus, "SB 238"))
    result = query_text(text, corpus, semantic=search)
    strict = query_text(text, corpus, semantic=search, semantic_threshold=0.999)

    assert result.hits is not None
    assert strict.hits is not None
    assert strict.hits.by_meaning == result.hits.by_meaning
    assert min(h[3] for h in result.hits.by_meaning) >= CORROBORATING
    assert regrouped(result, corpus, 0.999) == strict.matches
