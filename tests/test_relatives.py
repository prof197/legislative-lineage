"""A corpus bill's relatives from the build's stored pairs, scored as a query of its text is."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lineage import queries
from lineage.pairs import load_cross_state_pairs
from lineage.query import Match, query_text
from lineage.relatives import RELATIVES_SHOWN, bill_relatives
from lineage.semantic_pairs import build_semantic_pairs, section_keys
from tests.conftest import FAKE_MODEL

# States with their own heading rules: a query, which knows no state, splits their text apart.
_OWN_HEADINGS = {"ND", "NC", "NH", "MO"}


@pytest.fixture
def tables(embedded_commission: Path) -> Path:
    build_semantic_pairs(embedded_commission, model=FAKE_MODEL, floor_bytes=0)
    return embedded_commission


def bill_of(tables: Path, bill_id: str) -> tuple[str, str]:
    with queries.connect_quiet() as con:
        rows = con.execute(
            "SELECT state, text FROM read_parquet($p) WHERE id = $i",
            {"p": str(tables / "bills" / "*.parquet"), "i": bill_id},
        ).fetchall()
    return str(rows[0][0]), str(rows[0][1])


def most_related(tables: Path) -> str:
    """The bill with the most cross-state near-duplicate pairs, of a state with no heading
    rules of its own."""
    pairs = load_cross_state_pairs(tables)
    bills, _ = section_keys(tables)
    numbers: list[int] = np.concatenate([pairs.left, pairs.right]).tolist()
    counts = Counter(bills[s] for s in numbers)
    for bill_id, _ in counts.most_common():
        if bill_of(tables, bill_id)[0] not in _OWN_HEADINGS:
            return bill_id
    raise AssertionError("no related bill in the fixture")


def test_a_bills_relatives_score_as_a_query_of_its_text_does(tables: Path) -> None:
    bill_id = most_related(tables)
    state, text = bill_of(tables, bill_id)

    got = bill_relatives(tables, [bill_id], model=FAKE_MODEL, stages=("near_duplicate",))
    queried = [m for m in query_text(text, tables).matches if m.state != state]

    relatives = got[bill_id]
    assert relatives.matches, "the fixture's bill should have relatives in other states"

    def scored(matches: list[Match]) -> list[tuple[str, float, int, float]]:
        return [(m.family, m.share, m.sections_matched, m.best_estimate) for m in matches]

    assert scored(relatives.matches) == scored(queried)[: len(relatives.matches)]
    assert relatives.families == len(queried)


def test_relatives_by_meaning_carry_the_semantic_stage(tables: Path) -> None:
    bill_id = most_related(tables)

    near = bill_relatives(tables, [bill_id], model=FAKE_MODEL, stages=("near_duplicate",))
    both = bill_relatives(tables, [bill_id], model=FAKE_MODEL)

    assert any("semantic" in m.sections_by_stage for m in both[bill_id].matches)
    assert both[bill_id].families >= near[bill_id].families
    for m in both[bill_id].matches:
        if "semantic" in m.sections_by_stage:
            assert m.best_cosine is not None


def test_no_relative_is_of_the_bills_own_state(tables: Path) -> None:
    bills, _ = section_keys(tables)
    wanted = sorted(set(bills))[:20]

    got = bill_relatives(tables, wanted, model=FAKE_MODEL)

    for bill_id, relatives in got.items():
        state, _ = bill_of(tables, bill_id)
        assert all(m.state != state for m in relatives.matches)


def test_the_shown_relatives_are_capped_and_the_count_is_not(tables: Path) -> None:
    bill_id = most_related(tables)
    every = bill_relatives(tables, [bill_id], model=FAKE_MODEL, shown=10_000)[bill_id]
    assert every.families > 2

    capped = bill_relatives(tables, [bill_id], model=FAKE_MODEL, shown=2)[bill_id]

    assert capped.matches == every.matches[:2]
    assert capped.families == every.families
    assert RELATIVES_SHOWN == 50
