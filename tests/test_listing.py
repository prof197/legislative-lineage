"""What a page lists: matches at the stage's threshold, without a state's boilerplate unless
both stages found the pair."""

from pathlib import Path

import numpy as np
import pytest

from lineage.listing import listed_hits, listed_pairs
from lineage.pairs import load_cross_state_pairs
from lineage.query import Match, query_text
from lineage.relatives import bill_relatives
from lineage.semantic import BUILD_THRESHOLD, THRESHOLD
from lineage.semantic_pairs import build_semantic_pairs, load_semantic_pairs, section_keys
from tests.conftest import FAKE_MODEL
from tests.test_relatives import bill_of, most_related


@pytest.fixture
def tables(embedded_commission: Path) -> Path:
    build_semantic_pairs(embedded_commission, model=FAKE_MODEL, floor_bytes=0)
    return embedded_commission


def test_a_flagged_hit_stands_only_when_both_stages_hit_that_section_from_that_query_section() -> (
    None
):
    near = [(0, "a", 1, 90), (0, "b", 2, 80), (1, "b", 2, 70), (0, "c", 3, 100)]
    meaning = [(0, "b", 2, 0.95), (1, "b", 2, 0.91), (0, "a", 1, 0.99)]
    flagged = {("b", 2), ("a", 1)}

    got_near, got_meaning = listed_hits(near, meaning, threshold=0.93, flagged=flagged)

    # (1, b, 2) is hit by meaning only under the threshold, so its near hit falls with the flag.
    assert got_near == [(0, "a", 90), (0, "b", 80), (0, "c", 100)]
    assert got_meaning == [(0, "b", 0.95), (0, "a", 0.99)]
    alone_near, alone_meaning = listed_hits(
        near, meaning, threshold=0.93, flagged=flagged, exempt_both=False
    )
    assert alone_near == [(0, "c", 100)]
    assert alone_meaning == []
    assert listed_hits(near, meaning, threshold=0.93) == (
        [(q, b, a) for q, b, _, a in near],
        [(0, "b", 0.95), (0, "a", 0.99)],
    )


def _keys(left: np.ndarray, right: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(np.asarray(left).tolist(), np.asarray(right).tolist(), strict=True))


def test_the_flag_drops_a_flagged_sections_pairs_but_those_both_stages_found(
    tables: Path,
) -> None:
    near = load_cross_state_pairs(tables)
    meaning = load_semantic_pairs(tables, FAKE_MODEL, threshold=THRESHOLD)
    both = _keys(near.left, near.right) & _keys(meaning.left, meaning.right)
    assert both, "the fixture should have pairs both stages found"
    flag = np.zeros(len(section_keys(tables)[0]), dtype=bool)
    flag[[a for a, _ in sorted(both)[:3]]] = True
    flag[np.asarray(near.left)[:5]] = True

    got = listed_pairs(tables, model=FAKE_MODEL, flag=flag)
    alone = listed_pairs(tables, model=FAKE_MODEL, flag=flag, exempt_both=False)

    def touched(pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(a, b) for a, b in pairs if flag[a] or flag[b]}

    every_near, every_meaning = _keys(near.left, near.right), _keys(meaning.left, meaning.right)
    assert _keys(got.near.left, got.near.right) == every_near - (touched(every_near) - both)
    assert _keys(got.meaning.left, got.meaning.right) == every_meaning - (
        touched(every_meaning) - both
    )
    assert got.exempted == {
        "near_duplicate": len(touched(both)),
        "semantic": len(touched(both)),
    }
    assert not touched(_keys(alone.near.left, alone.near.right))
    assert not touched(_keys(alone.meaning.left, alone.meaning.right))
    assert listed_pairs(tables, model=FAKE_MODEL, flag=None).near.left.size == near.left.size


def test_the_stage_threshold_lists_fewer_by_meaning_than_the_build_holds(tables: Path) -> None:
    at_build = listed_pairs(tables, model=FAKE_MODEL, threshold=BUILD_THRESHOLD, flag=None)
    at_stage = listed_pairs(tables, model=FAKE_MODEL, flag=None)

    assert at_stage.meaning.left.size < at_build.meaning.left.size
    assert (at_stage.meaning.cosine >= THRESHOLD).all()


def test_a_flagged_bills_relatives_score_as_a_query_of_its_text_does(tables: Path) -> None:
    bill_id = most_related(tables)
    state, text = bill_of(tables, bill_id)
    bills, ordinals = section_keys(tables)
    before = bill_relatives(tables, [bill_id], model=FAKE_MODEL, stages=("near_duplicate",))
    # Flag the sections of the bill's best relative.
    best = before[bill_id].matches[0].bill_id
    flag = np.fromiter((b == best for b in bills), dtype=bool, count=len(bills))
    keys = {(b, int(o)) for b, o, f in zip(bills, ordinals.tolist(), flag, strict=True) if f}

    got = bill_relatives(
        tables, [bill_id], model=FAKE_MODEL, stages=("near_duplicate",), flag=flag,
        exempt_both=False,
    )  # fmt: skip
    queried = [m for m in query_text(text, tables, flagged=keys).matches if m.state != state]

    def scored(matches: list[Match]) -> list[tuple[str, float, int]]:
        return [(m.family, m.share, m.sections_matched) for m in matches]

    assert best not in {m.bill_id for m in got[bill_id].matches}
    assert scored(got[bill_id].matches) == scored(queried)[: len(got[bill_id].matches)]
