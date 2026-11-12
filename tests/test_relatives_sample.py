"""The precision of what a page lists: random bills of one state, one listed relative each,
judged on the section pair its listing rests on most, with the aligned passages."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lineage import queries
from lineage.labels import Item, Pair, Section, SectionKey, SectionRef, text_sha256
from lineage.pairs import load_cross_state_pairs
from lineage.relatives import bill_relatives
from lineage.relatives_sample import (
    TOP,
    RelativesSampleError,
    build_relatives_sample,
    estimate,
    with_passages,
)
from lineage.semantic import BUILD_THRESHOLD, THRESHOLD
from lineage.semantic_pairs import build_semantic_pairs, load_semantic_pairs, section_keys
from tests.conftest import FAKE_MODEL

STATE = "UT"  # the fixture state with the most bills


@pytest.fixture
def tables(embedded_commission: Path) -> Path:
    build_semantic_pairs(embedded_commission, model=FAKE_MODEL, floor_bytes=0)
    return embedded_commission


def _states(tables: Path) -> dict[str, str]:
    with queries.connect_quiet() as con:
        return queries.bill_states(con, bills_glob=str(tables / "bills" / "*.parquet"))


def _expected_pair(tables: Path, query_bill: str, match_bill: str) -> tuple[int, int]:
    """By brute force: the near-duplicate pair with the most agreements, else the semantic
    pair at the best cosine, the lowest sections on a tie."""
    bills, ordinals = section_keys(tables)
    near = load_cross_state_pairs(tables)
    meaning = load_semantic_pairs(tables, model=FAKE_MODEL, threshold=BUILD_THRESHOLD)
    for left, right, scores in (
        (near.left, near.right, near.agreements),
        (meaning.left, meaning.right, meaning.cosine),
    ):
        found: list[tuple[float, int, int]] = []
        for a, b, s in zip(left.tolist(), right.tolist(), scores.tolist(), strict=True):
            for q, m in ((a, b), (b, a)):
                if bills[q] == query_bill and bills[m] == match_bill:
                    found.append((-float(s), int(ordinals[q]), int(ordinals[m])))
        if found:
            _, q, m = min(found)
            return q, m
    raise AssertionError("no stored pair")


def test_each_row_is_one_listed_relative_of_a_random_bill_of_the_state(tables: Path) -> None:
    states = _states(tables)

    rows, report = build_relatives_sample(tables, model=FAKE_MODEL, state=STATE, bills=3)

    assert len(rows) == 3
    assert len({r["query"]["bill_id"] for r in rows}) == 3
    assert report["state_bills_with_a_relative"] >= 3
    # By default a draw lists pages as the tuning sample was drawn: at the build's threshold.
    listed = bill_relatives(
        tables, [r["query"]["bill_id"] for r in rows], model=FAKE_MODEL, threshold=BUILD_THRESHOLD
    )
    for row in rows:
        page: dict[str, Any] = row["page"]
        query, match = row["query"]["bill_id"], row["match"]["bill_id"]
        assert states[query] == STATE
        assert states[match] != STATE
        rank = int(page["rank"])
        relative = listed[query].matches[rank - 1]
        assert (relative.bill_id, relative.family) == (match, page["family"])
        assert page["listed"] == len(listed[query].matches)
        assert (row["query"]["ordinal"], row["match"]["ordinal"]) == _expected_pair(
            tables, query, match
        )
        assert str(row["pair"]).startswith("r")
    again, _ = build_relatives_sample(tables, model=FAKE_MODEL, state=STATE, bills=3)
    assert again == rows


def test_a_state_with_too_few_related_bills_is_refused(tables: Path) -> None:
    with pytest.raises(RelativesSampleError, match="fewer than"):
        build_relatives_sample(tables, model=FAKE_MODEL, state=STATE, bills=10_000)


def _item(n: int, a: str, b: str) -> Item:
    def section(bill: str, text: str) -> Section:
        return Section(SectionRef(bill, None, 1, text_sha256(text)), text)

    return Item(
        Pair(f"r{n:04d}", ("semantic",), SectionKey(f"q{n}", 1), SectionKey(f"m{n}", 1)),
        section(f"q{n}", a),
        section(f"m{n}", b),
    )


def test_the_passages_are_the_aligned_text_unmarked() -> None:
    shared = "the board shall adopt rules to carry out this act within ninety days"
    item = _item(0, f"Georgia preamble words. {shared}.", f"{shared}. Texas tail words here.")

    [got] = with_passages([item], embed=None)

    assert got.passages
    for a, b in got.passages:
        assert a in item.query.text
        assert b in item.match.text
    assert any(shared in a for a, _ in got.passages)
    [unrelated] = with_passages([_item(1, "one two three", "four five six")], embed=None)
    assert unrelated.passages == ()


def test_the_estimate_is_the_share_correct_by_relative_stage_and_rank() -> None:
    items = [_item(n, f"a {n}", f"b {n}") for n in range(4)]
    # Each page's list: how many it shows, and the cosines of those found by meaning alone.
    pages = {
        "r0000": {"rank": 1, "relative_stages": ["semantic"], "listed": 4,
                  "shown_semantic_only": [0.95]},
        "r0001": {"rank": TOP + 1, "relative_stages": ["semantic"], "listed": 8,
                  "shown_semantic_only": [0.94, 0.95, 0.96, 0.97]},
        "r0002": {"rank": 2, "relative_stages": ["near_duplicate", "semantic"], "listed": 2,
                  "shown_semantic_only": []},
        "r0003": {"rank": 3, "relative_stages": ["near_duplicate"], "listed": 3,
                  "shown_semantic_only": [0.95]},
    }  # fmt: skip
    labels = [
        {
            "query": i.query.ref.as_dict(),
            "match": i.match.ref.as_dict(),
            "label": label,
            "labeler": "model:claude-opus-5-5@relatives-v1",
            "at": "2026-11-11",
        }
        for i, label in zip(items, [3, 1, 4, 0], strict=True)
    ]

    got = estimate(items, labels, pages, labeler="model:")

    assert got["overall"]["labeled"] == 4
    assert got["overall"]["correct"] == 2
    assert got["relative_strata"]["semantic_only"]["correct"] == 1
    assert got["relative_strata"]["both"]["precision"] == 1.0
    assert got["relative_strata"]["near_duplicate_only"]["precision"] == 0.0
    assert got["rank"][f"first_{TOP}"]["labeled"] == 3
    assert got["rank"][f"after_{TOP}"]["correct"] == 0
    assert np.isclose(got["overall"]["precision"], 0.5)
    by_meaning, wording = got["lists"]["by_meaning_alone"], got["lists"]["sharing_wording"]
    assert (by_meaning["labeled"], by_meaning["correct"]) == (2, 1)
    assert (wording["labeled"], wording["correct"]) == (2, 1)
    # Weighted by list over that list's length: 4 / 1 for r0000, right; 8 / 4 for r0001, wrong.
    assert np.isclose(by_meaning["page_weighted"]["precision"], 4 / 6, atol=1e-4)
    # 2 / 2 for r0002, right; 3 / 2 for r0003, wrong: 1 / (1 + 1.5).
    assert np.isclose(wording["page_weighted"]["precision"], 1 / 2.5, atol=1e-4)


def test_a_part_draws_its_own_bills_from_pages_as_listed_and_records_what_a_threshold_needs(
    tables: Path,
) -> None:
    first, _ = build_relatives_sample(tables, model=FAKE_MODEL, state=STATE, bills=2)
    bills, _ = section_keys(tables)
    flag = np.fromiter((b == first[0]["match"]["bill_id"] for b in bills), bool, len(bills))

    second, report = build_relatives_sample(
        tables, model=FAKE_MODEL, state=STATE, bills=1, seed=7, threshold=THRESHOLD, flag=flag,
        excluded={r["query"]["bill_id"] for r in first}, first=101,
    )  # fmt: skip

    assert [r["pair"] for r in second] == ["r0101"]
    assert not {r["query"]["bill_id"] for r in second} & {r["query"]["bill_id"] for r in first}
    assert (report["threshold"], report["state_boilerplate_flag"]) == (THRESHOLD, True)
    assert 0 < report["bills_excluded"] <= 2
    listed = bill_relatives(
        tables, [r["query"]["bill_id"] for r in second], model=FAKE_MODEL, threshold=THRESHOLD,
        flag=flag,
    )  # fmt: skip
    for row in second:
        page: dict[str, Any] = row["page"]
        shown = listed[row["query"]["bill_id"]].matches
        assert len(shown) == page["listed"]
        assert page["best_cosine"] == shown[int(page["rank"]) - 1].best_cosine
        assert page["shown_semantic_only"] == sorted(
            float(m.best_cosine)
            for m in shown
            if "near_duplicate" not in m.sections_by_stage and m.best_cosine is not None
        )
        assert all(c >= THRESHOLD for c in page["shown_semantic_only"])
