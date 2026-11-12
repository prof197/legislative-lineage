"""The state boilerplate flag: a section counted by the other families of its own state that
hold a near-duplicate of it, and precision measured with the flag on."""

import json
from pathlib import Path

import numpy as np
import pytest

from lineage import queries
from lineage.labels import Item, Pair, Section, SectionKey, SectionRef, text_sha256
from lineage.pairs import load_families, load_same_state_pairs
from lineage.precision import STRATA, stored_strata, without_sections
from lineage.semantic_pairs import build_semantic_pairs, section_keys
from lineage.state_boilerplate import (
    COMPACT_PHRASES,
    MIN_FAMILIES,
    StateBoilerplateError,
    build_flagged_sample,
    build_state_boilerplate,
    families_reached,
    flagged,
    load_families_reached,
    section_families,
    with_flag,
)
from tests.conftest import FAKE_MODEL


def _u32(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.uint32)


def test_a_section_counts_each_other_family_it_reaches_once() -> None:
    family = np.array([0, 0, 1, 2, 3, 3], dtype=np.int64)
    left, right = _u32(0, 0, 0, 0, 1, 4), _u32(2, 3, 4, 5, 2, 5)

    got = families_reached(family, left, right)

    # Section 0 reaches families 1, 2 and 3, the last through two of its sections; 4 and 5
    # are of one family, so their pair reaches no other.
    assert got.tolist() == [3, 1, 1, 1, 1, 1]


def test_a_family_of_many_versions_counts_once() -> None:
    # One section copied into four bills of one family, and into one bill of another.
    family = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    left, right = _u32(0, 0, 0, 0), _u32(1, 2, 3, 4)

    assert families_reached(family, left, right).tolist() == [1, 0, 0, 0, 1]


def test_sections_take_their_bills_family() -> None:
    got = section_families(["b", "a", "c", "b"], {"a": "a", "b": "a", "c": "c"})

    assert got.tolist() == [0, 0, 1, 0]


def test_the_flag_is_a_count_at_or_over_the_threshold() -> None:
    counts = np.array([0, MIN_FAMILIES - 1, MIN_FAMILIES, MIN_FAMILIES + 5], dtype=np.int64)

    assert flagged(counts).tolist() == [False, False, True, True]


@pytest.fixture
def tables(embedded_commission: Path) -> Path:
    build_semantic_pairs(embedded_commission, model=FAKE_MODEL, floor_bytes=0)
    return embedded_commission


def test_the_build_stores_each_sections_count_and_the_strata_with_the_flag(
    tables: Path,
) -> None:
    doc = build_state_boilerplate(tables, model=FAKE_MODEL, floor_bytes=0)

    counts = load_families_reached(tables)
    bills, _ = section_keys(tables)
    family_of_bill = load_families(tables)
    same = load_same_state_pairs(tables)
    reached: list[set[str]] = [set() for _ in bills]
    lefts: list[int] = same.left.tolist()
    rights: list[int] = same.right.tolist()
    for a, b in zip(lefts, rights, strict=True):
        if family_of_bill[bills[a]] != family_of_bill[bills[b]]:
            reached[a].add(family_of_bill[bills[b]])
            reached[b].add(family_of_bill[bills[a]])
    assert counts.tolist() == [len(r) for r in reached]
    assert any(reached), "the fixture should have same-state near-duplicates across families"
    assert sum(h["sections"] for h in doc["histogram"]) == len(bills)
    with queries.connect_quiet() as con:
        texts: list[tuple[str, int, str]] = con.execute(
            "SELECT bill_id, ordinal, normalized FROM read_parquet($p)",
            {"p": str(tables / "sections" / "*.parquet")},
        ).fetchall()
    signed = set(zip(bills, section_keys(tables)[1].tolist(), strict=True))
    naming = [
        (b, o) for b, o, n in texts if (b, o) in signed and any(p in n for p in COMPACT_PHRASES)
    ]
    assert doc["compact_sections"]["sections"] == len(naming)
    off = {n: int(k.size) for n, k in stored_strata(tables, model=FAKE_MODEL).items()}
    assert doc["precision_strata"]["flag_off"] == off
    flag = flagged(counts)
    on = stored_strata(tables, model=FAKE_MODEL, dropped=flag)
    assert doc["precision_strata"]["flag_on"] == {n: int(k.size) for n, k in on.items()}
    for keys in on.values():
        left, right = (keys >> np.uint64(32)).astype(np.int64), (keys & np.uint64(0xFFFFFFFF))
        assert not (flag[left] | flag[right.astype(np.int64)]).any()


def test_counts_from_another_build_are_refused(tables: Path) -> None:
    build_state_boilerplate(tables, model=FAKE_MODEL, floor_bytes=0)
    path = tables / "state_boilerplate" / "families_reached.npy"
    np.save(path, np.load(path)[:-1])

    with pytest.raises(StateBoilerplateError, match="lineage state-boilerplate"):
        load_families_reached(tables)


def _item(n: int, query: SectionKey, match: SectionKey, stages: tuple[str, ...]) -> Item:
    def section(key: SectionKey) -> Section:
        text = f"{key.bill_id} {key.ordinal} text"
        return Section(SectionRef(key.bill_id, None, key.ordinal, text_sha256(text)), text)

    return Item(Pair(f"p{n:04d}", stages, query, match), section(query), section(match))


def test_pairs_with_a_flagged_side_are_left_out() -> None:
    a, b, c = SectionKey("a", 1), SectionKey("b", 1), SectionKey("c", 1)
    items = [_item(0, a, b, ("semantic",)), _item(1, b, c, ("semantic",))]

    assert [i.pair.pair for i in without_sections(items, {c})] == ["p0000"]


def _report(tables: Path, path: Path, min_families: int = MIN_FAMILIES, **changed: object) -> Path:
    doc = build_state_boilerplate(
        tables, model=FAKE_MODEL, floor_bytes=0, min_families=min_families
    )
    path.write_text(json.dumps(doc | changed))
    return path


def test_precision_with_the_flag_is_refused_at_another_threshold(
    tables: Path, tmp_path: Path
) -> None:
    report = _report(tables, tmp_path / "flag.json", min_families=MIN_FAMILIES + 1)

    with pytest.raises(StateBoilerplateError, match="families"):
        with_flag(tables, [], [], dict.fromkeys(STRATA, 1), report_path=report)


def test_precision_with_the_flag_is_refused_for_another_draw(tables: Path, tmp_path: Path) -> None:
    report = _report(tables, tmp_path / "flag.json")

    with pytest.raises(StateBoilerplateError, match="strata"):
        with_flag(tables, [], [], dict.fromkeys(STRATA, 10**9), report_path=report)


def test_precision_with_the_flag_weights_by_the_strata_without_flagged_pairs(
    tables: Path, tmp_path: Path
) -> None:
    # One family is enough here, so the fixture has flagged sections to leave out.
    report = _report(tables, tmp_path / "flag.json", min_families=1)
    doc = json.loads(report.read_text())
    counts = load_families_reached(tables)
    bills, ordinals = section_keys(tables)
    key = [SectionKey(bills[s], int(ordinals[s])) for s in range(len(bills))]
    hit = key[int(np.flatnonzero(counts >= 1)[0])]
    unflagged: list[int] = np.flatnonzero(counts == 0)[:3].tolist()
    a, b, c = (key[s] for s in unflagged)
    items = [
        _item(0, a, b, ("semantic",)),
        _item(1, b, c, ("semantic",)),
        _item(2, a, hit, ("semantic",)),
        _item(3, hit, c, ("near_duplicate",)),
    ]
    labels = [
        {
            "query": i.query.ref.as_dict(),
            "match": i.match.ref.as_dict(),
            "label": 4 if i.pair.pair in ("p0000", "p0001") else 0,
            "labeler": "github:a",
            "at": "2026-11-10",
        }
        for i in items
    ]

    got = with_flag(
        tables, items, labels, doc["precision_strata"]["flag_off"], report_path=report,
        min_families=1,
    )  # fmt: skip

    assert got is not None
    assert got["min_families"] == 1
    assert got["sample_pairs_dropped"] == {
        "near_duplicate_only": 1, "semantic_only": 1, "both": 0
    }  # fmt: skip
    on = got["flag_on"]["person"]["strata"]["semantic_only"]
    assert on["size"] == doc["precision_strata"]["flag_on"]["semantic_only"]
    assert (on["labeled"], on["correct"]) == (2, 2)
    assert got["flag_on"]["person"]["strata"]["near_duplicate_only"]["labeled"] == 0


def test_without_the_counts_there_is_no_flag_estimate(embedded_commission: Path) -> None:
    assert with_flag(embedded_commission, [], [], {}) is None


def test_the_flagged_sample_draws_only_pairs_the_flag_drops(tables: Path) -> None:
    doc = build_state_boilerplate(tables, model=FAKE_MODEL, floor_bytes=0, min_families=1)

    rows, report = build_flagged_sample(tables, model=FAKE_MODEL, per_stratum=5, min_families=1)

    counts = load_families_reached(tables)
    bills, ordinals = section_keys(tables)
    number = {(bills[s], int(ordinals[s])): s for s in range(len(bills))}
    assert rows, "the fixture should have pairs with a flagged side"
    for row in rows:
        sides = [number[(row[k]["bill_id"], row[k]["ordinal"])] for k in ("query", "match")]
        assert max(int(counts[s]) for s in sides) >= 1
        assert str(row["pair"]).startswith("f")
    strata = doc["precision_strata"]
    for name in STRATA:
        size = strata["flag_off"][name] - strata["flag_on"][name]
        assert report["strata"][name]["size"] == size
        assert report["strata"][name]["drawn"] == min(5, size)
    again, _ = build_flagged_sample(tables, model=FAKE_MODEL, per_stratum=5, min_families=1)
    assert again == rows


def test_the_dropped_pairs_precision_is_weighted_by_the_dropped_strata(
    tables: Path, tmp_path: Path
) -> None:
    report = _report(tables, tmp_path / "flag.json", min_families=1)
    doc = json.loads(report.read_text())
    rows, drawn = build_flagged_sample(tables, model=FAKE_MODEL, per_stratum=5, min_families=1)
    dropped_report = tmp_path / "flagged.json"
    dropped_report.write_text(json.dumps(drawn))
    items = [
        _item(
            n,
            SectionKey(r["query"]["bill_id"], r["query"]["ordinal"]),
            SectionKey(r["match"]["bill_id"], r["match"]["ordinal"]),
            tuple(r["stages"]),
        )
        for n, r in enumerate(rows)
    ]
    labels = [
        {
            "query": i.query.ref.as_dict(),
            "match": i.match.ref.as_dict(),
            "label": 0,
            "labeler": "github:a",
            "at": "2026-11-10",
        }
        for i in items
    ]

    got = with_flag(
        tables, [], labels, doc["precision_strata"]["flag_off"], report_path=report,
        min_families=1, dropped_items=items, dropped_report=dropped_report,
    )  # fmt: skip

    assert got is not None
    assert got["dropped_pairs"] is not None
    strata = got["dropped_pairs"]["person"]["strata"]
    for name in STRATA:
        assert strata[name]["size"] == drawn["strata"][name]["size"]
        assert strata[name]["labeled"] == drawn["strata"][name]["drawn"]
        assert strata[name]["correct"] == 0
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(drawn | {"min_families": 2}))
    with pytest.raises(StateBoilerplateError, match="flagged"):
        with_flag(
            tables, [], labels, doc["precision_strata"]["flag_off"], report_path=report,
            min_families=1, dropped_items=items, dropped_report=stale,
        )  # fmt: skip
