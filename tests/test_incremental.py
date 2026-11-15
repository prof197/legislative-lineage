"""`lineage update`'s incremental core (DESIGN.md#updates, DR-7), on the recorded days and on
two more that retire a text no other window carries and flip a record's eligibility: each
update gives what a full rebuild over the same days gives, its change log is the difference,
the same day twice changes nothing, and a build shared by a link is never written into."""

import copy
import hashlib
import json
import shutil
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from lineage import queries
from lineage.incremental import (
    NoEmbedderError,
    applied_days,
    apply_day,
    register_embedder,
    unregister_embedder,
)
from lineage.updates import (
    Change,
    RecordAdded,
    RecordReplaced,
    RecordUpdated,
    RelativeGained,
    RelativeLost,
    Snapshot,
    UpdateError,
    fold,
    load_day,
    load_days,
    origin_changes,
    read_changes,
    record_changes,
    relative_changes,
    snapshot,
    update,
    write_changes,
)
from lineage.vectors import load_plan, load_vectors, semantic_dir, shard_name
from tests.conftest import FAKE_MODEL
from tests.days import DAYS, GEORGIA_ID, UTAH_SB_18, rebuild, write_days
from tests.fakes import MARK, bag_of_words

EXTRA = ("2026-01-15", "2026-01-16")


def _day(root: Path, as_of: str, archives: dict[str, list[dict[str, Any]]]) -> None:
    day = root / as_of
    day.mkdir(parents=True)
    (day / "day.json").write_text(json.dumps({"as_of": as_of, "sources": ["fixture"]}))
    for name, rows in archives.items():
        (day / f"{name}_bills.json").write_text(json.dumps(rows))


def _marked(text: str) -> str:
    """The text with every word after the first two of each line marked, so its sections and
    windows are the same but no shingle is: the stand-in model reads it as the original."""
    lines: list[str] = []
    for line in text.split("\n"):
        words = line.split(" ")
        lines.append(" ".join(w if i < 2 or not w else w + MARK for i, w in enumerate(words)))
    return "\n".join(lines)


def _rows(day: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads((day / f"{name}_bills.json").read_text())
    return rows


def _write_all_days(root: Path) -> Path:
    """The recorded days, then day 4, when the Georgia record takes Utah SB 18's text marked
    (a match by meaning alone) and the record signed on day 3 is filed as a resolution, which
    no rule matches; and day 5, when Georgia's text is replaced by one of its own, retiring
    day 4's windows, which no other record carries, and the record is a bill again."""
    write_days(root)
    third = root / DAYS[2]
    [georgia] = _rows(third, "GA_2025_26")
    signed_file = next(p for p in third.glob("*_bills.json") if not p.name.startswith("GA_"))
    name = signed_file.name.removesuffix("_bills.json")
    [signed] = _rows(third, name)
    utah = next(
        r
        for p in (root / DAYS[1]).glob("UT_*_bills.json")
        for r in json.loads(p.read_text())
        if r["id"] == UTAH_SB_18
    )
    marked = copy.deepcopy(georgia) | {
        "raw_text": _marked(utah["raw_text"]),
        "title": "Synthetic fixture record: Utah SB 18's text, its words marked",
    }
    resolution = copy.deepcopy(signed) | {"classification": ["resolution"]}
    _day(root, EXTRA[0], {"GA_2025_26": [marked], name: [resolution]})
    own = copy.deepcopy(georgia) | {
        "raw_text": "Section 1. Synthetic fixture text.\nNo other record carries these words, "
        "which a test of retired windows needs: quartz lantern meadow ferry orchard.",
        "title": "Synthetic fixture record: a text of its own",
    }
    _day(root, EXTRA[1], {"GA_2025_26": [own], name: [signed]})
    return root


@pytest.fixture(scope="module")
def days(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_all_days(tmp_path_factory.mktemp("days"))


@pytest.fixture(scope="module")
def rebuilt(days: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    root = tmp_path_factory.mktemp("rebuilt")
    return {n: rebuild(days, n, root / f"day{n}") for n in range(1, 6)}


@pytest.fixture(scope="module")
def snapshots(rebuilt: dict[int, Path]) -> dict[int, Snapshot]:
    return {n: snapshot(t, model=FAKE_MODEL) for n, t in rebuilt.items()}


def _expected(days: Path, n: int, snapshots: dict[int, Snapshot]) -> list[Change]:
    all_days = load_days(days)
    as_of = all_days[n - 1].as_of
    out = record_changes(fold(all_days[: n - 1]), all_days[n - 1])
    out += relative_changes(snapshots[n - 1], snapshots[n], as_of)
    out += origin_changes(snapshots[n - 1], snapshots[n], as_of)
    return out


def _apply(tables: Path, days: Path, n: int, log: Path) -> list[Change]:
    day = load_days(days)[n - 1]
    return apply_day(
        tables, day, model=FAKE_MODEL, changes_dir=log, floor_bytes=0, embed=bag_of_words
    )


@pytest.mark.parametrize("n", [4, 5])
def test_an_update_that_retires_texts_or_flips_eligibility_gives_what_a_rebuild_gives(
    n: int, days: Path, rebuilt: dict[int, Path], snapshots: dict[int, Snapshot], tmp_path: Path
) -> None:
    tables = tmp_path / "tables"
    shutil.copytree(rebuilt[n - 1], tables, symlinks=False)

    changes = _apply(tables, days, n, tmp_path / "changes")

    assert snapshot(tables, model=FAKE_MODEL) == snapshots[n]
    assert changes == _expected(days, n, snapshots)


def test_the_georgia_record_carries_no_vector_once_its_windows_are_retired(
    days: Path, rebuilt: dict[int, Path], tmp_path: Path
) -> None:
    tables = tmp_path / "tables"
    shutil.copytree(rebuilt[4], tables, symlinks=False)
    units = str(semantic_dir(tables) / "units" / "*.parquet")
    with queries.connect_quiet() as con:
        marked = [
            t for _, t in queries.semantic_text_ids_of(con, units_glob=units, bill_ids=[GEORGIA_ID])
        ]
    # Day 4's windows, which only the Georgia record carries: its words are marked.
    assert marked
    assert all(MARK in t for t in _texts(tables, marked))

    _apply(tables, days, 5, tmp_path / "changes")

    plan = load_plan(tables)
    assert plan is not None
    vectors = np.concatenate([np.asarray(v) for v in load_vectors(tables, FAKE_MODEL, plan)])
    assert not vectors[marked].any()  # zeroed: no threshold reached, no neighbor's place taken
    assert _texts(tables, marked) == [""] * len(marked)  # blanked: no later window shares one
    with queries.connect_quiet() as con:
        assert queries.carried_text_ids(con, units_glob=units, text_ids=marked) == []
        now = [
            t for _, t in queries.semantic_text_ids_of(con, units_glob=units, bill_ids=[GEORGIA_ID])
        ]
    assert now
    assert min(now) >= vectors.shape[0] - len(now)


def _texts(tables: Path, ids: list[int]) -> list[str]:
    plan = load_plan(tables)
    assert plan is not None
    texts: list[str] = []
    with queries.connect_quiet() as con:
        for s in range(plan.shards):
            texts += queries.read_semantic_texts(
                con, parquet_path=semantic_dir(tables) / "texts" / f"{shard_name(s)}.parquet"
            )
    return [texts[i] for i in ids]


def test_the_change_log_tells_the_georgia_story(
    days: Path, rebuilt: dict[int, Path], snapshots: dict[int, Snapshot], tmp_path: Path
) -> None:
    # Day by day from the build over day 1: the Georgia record arrives before Utah's version,
    # so without Utah SB 18 among its relatives (it has those of the states day 1 brings), gains
    # it through both stages when Utah's version arrives, and the two lose each other when its
    # text is replaced.
    tables, log = tmp_path / "tables", tmp_path / "changes"
    shutil.copytree(rebuilt[1], tables, symlinks=False)
    assert UTAH_SB_18 not in {r[0] for r in snapshots[1].relatives[GEORGIA_ID]}

    for n in range(2, 6):
        assert _apply(tables, days, n, log) == _expected(days, n, snapshots)
        assert snapshot(tables, model=FAKE_MODEL) == snapshots[n]

    d2, d3 = (date.fromisoformat(d) for d in DAYS[1:3])
    logged = read_changes(log)
    both = ["near_duplicate", "semantic"]
    assert (
        RelativeGained(as_of=d2, bill_id=GEORGIA_ID, relative_id=UTAH_SB_18, stages=both) in logged
    )
    assert (
        RelativeGained(as_of=d2, bill_id=UTAH_SB_18, relative_id=GEORGIA_ID, stages=both) in logged
    )
    assert RelativeLost(as_of=d3, bill_id=GEORGIA_ID, relative_id=UTAH_SB_18) in logged
    assert RelativeLost(as_of=d3, bill_id=UTAH_SB_18, relative_id=GEORGIA_ID) in logged
    assert next(type(c) for c in logged if c.bill_id == GEORGIA_ID and c.as_of == d3) is (
        RecordReplaced
    )
    assert not any(isinstance(c, RecordAdded) and c.bill_id == GEORGIA_ID for c in logged)
    # Day 4's marked copy is found by meaning alone, and day 5 loses it.
    d4, d5 = (date.fromisoformat(d) for d in EXTRA)
    assert (
        RelativeGained(as_of=d4, bill_id=GEORGIA_ID, relative_id=UTAH_SB_18, stages=["semantic"])
        in logged
    )
    assert RelativeLost(as_of=d5, bill_id=GEORGIA_ID, relative_id=UTAH_SB_18) in logged


def _digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def test_the_same_day_twice_changes_nothing_and_a_different_one_is_refused(
    days: Path, rebuilt: dict[int, Path], tmp_path: Path
) -> None:
    tables, log = tmp_path / "tables", tmp_path / "changes"
    shutil.copytree(rebuilt[2], tables, symlinks=False)
    first = _apply(tables, days, 3, log)
    files, logged = _digest(tables), (log / f"{DAYS[2]}.jsonl").read_bytes()

    assert _apply(tables, days, 3, log) == []
    assert _digest(tables) == files
    assert (log / f"{DAYS[2]}.jsonl").read_bytes() == logged
    assert read_changes(log) == first
    assert set(applied_days(tables)) == {date.fromisoformat(DAYS[2])}

    other = tmp_path / "other"
    shutil.copytree(days / DAYS[2], other)
    rows = _rows(other, "GA_2025_26")
    (other / "GA_2025_26_bills.json").write_text(json.dumps([rows[0] | {"title": "Other"}]))
    with pytest.raises(UpdateError, match="other records"):
        apply_day(
            tables, load_day(other), model=FAKE_MODEL, changes_dir=log, floor_bytes=0,
            embed=bag_of_words,
        )  # fmt: skip
    with pytest.raises(UpdateError, match="out of order"):
        _apply(tables, days, 2, log)
    assert _digest(tables) == files

    base = tmp_path / "base"
    shutil.copytree(rebuilt[2], base, symlinks=False)
    with pytest.raises(UpdateError, match="already stands as of"):
        _apply(base, days, 1, log)  # the build over days 1 and 2 already holds day 1


def test_an_update_writes_nothing_into_a_build_it_shares_by_a_link(
    days: Path, rebuilt: dict[int, Path], tmp_path: Path
) -> None:
    # The reference builds hold their tables as links to an unembedded build; a copy that keeps
    # the links shares those files, and an update must replace, never write through, them.
    shared = rebuilt[2].parent
    before = _digest(shared)
    tables = tmp_path / "tables"
    shutil.copytree(rebuilt[2], tables, symlinks=True)

    _apply(tables, days, 3, tmp_path / "changes")

    assert _digest(shared) == before
    assert snapshot(tables, model=FAKE_MODEL) == snapshot(rebuilt[3], model=FAKE_MODEL)


def test_a_model_without_an_embedder_is_refused_before_anything_is_written(
    days: Path, rebuilt: dict[int, Path], tmp_path: Path
) -> None:
    # The stand-in's vectors under a model name no embedder is registered for.
    model = "unpinned"
    tables = tmp_path / "tables"
    shutil.copytree(rebuilt[2], tables, symlinks=False)
    (semantic_dir(tables) / FAKE_MODEL).rename(semantic_dir(tables) / model)
    fingerprint = semantic_dir(tables) / model / "pairs" / "fingerprint.json"
    fingerprint.write_text(json.dumps(json.loads(fingerprint.read_text()) | {"model": model}))
    files = _digest(tables)
    day = load_days(days)[2]
    with pytest.raises(NoEmbedderError):
        update(tables, day, model=model, changes_dir=tmp_path / "log", floor_bytes=0)
    assert _digest(tables) == files
    assert not (tmp_path / "log").exists()

    register_embedder(model, bag_of_words)
    try:
        update(tables, day, model=model, changes_dir=tmp_path / "log", floor_bytes=0)
    finally:
        unregister_embedder(model)
    assert snapshot(tables, model=model) == snapshot(rebuilt[3], model=FAKE_MODEL)


def test_an_update_that_stops_before_the_tables_change_leaves_them_as_they_were(
    days: Path, rebuilt: dict[int, Path], tmp_path: Path
) -> None:
    tables, log = tmp_path / "tables", tmp_path / "changes"
    shutil.copytree(rebuilt[1], tables, symlinks=False)
    files = _digest(tables)
    day = load_days(days)[1]

    def broken(texts: Sequence[str]) -> NDArray[np.float32]:
        raise RuntimeError("the model stopped")

    with pytest.raises(RuntimeError, match="stopped"):
        apply_day(tables, day, model=FAKE_MODEL, changes_dir=log, floor_bytes=0, embed=broken)
    assert _digest(tables) == files  # nothing staged is left, and nothing marks the tables
    # A log that already holds other changes for the day is refused before the tables change.
    write_changes(log, day.as_of, [RecordUpdated(as_of=day.as_of, bill_id="other")])
    with pytest.raises(UpdateError, match="different changes"):
        _apply(tables, days, 2, log)
    assert _digest(tables) == files

    (log / f"{day.as_of.isoformat()}.jsonl").unlink()
    _apply(tables, days, 2, log)
    assert snapshot(tables, model=FAKE_MODEL) == snapshot(rebuilt[2], model=FAKE_MODEL)
