"""The live track's contracts (DESIGN.md#updates, DR-7): days fold into records with versions,
the change log is append-only, and an update gives what a full rebuild gives. The last is lane
L1's to meet; until then it is an expected failure that fails the run once it passes."""

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from lineage.updates import (
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
    read_changes,
    record_changes,
    relative_changes,
    snapshot,
    update,
    write_changes,
)
from tests.conftest import FAKE_MODEL
from tests.days import DAYS, GEORGIA_ID, UTAH_SB_18, rebuild, write_days

D1, D2, D3 = (date.fromisoformat(d) for d in DAYS)


@pytest.fixture(scope="session")
def days(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_days(tmp_path_factory.mktemp("days"))


@pytest.fixture(scope="session")
def rebuilt(days: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    """The reference build after each day."""
    root = tmp_path_factory.mktemp("rebuilt")
    return {n: rebuild(days, n, root / f"day{n}") for n in (1, 2, 3)}


@pytest.fixture(scope="session")
def snapshots(rebuilt: dict[int, Path]) -> dict[int, Snapshot]:
    return {n: snapshot(t, model=FAKE_MODEL) for n, t in rebuilt.items()}


def test_a_record_seen_again_stands_as_last_seen_and_keeps_its_earlier_text(days: Path) -> None:
    folded = fold(load_days(days))

    assert folded.as_of == D3
    assert folded.first_seen[GEORGIA_ID] == D1
    assert "Special Deposits" in folded.records[GEORGIA_ID][2]["title"]
    georgia = [v for v in folded.versions if v.bill_id == GEORGIA_ID]
    assert [(v.first_seen, v.replaced_on) for v in georgia] == [(D1, D3), (D3, None)]
    assert georgia[0].text_sha256 != georgia[1].text_sha256
    signed = [v for v in folded.versions if v.bill_id != GEORGIA_ID and v.first_seen == D1]
    assert all(v.replaced_on is None for v in signed)  # new actions alone are not a new version


def test_a_day_says_what_it_did_to_each_record(days: Path) -> None:
    all_days = load_days(days)
    first = record_changes(None, all_days[0])
    third = record_changes(fold(all_days[:2]), all_days[2])

    assert all(isinstance(c, RecordAdded) for c in first)
    assert {c.bill_id for c in first} >= {GEORGIA_ID}
    kinds = {c.bill_id: type(c) for c in third}
    assert kinds.pop(GEORGIA_ID) is RecordReplaced
    assert list(kinds.values()) == [RecordUpdated]


def test_days_out_of_order_or_a_record_seen_twice_in_a_day_are_refused(
    days: Path, tmp_path: Path
) -> None:
    all_days = load_days(days)
    with pytest.raises(UpdateError, match="out of order"):
        fold([all_days[1], all_days[0]])
    twice = tmp_path / "twice"
    shutil.copytree(days / DAYS[2], twice)
    rows = json.loads((twice / "GA_2025_26_bills.json").read_text())
    (twice / "GA_2025_26_bills.json").write_text(json.dumps(rows * 2))
    with pytest.raises(UpdateError, match="twice"):
        load_day(twice)


def test_the_change_log_is_append_only(tmp_path: Path) -> None:
    gained = RelativeGained(as_of=D2, bill_id="a", relative_id="b", stages=["semantic"])
    lost = RelativeLost(as_of=D3, bill_id="a", relative_id="b")

    write_changes(tmp_path, D2, [gained])
    write_changes(tmp_path, D2, [gained])  # the same day again changes nothing
    write_changes(tmp_path, D3, [lost])

    assert read_changes(tmp_path) == [gained, lost]
    with pytest.raises(UpdateError, match="different changes"):
        write_changes(tmp_path, D2, [lost])


def test_the_recorded_days_do_what_they_say(snapshots: dict[int, Snapshot]) -> None:
    # The fixture's own check: the Georgia record has Utah's relative only once Utah arrives,
    # and loses its anti-SLAPP relatives when its text is replaced.
    def related(n: int) -> set[str]:
        return {r[0] for r in snapshots[n].relatives.get(GEORGIA_ID, ())}

    assert UTAH_SB_18 not in related(1)
    assert UTAH_SB_18 in related(2)
    assert UTAH_SB_18 not in related(3)
    gained = relative_changes(snapshots[1], snapshots[2], D2)
    assert (
        RelativeGained(
            as_of=D2,
            bill_id=GEORGIA_ID,
            relative_id=UTAH_SB_18,
            stages=["near_duplicate", "semantic"],
        )
        in gained
    )
    assert any(
        isinstance(c, RelativeGained) and (c.bill_id, c.relative_id) == (UTAH_SB_18, GEORGIA_ID)
        for c in gained
    )
    lost = relative_changes(snapshots[2], snapshots[3], D3)
    assert RelativeLost(as_of=D3, bill_id=UTAH_SB_18, relative_id=GEORGIA_ID) in lost


@pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="lane L1 (M10) builds update")
@pytest.mark.parametrize("n", [2, 3])
def test_an_update_gives_what_a_rebuild_gives(
    n: int, days: Path, rebuilt: dict[int, Path], snapshots: dict[int, Snapshot], tmp_path: Path
) -> None:
    tables = tmp_path / "tables"
    shutil.copytree(rebuilt[n - 1], tables, symlinks=False)
    all_days = load_days(days)
    log = tmp_path / "changes"

    changes = update(tables, all_days[n - 1], model=FAKE_MODEL, changes_dir=log, floor_bytes=0)

    assert snapshot(tables, model=FAKE_MODEL) == snapshots[n]
    expected = record_changes(fold(all_days[: n - 1]), all_days[n - 1])
    expected += relative_changes(snapshots[n - 1], snapshots[n], all_days[n - 1].as_of)
    assert changes == expected
    assert read_changes(log) == expected
    again = update(tables, all_days[n - 1], model=FAKE_MODEL, changes_dir=log, floor_bytes=0)
    assert again == []  # applied already
    assert snapshot(tables, model=FAKE_MODEL) == snapshots[n]
