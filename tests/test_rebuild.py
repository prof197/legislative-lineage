"""`lineage rebuild` (DESIGN.md#updates, DR-7): the folded days laid over the snapshot's archives
and the whole pipeline run again. With the updated tables' statistics pinned it must give what
the reference rebuild gives (tests/days.py) and what the updates gave, and with them counted
afresh what the refreshed reference gives (tests/live.py), at fixture scale with exact search.
"""

import json
import shutil
from pathlib import Path
from typing import TypedDict

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.rebuild import Base, RebuildError, lay_days, rebuild, work_dir
from lineage.snapshot import scan, write_manifest
from lineage.updates import fold, load_days, snapshot, write_archives
from tests.conftest import FAKE_MODEL
from tests.days import DAYS, sessions_csv
from tests.days import rebuild as reference
from tests.live import REFRESHED_SHARE, Recorded, recorded

D3 = DAYS[2]


@pytest.fixture(scope="module")
def rec(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    return recorded(tmp_path_factory)


class FixtureArgs(TypedDict):
    sessions: Path
    model: str
    text_plan: Path
    neighbors_spec: str
    ef_search: int | None
    floor_bytes: int


def _fixture_args(root: Path) -> FixtureArgs:
    """What the fixture's builds run with: its sessions, the stand-in model, no fetched text,
    exact neighbor search and no disk floor."""
    return {
        "sessions": sessions_csv(root),
        "model": FAKE_MODEL,
        "text_plan": root / "no-plan.parquet",
        "neighbors_spec": "Flat",
        "ef_search": None,
        "floor_bytes": 0,
    }


def _base(rec: Recorded, root: Path) -> Base:
    """Day 1's records as a snapshot of archives, pinned by a manifest, as the base build read
    them."""
    raw = write_archives(fold(load_days(rec.days)[:1]), root / "raw")
    manifest = root / "manifest.json"
    write_manifest(scan(raw), manifest)
    return Base(raw=raw, manifest=manifest)


def _later_days(rec: Recorded, root: Path) -> Path:
    for day in DAYS[1:]:
        shutil.copytree(rec.days / day, root / day)
    return root


def test_the_days_laid_over_the_snapshot_with_pinned_statistics_give_the_reference_rebuild(
    rec: Recorded, tmp_path: Path
) -> None:
    base = _base(rec, tmp_path / "base")
    out = tmp_path / "pinned"
    doc = rebuild(
        out,
        days_dir=_later_days(rec, tmp_path / "days"),
        base=base,
        pinned_from=rec.tables[3],
        **_fixture_args(tmp_path),
    )

    rebuilt = snapshot(out, model=FAKE_MODEL)
    assert rebuilt == snapshot(rec.rebuilt, model=FAKE_MODEL)
    assert rebuilt == snapshot(rec.tables[3], model=FAKE_MODEL)
    assert sum(len(r) for r in rebuilt.relatives.values()) > 0
    assert doc["statistics"] == "pinned"
    assert doc["days"] == list(DAYS[1:])
    assert doc["as_of"] == D3
    # Day 2 brings sessions day 1 lacked; day 3 touches two of day 1's archives.
    assert doc["archives"]["touched_by_days"] >= 2
    assert doc["archives"]["new_sessions"] > 0
    assert doc["vectors"]["shards_reused"] == doc["vectors"]["shards"]
    assert doc["vectors"]["shards_embedded"] == 0
    assert json.loads((work_dir(out) / "rebuild.json").read_text()) == doc
    assert not (work_dir(out) / "raw").exists()  # the archives' copy is not kept
    assert not out.with_name(out.name + ".building").exists()


def test_the_rebuild_with_statistics_counted_afresh_gives_the_refreshed_reference(
    rec: Recorded, tmp_path: Path
) -> None:
    out = tmp_path / "refreshed"
    doc = rebuild(
        out,
        days_dir=rec.days,
        base=None,
        vectors_from=rec.tables[3],
        max_share=float(REFRESHED_SHARE),
        **_fixture_args(tmp_path),
    )

    assert snapshot(out, model=FAKE_MODEL) == snapshot(rec.refreshed, model=FAKE_MODEL)
    assert doc["statistics"] == "refreshed"
    assert doc["days"] == list(DAYS)
    assert (out / "state_boilerplate" / "families_reached.npy").exists()


def test_a_rebuild_embeds_what_it_cannot_reuse(rec: Recorded, tmp_path: Path) -> None:
    out = tmp_path / "embedded"
    doc = rebuild(
        out,
        days_dir=rec.days,
        through=load_days(rec.days)[1].as_of,
        base=None,
        max_share=1.0,
        **_fixture_args(tmp_path),
    )
    assert doc["vectors"]["shards_reused"] == 0
    assert doc["vectors"]["shards_embedded"] == doc["vectors"]["shards"] > 0
    assert doc["days"] == list(DAYS[:2])
    # Over days 1 and 2 with the base build's empty stoplist, a rebuild is the reference's.
    assert snapshot(out, model=FAKE_MODEL) == snapshot(
        reference(rec.days, 2, tmp_path / "reference"), model=FAKE_MODEL
    )


def test_pinned_statistics_must_come_from_tables_at_the_same_day(
    rec: Recorded, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "kept").write_text("the last rebuild")
    with pytest.raises(RebuildError, match="same day"):
        rebuild(
            out,
            days_dir=rec.days,
            base=None,
            pinned_from=rec.tables[2],  # stands as of day 2; the days fold to day 3
            **_fixture_args(tmp_path),
        )
    assert (out / "kept").read_text() == "the last rebuild"


def test_an_archive_no_day_touches_is_copied_restamped(rec: Recorded, tmp_path: Path) -> None:
    base = _base(rec, tmp_path / "base")
    days = load_days(rec.days)
    laid = lay_days(fold(days[1:2]), base, tmp_path / "raw", as_of=days[1].as_of)
    before = {a.path: a for a in scan(base.raw)}
    after = {a.path: a for a in scan(tmp_path / "raw")}
    assert set(before) <= set(after)
    assert {a.generated_at for a in after.values()} == {f"{DAYS[1]} 00:00:00"}
    assert laid.touched == 0  # day 2 holds only states day 1 did not
    assert laid.new_sessions == len(after) - len(before) > 0
    assert all(ids for ids in laid.fresh.values())


def test_the_command_refuses_pinned_statistics_from_another_day(
    rec: Recorded, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "rebuild",
            "--days",
            str(rec.days),
            "--days-only",
            "--pinned-from",
            str(rec.tables[2]),
            "--out",
            str(tmp_path / "out"),
            "--text-plan",
            str(tmp_path / "no-plan.parquet"),
            "--min-free-gib",
            "0",
        ],
    )
    assert result.exit_code == 1
    assert "same day" in result.output
    assert not (tmp_path / "out").exists()
