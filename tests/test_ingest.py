import json
from pathlib import Path

import duckdb
import pytest

from lineage import queries
from lineage.disk import InsufficientDiskSpaceError
from lineage.ingest import ingest
from lineage.snapshot import SnapshotError, scan, write_manifest
from lineage.status import ENACTED_WORDING, OUTCOME_CLASSIFICATIONS
from tests.conftest import SESSIONS


@pytest.fixture
def manifest(raw_dir: Path, tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    write_manifest(scan(raw_dir), path)
    return path


def _bills(out: Path) -> duckdb.DuckDBPyConnection:
    con = queries.connect_quiet()
    con.execute(f"CREATE VIEW bills AS SELECT * FROM read_parquet('{out / 'bills' / '*.parquet'}')")
    return con


def test_writes_one_row_per_record(raw_dir: Path, manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "tables"

    result = ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)

    assert result.archives == 4
    assert result.rows == 5
    assert _bills(out).execute("SELECT count(*) FROM bills").fetchall() == [(5,)]


def test_rows_carry_state_session_and_generated_at(
    raw_dir: Path, manifest: Path, tmp_path: Path
) -> None:
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)

    rows = (
        _bills(out)
        .execute(
            "SELECT DISTINCT state, session, CAST(generated_at AS VARCHAR) "
            "FROM bills WHERE state = 'GA'"
        )
        .fetchall()
    )

    assert rows == [("GA", "2025_26", "2026-07-25 06:17:51.59996")]


def test_missing_text_is_null_and_blank_text_is_kept_blank(
    raw_dir: Path, manifest: Path, tmp_path: Path
) -> None:
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)

    rows = dict(
        _bills(out).execute("SELECT state, text FROM bills WHERE state IN ('HI', 'AK')").fetchall()
    )

    assert rows == {"HI": None, "AK": ""}


def test_keeps_actions_and_classification_for_later_steps(
    raw_dir: Path, manifest: Path, tmp_path: Path
) -> None:
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)
    con = _bills(out)

    (classes,) = con.execute(
        "SELECT list(DISTINCT c ORDER BY c) FROM (SELECT unnest(classification) AS c FROM bills)"
    ).fetchall()[0]
    dates = con.execute(
        "SELECT a.date FROM (SELECT unnest(actions) AS a FROM bills WHERE state = 'AL')"
    ).fetchall()

    assert classes == ["bill", "resolution"]
    # Open States mixes plain dates with zoned timestamps; ingest keeps them verbatim.
    assert {len(d) for (d,) in dates} == {25}


def test_derives_status_as_of_each_archive(raw_dir: Path, manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)

    rows = (
        _bills(out)
        .execute(
            "SELECT state, identifier, status, status_reason, CAST(status_date AS VARCHAR) "
            "FROM bills WHERE identifier IN ('SB 238', 'SR 1065', 'SB 47', 'HB 81') "
            "ORDER BY state, identifier"
        )
        .fetchall()
    )

    assert rows == [
        ("AK", "SB 47", "died", "session-ended", "2024-05-15"),
        ("AL", "HB 81", "died", "session-ended", "2023-06-08"),
        # Georgia files a line-item veto after the signature; the bill is still law.
        ("GA", "SB 238", "passed", "executive-signature", "2025-05-14"),
        # A simple resolution passes when its chamber adopts it.
        ("GA", "SR 1065", "passed", "passage", "2026-03-31"),
    ]


def test_refuses_archives_that_differ_from_the_manifest(
    raw_dir: Path, manifest: Path, tmp_path: Path
) -> None:
    with (raw_dir / "GA" / "GA_2025_26_json_fixture.zip").open("ab") as f:
        f.write(b"\0")
    out = tmp_path / "tables"

    with pytest.raises(SnapshotError):
        ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)
    assert not out.exists()


def test_refuses_to_run_below_the_disk_floor(raw_dir: Path, manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "tables"

    with pytest.raises(InsufficientDiskSpaceError):
        ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=1 << 60)
    assert not out.exists()


def test_rerun_replaces_the_previous_table(raw_dir: Path, manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "tables"
    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)
    stale = out / "bills" / "stale.parquet"
    stale.write_bytes(b"")

    ingest(raw_dir, manifest, SESSIONS, out, floor_bytes=0)

    assert not stale.exists()
    assert _bills(out).execute("SELECT count(*) FROM bills").fetchall() == [(5,)]


def test_status_columns_keep_their_types_when_every_value_is_null(
    raw_dir: Path, manifest: Path, tmp_path: Path
) -> None:
    out = tmp_path / "tables"
    # With no session dates, every fixture bill without an outcome stays introduced, so whole
    # archives have no status reason at all.
    no_dates = tmp_path / "sessions.csv"
    no_dates.write_text("state,session,name,start_date,end_date,carries_over_to\n")

    ingest(raw_dir, manifest, no_dates, out, floor_bytes=0)

    types = _bills(out).execute(
        "SELECT DISTINCT typeof(status_reason), typeof(status_date) FROM bills"
    )
    assert types.fetchall() == [("VARCHAR", "DATE")]


def test_status_reads_unclassified_actions_whose_wording_records_an_enactment(
    tmp_path: Path,
) -> None:
    def action(description: str, kinds: list[str], order: int) -> dict[str, object]:
        return {
            "organization__name": "House",
            "description": description,
            "date": "2025-05-20",
            "classification": kinds,
            "order": order,
        }

    bill = {
        "id": "ok-1",
        "classification": ["bill"],
        "actions": [
            action("Second Reading referred to Rules", [], 0),
            action("Third Reading, Measure passed", ["passage"], 1),
            action("Becomes law without Governor's signature 05/20/2025", [], 2),
        ],
    }
    path = tmp_path / "bills.json"
    path.write_text(json.dumps([bill]))

    with queries.connect_quiet() as con:
        ((_, _, actions),) = queries.outcome_actions(
            con, json_path=path, kinds=sorted(OUTCOME_CLASSIFICATIONS), wording=ENACTED_WORDING
        )

    assert [a["order"] for a in actions] == [1, 2]
