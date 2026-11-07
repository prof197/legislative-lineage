"""Load the pinned Open States archives into the `bills` Parquet table.

DuckDB reads each archive's JSON with a declared schema and writes one Parquet file per
archive under `<out>/bills/`; together they form the table. Each bill's status is derived on
the way in, as of its archive's Generated At. See DESIGN.md#dr-4 and DESIGN.md#bill-status.
"""

import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from lineage import queries
from lineage.disk import require_free_space
from lineage.sessions import Session, load_sessions, status_end_date
from lineage.snapshot import Archive, bills_member, load_manifest, verify
from lineage.status import OUTCOME_CLASSIFICATIONS, STATUS_WORDING, Action, bill_status

# Ample for one archive at a time; the largest bills file in the snapshot is under 300 MiB.
_MEMORY_LIMIT = "4GB"


@dataclass(frozen=True, slots=True)
class IngestResult:
    archives: int
    rows: int
    output_bytes: int
    seconds: float


def _status_columns(
    con: duckdb.DuckDBPyConnection,
    json_path: Path,
    archive: Archive,
    sessions: dict[tuple[str, str], Session],
) -> queries.StatusColumns:
    end = status_end_date(sessions, archive.state, archive.session)
    as_of = date.fromisoformat(archive.generated_at[:10])
    columns = queries.StatusColumns(id=[], status=[], reason=[], on=[])
    rows = queries.outcome_actions(
        con, json_path=json_path, kinds=sorted(OUTCOME_CLASSIFICATIONS), wording=STATUS_WORDING
    )
    for bill_id, kinds, actions in rows:
        outcome = bill_status(
            [
                Action(
                    date=a["date"],
                    classification=tuple(a["classification"]),
                    order=a["order"],
                    organization=a["organization__name"] or "",
                    description=a["description"] or "",
                )
                for a in actions
            ],
            kinds=kinds,
            state=archive.state,
            session_end=end,
            as_of=as_of,
        )
        columns.id.append(bill_id)
        columns.status.append(outcome.status.value)
        columns.reason.append(outcome.reason)
        columns.on.append(outcome.on)
    return columns


def _ingest_archive(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    archive: Archive,
    sessions: dict[tuple[str, str], Session],
    dest: Path,
    scratch: Path,
) -> int:
    # DuckDB cannot read inside a zip. The members are stored uncompressed, so copying one out
    # is cheap, and copying one at a time bounds the extra disk use to the largest member.
    with zipfile.ZipFile(raw_dir / archive.path) as z:
        json_path = scratch / "bills.json"
        with z.open(bills_member(z)) as src, json_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    try:
        return queries.copy_bills_to_parquet(
            con,
            json_path=json_path,
            parquet_path=dest / (Path(archive.path).stem + ".parquet"),
            state=archive.state,
            generated_at=archive.generated_at,
            archive=archive.path,
            status=_status_columns(con, json_path, archive, sessions),
        )
    finally:
        json_path.unlink()


def ingest(
    raw_dir: Path, manifest_path: Path, sessions_path: Path, out_dir: Path, *, floor_bytes: int
) -> IngestResult:
    """Verify the raw archives against the manifest, then rebuild `<out>/bills/` from them."""
    started = time.monotonic()
    archives = load_manifest(manifest_path)
    sessions = load_sessions(sessions_path)
    verify(raw_dir, archives)
    # Parquet output is smaller than its JSON input, so the raw total bounds what this writes,
    # with the largest member added for the extraction copy.
    raw_total = sum(a.bytes for a in archives)
    largest = max((a.bytes for a in archives), default=0)
    require_free_space(out_dir, needed=raw_total + largest, floor=floor_bytes)

    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "bills"
    # Build beside the old table and swap at the end, so a failed run leaves the old table whole
    # and a successful one leaves no files from an earlier manifest behind.
    building = Path(tempfile.mkdtemp(prefix="bills.", dir=out_dir))
    try:
        with tempfile.TemporaryDirectory(dir=out_dir) as scratch:
            con = queries.connect(memory_limit=_MEMORY_LIMIT, temp_directory=Path(scratch))
            try:
                rows = sum(
                    _ingest_archive(con, raw_dir, a, sessions, building, Path(scratch))
                    for a in archives
                )
            finally:
                con.close()
        if final.exists():
            shutil.rmtree(final)
        building.rename(final)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    output_bytes = sum(p.stat().st_size for p in final.glob("*.parquet"))
    return IngestResult(
        archives=len(archives),
        rows=rows,
        output_bytes=output_bytes,
        seconds=time.monotonic() - started,
    )
