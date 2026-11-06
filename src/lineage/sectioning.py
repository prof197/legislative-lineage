"""Build the `sections` table: every bill's text split into sections, one row per section,
each flagged with the kind of boilerplate it is, if any.

Splitting is CPU-bound Python, so bills files are split in parallel, one per worker process.
Each worker writes its own Parquet parts, and a bounded batch keeps its memory flat however
large the archive. See DESIGN.md#sections.
"""

import multiprocessing
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import duckdb

from lineage import queries
from lineage.boilerplate import boilerplate_kind
from lineage.disk import require_free_space
from lineage.sections import normalize, split_sections

_FETCH_BILLS = 500
_PART_SECTIONS = 10_000


# Fork is Linux's default, and forking after DuckDB has started threads can deadlock the
# child. Spawn starts each worker clean on every platform.
_SPAWN = multiprocessing.get_context("spawn")


class SectioningError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class FileCounts:
    bills: int
    with_text: int
    by_heading: int
    sections: int


@dataclass(frozen=True, slots=True)
class SectioningResult:
    bills: int
    with_text: int
    by_heading: int
    sections: int
    output_bytes: int
    seconds: float


def _empty() -> queries.SectionColumns:
    return queries.SectionColumns([], [], [], [], [], [], [])


def _split_file(bills_file: Path, dest: Path) -> FileCounts:
    """Split one bills file into `dest/<stem>-<n>.parquet` parts. Runs in a worker process."""
    bills = with_text = by_heading = sections = 0
    columns = _empty()
    parts = 0

    def flush(con: duckdb.DuckDBPyConnection) -> None:
        nonlocal columns, parts
        if columns.bill_id:
            path = dest / f"{bills_file.stem}-{parts:03d}.parquet"
            queries.write_sections(con, parquet_path=path, columns=columns)
            parts += 1
            columns = _empty()

    reader = queries.connect_quiet()
    writer = queries.connect_quiet()
    try:
        scan = queries.open_bill_texts(reader, parquet_path=bills_file)
        while batch := scan.fetchmany(_FETCH_BILLS):
            for bill_id, raw_text in batch:
                bills += 1
                split = split_sections(raw_text)
                if not split.sections:
                    continue
                with_text += 1
                by_heading += split.by_heading
                for s in split.sections:
                    normalized = normalize(s.text)
                    columns.bill_id.append(bill_id)
                    columns.ordinal.append(s.ordinal)
                    columns.heading.append(s.heading)
                    columns.text.append(s.text)
                    columns.normalized.append(normalized)
                    columns.by_heading.append(split.by_heading)
                    columns.boilerplate.append(boilerplate_kind(normalized))
                sections += len(split.sections)
            if len(columns.bill_id) >= _PART_SECTIONS:
                flush(writer)
        flush(writer)
    finally:
        reader.close()
        writer.close()
    return FileCounts(bills, with_text, by_heading, sections)


def build_sections(tables_dir: Path, *, workers: int, floor_bytes: int) -> SectioningResult:
    """Rebuild `<tables>/sections/` from `<tables>/bills/`."""
    started = time.monotonic()
    # Largest first, so the slowest files do not start last and leave the other workers idle.
    files = sorted(
        (tables_dir / "bills").glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True
    )
    if not files:
        raise SectioningError(f"no bills table in {tables_dir}; run `lineage ingest` first")
    with queries.connect_quiet() as con:
        text = queries.text_bytes(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    # Each section is stored twice, as display text and as normalized text, and neither copy is
    # longer than the input; compressed Parquet stays under that bound.
    require_free_space(tables_dir, needed=2 * text, floor=floor_bytes)

    final = tables_dir / "sections"
    building = Path(tempfile.mkdtemp(prefix="sections.", dir=tables_dir))
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_SPAWN) as pool:
            counts = list(pool.map(_split_file, files, repeat(building)))
        if final.exists():
            shutil.rmtree(final)
        building.rename(final)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return SectioningResult(
        bills=sum(c.bills for c in counts),
        with_text=sum(c.with_text for c in counts),
        by_heading=sum(c.by_heading for c in counts),
        sections=sum(c.sections for c in counts),
        output_bytes=sum(p.stat().st_size for p in final.glob("*.parquet")),
        seconds=time.monotonic() - started,
    )
