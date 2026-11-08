"""Build the signatures table: one MinHash signature for every window of a matchable section.

A section of up to `WHOLE_MAX` words is one window; a longer one is cut into overlapping
windows (lineage.windows). Sections files are signed in parallel, one per worker process. Each
writes a dense uint32 matrix with one row per window (`<stem>.npy`, which memory-maps without
conversion) and the keys of those rows (`<stem>.parquet`: bill id, section ordinal, window).
Shingles are stoplisted before signing. A window left with no shingle has no signature. A
section with no signed window is not signed, and the report counts why: too short to form a
shingle, or every shingle stoplisted. See DESIGN.md#dr-3 and DESIGN.md#windows.
"""

import hashlib
import multiprocessing
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.disk import require_free_space
from lineage.minhash import NUM_PERM, SEED, Permutations, permutations, signatures
from lineage.shingles import section_shingles
from lineage.stoplist import apply_stoplist, load_stoplist
from lineage.windows import WHOLE_MAX, WINDOW_STRIDE, WINDOW_WORDS, normalized_windows

# Fork after DuckDB has started threads can deadlock the child; spawn starts each worker clean.
_SPAWN = multiprocessing.get_context("spawn")

_FETCH_SECTIONS = 5_000
_ROW_BYTES = NUM_PERM * 4
# Generous per-row and per-file allowances for the compressed key files.
_KEY_ROW_BYTES = 128
_FILE_BYTES = 64 << 10


@dataclass(frozen=True, slots=True)
class PartCounts:
    matchable: int
    signed: int
    too_short: int
    all_stoplisted: int
    windowed: int  # sections cut into more than one window
    windows_stoplisted: int  # windows of a signed section left with no shingle
    set_sizes: NDArray[np.int64]  # shingles per signed window, after the stoplist


@dataclass(frozen=True, slots=True)
class SignatureTable:
    bill_ids: list[str]
    ordinals: NDArray[np.int64]
    windows: NDArray[np.int64]
    values: NDArray[np.uint32]  # one row per window, NUM_PERM columns

    def sections(self) -> NDArray[np.int64]:
        """Each row's section, numbered from 0 in row order; a section's windows are adjacent."""
        return section_numbers(self.bill_ids, self.ordinals)


def section_numbers(bill_ids: list[str], ordinals: NDArray[np.int64]) -> NDArray[np.int64]:
    """Number the sections of signature rows in order: a new number wherever the key changes."""
    if not bill_ids:
        return np.empty(0, dtype=np.int64)
    bills = np.fromiter(
        (a != b for a, b in zip(bill_ids[1:], bill_ids[:-1], strict=True)),
        dtype=bool,
        count=len(bill_ids) - 1,
    )
    new = np.concatenate([[True], bills | (ordinals[1:] != ordinals[:-1])])
    return np.cumsum(new) - 1


def _sign_part(
    part: Path, eligible: Path, stoplist: NDArray[np.uint32], perms: Permutations, dest: Path
) -> PartCounts:
    """Sign one sections file into `dest/<stem>.npy` and its keys. Runs in a worker process."""
    matchable = signed = too_short = all_stoplisted = windowed = windows_stoplisted = 0
    bill_ids: list[str] = []
    ordinals: list[int] = []
    windows: list[int] = []
    rows: list[NDArray[np.uint32]] = []
    sizes: list[int] = []
    with queries.connect_quiet() as con:
        scan = queries.matchable_sections(
            con, sections_path=part, eligible_path=eligible, display=True
        )
        while batch := scan.fetchmany(_FETCH_SECTIONS):
            cut = [normalized_windows(text, normalized) for *_, text, normalized in batch]
            shingled = section_shingles([w for section in cut for w in section])
            kept = apply_stoplist(shingled, stoplist)
            signable: list[NDArray[np.uint32]] = []
            at = 0
            for (bill_id, ordinal, *_), section in zip(batch, cut, strict=True):
                befores, afters = shingled[at : at + len(section)], kept[at : at + len(section)]
                at += len(section)
                windowed += len(section) > 1
                if not any(a.size for a in afters):
                    if any(b.size for b in befores):
                        all_stoplisted += 1
                    else:
                        too_short += 1
                    continue
                signed += 1
                for window, (before, after) in enumerate(zip(befores, afters, strict=True)):
                    if after.size:
                        bill_ids.append(bill_id)
                        ordinals.append(ordinal)
                        windows.append(window)
                        signable.append(after)
                        sizes.append(after.size)
                    elif before.size:
                        windows_stoplisted += 1
            matchable += len(batch)
            if signable:
                rows.append(signatures(signable, perms))
        if rows:
            np.save(dest / f"{part.stem}.npy", np.concatenate(rows))
            queries.write_signature_keys(
                con,
                parquet_path=dest / f"{part.stem}.parquet",
                bill_ids=bill_ids,
                ordinals=ordinals,
                windows=windows,
            )
    return PartCounts(
        matchable,
        signed,
        too_short,
        all_stoplisted,
        windowed,
        windows_stoplisted,
        np.array(sizes, dtype=np.int64),
    )


def _quantile(values: NDArray[np.int64], q: float) -> int:
    # An observed value, not an interpolation, so the report shows real set sizes.
    return int(np.quantile(values, q, method="inverted_cdf")) if values.size else 0


def build_signatures(tables_dir: Path, *, workers: int, floor_bytes: int) -> dict[str, Any]:
    """Rebuild `<tables>/signatures/` from the matchable sections and return its report."""
    # Largest first, so the slowest files do not start last and leave the other workers idle.
    parts = sorted(
        (tables_dir / "sections").glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True
    )
    eligible = tables_dir / "eligible.parquet"
    stoplist = load_stoplist(tables_dir)
    with queries.connect_quiet() as con:
        # Windows overlap by half, so a section yields at most one per stride of its words,
        # plus the last.
        windows = queries.window_bound(
            con, sections_glob=str(tables_dir / "sections" / "*.parquet"), stride=WINDOW_STRIDE
        )
        words = queries.section_word_quantiles(
            con, sections_glob=str(tables_dir / "sections" / "*.parquet"), eligible_path=eligible
        )
    require_free_space(
        tables_dir,
        needed=windows * (_ROW_BYTES + _KEY_ROW_BYTES) + 2 * len(parts) * _FILE_BYTES,
        floor=floor_bytes,
    )

    perms = permutations()
    final = tables_dir / "signatures"
    building = Path(tempfile.mkdtemp(prefix="signatures.", dir=tables_dir))
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_SPAWN) as pool:
            counts = list(
                pool.map(
                    _sign_part,
                    parts,
                    repeat(eligible),
                    repeat(stoplist),
                    repeat(perms),
                    repeat(building),
                )
            )
        if final.exists():
            shutil.rmtree(final)
        building.rename(final)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise

    sizes = np.concatenate([c.set_sizes for c in counts]) if counts else np.empty(0, np.int64)
    return {
        # The stoplist's content, which every signature depends on.
        "stoplist_sha256": hashlib.sha256(stoplist.tobytes()).hexdigest(),
        "minhash": {"num_perm": NUM_PERM, "seed": SEED, "prime": "2^61 - 1"},
        "sections": {
            "matchable": sum(c.matchable for c in counts),
            "signed": sum(c.signed for c in counts),
            "too_short": sum(c.too_short for c in counts),
            "all_stoplisted": sum(c.all_stoplisted for c in counts),
        },
        # Display words per matchable section, which the window size is set from.
        "section_words": words,
        "windows": {
            "words": WINDOW_WORDS,
            "stride": WINDOW_STRIDE,
            "whole_max": WHOLE_MAX,
            "windowed_sections": sum(c.windowed for c in counts),
            "signed": int(sizes.size),
            "all_stoplisted": sum(c.windows_stoplisted for c in counts),
        },
        # Per signed window.
        "shingle_set_size": {
            "median": _quantile(sizes, 0.5),
            "p90": _quantile(sizes, 0.9),
            "p99": _quantile(sizes, 0.99),
            "max": int(sizes.max()) if sizes.size else 0,
        },
        "signature_bytes": int(sizes.size) * _ROW_BYTES,
    }


def load_signatures(tables_dir: Path) -> SignatureTable:
    """The whole signatures table in memory, rows in file order."""
    files = sorted((tables_dir / "signatures").glob("*.npy"))
    bill_ids: list[str] = []
    ordinals: list[int] = []
    windows: list[int] = []
    matrices: list[NDArray[np.uint32]] = []
    with queries.connect_quiet() as con:
        for path in files:
            ids, ords, wins = queries.read_signature_keys(
                con, parquet_path=path.with_suffix(".parquet")
            )
            bill_ids += ids
            ordinals += ords
            windows += wins
            matrices.append(cast("NDArray[np.uint32]", np.load(path)))
    values = np.concatenate(matrices) if matrices else np.empty((0, NUM_PERM), dtype=np.uint32)
    return SignatureTable(
        bill_ids, np.array(ordinals, dtype=np.int64), np.array(windows, dtype=np.int64), values
    )


@dataclass(frozen=True, slots=True)
class SignatureFiles:
    """The signatures table on disk, memory-mapped. Rows are numbered across files in order."""

    matrices: list[NDArray[np.uint32]]
    keys: list[Path]
    starts: NDArray[np.int64]  # the first row of each file


def open_signatures(tables_dir: Path) -> SignatureFiles:
    """Memory-map every signatures file, reading no rows until they are asked for."""
    files = sorted((tables_dir / "signatures").glob("*.npy"))
    matrices = [cast("NDArray[np.uint32]", np.load(path, mmap_mode="r")) for path in files]
    sizes = np.array([m.shape[0] for m in matrices], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    return SignatureFiles(matrices, [p.with_suffix(".parquet") for p in files], starts)


def _by_file(files: SignatureFiles, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.searchsorted(files.starts, rows, side="right") - 1


def signature_values(files: SignatureFiles, rows: NDArray[np.int64]) -> NDArray[np.uint32]:
    """The signatures of the given rows, in the order given."""
    out = np.empty((rows.size, NUM_PERM), dtype=np.uint32)
    which = _by_file(files, rows)
    used: list[int] = np.unique(which).tolist()
    for f in used:
        here = which == f
        out[here] = files.matrices[f][rows[here] - files.starts[f]]
    return out


def signature_bill_ids(files: SignatureFiles, rows: NDArray[np.int64]) -> list[str]:
    """The bill id behind each of the given rows, in the order given."""
    out = [""] * rows.size
    which = _by_file(files, rows)
    used: list[int] = np.unique(which).tolist()
    with queries.connect_quiet() as con:
        for f in used:
            positions = np.flatnonzero(which == f)
            local = (rows[positions] - files.starts[f]).tolist()
            ids = queries.signature_bill_ids_at(con, parquet_path=files.keys[f], rows=local)
            for position, row in zip(positions.tolist(), local, strict=True):
                out[position] = ids[row]
    return out
