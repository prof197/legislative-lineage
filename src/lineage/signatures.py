"""Build the signatures table: one MinHash signature for every matchable section.

Sections files are signed in parallel, one per worker process. Each writes a dense uint32
matrix with one row per section (`<stem>.npy`, which memory-maps without conversion) and the
keys of those rows (`<stem>.parquet`: bill id and ordinal). Shingles are stoplisted before
signing. A section left with no shingle has no signature, and the report counts why: too short
to form a shingle, or every shingle stoplisted. See DESIGN.md#dr-3.
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
    too_short: int
    all_stoplisted: int
    set_sizes: NDArray[np.int64]  # shingles per signed section, after the stoplist


@dataclass(frozen=True, slots=True)
class SignatureTable:
    bill_ids: list[str]
    ordinals: NDArray[np.int64]
    values: NDArray[np.uint32]  # one row per section, NUM_PERM columns


def _sign_part(
    part: Path, eligible: Path, stoplist: NDArray[np.uint32], perms: Permutations, dest: Path
) -> PartCounts:
    """Sign one sections file into `dest/<stem>.npy` and its keys. Runs in a worker process."""
    matchable = too_short = all_stoplisted = 0
    bill_ids: list[str] = []
    ordinals: list[int] = []
    rows: list[NDArray[np.uint32]] = []
    sizes: list[int] = []
    with queries.connect_quiet() as con:
        scan = queries.matchable_sections(con, sections_path=part, eligible_path=eligible)
        while batch := scan.fetchmany(_FETCH_SECTIONS):
            shingled = section_shingles([text for *_, text in batch])
            kept = apply_stoplist(shingled, stoplist)
            signable: list[NDArray[np.uint32]] = []
            for (bill_id, ordinal, _), before, after in zip(batch, shingled, kept, strict=True):
                if after.size:
                    bill_ids.append(bill_id)
                    ordinals.append(ordinal)
                    signable.append(after)
                    sizes.append(after.size)
                elif before.size:
                    all_stoplisted += 1
                else:
                    too_short += 1
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
            )
    return PartCounts(matchable, too_short, all_stoplisted, np.array(sizes, dtype=np.int64))


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
        sections = queries.count_sections(
            con, sections_glob=str(tables_dir / "sections" / "*.parquet")
        )
    require_free_space(
        tables_dir,
        needed=sections * (_ROW_BYTES + _KEY_ROW_BYTES) + 2 * len(parts) * _FILE_BYTES,
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
            "signed": int(sizes.size),
            "too_short": sum(c.too_short for c in counts),
            "all_stoplisted": sum(c.all_stoplisted for c in counts),
        },
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
    matrices: list[NDArray[np.uint32]] = []
    with queries.connect_quiet() as con:
        for path in files:
            ids, ords = queries.read_signature_keys(con, parquet_path=path.with_suffix(".parquet"))
            bill_ids += ids
            ordinals += ords
            matrices.append(cast("NDArray[np.uint32]", np.load(path)))
    values = np.concatenate(matrices) if matrices else np.empty((0, NUM_PERM), dtype=np.uint32)
    return SignatureTable(bill_ids, np.array(ordinals, dtype=np.int64), values)
