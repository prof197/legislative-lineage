"""Build the LSH band index from the signatures table, and measure what the bucket cap skips.

The index is two (BANDS, n) matrices in `<tables>/bands/`: each band's sorted keys
(`keys.npy`) and the signature row behind each key (`rows.npy`). Rows number the signatures,
one per window, in the order `load_signatures` reads them. The report counts the candidate
pairs and what the bucket cap skipped, including the cross-state candidates it drops for good.
For a range of caps, it also counts the cross-state candidates each would lose. See
DESIGN.md#dr-3.
"""

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.disk import require_free_space
from lineage.lsh import (
    BANDS,
    BUCKET_CAP,
    ROWS,
    BandIndex,
    cap_sweep,
    fill_band_index,
    threshold,
)
from lineage.signatures import RowLabels, SignatureFiles

# Each row (a window) costs a uint64 key and a uint32 row number in every band.
INDEX_ROW_BYTES = BANDS * (8 + 4)
SWEEP_CAPS = (100, 200, 500, 1000, 2000, 5000)
# Signature rows read per step while the band keys are computed: 512 MiB.
_CHUNK_ROWS = 1 << 20


def row_states(tables_dir: Path, labels: RowLabels) -> NDArray[np.int32]:
    """Each signature row's state, as a code that is equal exactly when the states are."""
    with queries.connect_quiet() as con:
        by_bill = queries.bill_states(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    code = {state: i for i, state in enumerate(sorted(set(by_bill.values())))}
    per_section = np.fromiter(
        (code[by_bill[b]] for b in labels.section_bills),
        dtype=np.int32,
        count=len(labels.section_bills),
    )
    return per_section[labels.section]


def _chunks(files: SignatureFiles) -> Iterator[NDArray[np.uint32]]:
    for matrix in files.matrices:
        for start in range(0, matrix.shape[0], _CHUNK_ROWS):
            yield np.asarray(matrix[start : start + _CHUNK_ROWS])


def build_bands(
    tables_dir: Path, files: SignatureFiles, states: NDArray[np.int32], *, floor_bytes: int
) -> dict[str, Any]:
    """Rebuild `<tables>/bands/` from the memory-mapped signatures; return its report section.

    The index is written straight into memory-mapped files, and the cap sweep's per-row tables
    live in files beside it, so the operating system pages them rather than the process
    holding them (DESIGN.md#dr-3). Candidate pairs and what the cap skips are counted when the
    pairs are formed and verified (lineage.pairs).
    """
    n = int(sum(m.shape[0] for m in files.matrices))
    # The index, and the cap sweep's tables of the same size while it runs.
    require_free_space(tables_dir, needed=2 * n * INDEX_ROW_BYTES + (1 << 20), floor=floor_bytes)

    def write(directory: Path) -> None:
        if n == 0:  # a file of no bytes cannot be memory-mapped
            np.save(directory / "keys.npy", np.empty((BANDS, 0), dtype=np.uint64))
            np.save(directory / "rows.npy", np.empty((BANDS, 0), dtype=np.uint32))
            return
        keys = np.lib.format.open_memmap(
            directory / "keys.npy", mode="w+", dtype=np.uint64, shape=(BANDS, n)
        )
        rows = np.lib.format.open_memmap(
            directory / "rows.npy", mode="w+", dtype=np.uint32, shape=(BANDS, n)
        )
        fill_band_index(BandIndex(keys, rows), _chunks(files))
        keys.flush()
        rows.flush()

    final = tables_dir / "bands"
    building = Path(tempfile.mkdtemp(prefix="bands.", dir=tables_dir))
    try:
        write(building)
        if final.exists():
            shutil.rmtree(final)
        building.rename(final)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise

    index = load_band_index(tables_dir)
    caps = sorted({*SWEEP_CAPS, BUCKET_CAP})
    # The build's own cap joins the sweep, so its losses are counted whatever the sweep lists.
    effects = cap_sweep(index, states, caps)
    return {
        "bands": BANDS,
        "rows": ROWS,
        "threshold": round(threshold(), 3),
        "index_bytes": n * INDEX_ROW_BYTES,
        "bucket_cap": BUCKET_CAP,
        "cross_state_pairs_lost": effects[BUCKET_CAP].cross_group_pairs_lost,
        "cap_sweep": [
            {
                "cap": cap,
                "buckets_over_cap": effect.buckets_over_cap,
                "multi_state_buckets_over_cap": effect.multi_group_buckets_over_cap,
                "most_states_in_a_capped_bucket": effect.most_groups_in_a_capped_bucket,
                "cross_state_pairs_lost": effect.cross_group_pairs_lost,
            }
            for cap, effect in effects.items()
            if cap in SWEEP_CAPS
        ],
    }


def lsh_report(bands: dict[str, Any], found: dict[str, Any]) -> dict[str, Any]:
    """The report's `lsh` section, from the index build and the band-by-band pair formation."""
    return {
        "bands": bands["bands"],
        "rows": bands["rows"],
        "threshold": bands["threshold"],
        "index_bytes": bands["index_bytes"],
        "bucket_cap": bands["bucket_cap"],
        "largest_bucket": found["largest_bucket"],
        # Summed over bands: a window in capped buckets of three bands counts three times.
        "capped": {
            "buckets": found["capped"]["buckets"],
            "window_memberships": found["capped"]["window_memberships"],
            "pairs": found["capped"]["pairs"],
            "cross_state_pairs_lost": bands["cross_state_pairs_lost"],
        },
        "candidate_pairs": found["candidate_pairs"],
        "cap_sweep": bands["cap_sweep"],
    }


def load_band_index(tables_dir: Path) -> BandIndex:
    """The band index, memory-mapped: a query reads only the pages its keys land on."""
    bands = tables_dir / "bands"
    keys = cast("NDArray[np.uint64]", np.load(bands / "keys.npy", mmap_mode="r"))
    rows = cast("NDArray[np.uint32]", np.load(bands / "rows.npy", mmap_mode="r"))
    return BandIndex(keys, rows)
