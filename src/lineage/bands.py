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
    build_band_index,
    candidate_pairs,
    cap_sweep,
    threshold,
)
from lineage.signatures import SignatureTable, load_signatures

# Each row (a window) costs a uint64 key and a uint32 row number in every band.
INDEX_ROW_BYTES = BANDS * (8 + 4)
SWEEP_CAPS = (100, 200, 500, 1000, 2000, 5000)


def row_states(tables_dir: Path, table: SignatureTable) -> NDArray[np.int32]:
    """Each signature row's state, as a code that is equal exactly when the states are."""
    with queries.connect_quiet() as con:
        by_bill = queries.bill_states(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    code = {state: i for i, state in enumerate(sorted(set(by_bill.values())))}
    return np.fromiter(
        (code[by_bill[b]] for b in table.bill_ids), dtype=np.int32, count=len(table.bill_ids)
    )


def build_bands(tables_dir: Path, *, floor_bytes: int) -> dict[str, Any]:
    """Rebuild `<tables>/bands/` from the signatures table and return its report section."""
    table = load_signatures(tables_dir)
    n = len(table.bill_ids)
    require_free_space(tables_dir, needed=n * INDEX_ROW_BYTES + (1 << 20), floor=floor_bytes)
    index = build_band_index(table.values)
    states = row_states(tables_dir, table)
    # The signatures are not read again here, and the cap sweep below needs the room.
    del table

    final = tables_dir / "bands"
    building = Path(tempfile.mkdtemp(prefix="bands.", dir=tables_dir))
    try:
        np.save(building / "keys.npy", index.keys)
        np.save(building / "rows.npy", index.rows)
        if final.exists():
            shutil.rmtree(final)
        building.rename(final)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise

    found = candidate_pairs(index, cap=BUCKET_CAP)
    cross = int(np.count_nonzero(states[found.left] != states[found.right]))
    # The build's own cap joins the sweep, so its losses are counted whatever the sweep lists.
    effects = cap_sweep(index, states, sorted({*SWEEP_CAPS, BUCKET_CAP}))
    return {
        "bands": BANDS,
        "rows": ROWS,
        "threshold": round(threshold(), 3),
        "index_bytes": n * INDEX_ROW_BYTES,
        "bucket_cap": BUCKET_CAP,
        "largest_bucket": found.largest_bucket,
        # Summed over bands: a window in capped buckets of three bands counts three times.
        "capped": {
            "buckets": found.buckets_over_cap,
            "window_memberships": found.rows_in_capped_buckets,
            "pairs": found.pairs_in_capped_buckets,
            "cross_state_pairs_lost": effects[BUCKET_CAP].cross_group_pairs_lost,
        },
        "candidate_pairs": {
            "all": int(found.left.size),
            "cross_state": cross,
            "same_state": int(found.left.size) - cross,
        },
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


def load_band_index(tables_dir: Path) -> BandIndex:
    """The band index, memory-mapped: a query reads only the pages its keys land on."""
    bands = tables_dir / "bands"
    keys = cast("NDArray[np.uint64]", np.load(bands / "keys.npy", mmap_mode="r"))
    rows = cast("NDArray[np.uint32]", np.load(bands / "rows.npy", mmap_mode="r"))
    return BandIndex(keys, rows)
