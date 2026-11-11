"""`lineage semantic-pairs`: the semantic stage's matches as cross-state section pairs.

`lineage neighbors` keeps each distinct text's K nearest texts at or above the threshold. A text
stands for every window that carries it, so two sections match by meaning when a window of one
and a window of the other carry the same text, or two texts one keeps as a neighbor of the
other. Each such pair across state lines is stored once, with its best cosine and the window
pair that gave it, numbered as the near-duplicate stage's pairs are, by section in the
signatures' order, so the two stages' pairs join directly: the precision sample's strata, bill
scoring and the export read them alike. Window numbers are the semantic stage's own, 94 words
without overlap (lineage.semantic.query_windows), which point alignment and explanations at the
text that matched, and keep the effect of the cap of K traceable.

Same-state pairs are not stored, since families come from the near-duplicate stage. A section
the near-duplicate stage did not sign, being shorter than one shingle, has no number; its
windows are left out and counted.

The table records the model, K and threshold it was built under, and refuses to load when any
of them differs from the stage's own, so changing one means rebuilding it. A single query never
reads it: it searches the vectors exactly. See DESIGN.md#semantic-stage.
"""

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.disk import require_free_space
from lineage.memory import BOUND_BYTES, peak_bytes, require_within
from lineage.semantic import MODEL, THRESHOLD, K
from lineage.signatures import open_signatures, row_labels
from lineage.vectors import semantic_dir, vectors_dir

REPORT = Path("reports/semantic_pairs.json")
COMMAND = "uv run lineage semantic-pairs"
# Neighbor scores are stored as float16, one step of which near 0.89 is 2**-11.
_FLOAT16_STEP = 2.0**-11
_PAIR_BYTES = 4 + 4 + 4 + 4 + 4
_FILES = ("left", "right", "cosine", "left_window", "right_window")
# DuckDB's share of the memory bound (DR-3); past it, the join spills to the data disk.
_DUCKDB_MEMORY = "6GB"


class SemanticPairsError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class SemanticPairs:
    left: NDArray[np.uint32]  # section numbers in the signatures' order, left < right
    right: NDArray[np.uint32]
    cosine: NDArray[np.float32]  # the best cosine between a window of each
    left_window: NDArray[np.uint32]  # the windows that gave it, numbered within each section
    right_window: NDArray[np.uint32]


def pairs_dir(tables_dir: Path, model: str = MODEL) -> Path:
    return vectors_dir(tables_dir, model) / "pairs"


def fingerprint(model: str, *, k: int = K, threshold: float = THRESHOLD) -> dict[str, Any]:
    return {"model": model, "k": k, "threshold": threshold}


def section_keys(tables_dir: Path) -> tuple[list[str], NDArray[np.int64]]:
    """Each signed section's bill id and ordinal, indexed by its section number."""
    labels = row_labels(open_signatures(tables_dir))
    return labels.section_bills, labels.section_ordinals


def _kept_neighbors(tables_dir: Path, model: str, k: int, threshold: float) -> dict[str, Any]:
    neighbors = vectors_dir(tables_dir, model) / "neighbors"
    if not (neighbors / "ids.npy").exists():
        raise SemanticPairsError(f"no neighbors in {neighbors}; run `lineage neighbors`")
    ids = np.load(neighbors / "ids.npy", mmap_mode="r")
    scores = np.load(neighbors / "scores.npy", mmap_mode="r")
    if ids.shape[1] != k:
        raise SemanticPairsError(
            f"the neighbors keep {ids.shape[1]} per text, not K = {k}; run `lineage neighbors`"
        )
    kept = np.asarray(ids) >= 0
    a = np.nonzero(kept)[0].astype(np.int64)
    b = np.asarray(ids)[kept].astype(np.int64)
    cosine = np.asarray(scores)[kept].astype(np.float32)
    if cosine.size and float(cosine.min()) < threshold - _FLOAT16_STEP:
        raise SemanticPairsError(
            f"the neighbors keep cosines down to {float(cosine.min()):.4f}, under the threshold "
            f"of {threshold}; run `lineage neighbors`"
        )
    return {"a": a, "b": b, "cosine": cosine}


def _replace_dir(final: Path, write: Any) -> None:
    building = final.with_name(final.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    write(building)
    if final.exists():
        shutil.rmtree(final)
    building.rename(final)


def build_semantic_pairs(
    tables_dir: Path,
    *,
    model: str = MODEL,
    floor_bytes: int,
    k: int = K,
    threshold: float = THRESHOLD,
    bound_bytes: int = BOUND_BYTES,
) -> dict[str, Any]:
    """Expand the kept neighbors into cross-state section pairs and store them; returns the
    report. Stops if the process's footprint crosses `bound_bytes`."""
    started = time.monotonic()
    neighbors = _kept_neighbors(tables_dir, model, k, threshold)
    bills, ordinals = section_keys(tables_dir)
    sections = {
        "section": np.arange(len(bills), dtype=np.int64),
        "bill_id": np.array(bills, dtype=object),
        "ordinal": ordinals,
    }
    units_glob = str(semantic_dir(tables_dir) / "units" / "*.parquet")
    final = pairs_dir(tables_dir, model)
    spill = final.with_name(final.name + ".spill")
    spill.mkdir(parents=True, exist_ok=True)
    with queries.connect(memory_limit=_DUCKDB_MEMORY, temp_directory=spill) as con:
        found = queries.semantic_section_pairs(
            con,
            units_glob=units_glob,
            bills_glob=str(tables_dir / "bills" / "*.parquet"),
            sections=sections,
            neighbors=neighbors,
        )
        windows_out, sections_out = queries.semantic_units_left_out(
            con, units_glob=units_glob, sections=sections
        )
    shutil.rmtree(spill)
    require_within(bound_bytes, "expanding the neighbors into section pairs")
    pairs = SemanticPairs(
        found["l"].astype(np.uint32),
        found["r"].astype(np.uint32),
        found["cosine"].astype(np.float32),
        found["lw"].astype(np.uint32),
        found["rw"].astype(np.uint32),
    )
    require_free_space(
        final.parent, needed=pairs.left.size * _PAIR_BYTES + (1 << 20), floor=floor_bytes
    )

    def write(dest: Path) -> None:
        for name in _FILES:
            np.save(dest / f"{name}.npy", getattr(pairs, name))
        doc = fingerprint(model, k=k, threshold=threshold)
        (dest / "fingerprint.json").write_text(json.dumps(doc, indent=1) + "\n")

    _replace_dir(final, write)
    shared = int((pairs.cosine >= 1.0 - _FLOAT16_STEP).sum())
    return {
        "command": COMMAND,
        **fingerprint(model, k=k, threshold=threshold),
        "neighbors_read": int(neighbors["a"].size),
        "pairs": int(pairs.left.size),
        "pairs_at_cosine_1": shared,
        "cosine_histogram": {
            "edges": [round(float(e), 4) for e in np.linspace(threshold, 1.0, 11)],
            "counts": np.histogram(pairs.cosine, bins=np.linspace(threshold, 1.0, 11))[0].tolist(),
        },
        "windows_left_out": windows_out,
        "sections_left_out": sections_out,
        "bytes": int(pairs.left.size * _PAIR_BYTES),
        "seconds": round(time.monotonic() - started),
        "peak_memory_bytes": peak_bytes(),
        "memory_bound_bytes": bound_bytes,
    }


def load_semantic_pairs(tables_dir: Path, model: str = MODEL) -> SemanticPairs:
    """The stored pairs, memory-mapped; refused when built under another model, K or threshold."""
    where = pairs_dir(tables_dir, model)
    path = where / "fingerprint.json"
    built = json.loads(path.read_text()) if path.exists() else None
    if built != fingerprint(model):
        raise SemanticPairsError(
            f"{where} was built under {built}, not {fingerprint(model)}; run `{COMMAND}`"
        )
    arrays = [np.load(where / f"{name}.npy", mmap_mode="r") for name in _FILES]
    return SemanticPairs(
        cast("NDArray[np.uint32]", arrays[0]),
        cast("NDArray[np.uint32]", arrays[1]),
        cast("NDArray[np.float32]", arrays[2]),
        cast("NDArray[np.uint32]", arrays[3]),
        cast("NDArray[np.uint32]", arrays[4]),
    )


def write_report(doc: dict[str, Any], path: Path = REPORT) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")
