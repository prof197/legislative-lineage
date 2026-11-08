"""`lineage index`: sign every matchable section, band and verify, and write the report.

The report also records the build's wall time and peak memory beside the machine's memory, the
figures DR-3's bound is checked against: a full build within half the reference laptop's 32 GiB.
The main process, which holds the signatures while it bands and verifies them, is measured by
its physical footprint (lineage.memory); the signing workers, gone by the end, by the largest
resident set among them. See DESIGN.md#dr-3.
"""

import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

from lineage import memory
from lineage.bands import build_bands
from lineage.pairs import build_pairs
from lineage.signatures import build_signatures

COMMAND = "uv run lineage index"


def build_index(
    tables_dir: Path, *, workers: int, floor_bytes: int, report_path: Path
) -> dict[str, Any]:
    """Rebuild the signatures, band index, cross-state pairs and families; write the report."""
    started = time.monotonic()
    doc: dict[str, Any] = {"command": COMMAND}
    doc |= build_signatures(tables_dir, workers=workers, floor_bytes=floor_bytes)
    doc["lsh"] = build_bands(tables_dir, floor_bytes=floor_bytes)
    doc |= build_pairs(tables_dir, floor_bytes=floor_bytes)
    doc["build"] = {
        "seconds": round(time.monotonic() - started),
        "workers": workers,
        "peak_memory_bytes": memory.peak_bytes(),
        "peak_worker_resident_bytes": peak_worker_resident_bytes(),
        "machine": {"platform": platform.platform(), "memory_bytes": machine_memory_bytes()},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def peak_worker_resident_bytes() -> int:
    """The largest resident set of any worker process this one waited for."""
    # Linux reports kilobytes, macOS bytes.
    unit = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit


def machine_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
