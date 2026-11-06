"""`lineage index`: sign every matchable section, band the signatures, and write the report.

See DESIGN.md#dr-3.
"""

import json
from pathlib import Path
from typing import Any

from lineage.bands import build_bands
from lineage.signatures import build_signatures

COMMAND = "uv run lineage index"


def build_index(
    tables_dir: Path, *, workers: int, floor_bytes: int, report_path: Path
) -> dict[str, Any]:
    """Rebuild the signatures and the band index, and write `report_path`."""
    doc: dict[str, Any] = {"command": COMMAND}
    doc |= build_signatures(tables_dir, workers=workers, floor_bytes=floor_bytes)
    doc["lsh"] = build_bands(tables_dir, floor_bytes=floor_bytes)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc
