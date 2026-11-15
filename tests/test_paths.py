"""Local data defaults to `data` beside the checkout, and `LINEAGE_DATA` moves all of it, so
worktrees can share one directory (DESIGN.md#lanes)."""

import os
import subprocess
import sys
from pathlib import Path


def _defaults(env: dict[str, str]) -> list[str]:
    code = (
        "from lineage.export import EXPORT_DIR; from lineage.modeldocs import MODELDOCS_DIR; "
        "from lineage.updates import CHANGES_DIR; "
        "print(EXPORT_DIR, MODELDOCS_DIR, CHANGES_DIR)"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
    return out.stdout.decode().split()


def test_one_variable_moves_every_default(tmp_path: Path) -> None:
    unset = {k: v for k, v in os.environ.items() if k != "LINEAGE_DATA"}
    assert _defaults(unset) == ["data/export", "data/modeldocs", "data/changes"]
    shared = str(tmp_path / "shared")
    assert _defaults({**unset, "LINEAGE_DATA": shared}) == [
        f"{shared}/export",
        f"{shared}/modeldocs",
        f"{shared}/changes",
    ]
