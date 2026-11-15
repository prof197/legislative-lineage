"""Lane L1's incremental core: one day applied to a built corpus, matched in both directions.

Owned by lane L1 (M10). The contract it meets is `lineage.updates.update` and the tests in
tests/test_updates.py. See DESIGN.md#updates, DR-7.
"""

from pathlib import Path

from lineage.updates import Change, Day


def apply_day(
    tables_dir: Path, day: Day, *, model: str, changes_dir: Path, floor_bytes: int
) -> list[Change]:
    """See `lineage.updates.update`."""
    raise NotImplementedError("lineage update is lane L1's (M10)")
