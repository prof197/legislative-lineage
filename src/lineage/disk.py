"""Refuse writes that would leave the disk below a free-space floor.

Every step that writes large derived data checks here first, with an upper bound on what it
will write, so a full disk is prevented rather than recovered from.
"""

import shutil
from pathlib import Path

GiB = 1 << 30
DEFAULT_FLOOR_BYTES = 20 * GiB


class InsufficientDiskSpaceError(Exception):
    pass


def check_free_space(*, free: int, needed: int, floor: int) -> None:
    if free - needed < floor:
        raise InsufficientDiskSpaceError(
            f"writing up to {needed / GiB:.1f} GiB would leave {(free - needed) / GiB:.1f} GiB "
            f"free, below the {floor / GiB:.1f} GiB floor"
        )


def require_free_space(path: Path, *, needed: int, floor: int) -> None:
    # Measure the nearest existing ancestor: the target directory may not exist yet.
    existing = next(p for p in (path, *path.parents) if p.exists())
    check_free_space(free=shutil.disk_usage(existing).free, needed=needed, floor=floor)
