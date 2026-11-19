"""Table directories replaced whole: each table is written into a directory beside its own,
then swapped in, so a reader never sees half of one. Every layer's tables go through these."""

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path


def replace_dir[T](final: Path, write: Callable[[Path], T]) -> T:
    """Write a table into a fresh directory beside `final`, then swap it in, and return what
    the write returned; the directory is removed if the write fails."""
    building = Path(tempfile.mkdtemp(prefix=f"{final.name}.", dir=final.parent))
    try:
        result = write(building)
        if final.exists():
            shutil.rmtree(final)
        building.rename(final)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return result


def rebuild_dir(final: Path, write: Callable[[Path], None]) -> None:
    """Write a table into `<final>.building`, cleared first, then swap it in; a failed write
    leaves it there for the next build to clear."""
    building = final.with_name(final.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    write(building)
    if final.exists():
        shutil.rmtree(final)
    building.rename(final)
