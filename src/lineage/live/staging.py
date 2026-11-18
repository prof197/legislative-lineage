"""The staging view an update builds the tables in beside them, swapped in only at the end.

Owned by lane L1 (M10). See DESIGN.md#updates, DR-7.
"""

import shutil
from pathlib import Path

UPDATES_DIR = "updates"


class Staging:
    """The tables after an update, built beside them: every entry starts as a link to the one
    in place, a directory written into is made real with links to its entries, and a file
    written replaces its link. `commit` moves what was written into place and removes what was
    dropped, so the tables change only at the end."""

    def __init__(self, tables_dir: Path, root: Path) -> None:
        self.tables = tables_dir
        self.root = root
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        for entry in sorted(tables_dir.iterdir()):
            if entry.name != UPDATES_DIR:
                (root / entry.name).symlink_to(entry.resolve())

    def own(self, rel: str) -> Path:
        """`rel` as a real directory in the view, its entries links to those in place. Its
        parents are made real first, so nothing below a link is ever written into."""
        parent = rel.rpartition("/")[0]
        if parent:
            self.own(parent)
        path = self.root / rel
        if path.is_symlink():
            target = path.resolve()
            path.unlink()
            path.mkdir()
            for entry in sorted(target.iterdir()):
                (path / entry.name).symlink_to(entry.resolve())
        elif not path.exists():
            path.mkdir()
        return path

    def fresh(self, rel: str) -> Path:
        """A path to write the file `rel` at, never through a link to the file in place."""
        parent, _, _ = rel.rpartition("/")
        if parent:
            self.own(parent)
        path = self.root / rel
        if path.is_symlink():
            path.unlink()
        return path

    def fresh_dir(self, rel: str) -> Path:
        """An empty real directory at `rel`, replacing the table in place on commit."""
        path = self.fresh(rel)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        return path

    def remove(self, rel: str) -> None:
        path = self.fresh(rel)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def commit(self) -> None:
        self._commit(self.root, self.tables)
        shutil.rmtree(self.root)

    def _commit(self, view: Path, place: Path) -> None:
        if place.is_symlink():
            # A table shared with another build by a link: what is unchanged stays shared.
            place.unlink()
        place.mkdir(parents=True, exist_ok=True)
        names = {p.name for p in view.iterdir()}
        for old in place.iterdir():
            if old.name not in names and not (place == self.tables and old.name == UPDATES_DIR):
                if old.is_dir() and not old.is_symlink():
                    shutil.rmtree(old)
                else:
                    old.unlink()
        for entry in sorted(view.iterdir()):
            target = place / entry.name
            if entry.is_symlink():
                if not target.exists() or target.resolve() != entry.resolve():
                    if target.is_symlink() or target.is_file():
                        target.unlink()
                    target.symlink_to(entry.resolve())
            elif entry.is_dir():
                if target.exists() and not target.is_dir():
                    target.unlink()
                self._commit(entry, target)
            else:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                entry.replace(target)
