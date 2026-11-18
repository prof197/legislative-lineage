"""Download the archives the manifest pins, so anyone can rebuild `data/raw/` byte for byte.

Open States publishes the session archives without an account or key. Each file lands only if
its SHA-256 matches the manifest; `repin` accepts a changed file and records its new hash. Every
request goes through lineage.fetch. See DESIGN.md#data-snapshot.
"""

import urllib.error
import urllib.parse
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from lineage.disk import require_free_space
from lineage.fetch import FetchError, open_url, stream
from lineage.snapshot import Archive, SnapshotError, load_manifest, read_archive, write_manifest

SOURCE = "https://data.openstates.org/json/latest/"


@dataclass(slots=True)
class FetchResult:
    fetched: int = 0
    present: int = 0
    bytes: int = 0
    repinned: list[str] = field(default_factory=list[str])
    problems: list[str] = field(default_factory=list[str])


def url_for(archive: Archive, source: str) -> str:
    # A few session names contain spaces, which a request line cannot carry unencoded.
    return source + urllib.parse.quote(Path(archive.path).name)


def _fetch_one(
    archive: Archive, raw_dir: Path, *, source: str, floor: int, repin: bool, pause: float
) -> Archive:
    """Download one archive into place; return the archive as it now stands on disk."""
    dest = raw_dir / archive.path
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_url(url_for(archive, source), pause, interval=pause) as response:
            length = response.headers.get("Content-Length")
            size = int(length) if length is not None else archive.bytes
            if size != archive.bytes and not repin:
                raise FetchError(f"server offers {size} bytes, manifest pins {archive.bytes}")
            require_free_space(dest.parent, needed=size, floor=floor)
            sha256 = stream(response, part, cap=size, floor=floor)
        if sha256 == archive.sha256:
            landed = archive
        elif repin:
            with zipfile.ZipFile(part) as z:
                if (bad := z.testzip()) is not None:
                    raise FetchError(f"CRC failure in {bad}")
            # Re-read the README too: a regenerated archive carries a new Generated At.
            landed = replace(read_archive(raw_dir, part), path=archive.path)
        else:
            raise FetchError("SHA-256 differs from the manifest")
        part.rename(dest)
    finally:
        part.unlink(missing_ok=True)
    return landed


def fetch_raw(
    raw_dir: Path,
    manifest_path: Path,
    *,
    source: str = SOURCE,
    floor_bytes: int,
    repin: bool = False,
    pause: float = 1.0,
) -> FetchResult:
    """Download every pinned archive missing from `raw_dir`, one at a time, `pause` apart
    (lineage.pace)."""
    pinned = load_manifest(manifest_path)
    result = FetchResult()
    todo = [a for a in pinned if not (raw_dir / a.path).exists()]
    result.present = len(pinned) - len(todo)
    # The manifest knows every size, so the whole run is checked before the first request.
    require_free_space(raw_dir, needed=sum(a.bytes for a in todo), floor=floor_bytes)

    current = {a.path: a for a in pinned}
    try:
        for archive in todo:
            try:
                landed = _fetch_one(
                    archive, raw_dir, source=source, floor=floor_bytes, repin=repin, pause=pause
                )
            except (FetchError, SnapshotError, urllib.error.URLError) as e:
                if isinstance(e, urllib.error.HTTPError):
                    e.close()
                result.problems.append(f"{archive.path}: {e}")
                continue
            result.fetched += 1
            result.bytes += landed.bytes
            if landed != archive:
                current[archive.path] = landed
                result.repinned.append(archive.path)
    finally:
        # Even if the disk floor stops the run, files already re-pinned must reach the manifest,
        # or the next ingest would reject them as changed.
        if result.repinned:
            write_manifest(list(current.values()), manifest_path)
    return result
