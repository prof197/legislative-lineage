"""Download the archives the manifest pins, so anyone can rebuild `data/raw/` byte for byte.

Open States publishes the session archives without an account or key. Each file lands only if
its SHA-256 matches the manifest; `repin` accepts a changed file and records its new hash. See
DESIGN.md#data-snapshot.
"""

import hashlib
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field, replace
from http.client import HTTPResponse
from pathlib import Path

from lineage import __version__
from lineage.disk import require_free_space
from lineage.snapshot import Archive, SnapshotError, load_manifest, read_archive, write_manifest

SOURCE = "https://data.openstates.org/json/latest/"
USER_AGENT = f"legislative-lineage/{__version__} (+https://github.com/prof197/legislative-lineage)"

_CHUNK = 1 << 20
# Free space is re-measured this often while streaming: other processes write to the disk too.
_RECHECK_BYTES = 64 * _CHUNK
_ATTEMPTS = 3
_TIMEOUT_SECONDS = 120
# Some legislature servers send their certificate without its intermediate. The bundle supplies
# those intermediates; chains must still end at a root in the default store.
INTERMEDIATES = Path(__file__).parent / "certs" / "intermediates.pem"


def tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=INTERMEDIATES)
    return context


_TLS = tls_context()


class FetchError(Exception):
    pass


@dataclass(slots=True)
class FetchResult:
    fetched: int = 0
    present: int = 0
    bytes: int = 0
    repinned: list[str] = field(default_factory=list[str])
    problems: list[str] = field(default_factory=list[str])


# Illinois retired beta.ilga.gov; the same paths are served from ilga.gov.
HOST_MOVES = {"beta.ilga.gov": "ilga.gov"}


def moved(url: str) -> str:
    """`url` on the host that now serves it, for hosts a legislature has retired."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if host not in HOST_MOVES:
        return url
    return urllib.parse.urlunsplit(parts._replace(netloc=HOST_MOVES[host]))


def url_for(archive: Archive, source: str) -> str:
    # A few session names contain spaces, which a request line cannot carry unencoded.
    return source + urllib.parse.quote(Path(archive.path).name)


def _retry_after(error: urllib.error.HTTPError, default: float) -> float:
    value = error.headers.get("Retry-After", "")
    return float(value) if value.isdigit() else default


def open_url(url: str, pause: float) -> HTTPResponse:
    """GET `url` as this project, retrying overload and server errors with backoff."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(_ATTEMPTS):
        try:
            response: HTTPResponse = urllib.request.urlopen(
                request, timeout=_TIMEOUT_SECONDS, context=_TLS
            )
            return response
        except urllib.error.HTTPError as e:
            # Only overload and server faults are worth repeating; a 404 will not change.
            if e.code not in {429, 500, 502, 503, 504} or attempt == _ATTEMPTS - 1:
                raise
            time.sleep(_retry_after(e, pause * 2 ** (attempt + 1)))
        except urllib.error.URLError:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(pause * 2 ** (attempt + 1))
    raise AssertionError("unreachable")


def _stream(response: HTTPResponse, part: Path, *, cap: int, floor: int) -> str:
    """Write the body to `part`, refusing more than `cap` bytes; return its SHA-256."""
    digest = hashlib.sha256()
    written = 0
    since_check = 0
    with part.open("wb") as out:
        while chunk := response.read(_CHUNK):
            written += len(chunk)
            if written > cap:
                raise FetchError(f"server sent more than the expected {cap} bytes")
            out.write(chunk)
            digest.update(chunk)
            since_check += len(chunk)
            if since_check >= _RECHECK_BYTES:
                since_check = 0
                require_free_space(part.parent, needed=cap - written, floor=floor)
    if written != cap:
        raise FetchError(f"received {written} of {cap} bytes")
    return digest.hexdigest()


def _fetch_one(
    archive: Archive, raw_dir: Path, *, source: str, floor: int, repin: bool, pause: float
) -> Archive:
    """Download one archive into place; return the archive as it now stands on disk."""
    dest = raw_dir / archive.path
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_url(url_for(archive, source), pause) as response:
            length = response.headers.get("Content-Length")
            size = int(length) if length is not None else archive.bytes
            if size != archive.bytes and not repin:
                raise FetchError(f"server offers {size} bytes, manifest pins {archive.bytes}")
            require_free_space(dest.parent, needed=size, floor=floor)
            sha256 = _stream(response, part, cap=size, floor=floor)
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
    """Download every pinned archive missing from `raw_dir`, one at a time, `pause` apart."""
    pinned = load_manifest(manifest_path)
    result = FetchResult()
    todo = [a for a in pinned if not (raw_dir / a.path).exists()]
    result.present = len(pinned) - len(todo)
    # The manifest knows every size, so the whole run is checked before the first request.
    require_free_space(raw_dir, needed=sum(a.bytes for a in todo), floor=floor_bytes)

    current = {a.path: a for a in pinned}
    try:
        for i, archive in enumerate(todo):
            if i:
                time.sleep(pause)
            try:
                landed = _fetch_one(
                    archive, raw_dir, source=source, floor=floor_bytes, repin=repin, pause=pause
                )
            except (FetchError, SnapshotError, urllib.error.URLError) as e:
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
