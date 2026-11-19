"""HTTP as this project makes it: its User-Agent, the TLS intermediates some legislature servers
omit, hosts that moved, retries, the host's turn (lineage.core.pace), and downloads streamed to disk
under the free-space floor. Every request the pipeline makes goes through open_url.
See DESIGN.md#data-snapshot.
"""

import hashlib
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from http.client import HTTPResponse
from pathlib import Path

from lineage import __version__
from lineage.core import pace
from lineage.core.disk import require_free_space
from lineage.core.pace import MIN_INTERVAL_SECONDS

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


# Illinois retired beta.ilga.gov; the same paths are served from ilga.gov.
HOST_MOVES = {"beta.ilga.gov": "ilga.gov"}
# Link schemes a legislature has retired, rewritten to the scheme that serves the same document.
# Ohio's export names its 135th General Assembly's PDFs by the retired solarapi/v1 path, which
# answers 404 for every one; api/v2 serves each version as <number>_<code>/pdf/ (or html/).
SCHEME_MOVES = (
    (
        re.compile(
            r"^https://search-prod\.lis\.state\.oh\.us/solarapi/v1/([^/]+)/bills/([a-z]+\d+)/"
            r"([A-Z]+)/(\d+)/\2_\4_\3\?format=pdf$"
        ),
        r"https://search-prod.lis.state.oh.us/api/v2/\1/legislation/\2/\4_\3/pdf/",
    ),
)


def moved(url: str) -> str:
    """`url` where the legislature now serves it, for retired hosts and link schemes."""
    for pattern, replacement in SCHEME_MOVES:
        url = pattern.sub(replacement, url)
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if host not in HOST_MOVES:
        return url
    return urllib.parse.urlunsplit(parts._replace(netloc=HOST_MOVES[host]))


# Characters a URL may carry as they are; anything else, such as the spaces in Oklahoma's and
# West Virginia's links, is percent-encoded. "%" is kept so existing escapes stay single.
_URL_SAFE = ":/?#[]@!$&'()*+,;=%~"


def request_url(url: str) -> str:
    return urllib.parse.quote(url, safe=_URL_SAFE)


def _retry_after(error: urllib.error.HTTPError, default: float) -> float:
    value = error.headers.get("Retry-After", "")
    return float(value) if value.isdigit() else default


def open_url(
    url: str,
    pause: float,
    *,
    interval: float = MIN_INTERVAL_SECONDS,
    headers: Mapping[str, str] | None = None,
) -> HTTPResponse:
    """GET `url` as this project, retrying overload and server errors with backoff `pause`.

    Every attempt waits for the host's turn (lineage.core.pace), at least `interval` after the last
    request any process sent it. `headers` go with the User-Agent, which they cannot replace:
    an API key travels this way, never in a URL that could be logged.
    """
    request = urllib.request.Request(
        request_url(url), headers={**(headers or {}), "User-Agent": USER_AGENT}
    )
    host = urllib.parse.urlsplit(url).hostname or ""
    for attempt in range(_ATTEMPTS):
        try:
            with pace.turn(host, interval):
                response: HTTPResponse = urllib.request.urlopen(
                    request, timeout=_TIMEOUT_SECONDS, context=_TLS
                )
            return response
        except urllib.error.HTTPError as e:
            # Only overload and server faults are worth repeating; a 404 will not change.
            if e.code not in {429, 500, 502, 503, 504} or attempt == _ATTEMPTS - 1:
                raise
            # An HTTPError holds the response open; close it before the next attempt.
            wait = _retry_after(e, pause * 2 ** (attempt + 1))
            e.close()
            time.sleep(wait)
        except urllib.error.URLError:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(pause * 2 ** (attempt + 1))
    raise AssertionError("unreachable")


def stream(response: HTTPResponse, part: Path, *, cap: int, floor: int) -> str:
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


def download(url: str, dest: Path, *, floor: int, pause: float = 1.0) -> tuple[int, str]:
    """Download `url` to `dest` as this project, checking the free space against the size the
    server declares first; return the size and SHA-256. A server that declares no size is
    refused, because the space could not be checked."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open_url(url, pause) as response:
            length = response.headers.get("Content-Length")
            if length is None:
                raise FetchError(f"{url} declares no Content-Length")
            size = int(length)
            require_free_space(dest.parent, needed=size, floor=floor)
            sha256 = stream(response, part, cap=size, floor=floor)
        part.rename(dest)
    finally:
        part.unlink(missing_ok=True)
    return size, sha256
