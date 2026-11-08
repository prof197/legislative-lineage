"""Fetch planned bill versions from the legislatures, politely, and keep only their text.

Each host gets one worker, and hosts run in parallel. A host sees at most one request a second,
or its robots.txt crawl delay if longer, from every process together (lineage.pace). Every URL
is checked against the host's robots.txt for this project's User-Agent before it is requested;
each read of robots.txt is recorded in `<store>/robots/<host>.jsonl`, and a host whose
robots.txt refuses only for now is left untouched, for a later run. Overload and server errors
are retried with backoff (lineage.fetch.open_url), and a host that keeps failing is stopped for
the run. The response body is extracted in memory and dropped: the store keeps the text, its
SHA-256, the URL fetched and the version's label, one JSON line per document in
`<store>/<host>.jsonl`. A rerun skips what is stored, except network and server failures, which
it tries again. See DESIGN.md#dr-6.
"""

import hashlib
import json
import multiprocessing
import os
import threading
import time
import urllib.error
import zlib
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from lineage.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.extract import ExtractError, extract
from lineage.fetch import moved, open_url
from lineage.pace import MIN_INTERVAL_SECONDS
from lineage.robots import Robots, read_robots

# A bill's PDF is rarely over a few megabytes; an appropriations act can pass 30.
MAX_BODY_BYTES = 64 << 20
# Consecutive failures after which a host is left alone for the rest of the run.
STOP_AFTER_FAILURES = 20
_RETRIED = frozenset({"network_error", "server_error"})
_PROBE = b"<p>Section 1. The extraction workers are running.</p>"


class FetchInWorkerError(Exception):
    """fetch_all was called inside a worker process, as a spawned worker re-running a script
    with no main guard would: each such worker would fetch every host again."""


class ExtractionWorkersError(Exception):
    """The extraction workers, not the document, failed: the run stops instead of recording
    every document as an extraction error, which a rerun would not retry."""


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    url: str
    format: str  # "html" or "pdf": the link's format as the export lists it
    note: str  # the version's label
    date: str
    layout: bool = False  # read a PDF by word position (lineage.extract.pdf_text)


@dataclass(frozen=True, slots=True)
class Fetched:
    id: str
    url: str
    fetched_url: str
    format: str
    note: str
    date: str
    outcome: str  # ok, not_usable, robots, too_large, http_error, server_error, network_error,
    # extract_error; not_in_bulk (lineage.ca_bulk)
    status: int | None  # the response's HTTP status; for outcome robots, robots.txt's
    error: str | None
    body_bytes: int
    text: str | None
    sha256: str | None
    seconds: float
    fetched_at: str
    layout_text: str | None = None  # a PDF read by word position too, when asked (quality test)


def host_of(url: str) -> str:
    return (urlsplit(moved(url)).hostname or "").lower()


def _decoded(body: bytes, encoding: str) -> bytes | None:
    """The body with its Content-Encoding undone, or None past the size limit."""
    if encoding not in {"gzip", "x-gzip", "deflate"}:
        return body
    # 16 + MAX_WBITS reads a gzip wrapper, MAX_WBITS a zlib one; max_length stops a bomb.
    wbits = 16 + zlib.MAX_WBITS if "gzip" in encoding else zlib.MAX_WBITS
    inflater = zlib.decompressobj(wbits)
    out = inflater.decompress(body, MAX_BODY_BYTES + 1)
    return None if len(out) > MAX_BODY_BYTES or inflater.unconsumed_tail else out


def _read(response: HTTPResponse) -> bytes | None:
    length = response.headers.get("Content-Length", "")
    if length.isdigit() and int(length) > MAX_BODY_BYTES:
        return None
    body = response.read(MAX_BODY_BYTES + 1)
    if len(body) > MAX_BODY_BYTES:
        return None
    # Some servers compress a response the request never said it could take (South Dakota).
    encoding = response.headers.get("Content-Encoding", "").strip().lower()
    return _decoded(body, encoding)


def fetch_one(
    job: Job,
    robots: Robots,
    *,
    opener: Callable[[str, float], HTTPResponse] = open_url,
    layout_too: bool = False,
    container: str | None = None,
    extractor: Callable[..., str] = extract,
    check: Callable[[str], bool] | None = None,
) -> Fetched:
    """Fetch one job and extract its text. `container` is the host's content element;
    `extractor` runs the extraction, in a worker process during a full run; text that fails
    `check` (the usable-text bar) is kept but marked not_usable."""
    url = moved(job.url)
    started = time.monotonic()
    at = datetime.now(UTC).isoformat(timespec="seconds")

    def result(
        outcome: str, *, status: int | None = None, error: str | None = None, size: int = 0,
        text: str | None = None, layout: str | None = None,
    ) -> Fetched:  # fmt: skip
        digest = hashlib.sha256(text.encode()).hexdigest() if text is not None else None
        return Fetched(
            job.id, job.url, url, job.format, job.note, job.date, outcome, status, error, size,
            text, digest, round(time.monotonic() - started, 3), at, layout,
        )  # fmt: skip

    if not robots.allows(url):
        return result("robots", status=robots.status, error=f"{robots.url}: {robots.rule}")
    try:
        with opener(url, MIN_INTERVAL_SECONDS * 5) as response:
            body = _read(response)
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        e.close()
        kind = "server_error" if e.code >= 500 or e.code == 429 else "http_error"
        return result(kind, status=e.code, error=str(e.reason))
    except (urllib.error.URLError, TimeoutError, ConnectionError, HTTPException) as e:
        return result("network_error", error=f"{type(e).__name__}: {e}")
    if body is None:
        return result("too_large", status=status, error=f"over {MAX_BODY_BYTES} bytes")
    try:
        text = extractor(body, content_type, layout=job.layout, container=container)
    except ExtractError as e:
        return result("extract_error", status=status, error=str(e), size=len(body))
    except ExtractionWorkersError:
        raise
    except Exception as e:
        error = f"extractor failed: {type(e).__name__}: {e}"
        return result("extract_error", status=status, error=error, size=len(body))
    layout = None
    if layout_too and body.startswith(b"%PDF"):
        try:
            layout = extractor(body, content_type, layout=True)
        except Exception:
            layout = None
    outcome = "ok" if check is None or check(text) else "not_usable"
    return result(outcome, status=status, size=len(body), text=text, layout=layout)


def stored(path: Path) -> set[tuple[str, str, str]]:
    """(id, format, address requested) already in a host's store and not worth retrying.

    The address is the one the request went to, so a link that has since moved (lineage.fetch,
    moved) is fetched again from where it now leads."""
    done: set[tuple[str, str, str]] = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if row["outcome"] not in _RETRIED:
                    done.add((row["id"], row["format"], row["fetched_url"]))
    return done


def run_host(
    host: str,
    jobs: list[Job],
    store_dir: Path,
    *,
    read: Callable[[str, str], Robots] = read_robots,
    fetch: Callable[[Job, Robots, float], Fetched] | None = None,
    layout_too: bool = False,
    container: str | None = None,
    extractor: Callable[..., str] = extract,
    check: Callable[[str], bool] | None = None,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    stop: threading.Event | None = None,
) -> Counter[str]:
    """Fetch one host's jobs in order, appending each result to the host's store. `fetch`
    replaces the request in tests; it is given the host's interval."""
    path = store_dir / f"{host}.jsonl"
    done = stored(path)
    # A changed plan (another version, another link) is fetched again.
    todo = [j for j in jobs if (j.id, j.format, moved(j.url)) not in done]
    outcomes: Counter[str] = Counter(skipped=len(jobs) - len(todo))
    if not todo:
        return outcomes
    robots = read(urlsplit(moved(todo[0].url)).scheme or "https", host)
    log = store_dir / "robots" / f"{host}.jsonl"
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(robots.record()) + "\n")
    if robots.temporary:
        # Nothing is stored, so a later run asks again; a deliberate refusal is stored below.
        outcomes["robots_unavailable"] += len(todo)
        return outcomes
    interval = max(MIN_INTERVAL_SECONDS, robots.crawl_delay() or 0.0)

    def opener(url: str, pause: float) -> HTTPResponse:
        return open_url(url, pause, interval=interval)

    failures = 0
    with path.open("a") as out:
        for job in todo:
            if stop is not None and stop.is_set():
                break
            fetched = (
                fetch(job, robots, interval)
                if fetch
                else fetch_one(
                    job,
                    robots,
                    opener=opener,
                    layout_too=layout_too,
                    container=container,
                    extractor=extractor,
                    check=check,
                )
            )
            line = json.dumps(asdict(fetched), ensure_ascii=False) + "\n"
            require_free_space(store_dir, needed=len(line.encode()), floor=floor_bytes)
            out.write(line)
            out.flush()
            outcomes[fetched.outcome] += 1
            failures = failures + 1 if fetched.outcome in _RETRIED else 0
            if failures >= STOP_AFTER_FAILURES:
                outcomes["host_stopped"] += 1
                break
    return outcomes


def fetch_all(
    jobs: Iterable[Job],
    store_dir: Path,
    *,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    layout_too: bool = False,
    containers: dict[str, str] | None = None,
    check: Callable[[str], bool] | None = None,
    processes: int | None = None,
) -> dict[str, Counter[str]]:
    """Run every host's jobs, hosts in parallel; return the outcome counts per host.

    Hosts wait on the network, so each gets a thread. Extraction is CPU-bound pure Python,
    which threads would run one at a time, so it goes to a pool of worker processes.
    """
    if multiprocessing.parent_process() is not None:
        raise FetchInWorkerError(
            "fetch_all ran in a worker process; put the calling script under "
            'if __name__ == "__main__":'
        )
    by_host: dict[str, list[Job]] = {}
    for job in jobs:
        by_host.setdefault(host_of(job.url), []).append(job)
    store_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    # The slowest hosts (most jobs, longest crawl delay) start first.
    hosts = sorted(by_host, key=lambda h: -len(by_host[h]))
    workers = processes or max(1, (os.cpu_count() or 2) - 2)
    with (
        ProcessPoolExecutor(max_workers=workers) as extractors,
        ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool,
    ):

        def extractor(body: bytes, content_type: str, **options: Any) -> str:
            try:
                return extractors.submit(extract, body, content_type, **options).result()
            except BrokenProcessPool as e:
                raise ExtractionWorkersError(str(e)) from e

        # Before any request: workers that cannot start (a script run without a main guard,
        # under spawn) would otherwise fail every document after its body was fetched.
        try:
            probe = extractor(_PROBE, "text/html")
        except Exception as e:
            raise ExtractionWorkersError(f"extraction workers did not start: {e}") from e
        if "extraction workers are running" not in probe:
            raise ExtractionWorkersError(f"extraction workers returned {probe!r}")

        futures = {
            h: pool.submit(
                run_host,
                h,
                by_host[h],
                store_dir,
                floor_bytes=floor_bytes,
                stop=stop,
                layout_too=layout_too,
                container=(containers or {}).get(h),
                extractor=extractor,
                check=check,
            )
            for h in hosts
        }
        try:
            return {h: f.result() for h, f in futures.items()}
        except BaseException:
            stop.set()
            raise
