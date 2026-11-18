"""One request at a time per host, spaced by its interval, across every thread and process.

Pacing kept inside a process paces only that process. On 2026-11-06, spawned extraction workers
each re-ran a fetch with its own pacing, which multiplied every host's rate by their number.
Here the time of each host's last request is kept in a file under an exclusive `flock`. Every
request this project makes passes through `turn`: a fetch run's documents, the inventory's
robots.txt reads, archive downloads and every retry. So every process the user runs waits for
the same host's turn. The longest interval a host has been given, such as its crawl delay, is
kept with it, so a caller that asks for a shorter one still waits as long. See DESIGN.md#dr-6.
"""

import fcntl
import json
import os
import re
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

MIN_INTERVAL_SECONDS = 1.0
# Shared by every checkout and working directory of the user, whatever TMPDIR says;
# LINEAGE_PACE_DIR overrides it.
DIRECTORY = Path.home() / ".cache" / "legislative-lineage" / "pace"


def directory() -> Path:
    return Path(os.environ.get("LINEAGE_PACE_DIR") or DIRECTORY)


def _path(host: str, where: Path) -> Path:
    name = re.sub(r"[^a-z0-9.-]", "_", host.lower()) or "_"
    return where / f"{name}.json"


@contextmanager
def turn(
    host: str,
    interval: float = MIN_INTERVAL_SECONDS,
    *,
    where: Path | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> Generator[None]:
    """Wait for `host`'s turn, then hold it while the request is sent.

    The request's end is recorded, and the next request to the host waits until the interval
    has passed. A wall clock is used because a monotonic clock is not comparable between
    processes.
    """
    where = where or directory()
    where.mkdir(parents=True, exist_ok=True)
    fd = os.open(_path(host, where), os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        raw = f.read()
        state: dict[str, float] = json.loads(raw) if raw.strip() else {}
        spacing = max(interval, state.get("interval", 0.0))
        wait = state.get("last", 0.0) + spacing - clock()
        if wait > 0:
            sleep(wait)
        try:
            yield
        finally:
            f.seek(0)
            f.truncate()
            f.write(json.dumps({"last": clock(), "interval": spacing}))
            f.flush()
