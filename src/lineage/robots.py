"""Each host's robots.txt rules, read for this project's own User-Agent.

The text layer fetches bill versions from the legislatures' own sites, so it asks each host
first. The rules are read as RFC 9309 describes, with one stricter reading: a robots.txt that
answers 401 or 403 bars the whole host, as Python's parser treats it, and is never asked again
in the run. Any other 4xx means the host publishes no rules. An overload (429), a server error
or no answer is temporary: robots.txt is asked again after a backoff, and a host still refusing
is left for a later run. See DESIGN.md#dr-6.
"""

import hashlib
import time
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from urllib.robotparser import RobotFileParser

from lineage.fetch import USER_AGENT, open_url
from lineage.pace import MIN_INTERVAL_SECONDS

# RFC 9309 asks crawlers to parse at least the first 500 KiB.
_MAX_BYTES = 512 * 1024
# Waits before asking a temporarily refusing robots.txt again, after open_url's own retries.
RETRY_SECONDS = (60.0, 300.0)


@dataclass(frozen=True, slots=True)
class Robots:
    url: str
    status: int | None  # robots.txt's HTTP status; None if the host did not answer
    rule: str  # "parsed", "no rules", "disallow all"
    sha256: str | None
    read_at: str
    parser: RobotFileParser | None  # set when rule == "parsed"
    attempts: int = 1

    @property
    def temporary(self) -> bool:
        """The host said "later", not "no": overloaded, failing or not answering."""
        return self.status is None or self.status >= 500 or self.status == 429

    def record(self) -> dict[str, object]:
        return {
            "url": self.url, "status": self.status, "rule": self.rule, "sha256": self.sha256,
            "read_at": self.read_at, "attempts": self.attempts, "crawl_delay": self.crawl_delay(),
        }  # fmt: skip

    def allows(self, url: str) -> bool:
        if self.rule == "no rules":
            return True
        if self.parser is None:
            return False
        return self.parser.can_fetch(USER_AGENT, url)

    def crawl_delay(self) -> float | None:
        if self.parser is None:
            return None
        delay = self.parser.crawl_delay(USER_AGENT)
        return float(delay) if delay is not None else None


def parse_robots(url: str, status: int | None, body: bytes, read_at: str) -> Robots:
    if status is None or status >= 500 or status in {401, 403, 429}:
        return Robots(url, status, "disallow all", None, read_at, None)
    if status >= 400:
        return Robots(url, status, "no rules", None, read_at, None)
    parser = RobotFileParser(url)
    parser.parse(body[:_MAX_BYTES].decode("utf-8", errors="replace").splitlines())
    return Robots(url, status, "parsed", hashlib.sha256(body).hexdigest(), read_at, parser)


def _read_once(url: str, pause: float, interval: float) -> Robots:
    read_at = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        with open_url(url, pause, interval=interval) as response:
            return parse_robots(url, response.status, response.read(_MAX_BYTES), read_at)
    except urllib.error.HTTPError as e:
        return parse_robots(url, e.code, b"", read_at)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return parse_robots(url, None, b"", read_at)


def read_robots(
    scheme: str,
    host: str,
    *,
    pause: float = 1.0,
    interval: float = MIN_INTERVAL_SECONDS,
    retry_seconds: tuple[float, ...] = RETRY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Robots:
    """Fetch and parse `scheme://host/robots.txt` as this project, asking again after each of
    `retry_seconds` while the refusal is temporary."""
    url = f"{scheme}://{host}/robots.txt"
    robots = _read_once(url, pause, interval)
    for attempt, wait in enumerate(retry_seconds, start=2):
        if not robots.temporary:
            break
        sleep(wait)
        robots = replace(_read_once(url, pause, interval), attempts=attempt)
    return robots
