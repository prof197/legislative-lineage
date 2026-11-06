"""Each host's robots.txt rules, read for this project's own User-Agent.

The text layer fetches bill versions from the legislatures' own sites, so it asks each host
first. The rules are read as RFC 9309 describes, with one stricter reading: a robots.txt that
answers 401 or 403 bars the whole host, as Python's parser treats it. Any other 4xx means the
host publishes no rules. A server error or no answer bars the host until a later run reads its
rules. See DESIGN.md#dr-6.
"""

import hashlib
import urllib.error
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.robotparser import RobotFileParser

from lineage.fetch import USER_AGENT, open_url

# RFC 9309 asks crawlers to parse at least the first 500 KiB.
_MAX_BYTES = 512 * 1024


@dataclass(frozen=True, slots=True)
class Robots:
    url: str
    status: int | None  # robots.txt's HTTP status; None if the host did not answer
    rule: str  # "parsed", "no rules", "disallow all"
    sha256: str | None
    read_at: str
    parser: RobotFileParser | None  # set when rule == "parsed"

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


def read_robots(scheme: str, host: str, *, pause: float = 1.0) -> Robots:
    """Fetch and parse `scheme://host/robots.txt` as this project."""
    url = f"{scheme}://{host}/robots.txt"
    read_at = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        with open_url(url, pause) as response:
            return parse_robots(url, response.status, response.read(_MAX_BYTES), read_at)
    except urllib.error.HTTPError as e:
        return parse_robots(url, e.code, b"", read_at)
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return parse_robots(url, None, b"", read_at)
