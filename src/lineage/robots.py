"""Each host's robots.txt rules, read for this project's own User-Agent.

The text layer fetches bill versions from the legislatures' own sites, and the collectors model
documents from their libraries, so each asks a host first. The rules are read as RFC 9309
describes: the group naming this project's product token, or else the "*" group, with groups for
the same agent combined; "*" and "$" in a rule's path; the longest matching rule decides, and an
Allow wins a tie. Python's `urllib.robotparser` applies the first rule that fits and reads "*"
literally, so it would fetch a path a wildcard rule bars; it is not used. Crawl-delay is not in
the RFC; it is honored from the chosen group, or from before any group, where some generators
write it. One stricter reading: a robots.txt that answers 401 or 403 bars the whole host and is
never asked again in the run. Any other 4xx means the host publishes no rules. An overload
(429), a server error or no answer is temporary: robots.txt is asked again after a backoff, and
a host still refusing is left for a later run. See DESIGN.md#dr-6.
"""

import hashlib
import re
import time
import urllib.error
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from lineage.fetch import USER_AGENT, open_url
from lineage.pace import MIN_INTERVAL_SECONDS

# RFC 9309 asks crawlers to parse at least the first 500 KiB.
_MAX_BYTES = 512 * 1024
# Waits before asking a temporarily refusing robots.txt again, after open_url's own retries.
RETRY_SECONDS = (60.0, 300.0)


# The product token a group's User-agent line names: this project's, before its version.
PRODUCT_TOKEN = USER_AGENT.partition("/")[0].lower()


@dataclass(frozen=True, slots=True)
class Rule:
    allow: bool
    path: str  # as written, "*" and a final "$" included
    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class Rules:
    """The rules of the group that applies to this project."""

    rules: tuple[Rule, ...]
    crawl_delay: float | None

    def allows(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        if parts.path == "/robots.txt":
            return True
        best: Rule | None = None
        for rule in self.rules:
            if not rule.pattern.match(target):
                continue
            longer = best is None or len(rule.path) > len(best.path)
            tie_to_allow = best is not None and len(rule.path) == len(best.path) and rule.allow
            if longer or tie_to_allow:
                best = rule
        return best is None or best.allow


def _pattern(path: str) -> re.Pattern[str]:
    anchored = path.endswith("$")
    body = path[:-1] if anchored else path
    # Paths compare with percent-escapes decoded, so "/a%2Db" and "/a-b" are one path.
    regex = ".*".join(re.escape(urllib.parse.unquote(part)) for part in body.split("*"))
    return re.compile(regex + ("$" if anchored else ""))


def _delay(value: str) -> float | None:
    try:
        delay = float(value)
    except ValueError:
        return None
    return delay if delay >= 0 else None


def parse_rules(text: str) -> Rules:
    """The rules for this project in a robots.txt (RFC 9309, plus Crawl-delay)."""
    groups: list[tuple[list[str], list[Rule], list[float]]] = []
    before_any: float | None = None
    agents: list[str] | None = None  # the open group's agents while its User-agent lines run
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if agents is None:
                agents = []
                groups.append((agents, [], []))
            agents.append(value.lower())
            continue
        if key in {"allow", "disallow"} and groups:
            agents = None
            if value:  # an empty Disallow allows everything, as no rule does
                groups[-1][1].append(Rule(key == "allow", value, _pattern(value)))
        elif key == "crawl-delay":
            delay = _delay(value)
            if not groups:
                before_any = delay if delay is not None else before_any
            else:
                agents = None
                if delay is not None:
                    groups[-1][2].append(delay)
    mine = [g for g in groups if PRODUCT_TOKEN in g[0]]
    chosen = mine or [g for g in groups if "*" in g[0]]
    rules = tuple(r for g in chosen for r in g[1])
    delays = [d for g in chosen for d in g[2]]
    return Rules(rules, max(delays) if delays else before_any)


@dataclass(frozen=True, slots=True)
class Robots:
    url: str
    status: int | None  # robots.txt's HTTP status; None if the host did not answer
    rule: str  # "parsed", "no rules", "disallow all"
    sha256: str | None
    read_at: str
    rules: Rules | None  # set when rule == "parsed"
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
        if self.rules is None:
            return False
        return self.rules.allows(url)

    def crawl_delay(self) -> float | None:
        return None if self.rules is None else self.rules.crawl_delay


def parse_robots(url: str, status: int | None, body: bytes, read_at: str) -> Robots:
    if status is None or status >= 500 or status in {401, 403, 429}:
        return Robots(url, status, "disallow all", None, read_at, None)
    if status >= 400:
        return Robots(url, status, "no rules", None, read_at, None)
    rules = parse_rules(body[:_MAX_BYTES].decode("utf-8", errors="replace"))
    return Robots(url, status, "parsed", hashlib.sha256(body).hexdigest(), read_at, rules)


def _read_once(url: str, pause: float, interval: float) -> Robots:
    read_at = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        with open_url(url, pause, interval=interval) as response:
            return parse_robots(url, response.status, response.read(_MAX_BYTES), read_at)
    except urllib.error.HTTPError as e:
        e.close()
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
