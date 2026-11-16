"""What every collector shares: robots.txt, paced requests, a document's provenance, storing it,
and the coverage report.

A collector finds documents and hands each to a `Collection`, which settles it one way: stored
(new, or with a changed text), unchanged, blocked by robots.txt, or failed. `found` is counted
from those outcomes, so the report accounts for every document by construction. robots.txt is
read through `lineage.robots`, once per host per run, and every request goes through
`lineage.fetch.open_url`, which waits for the host's turn across every process
(`lineage.pace`), at least the host's crawl delay apart. A URL robots.txt disallows is reported,
never fetched; one whose robots.txt could not be read is a failure to try again in a later run,
not a block. See DESIGN.md#model-documents.
"""

import urllib.error
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from lineage import robots as robots_txt
from lineage.collect import CollectContext, CollectFailure, CollectReport
from lineage.disk import require_free_space
from lineage.fetch import open_url
from lineage.modeldocs import ModelDoc, Source, sha256, store, withdraw
from lineage.pace import MIN_INTERVAL_SECONDS
from lineage.robots import Robots

RobotsStatus = Literal["allowed", "no_robots_txt"]
# A model act is a few hundred KiB; anything far larger is not the document asked for.
MAX_BYTES = 64 << 20
# Backoff for overloaded or failing hosts, as the text layer's fetches use.
PAUSE_SECONDS = 1.0


class CollectFetchError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Fetched:
    url: str  # as asked for; a redirect's target is not recorded
    content_type: str
    body: bytes


# (url, interval) -> the response. `fetch_url` in a run, recorded responses in tests.
Fetch = Callable[[str, float], Fetched]
# (scheme, host) -> the host's rules. `lineage.robots.read_robots` in a run.
ReadRobots = Callable[[str, str], Robots]


def fetch_url(url: str, interval: float) -> Fetched:
    """GET `url` as this project, through `open_url` and the host's pacing."""
    try:
        with open_url(url, PAUSE_SECONDS, interval=interval) as response:
            body = response.read(MAX_BYTES + 1)
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        e.close()
        raise CollectFetchError(f"HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise CollectFetchError(f"no answer: {e}") from e
    if len(body) > MAX_BYTES:
        raise CollectFetchError(f"more than {MAX_BYTES} bytes")
    return Fetched(url, content_type, body)


def read_robots(scheme: str, host: str) -> Robots:
    return robots_txt.read_robots(scheme, host)


def slugify(name: str) -> str:
    """A document's slug from its name: lowercase words joined by hyphens."""
    words = "".join(c if c.isalnum() else " " for c in name.lower()).split()
    if not words:
        raise ValueError(f"no slug in {name!r}")
    return "-".join(words)


def provenance(
    source: Source,
    *,
    slug: str,
    title: str,
    url: str,
    raw: bytes,
    media_type: str,
    text: str,
    robots: RobotsStatus,
    collected_on: date,
    published: date | None = None,
    co_publishers: tuple[str, ...] = (),
) -> ModelDoc:
    """The record of one collected document: where it came from, what robots.txt said, the
    organizations it names as publishing it with the source's, and the digests of its bytes as
    fetched and of the cleaned text."""
    return ModelDoc(
        id=f"{source.key}/{slug}",
        source=source.key,
        title=title,
        url=url,
        published=published,
        collected_on=collected_on,
        robots=robots,
        leaked=source.leaked,
        raw_sha256=sha256(raw),
        media_type=media_type,
        text_sha256=sha256(text.encode()),
        co_publishers=co_publishers,
    )


@dataclass(slots=True)
class Collection:
    """One run of one source: robots.txt per host, and the outcome of every document found."""

    source: Source
    context: CollectContext
    fetch: Fetch = fetch_url
    robots_reader: ReadRobots = read_robots
    collected: int = 0
    unchanged: int = 0
    blocked: list[str] = field(default_factory=list[str])
    failed: list[CollectFailure] = field(default_factory=list[CollectFailure])
    withdrawn: list[str] = field(default_factory=list[str])
    _robots: dict[tuple[str, str], Robots] = field(default_factory=dict[tuple[str, str], Robots])

    def robots_for(self, url: str) -> Robots:
        parts = urllib.parse.urlsplit(url)
        key = (parts.scheme, (parts.hostname or "").lower())
        if key not in self._robots:
            self._robots[key] = self.robots_reader(*key)
        return self._robots[key]

    def permission(self, url: str) -> RobotsStatus | Literal["blocked", "unread"]:
        """What robots.txt says about `url` for this project."""
        robots = self.robots_for(url)
        if robots.temporary:
            return "unread"
        if not robots.allows(url):
            return "blocked"
        return "no_robots_txt" if robots.rule == "no rules" else "allowed"

    def get(self, url: str) -> Fetched:
        """Fetch a URL robots.txt allows, at least its host's crawl delay after the last request.
        Raises CollectFetchError for a URL it may not or could not fetch; the caller decides
        which document that fails."""
        permission = self.permission(url)
        if permission == "blocked":
            raise CollectFetchError("robots.txt disallows it")
        if permission == "unread":
            raise CollectFetchError("robots.txt could not be read; try again in a later run")
        delay = self.robots_for(url).crawl_delay() or 0.0
        return self.fetch(url, max(MIN_INTERVAL_SECONDS, delay))

    def block(self, url: str) -> None:
        self.blocked.append(url)

    def fail(self, url: str, reason: str) -> None:
        self.failed.append(CollectFailure(url=url, reason=reason))

    def check(self, url: str) -> RobotsStatus | None:
        """robots.txt's word on a document's URL; a URL it disallows, or a host whose robots.txt
        could not be read, settles the document as blocked or failed and gives None."""
        permission = self.permission(url)
        if permission == "blocked":
            self.block(url)
            return None
        if permission == "unread":
            self.fail(url, "robots.txt could not be read; try again in a later run")
            return None
        return permission

    def keep(self, doc: ModelDoc, *, raw: bytes, text: str) -> bool:
        """Store a document through `lineage.modeldocs.store`; returns whether anything changed.
        The disk is checked first against the bytes this could write."""
        require_free_space(
            self.context.root,
            needed=len(raw) + len(text.encode()) + (1 << 20),
            floor=self.context.floor_bytes,
        )
        changed = store(self.source, doc, raw=raw, text=text, root=self.context.root)
        if changed:
            self.collected += 1
        else:
            self.unchanged += 1
        return changed

    def out_of_scope(self, slug: str) -> None:
        """A document this run read and found not to be the source's model state legislation:
        not counted as found, and withdrawn from the current list if an earlier run stored it.
        Only a collector that has read the document says so; a document a run could not read
        is failed, never withdrawn."""
        doc_id = f"{self.source.key}/{slug}"
        if withdraw(self.source, doc_id, on=self.context.on, root=self.context.root):
            self.withdrawn.append(doc_id)

    def report(self) -> CollectReport:
        return CollectReport(
            source=self.source.key,
            run_on=self.context.on,
            found=self.collected + self.unchanged + len(self.blocked) + len(self.failed),
            collected=self.collected,
            unchanged=self.unchanged,
            blocked=sorted(self.blocked),
            failed=sorted(self.failed, key=lambda f: f.url),
            withdrawn=sorted(self.withdrawn),
        )
