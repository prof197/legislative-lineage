"""What collectors share (DESIGN.md#model-documents): robots.txt per host, paced requests through
open_url, provenance, storing, and a report that accounts for every document found."""

import urllib.error
from datetime import date
from email.message import Message
from io import BytesIO
from pathlib import Path

import pytest

from lineage import collect_base
from lineage.collect import CollectContext
from lineage.collect_base import CollectFetchError, Collection, Fetched, provenance, slugify
from lineage.core.disk import InsufficientDiskSpaceError
from lineage.core.robots import Robots, parse_robots
from lineage.modeldocs import ModelDoc, Source, load_text, read_docs, read_versions

SOURCE = Source(
    key="example-library",
    organization="Example Council",
    descriptor="nonprofit membership organization, 501(c)(3)",
    library_url="https://example.org/models",
    terms_note="viewing and copying with attribution",
    terms_read_on=date(2026, 11, 15),
)
DAY = date(2026, 11, 15)
READ_AT = "2026-11-15T00:00:00+00:00"
RULES = b"User-agent: *\nCrawl-delay: 3\nDisallow: /private/\n"


def _robots(status: int | None, body: bytes = b"") -> Robots:
    return parse_robots("https://example.org/robots.txt", status, body, READ_AT)


def _collection(tmp_path: Path, robots: Robots, fetched: list[tuple[str, float]]) -> Collection:
    def fetch(url: str, interval: float) -> Fetched:
        fetched.append((url, interval))
        return Fetched(url, "text/plain", b"model text")

    reads: list[str] = []

    def reader(scheme: str, host: str) -> Robots:
        reads.append(host)
        assert len(reads) == 1, "robots.txt is read once per host per run"
        return robots

    return Collection(
        SOURCE, CollectContext(root=tmp_path, on=DAY), fetch=fetch, robots_reader=reader
    )


def _doc(text: str, robots: collect_base.RobotsStatus = "allowed") -> tuple[bytes, str, ModelDoc]:
    raw = f"raw {text}".encode()
    doc = provenance(
        SOURCE,
        slug="lantern-act",
        title="Lantern Act",
        url="https://example.org/models/lantern",
        raw=raw,
        media_type="text/plain",
        text=text,
        robots=robots,
        collected_on=DAY,
    )
    return raw, text, doc


def test_robots_txt_decides_each_url_and_is_read_once(tmp_path: Path) -> None:
    fetched: list[tuple[str, float]] = []
    c = _collection(tmp_path, _robots(200, RULES), fetched)

    assert c.check("https://example.org/models/lantern") == "allowed"
    assert c.check("https://example.org/private/lantern") is None
    assert c.get("https://example.org/models/lantern").body == b"model text"
    with pytest.raises(CollectFetchError, match="disallows"):
        c.get("https://example.org/private/other")

    # The crawl delay spaces requests, never less than the project's own interval.
    assert fetched == [("https://example.org/models/lantern", 3.0)]
    report = c.report()
    assert report.blocked == ["https://example.org/private/lantern"]
    assert report.found == 1


@pytest.mark.parametrize(
    ("robots", "outcome"),
    [
        (_robots(404), "no_robots_txt"),
        (_robots(403), "blocked"),
        (_robots(503), "failed"),
        (_robots(None), "failed"),
    ],
)
def test_a_host_without_rules_a_barred_host_and_a_silent_one(
    tmp_path: Path, robots: Robots, outcome: str
) -> None:
    c = _collection(tmp_path, robots, [])
    status = c.check("https://example.org/models/lantern")
    report = c.report()

    if outcome == "no_robots_txt":
        assert status == "no_robots_txt"
        assert report.found == 0
    elif outcome == "blocked":
        assert status is None
        assert report.blocked == ["https://example.org/models/lantern"]
    else:
        # A robots.txt that could not be read is not a refusal: it is tried again next run.
        assert status is None
        assert report.blocked == []
        assert "try again" in report.failed[0].reason


def test_keeping_a_document_twice_changes_nothing_and_a_new_text_is_a_version(
    tmp_path: Path,
) -> None:
    c = _collection(tmp_path, _robots(404), [])
    raw, text, doc = _doc("Section 1. Lanterns.\n")
    assert c.keep(doc, raw=raw, text=text)
    assert not c.keep(doc, raw=raw, text=text)
    raw2, text2, doc2 = _doc("Section 1. Lanterns, lit.\n")
    assert c.keep(doc2, raw=raw2, text=text2)

    report = c.report()
    assert (report.found, report.collected, report.unchanged) == (3, 2, 1)
    [current] = read_docs(SOURCE, tmp_path)
    assert current.text_sha256 == doc2.text_sha256
    assert load_text(current, tmp_path) == text2
    assert [v.replaced_on for v in read_versions(SOURCE, tmp_path)] == [DAY, None]


def test_provenance_records_the_digests_and_robots(tmp_path: Path) -> None:
    _, _, doc = _doc("text\n", robots="no_robots_txt")
    assert doc.id == "example-library/lantern-act"
    assert doc.robots == "no_robots_txt"
    assert doc.leaked is False
    assert slugify("Health-Care Decisions Act (2026)") == "health-care-decisions-act-2026"
    assert doc.co_publishers == ()
    joint = provenance(
        SOURCE,
        slug="joint-act",
        title="Joint Act",
        url="https://example.org/models/joint",
        raw=b"raw",
        media_type="text/plain",
        text="text\n",
        robots="allowed",
        collected_on=DAY,
        co_publishers=("Second Institute",),
    )
    assert joint.co_publishers == ("Second Institute",)


def test_the_disk_floor_is_checked_before_storing(tmp_path: Path) -> None:
    c = Collection(
        SOURCE,
        CollectContext(root=tmp_path, on=DAY, floor_bytes=1 << 62),
        robots_reader=lambda scheme, host: _robots(404),
    )
    raw, text, doc = _doc("text\n")
    with pytest.raises(InsufficientDiskSpaceError):
        c.keep(doc, raw=raw, text=text)
    assert read_docs(SOURCE, tmp_path) == []


class _Response(BytesIO):
    def __init__(self, body: bytes, content_type: str) -> None:
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> "_Response":
        return self


def test_fetch_url_goes_through_open_url_and_its_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float, float]] = []

    def open_url(url: str, pause: float, *, interval: float) -> _Response:
        calls.append((url, pause, interval))
        if url.endswith("missing"):
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        return _Response(b"body", "application/pdf")

    monkeypatch.setattr(collect_base, "open_url", open_url)
    got = collect_base.fetch_url("https://example.org/a.pdf", 2.0)
    assert (got.body, got.content_type) == (b"body", "application/pdf")
    assert calls == [("https://example.org/a.pdf", collect_base.PAUSE_SECONDS, 2.0)]
    with pytest.raises(CollectFetchError, match="HTTP 404"):
        collect_base.fetch_url("https://example.org/missing", 1.0)
    monkeypatch.setattr(collect_base, "MAX_BYTES", 3)
    with pytest.raises(CollectFetchError, match="more than 3 bytes"):
        collect_base.fetch_url("https://example.org/a.pdf", 1.0)
