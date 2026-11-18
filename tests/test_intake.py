"""The daily intake: API v3 pages into days `load_day` reads, text through the text layer.

The API is never called: `FakeApi` answers from constructed bills (tests/fixtures/intake/api.json)
the way the API's own code does (updated_at at or after `updated_since`, sorted, paged), and a
local server stands in for it where `lineage.core.http.open_url`'s pacing and backoff are tested.
"""

import json
import math
import threading
import time
import urllib.error
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from email.message import Message
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.core import db, pace
from lineage.core.http import open_url, request_url
from lineage.core.robots import parse_robots
from lineage.intake import (
    IntakeError,
    Tier,
    january_estimate,
    pages_for,
    requests_per_day,
    run_intake,
)
from lineage.openstates_api import INCLUDE, PER_PAGE, Bill, bills_url, bulk_record
from lineage.text import queries as text_queries
from lineage.text.fetch import Job, fetch_one, host_of, run_host
from lineage.text.rules import load_rules, usable
from lineage.updates import fold, load_day, record_changes, text_hash, write_archives
from tests.conftest import SESSIONS

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "intake" / "api.json").read_text())
D1, D2, D3 = date(2026, 1, 12), date(2026, 1, 13), date(2026, 1, 14)
START = datetime(2026, 1, 11, tzinfo=UTC)
STATES = ("CA", "DE", "IN", "MI")
KEY = "fixture-key-not-a-real-one"
FAST = Tier(per_minute=6000, per_day=1000)
AT = "2026-01-12T00:00:00+00:00"
DE_HB1, DE_SB2 = "ocd-bill/fixture-de-0001", "ocd-bill/fixture-de-0002"
MI_SB1, MI_HB4001 = "ocd-bill/fixture-mi-0001", "ocd-bill/fixture-mi-0002"
# Bulk records' keys, in the order of a real session archive's records.
ARCHIVE_KEYS = [
    "id", "legislative_session", "jurisdiction_name", "identifier", "title", "chamber",
    "classification", "subject", "abstracts", "other_titles", "actions", "related_bills",
    "sponsors", "documents", "versions", "sources", "votes", "raw_text", "raw_text_url",
]  # fmt: skip


class _Response:
    def __init__(self, body: bytes, content_type: str = "application/json") -> None:
        self.body, self.status, self.headers = body, 200, Message()
        self.headers["Content-Type"] = content_type

    def read(self, n: int = -1) -> bytes:
        return self.body if n < 0 else self.body[:n]

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "fixture", Message(), None)


def answer(bills: dict[str, dict[str, Any]], query: dict[str, list[str]]) -> dict[str, Any]:
    """GET /bills over `bills`, as api-v3 computes it."""
    where = query["jurisdiction"][0]
    since = datetime.fromisoformat(query["updated_since"][0])
    per_page, page = int(query["per_page"][0]), int(query["page"][0])
    matching = sorted(
        (
            b
            for b in bills.values()
            if b["jurisdiction"]["id"] == where and datetime.fromisoformat(b["updated_at"]) >= since
        ),
        key=lambda b: datetime.fromisoformat(b["updated_at"]),
    )
    pages = math.ceil(len(matching) / per_page) or 1
    return {
        "results": matching[(page - 1) * per_page : page * per_page],
        "pagination": {
            "per_page": per_page, "page": page, "max_page": pages, "total_items": len(matching)
        },
    }  # fmt: skip


@dataclass
class FakeApi:
    bills: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    calls: list[tuple[str, dict[str, str], float]] = field(
        default_factory=list[tuple[str, dict[str, str], float]]
    )
    fail: dict[str, list[int]] = field(default_factory=dict[str, list[int]])  # state -> codes

    def advance(self, day: date) -> None:
        for bill in FIXTURE["days"][day.isoformat()]:
            self.bills[bill["id"]] = bill

    def __call__(
        self,
        url: str,
        pause: float,
        *,
        interval: float = 1.0,
        headers: Mapping[str, str] | None = None,
    ) -> HTTPResponse:
        self.calls.append((url, dict(headers or {}), interval))
        query = parse_qs(urlsplit(url).query)
        state = query["jurisdiction"][0].split("state:")[1].split("/")[0].upper()
        if self.fail.get(state):
            raise _http_error(url, self.fail[state].pop(0))
        return cast(HTTPResponse, _Response(json.dumps(answer(self.bills, query)).encode()))


@dataclass
class FakeText:
    """The text layer with its network replaced: run_host and fetch_one, robots.txt read from a
    constructed body, documents served from the fixture. `fail` counts the network errors a
    URL answers before its text."""

    robots: bytes = b"User-agent: *\nDisallow: /private/\n"
    fail: dict[str, int] = field(default_factory=dict[str, int])
    asked: list[list[str]] = field(default_factory=list[list[str]])
    served: list[str] = field(default_factory=list[str])

    def _opener(self, url: str, pause: float) -> HTTPResponse:
        self.served.append(url)
        if self.fail.get(url):
            self.fail[url] -= 1
            raise urllib.error.URLError("fixture network error")
        return cast(HTTPResponse, _Response(FIXTURE["documents"][url].encode(), "text/html"))

    def __call__(self, jobs: list[Job], store: Path) -> object:
        self.asked.append([j.url for j in jobs])
        rules = load_rules()
        by_host: dict[str, list[Job]] = {}
        for job in jobs:
            by_host.setdefault(host_of(job.url), []).append(job)
        for host, host_jobs in sorted(by_host.items()):
            run_host(
                host, host_jobs, store,
                read=lambda scheme, h: parse_robots(f"{scheme}://{h}/robots.txt", 200,
                                                    self.robots, AT),
                fetch=lambda job, robots, interval: fetch_one(
                    job, robots, opener=self._opener, check=lambda t: usable(t, rules)
                ),
                floor_bytes=0,
            )  # fmt: skip
        return None


def run(tmp: Path, as_of: date, api: Any, text: Any, **options: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "opener": api, "key": KEY, "start": START, "states": STATES, "runs": tmp / "runs",
        "days": tmp / "days", "store": tmp / "store", "reports": tmp / "reports",
        "tier": FAST, "pause": 0.0, "fetch_texts": text, "floor_bytes": 0,
    }  # fmt: skip
    return run_intake(as_of, **(settings | options))


def records(day_dir: Path) -> dict[str, Mapping[str, Any]]:
    return {str(r["id"]): r for rows in load_day(day_dir).records.values() for r in rows}


def files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


# The request and the mapping.


def test_the_request_names_the_documented_parameters_and_keeps_the_key_out_of_the_url() -> None:
    since = datetime(2026, 1, 12, 9, 0, 0, 1, tzinfo=UTC)
    url = bills_url("UT", since, 2)
    query = parse_qs(urlsplit(url).query)

    assert url.startswith("https://v3.openstates.org/bills?")
    assert query == {
        "jurisdiction": ["ocd-jurisdiction/country:us/state:ut/government"],
        "updated_since": ["2026-01-12T09:00:00.000001+00:00"],
        "sort": ["updated_asc"],
        "per_page": ["20"],
        "page": ["2"],
        "include": list(INCLUDE),
    }
    assert {"sponsorships", "actions", "versions"} <= set(INCLUDE)
    # The offset's "+" is escaped, or the server would read a space; open_url keeps escapes.
    assert "%2B00%3A00" in url
    assert request_url(url) == url


def test_a_v3_bill_maps_to_an_archive_record() -> None:
    bill = Bill.model_validate(FIXTURE["days"]["2026-01-13"][0])
    record = bulk_record(bill)

    assert list(record) == ARCHIVE_KEYS
    assert record["legislative_session"] == "153"
    assert record["jurisdiction_name"] == "Delaware"
    assert record["chamber"] == "lower"
    assert record["actions"][1] == {
        "organization__name": "House", "description": "Passed In House", "date": "2026-01-13",
        "classification": ["passage"], "order": 2,
    }  # fmt: skip
    assert record["sponsors"] == [
        {"name": "Fixture Sponsor One", "primary": True, "classification": "primary"}
    ]
    assert record["versions"][1] == {
        "note": "Engrossed", "date": "2026-01-13",
        "links": [{"url": "https://legis.example/de/hb1-eng.html", "media_type": "text/html"}],
    }  # fmt: skip
    assert record["sources"] == [{"url": "https://legis.example/de/hb1"}]
    assert (record["raw_text"], record["raw_text_url"]) == ("", None)


# Days.


def test_days_written_from_the_fixtures_load_and_fold(tmp_path: Path) -> None:
    api, text = FakeApi(), FakeText()
    api.advance(D1)
    first = run(tmp_path, D1, api, text)
    api.advance(D2)
    run(tmp_path, D2, api, text)

    day1, day2 = (
        load_day(tmp_path / "days" / "2026-01-12"),
        load_day(tmp_path / "days" / "2026-01-13"),
    )
    assert day1.as_of == D1
    assert day1.sources == ("openstates_api_v3", "legislature")
    assert sorted(day1.records) == [
        ("CA", "20252026"), ("DE", "153"), ("IN", "2026"), ("MI", "2025-2026")
    ]  # fmt: skip
    one, two = records(tmp_path / "days" / "2026-01-12"), records(tmp_path / "days" / "2026-01-13")
    # Delaware's rule prefers HTML; the text is the extracted bill, in the record's text fields.
    assert one[DE_HB1]["raw_text_url"] == "https://legis.example/de/hb1.html"
    assert "every five years" in one[DE_HB1]["raw_text"]
    assert list(one[DE_HB1]) == ARCHIVE_KEYS
    for none in (DE_SB2, "ocd-bill/fixture-ca-0001", "ocd-bill/fixture-in-0001"):
        assert (one[none]["raw_text"], one[none]["raw_text_url"]) == ("", None)
    assert first["states"]["DE"]["texts"] == {"fetched": 1, "none: robots": 1}
    assert first["states"]["CA"]["texts"] == {"none: bulk download only, not fetched daily": 1}
    assert first["states"]["IN"]["texts"] == {
        "none: documents are served only through an API that needs a registered key": 1
    }
    # Day 2: Delaware's bill is engrossed, Michigan's gains an action, a new bill arrives.
    assert sorted(two) == [DE_HB1, MI_SB1, MI_HB4001]
    assert "every three years" in two[DE_HB1]["raw_text"]
    assert text_hash(two[MI_SB1]) == text_hash(one[MI_SB1])
    assert text.asked[1] == [
        "https://legis.example/de/hb1-eng.html",
        "https://mi.example/hb4001.htm",
    ]

    folded = fold([day1, day2])
    assert {c.bill_id: c.kind for c in record_changes(fold([day1]), day2)} == {
        DE_HB1: "replaced", MI_SB1: "updated", MI_HB4001: "added"
    }  # fmt: skip
    versions = [v for v in folded.versions if v.bill_id == DE_HB1]
    assert [(v.first_seen, v.replaced_on) for v in versions] == [(D1, D2), (D2, None)]


def test_re_running_a_day_changes_nothing_and_sends_no_request(tmp_path: Path) -> None:
    api, text = FakeApi(), FakeText()
    api.advance(D1)
    run(tmp_path, D1, api, text)
    written, calls, asked = files(tmp_path), len(api.calls), len(text.asked)
    # Even with more bills waiting upstream, the written day stands as it was.
    api.advance(D2)

    again = run(tmp_path, D1, api, text)

    assert again == {"command": "uv run lineage intake", "as_of": "2026-01-12", "day": "unchanged"}
    assert files(tmp_path) == written
    assert (len(api.calls), len(text.asked)) == (calls, asked)


def test_a_changed_written_day_is_refused_not_overwritten(tmp_path: Path) -> None:
    api, text = FakeApi(), FakeText()
    api.advance(D1)
    run(tmp_path, D1, api, text)
    path = tmp_path / "days" / "2026-01-12" / "DE_153_bills.json"
    path.write_text("[]")

    with pytest.raises(IntakeError, match="written already, and differently"):
        run(tmp_path, D1, api, text)
    assert path.read_text() == "[]"


def test_a_run_stopped_before_its_day_was_written_resumes_from_its_recorded_pages(
    tmp_path: Path,
) -> None:
    api, text = FakeApi(), FakeText()
    api.advance(D1)

    def crash(jobs: list[Job], store: Path) -> object:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, D1, api, crash)
    assert not (tmp_path / "days").exists()
    calls = len(api.calls)

    report = run(tmp_path, D1, api, text)

    # Every state's list had ended: the pages recorded before the stop are the day's.
    assert len(api.calls) == calls
    assert report["outcomes"] == {"complete": 4}
    assert report["records"] == 5


def test_the_rebuild_reads_the_fetched_text(tmp_path: Path) -> None:
    api, text = FakeApi(), FakeText()
    api.advance(D1)
    run(tmp_path, D1, api, text)
    api.advance(D2)
    run(tmp_path, D2, api, text)
    days = [load_day(tmp_path / "days" / d) for d in ("2026-01-12", "2026-01-13")]
    raw = write_archives(fold(days), tmp_path / "raw")
    sessions = tmp_path / "sessions.csv"
    sessions.write_text(
        "state,session,name,start_date,end_date,carries_over_to\n"
        "CA,20252026,2025-2026 Regular Session,2024-12-02,2026-11-30,\n"
        "DE,153,153rd General Assembly (2025-2026),2025-01-14,2026-06-30,\n"
        "IN,2026,2026 Regular Session,2026-01-09,2026-04-29,\n"
        "MI,2025-2026,2025-2026 Regular Session,2025-01-08,2026-12-31,\n"
    )
    manifest, tables = tmp_path / "manifest.json", tmp_path / "tables"
    runner = CliRunner()
    for args in (
        ["snapshot", "--raw", str(raw), "--manifest", str(manifest)],
        ["ingest", "--raw", str(raw), "--manifest", str(manifest), "--out", str(tables),
         "--sessions", str(sessions), "--min-free-gib", "0"],
    ):  # fmt: skip
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output

    with db.connect_quiet() as con:
        stored = text_queries.stored_texts(
            con, bills_glob=str(tables / "bills" / "*.parquet"), ids=[DE_HB1, MI_HB4001]
        )
    latest = records(tmp_path / "days" / "2026-01-13")
    assert stored == {DE_HB1: latest[DE_HB1]["raw_text"], MI_HB4001: latest[MI_HB4001]["raw_text"]}


# Failures, per state.


def test_a_failing_state_is_reported_and_retried_next_run_from_its_own_last_success(
    tmp_path: Path,
) -> None:
    api, text = FakeApi(), FakeText()
    api.advance(D1)
    api.fail["DE"] = [500]
    first = run(tmp_path, D1, api, text)

    assert first["states"]["DE"] | {"since": None} == {
        "outcome": "failed", "since": None, "pages": 0, "error": "HTTP 500", "records": 0,
        "texts": {},
    }  # fmt: skip
    assert first["outcomes"] == {"complete": 3, "failed": 1}
    assert json.loads((tmp_path / "reports" / "2026-01-12.json").read_text()) == first
    assert ("DE", "153") not in load_day(tmp_path / "days" / "2026-01-12").records

    api.advance(D2)
    second = run(tmp_path, D2, api, text)

    # Delaware starts again where it last succeeded, here never; Michigan past its last bill.
    assert second["states"]["DE"]["since"] == START.isoformat()
    assert second["states"]["MI"]["since"] == "2026-01-12T11:00:00.000004+00:00"
    assert second["outcomes"] == {"complete": 4}
    assert set(records(tmp_path / "days" / "2026-01-13")) == {DE_HB1, DE_SB2, MI_SB1, MI_HB4001}


def test_a_refused_key_stops_the_run_before_anything_is_written(tmp_path: Path) -> None:
    api = FakeApi(fail={"CA": [401]})
    with pytest.raises(IntakeError, match="refused the key: HTTP 401"):
        run(tmp_path, D1, api, FakeText())
    assert not (tmp_path / "days").exists()


def test_the_rate_limit_stops_the_api_side_and_the_states_left_resume(tmp_path: Path) -> None:
    api, text = FakeApi(), FakeText()
    api.advance(D1)
    api.fail["DE"] = [429]
    first = run(tmp_path, D1, api, text)

    assert {s: r["outcome"] for s, r in first["states"].items()} == {
        "CA": "complete", "DE": "rate_limited", "IN": "not_run", "MI": "not_run"
    }  # fmt: skip
    assert len(api.calls) == 2

    second = run(tmp_path, D2, api, text)

    # The states furthest behind are asked first.
    asked = [parse_qs(urlsplit(u).query)["jurisdiction"][0][-13:-11] for u, _, _ in api.calls[2:]]
    assert asked == ["de", "in", "mi", "ca"]
    assert set(records(tmp_path / "days" / "2026-01-13")) == {
        DE_HB1, DE_SB2, MI_SB1, "ocd-bill/fixture-in-0001"
    }  # fmt: skip
    assert second["outcomes"] == {"complete": 4}


def _many(n: int, tied: int) -> dict[str, dict[str, Any]]:
    """`n` Delaware bills, the first `tied` updated at one instant."""
    template = FIXTURE["days"]["2026-01-12"][1]
    base = datetime(2026, 1, 12, 8, tzinfo=UTC)
    out: dict[str, dict[str, Any]] = {}
    for i in range(n):
        at = base if i < tied else base + timedelta(seconds=i)
        bill_id = f"ocd-bill/fixture-de-{i + 100:04}"
        out[bill_id] = template | {"id": bill_id, "identifier": f"HB {i + 100}",
                                   "updated_at": at.isoformat(), "versions": []}  # fmt: skip
    return out


def test_paging_follows_the_cursor_and_ties_page_on(tmp_path: Path) -> None:
    api = FakeApi(bills=_many(45, tied=30))
    report = run(tmp_path, D1, api, FakeText(), states=("DE",))

    pages = [(q["updated_since"][0][11:19], q["page"][0])
             for q in (parse_qs(urlsplit(u).query) for u, _, _ in api.calls)]  # fmt: skip
    # Thirty bills at one instant: the cursor cannot move past them, so the request pages on.
    assert pages == [("00:00:00", "1"), ("08:00:00", "1"), ("08:00:00", "2"), ("08:00:39", "1")]
    assert report["states"]["DE"]["pages"] == 4
    assert set(records(tmp_path / "days" / "2026-01-12")) == set(api.bills)


def test_a_day_s_request_budget_stops_a_state_which_resumes_the_next_day(tmp_path: Path) -> None:
    api = FakeApi(bills=_many(45, tied=30))
    first = run(tmp_path, D1, api, FakeText(), states=("DE",), tier=Tier(6000, per_day=2))
    second = run(tmp_path, D2, api, FakeText(), states=("DE",), tier=Tier(6000, per_day=2))

    assert first["states"]["DE"]["outcome"] == "budget"
    assert second["states"]["DE"] | {"since": None} == first["states"]["DE"] | {
        "outcome": "complete", "since": None, "records": 25, "texts": {"none: no bill version": 25}
    }  # fmt: skip
    one, two = records(tmp_path / "days" / "2026-01-12"), records(tmp_path / "days" / "2026-01-13")
    assert (len(one), len(two)) == (20, 25)
    assert set(one) | set(two) == set(api.bills)


def test_a_new_version_that_fails_keeps_its_earlier_text_until_it_is_fetched(
    tmp_path: Path,
) -> None:
    api, text = FakeApi(), FakeText(fail={"https://legis.example/de/hb1-eng.html": 1})
    api.advance(D1)
    run(tmp_path, D1, api, text)
    api.advance(D2)
    second = run(tmp_path, D2, api, text)
    third = run(tmp_path, D3, api, text)

    one, two, three = (records(tmp_path / "days" / d)
                       for d in ("2026-01-12", "2026-01-13", "2026-01-14"))  # fmt: skip
    assert second["states"]["DE"]["texts"] == {"kept_earlier_text": 1}
    assert (two[DE_HB1]["raw_text"], two[DE_HB1]["raw_text_url"]) == (
        one[DE_HB1]["raw_text"], one[DE_HB1]["raw_text_url"]
    )  # fmt: skip
    # Nothing changed upstream on day 3; the version is fetched again and the record joins.
    assert set(three) == {DE_HB1}
    assert "every three years" in three[DE_HB1]["raw_text"]
    assert third["earlier_records_given_text"] == 1
    assert text.asked[2] == ["https://legis.example/de/hb1-eng.html"]


# Robots and rate limits.


def test_a_version_robots_txt_disallows_is_never_requested(tmp_path: Path) -> None:
    api, text = FakeApi(), FakeText()
    api.advance(D1)
    run(tmp_path, D1, api, text)

    assert "https://legis.example/private/sb2.html" in text.asked[0]
    assert "https://legis.example/private/sb2.html" not in text.served
    assert "https://legis.example/de/hb1.html" in text.served
    assert records(tmp_path / "days" / "2026-01-12")[DE_SB2]["raw_text"] == ""


def test_the_key_travels_in_its_header_at_the_tier_s_pace(tmp_path: Path) -> None:
    api = FakeApi()
    api.advance(D1)
    run(tmp_path, D1, api, FakeText(), tier=Tier(per_minute=10, per_day=250))

    assert {h["X-API-KEY"] for _, h, _ in api.calls} == {KEY}
    assert all(KEY not in url for url, _, _ in api.calls)
    assert {interval for _, _, interval in api.calls} == {6.0}


@dataclass
class Server:
    url: str
    times: list[float]
    answers: list[tuple[int, dict[str, str]]]  # status and headers for the next requests


@pytest.fixture
def server() -> Iterator[Server]:
    """A local stand-in for the API, so open_url's pacing and backoff run for real."""
    times: list[float] = []
    answers: list[tuple[int, dict[str, str]]] = []
    bills: dict[str, dict[str, Any]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            times.append(time.monotonic())
            status, headers = answers.pop(0) if answers else (200, {})
            body = (
                json.dumps(answer(bills, parse_qs(urlsplit(self.path).query))).encode()
                if status == 200
                else b'{"detail": "exceeded limit of 10/min: 11"}'
            )
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    yield Server(f"http://127.0.0.1:{http.server_address[1]}/", times, answers)
    http.shutdown()
    http.server_close()


def _through_open_url(
    url: str, pause: float, *, interval: float = 1.0, headers: Mapping[str, str] | None = None
) -> HTTPResponse:
    # The pacing and backoff under test are open_url's own.
    return open_url(url, pause, interval=interval, headers=headers)


def _waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []
    monkeypatch.setattr("lineage.core.http.time.sleep", waits.append)
    return waits


def test_a_429_s_retry_after_is_waited_out(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits = _waits(monkeypatch)
    server.answers.append((429, {"Retry-After": "7"}))
    report = run(tmp_path, D1, _through_open_url, FakeText(), states=("DE",), root=server.url)

    assert waits == [7.0]
    assert len(server.times) == 2
    assert report["states"]["DE"]["outcome"] == "complete"


def test_a_429_without_retry_after_backs_off_then_stops_the_run(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits = _waits(monkeypatch)
    server.answers.extend((429, dict[str, str]()) for _ in range(3))
    report = run(tmp_path, D1, _through_open_url, FakeText(), states=("DE", "MI"),
                 root=server.url, pause=30.0)  # fmt: skip

    # open_url's backoff: twice the pause, then four times, then the run stops asking.
    assert waits == [60.0, 120.0]
    assert {s: r["outcome"] for s, r in report["states"].items()} == {
        "DE": "rate_limited", "MI": "not_run"
    }  # fmt: skip
    assert len(server.times) == 3


def test_requests_wait_for_the_host_s_turn_at_the_tier_s_interval(
    tmp_path: Path, server: Server
) -> None:
    tier = Tier(per_minute=200, per_day=250)  # 0.3 seconds apart
    run(tmp_path, D1, _through_open_url, FakeText(), states=("DE", "MI"), root=server.url,
        tier=tier)  # fmt: skip

    gaps = [b - a for a, b in zip(server.times, server.times[1:], strict=False)]
    assert len(server.times) == 2
    assert min(gaps) >= tier.interval - 0.02
    state = json.loads((pace.directory() / "127.0.0.1.json").read_text())
    assert state["interval"] == pytest.approx(tier.interval)


# The January estimate.


def test_requests_count_twenty_bills_on_the_first_page_and_nineteen_after() -> None:
    assert [pages_for(n) for n in (0, 1, 19, 20, 38, 39, 40)] == [1, 1, 1, 2, 2, 3, 3]
    assert PER_PAGE == 20
    rows = [("2026-01-07", "NY", 39), ("2026-01-07", "NJ", 5), ("2026-01-08", "NY", 0)]
    assert requests_per_day(rows, ["2026-01-07", "2026-01-08", "2026-01-09"], states=50) == {
        "2026-01-07": 3 + 1 + 48, "2026-01-08": 1 + 49, "2026-01-09": 50
    }  # fmt: skip


def test_the_january_estimate_counts_each_bill_once_a_day_it_acts(
    raw_dir: Path, tmp_path: Path
) -> None:
    manifest, tables = tmp_path / "manifest.json", tmp_path / "tables"
    runner = CliRunner()
    for args in (
        ["snapshot", "--raw", str(raw_dir), "--manifest", str(manifest)],
        ["ingest", "--raw", str(raw_dir), "--manifest", str(manifest), "--out", str(tables),
         "--sessions", str(SESSIONS), "--min-free-gib", "0"],
    ):  # fmt: skip
        assert runner.invoke(app, args).exit_code == 0
    expected: dict[str, set[tuple[str, str]]] = {}
    for path in sorted((Path(__file__).parent / "fixtures" / "openstates").glob("*/*_bills.json")):
        for bill in json.loads(path.read_text()):
            for action in bill["actions"]:
                if action["date"].startswith("2023-01"):
                    day = action["date"][:10]
                    expected.setdefault(day, set()).add((path.name[:2], bill["id"]))

    estimate = january_estimate(tables, 2023, tiers=[Tier(10, 250)])

    assert estimate["bills_updated"] == sum(len(v) for v in expected.values()) > 0
    assert estimate["requests"] == sum(estimate["per_day"].values()) >= 50 * 31
    assert estimate["days_over"] == {"250/day": 0}


def test_the_command_needs_the_key_and_says_so_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from lineage.cli import app

    monkeypatch.delenv("OPENSTATES_API_KEY", raising=False)
    run = CliRunner().invoke(app, ["intake", "--day", "2026-01-12", "--days", str(tmp_path / "d")])

    assert run.exit_code == 1
    assert "OPENSTATES_API_KEY is not set" in run.output
    assert not (tmp_path / "d").exists()
