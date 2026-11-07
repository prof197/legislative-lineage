"""The text fetcher: robots first, paced per host, text kept, failures retried on a rerun."""

import gzip
import json
import multiprocessing
import urllib.error
from email.message import Message
from http.client import HTTPResponse
from pathlib import Path
from typing import Any, cast

import pytest

from lineage.robots import Robots, parse_robots
from lineage.text_fetch import (
    ExtractionWorkersError,
    Fetched,
    FetchInWorkerError,
    Job,
    fetch_all,
    fetch_one,
    run_host,
)

AT = "2099-01-02T00:00:00+00:00"
PAGE = b"<html><body><p>Section 1. Short title.</p><p>Section 2. Fees.</p></body></html>"


class _Response:
    def __init__(self, body: bytes, content_type: str, encoding: str = "") -> None:
        self.body, self.status, self.headers = body, 200, Message()
        self.headers["Content-Type"] = content_type
        if encoding:
            self.headers["Content-Encoding"] = encoding

    def read(self, n: int) -> bytes:
        return self.body[:n]

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _robots(body: bytes = b"") -> Robots:
    return parse_robots("https://leg.example/robots.txt", 200, body, AT)


def _opener(body: bytes = PAGE, content_type: str = "text/html", encoding: str = "") -> Any:
    def opener(url: str, pause: float) -> HTTPResponse:
        return cast(HTTPResponse, _Response(body, content_type, encoding))

    return opener


JOB = Job("b1", "https://leg.example/bills/hb12.htm", "html", "Enrolled", "2099-03-01")


def test_a_fetched_page_keeps_its_text_hash_and_version() -> None:
    got = fetch_one(JOB, _robots(), opener=_opener())

    assert (got.outcome, got.note, got.body_bytes) == ("ok", "Enrolled", len(PAGE))
    assert got.text == "Section 1. Short title.\nSection 2. Fees."
    assert got.sha256 is not None
    assert len(got.sha256) == 64


def test_an_extractor_crash_is_recorded_not_raised() -> None:
    got = fetch_one(JOB, _robots(), opener=_opener(b"%PDF-1.7\n" + b"\x00" * 64, "application/pdf"))

    assert got.outcome == "extract_error"


def test_a_body_compressed_without_being_asked_is_decompressed() -> None:
    # South Dakota's document API gzips its PDFs whatever the request accepts.
    got = fetch_one(JOB, _robots(), opener=_opener(gzip.compress(PAGE), "text/html", "gzip"))

    assert (got.outcome, got.body_bytes) == ("ok", len(PAGE))


def test_a_url_robots_txt_disallows_is_not_requested() -> None:
    def opener(url: str, pause: float) -> HTTPResponse:
        raise AssertionError("requested")

    got = fetch_one(JOB, _robots(b"User-agent: *\nDisallow: /bills/\n"), opener=opener)

    assert got.outcome == "robots"


def test_server_errors_and_client_errors_are_told_apart() -> None:
    def failing(code: int) -> Any:
        def opener(url: str, pause: float) -> HTTPResponse:
            raise urllib.error.HTTPError(url, code, "nope", Message(), None)

        return opener

    assert fetch_one(JOB, _robots(), opener=failing(503)).outcome == "server_error"
    assert fetch_one(JOB, _robots(), opener=failing(404)).outcome == "http_error"


def test_a_page_with_no_extractable_text_is_an_extract_error() -> None:
    got = fetch_one(JOB, _robots(), opener=_opener(b"PK\x03\x04", "application/msword"))

    assert (got.outcome, got.text) == ("extract_error", None)


def _fake_fetch(outcomes: dict[str, str]) -> Any:
    def fetch(job: Job, robots: Robots, interval: float) -> Fetched:
        intervals.append(interval)
        outcome = outcomes.get(job.id, "ok")
        text = "t" if outcome == "ok" else None
        return Fetched(
            job.id, job.url, job.url, job.format, job.note, job.date, outcome, None, None, 0,
            text, None, 0.0, AT,
        )  # fmt: skip

    return fetch


intervals: list[float] = []


def _jobs(n: int) -> list[Job]:
    return [Job(f"b{i}", f"https://leg.example/{i}", "pdf", "Introduced", "") for i in range(n)]


def test_a_host_is_paced_at_its_crawl_delay_and_a_rerun_retries_only_failures(
    tmp_path: Path,
) -> None:
    def run(outcomes: dict[str, str]) -> dict[str, int]:
        return dict(
            run_host(
                "leg.example", _jobs(3), tmp_path,
                read=lambda s, h: _robots(b"User-agent: *\nCrawl-delay: 4\n"),
                fetch=_fake_fetch(outcomes), floor_bytes=0,
            )
        )  # fmt: skip

    intervals.clear()
    first = run({"b1": "network_error", "b2": "http_error"})
    second = run({})

    assert intervals == [4.0, 4.0, 4.0, 4.0]
    assert first == {"skipped": 0, "ok": 1, "network_error": 1, "http_error": 1}
    assert second == {"skipped": 2, "ok": 1}
    rows = [json.loads(line) for line in (tmp_path / "leg.example.jsonl").read_text().splitlines()]
    assert [(r["id"], r["outcome"]) for r in rows][-1] == ("b1", "ok")


def test_a_host_that_keeps_failing_is_stopped(tmp_path: Path) -> None:
    jobs = _jobs(30)

    got = run_host(
        "leg.example", jobs, tmp_path, read=lambda s, h: _robots(),
        fetch=_fake_fetch({j.id: "server_error" for j in jobs}), floor_bytes=0,
    )  # fmt: skip

    assert (got["server_error"], got["host_stopped"]) == (20, 1)


def _host(tmp_path: Path, status: int | None) -> dict[str, int]:
    # No fake fetch: a barred host must not reach the request at all.
    robots = parse_robots("https://leg.example/robots.txt", status, b"", AT)
    return dict(
        run_host(
            "leg.example", _jobs(2), tmp_path, read=lambda s, h: robots, floor_bytes=0,
        )
    )  # fmt: skip


@pytest.mark.parametrize("status", [None, 429, 503])
def test_a_temporary_robots_refusal_stores_nothing_so_a_later_run_asks_again(
    tmp_path: Path, status: int | None
) -> None:
    assert _host(tmp_path, status) == {"skipped": 0, "robots_unavailable": 2}
    assert _host(tmp_path, status) == {"skipped": 0, "robots_unavailable": 2}

    assert not (tmp_path / "leg.example.jsonl").exists()
    reads = (tmp_path / "robots" / "leg.example.jsonl").read_text().splitlines()
    assert [json.loads(r)["status"] for r in reads] == [status, status]


@pytest.mark.parametrize("status", [401, 403])
def test_a_deliberate_robots_refusal_is_stored_and_not_asked_again(
    tmp_path: Path, status: int
) -> None:
    assert _host(tmp_path, status) == {"skipped": 0, "robots": 2}
    assert _host(tmp_path, status) == {"skipped": 2}

    rows = [json.loads(r) for r in (tmp_path / "leg.example.jsonl").read_text().splitlines()]
    assert {(r["outcome"], r["status"]) for r in rows} == {("robots", status)}
    reads = (tmp_path / "robots" / "leg.example.jsonl").read_text().splitlines()
    assert len(reads) == 1


def test_fetch_all_refuses_to_run_inside_a_worker_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A spawned worker that re-ran a script without a main guard would call fetch_all here.
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())

    with pytest.raises(FetchInWorkerError, match="__main__"):
        fetch_all([JOB], tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_failed_extraction_workers_stop_the_run_instead_of_failing_each_document() -> None:
    def extractor(body: bytes, content_type: str, **options: Any) -> str:
        raise ExtractionWorkersError("a process in the pool terminated abruptly")

    with pytest.raises(ExtractionWorkersError):
        fetch_one(JOB, _robots(), opener=_opener(), extractor=extractor)
