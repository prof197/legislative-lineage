"""One limiter per host, shared by every process: the rate holds however many workers run."""

import itertools
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lineage import pace

INTERVAL = 0.2
WORKERS = 4
DOCUMENTS = 3


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(round(seconds, 6))
        self.now += seconds


def test_a_second_request_waits_out_the_interval(tmp_path: Path) -> None:
    clock = _Clock()

    with pace.turn("leg.example", 1.0, where=tmp_path, clock=clock, sleep=clock.sleep):
        clock.now += 0.25  # the request took a quarter of a second
    with pace.turn("leg.example", 1.0, where=tmp_path, clock=clock, sleep=clock.sleep):
        pass
    with pace.turn("other.example", 1.0, where=tmp_path, clock=clock, sleep=clock.sleep):
        pass

    assert clock.waits == [1.0]


def test_a_host_keeps_its_longest_interval(tmp_path: Path) -> None:
    clock = _Clock()

    with pace.turn("leg.example", 4.0, where=tmp_path, clock=clock, sleep=clock.sleep):
        pass
    # robots.txt is read at the default interval, yet the host's crawl delay still holds.
    with pace.turn("leg.example", 1.0, where=tmp_path, clock=clock, sleep=clock.sleep):
        pass

    assert clock.waits == [4.0]


@pytest.fixture
def arrivals() -> Iterator[tuple[str, list[float]]]:
    times: list[float] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            times.append(time.time())
            body = b"User-agent: *\nAllow: /\n" if self.path == "/robots.txt" else b"text"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    yield f"127.0.0.1:{server.server_address[1]}", times
    server.shutdown()
    server.server_close()


# Each worker does what a fetch run does: robots.txt first, then the documents.
_WORKER = f"""
import sys
from lineage.fetch import open_url
from lineage.robots import read_robots

host = sys.argv[1]
read_robots("http", host, pause=0, interval={INTERVAL}, retry_seconds=())
for i in range({DOCUMENTS}):
    with open_url(f"http://{{host}}/bill/{{i}}", 0, interval={INTERVAL}) as response:
        response.read()
"""


def _run_workers(host: str, pace_dirs: list[Path]) -> None:
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, host],
            env=os.environ | {"LINEAGE_PACE_DIR": str(d)},
        )
        for d in pace_dirs
    ]
    assert [w.wait(timeout=60) for w in workers] == [0] * len(workers)


def _gaps(times: list[float]) -> list[float]:
    ordered = sorted(times)
    return [b - a for a, b in itertools.pairwise(ordered)]


def test_parallel_worker_processes_share_one_rate_per_host(
    tmp_path: Path, arrivals: tuple[str, list[float]]
) -> None:
    host, times = arrivals

    _run_workers(host, [tmp_path / "pace"] * WORKERS)

    assert len(times) == WORKERS * (1 + DOCUMENTS)
    assert min(_gaps(times)) >= INTERVAL * 0.95


def test_the_rate_test_catches_processes_that_each_pace_themselves(
    tmp_path: Path, arrivals: tuple[str, list[float]]
) -> None:
    host, times = arrivals

    # A limiter per process, as each spawned worker had on 2026-11-06.
    _run_workers(host, [tmp_path / f"pace{i}" for i in range(WORKERS)])

    assert len(times) == WORKERS * (1 + DOCUMENTS)
    assert min(_gaps(times)) < INTERVAL * 0.95
