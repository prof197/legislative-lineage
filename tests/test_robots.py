"""robots.txt rules are read for this project's User-Agent, and an unreadable file bars the host;
a temporary refusal is asked again, a deliberate one is not."""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lineage.robots import parse_robots, read_robots

URL = "https://leg.example/robots.txt"
BILL = "https://leg.example/bills/getPDF.asp?paper=SP0891&item=1"
AT = "2099-01-02T00:00:00+00:00"


def test_rules_for_other_named_crawlers_do_not_apply_to_this_project() -> None:
    body = b"User-agent: ClaudeBot\nDisallow: /\n\nUser-agent: anthropic-ai\nDisallow: /\n"

    robots = parse_robots(URL, 200, body, AT)

    assert robots.allows(BILL)


def test_rules_for_every_agent_apply() -> None:
    body = b"User-agent: *\nDisallow: /bills/\nCrawl-delay: 5\n"

    robots = parse_robots(URL, 200, body, AT)

    assert not robots.allows(BILL)
    assert robots.allows("https://leg.example/about")
    assert robots.crawl_delay() == 5.0


def test_rules_naming_this_project_apply() -> None:
    body = b"User-agent: legislative-lineage\nDisallow: /\n\nUser-agent: *\nAllow: /\n"

    assert not parse_robots(URL, 200, body, AT).allows(BILL)


def test_a_wildcard_rule_bars_what_it_matches() -> None:
    body = b"User-agent: *\nDisallow: /reports/world-map/*\nDisallow: /*.pdf$\n"
    robots = parse_robots(URL, 200, body, AT)

    assert not robots.allows("https://leg.example/reports/world-map/2024/index.html")
    assert not robots.allows("https://leg.example/files/bill.pdf")
    assert robots.allows("https://leg.example/files/bill.pdf?download=1")  # "$" ends the path
    assert robots.allows("https://leg.example/reports/other/")


def test_the_longest_rule_decides_and_an_allow_wins_a_tie() -> None:
    body = b"User-agent: *\nDisallow: /\nAllow: /models/\nDisallow: /models/private/\n"
    robots = parse_robots(URL, 200, body, AT)

    assert robots.allows("https://leg.example/models/act.pdf")
    assert not robots.allows("https://leg.example/models/private/act.pdf")
    assert not robots.allows("https://leg.example/about")
    tie = parse_robots(URL, 200, b"User-agent: *\nDisallow: /a\nAllow: /a\n", AT)
    assert tie.allows("https://leg.example/a")


def test_an_empty_disallow_allows_everything_wherever_it_stands() -> None:
    body = b"User-agent: *\nDisallow:\nDisallow: /withheld-\n"
    robots = parse_robots(URL, 200, body, AT)

    assert not robots.allows("https://leg.example/withheld-act/")
    assert robots.allows("https://leg.example/act/")


def test_groups_for_the_same_agent_combine_and_lines_can_share_a_group() -> None:
    body = (
        b"User-agent: legislative-lineage\nDisallow: /a/\n\n"
        b"User-agent: SomeBot\nUser-agent: legislative-lineage\nDisallow: /b/\n"
    )
    robots = parse_robots(URL, 200, body, AT)

    assert not robots.allows("https://leg.example/a/x")
    assert not robots.allows("https://leg.example/b/x")
    assert robots.allows("https://leg.example/c/x")


def test_a_crawl_delay_before_any_group_still_applies() -> None:
    body = b"Crawl-delay: 10\nUser-agent: *\nDisallow:\n"
    assert parse_robots(URL, 200, body, AT).crawl_delay() == 10.0
    own = b"Crawl-delay: 10\nUser-agent: *\nCrawl-delay: 2\n"
    assert parse_robots(URL, 200, own, AT).crawl_delay() == 2.0


def test_robots_txt_itself_is_always_allowed() -> None:
    robots = parse_robots(URL, 200, b"User-agent: *\nDisallow: /\n", AT)
    assert robots.allows("https://leg.example/robots.txt")


def test_a_missing_robots_txt_publishes_no_rules() -> None:
    robots = parse_robots(URL, 404, b"", AT)

    assert (robots.rule, robots.allows(BILL)) == ("no rules", True)


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503, None])
def test_a_refused_or_unreachable_robots_txt_bars_the_host(status: int | None) -> None:
    robots = parse_robots(URL, status, b"", AT)

    assert (robots.rule, robots.allows(BILL)) == ("disallow all", False)


@pytest.fixture
def robots_host() -> Iterator[tuple[str, list[int]]]:
    """A host whose robots.txt answers each status in turn, then 200 with rules."""
    statuses: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status = statuses.pop(0) if statuses else 200
            body = b"User-agent: *\nDisallow: /private/\n" if status == 200 else b""
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    host = f"127.0.0.1:{server.server_address[1]}"
    yield host, statuses
    server.shutdown()
    server.server_close()


def test_a_temporary_refusal_is_asked_again_after_a_backoff(
    robots_host: tuple[str, list[int]],
) -> None:
    host, statuses = robots_host
    # open_url retries a 503 three times itself; the fourth answer comes after the backoff.
    statuses.extend([503, 503, 503])
    waits: list[float] = []

    robots = read_robots(
        "http", host, pause=0, interval=0, retry_seconds=(60.0,), sleep=waits.append
    )

    assert (robots.status, robots.rule, robots.attempts) == (200, "parsed", 2)
    assert waits == [60.0]
    assert robots.allows(f"http://{host}/bills/1")
    assert not robots.allows(f"http://{host}/private/")


def test_a_deliberate_refusal_is_not_asked_again(robots_host: tuple[str, list[int]]) -> None:
    host, statuses = robots_host
    statuses.extend([403])
    waits: list[float] = []

    robots = read_robots(
        "http", host, pause=0, interval=0, retry_seconds=(60.0,), sleep=waits.append
    )

    assert (robots.status, robots.rule, robots.attempts, waits) == (403, "disallow all", 1, [])
    assert not robots.temporary
