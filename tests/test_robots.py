"""robots.txt rules are read for this project's User-Agent, and an unreadable file bars the host."""

import pytest

from lineage.robots import parse_robots

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


def test_a_missing_robots_txt_publishes_no_rules() -> None:
    robots = parse_robots(URL, 404, b"", AT)

    assert (robots.rule, robots.allows(BILL)) == ("no rules", True)


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503, None])
def test_a_refused_or_unreachable_robots_txt_bars_the_host(status: int | None) -> None:
    robots = parse_robots(URL, status, b"", AT)

    assert (robots.rule, robots.allows(BILL)) == ("disallow all", False)
