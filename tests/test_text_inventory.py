"""The text layer's inventory: a version per record, counted by host, checked by robots.txt."""

import json
from pathlib import Path
from typing import Any

from lineage import queries
from lineage.robots import Robots, parse_robots
from lineage.text_inventory import build_inventory, plan_row

AT = "2099-01-02T00:00:00+00:00"


def _version(note: str, day: str, *links: tuple[str, str]) -> dict[str, Any]:
    return {"note": note, "date": day, "links": [{"url": u, "media_type": m} for u, m in links]}


PDF = "application/pdf"


def test_a_record_with_an_amendment_stored_is_planned_for_its_latest_bill_version() -> None:
    versions = [
        _version("HB459 Original", "2026-03-01", ("https://la.example/o.pdf", PDF)),
        _version("HB459 Enrolled", "2026-05-20", ("https://la.example/e.pdf", PDF)),
        _version("House Committee Amendment, 12", "2026-05-21", ("https://la.example/a.pdf", PDF)),
    ]

    row = plan_row(("la-1", "LA", "2026", "passed", "amendment_only", versions))

    assert (row["outcome"], row["note"], row["pdf_url"], row["html_url"]) == (
        "planned", "HB459 Enrolled", "https://la.example/e.pdf", None,
    )  # fmt: skip


def test_records_without_a_link_or_a_bill_version_are_not_planned() -> None:
    unlinked = [_version("Introduced", "")]
    amendment = [_version("Amendment 2023-0347h", "", ("https://nh.example/a", "text/html"))]

    no_link = plan_row(("a", "NH", "2023", "died", "blank", unlinked))
    no_bill = plan_row(("b", "NH", "2023", "died", "blank", amendment))

    assert (no_link["outcome"], no_bill["outcome"]) == ("no_link", "no_bill_version")


def _robots(scheme: str, host: str) -> Robots:
    body = b"User-agent: *\nDisallow: /\n" if host.startswith("blocked") else b""
    return parse_robots(f"{scheme}://{host}/robots.txt", 200, body, AT)


def test_the_inventory_counts_the_fixture_corpus_and_writes_the_plan(
    corpus: Path, tmp_path: Path
) -> None:
    plan, report = tmp_path / "plan.parquet", tmp_path / "inventory.json"

    doc = build_inventory(corpus, plan_path=plan, report_path=report, read=_robots)

    assert json.loads(report.read_text()) == doc
    assert doc["totals"]["eligible"] == sum(s["eligible"] for s in doc["states"].values())
    with queries.connect_quiet() as con:
        (rows,) = con.execute(f"SELECT count(*) FROM '{plan}'").fetchone() or (0,)
    assert rows == doc["totals"]["eligible"]
    for host in doc["hosts"].values():
        assert host["robots"]["rule"] == "parsed"
        for key, n in host["links"].items():
            if not key.endswith(":allowed"):
                assert host["links"].get(f"{key}:allowed") == n


def test_stored_text_the_sections_no_longer_hold_is_split_again(tmp_path: Path) -> None:
    # Once a record's fetched text is split, the sections table says nothing about whether
    # its stored text yields a section; the inventory splits the stored text itself.
    from lineage.text_inventory import _rechecked  # pyright: ignore[reportPrivateUsage]

    bills = tmp_path / "bills.parquet"
    with queries.connect_quiet() as con:
        con.execute(
            "COPY (SELECT * FROM (VALUES ('a', 'Section 1. Fees. Section 2. Rules.'), "
            "('b', '   ')) t(id, text)) TO '" + str(bills) + "' (FORMAT parquet)"
        )
        none: list[dict[str, Any]] = []
        records = [
            ("a", "NJ", "2025", "passed", "recheck:amendment_only", none),
            ("b", "NJ", "2025", "passed", "recheck:usable", none),
            ("c", "NJ", "2025", "passed", "usable", none),
        ]
        kinds = [r[4] for r in _rechecked(con, records, str(bills))]

    assert kinds == ["amendment_only", "no_sections", "usable"]
