"""`lineage text inventory`: which records need text, which version each would fetch, and where.

Open States stays the source for bills, actions and version links; its stored text is unusable
for about a third of eligible records. The inventory chooses each eligible record's version to
fetch (lineage.versions), groups the records by host and link format, and reads each host's
robots.txt for this project's User-Agent. It writes the fetch plan to `data/text/plan.parquet`
and the counts to `reports/text_inventory.json`. Only robots.txt is requested. See
DESIGN.md#dr-6.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from lineage import queries
from lineage.fetch import moved
from lineage.robots import Robots, read_robots
from lineage.stats import (
    AMENDMENT_CHARS,
    AMENDMENT_OPENING,
    AMENDMENT_SUBSTITUTE,
    BLANK_PATTERN,
    GEORGIA_ENACTING,
    GEORGIA_NOTICE,
)
from lineage.versions import Link, Version, choose_version

COMMAND = "uv run lineage text inventory"
REPORT = Path("reports/text_inventory.json")
PLAN = Path("data/text/plan.parquet")
PATTERNS = queries.TextPatterns(
    blank=BLANK_PATTERN,
    notice=GEORGIA_NOTICE,
    enacting=GEORGIA_ENACTING,
    opening=AMENDMENT_OPENING,
    chars=AMENDMENT_CHARS,
    substitute=AMENDMENT_SUBSTITUTE,
)
HTML, PDF = "text/html", "application/pdf"
PLAN_COLUMNS = (
    "id", "state", "session", "status", "kind", "outcome", "note", "date", "html_url", "pdf_url",
)  # fmt: skip


def _versions(raw: list[dict[str, Any]]) -> list[Version]:
    out: list[Version] = []
    for v in raw:
        links: list[dict[str, str | None]] = v["links"] or []
        kept = tuple(Link(lk["url"], lk["media_type"] or "") for lk in links if lk["url"])
        out.append(Version(v["note"] or "", v["date"] or "", kept))
    return out


def plan_row(
    record: tuple[str, str, str, str, str, list[dict[str, Any]]],
) -> dict[str, str | None]:
    """One record's fetch plan: the version chosen, its HTML and PDF links, and the outcome."""
    bill_id, state, session, status, kind, raw = record
    versions = _versions(raw)
    row: dict[str, str | None] = dict.fromkeys(PLAN_COLUMNS)
    row |= {"id": bill_id, "state": state, "session": session, "status": status, "kind": kind}
    choice = choose_version(versions, state)
    if not any(v.links for v in versions):
        row["outcome"] = "no_link"
    elif choice is None:
        row["outcome"] = "no_bill_version"
    elif not choice.links:
        row["outcome"] = "unsupported_format"
        row |= {"note": choice.note, "date": choice.date}
    else:
        urls = {lk.media_type: lk.url for lk in reversed(choice.links)}
        row |= {"outcome": "planned", "note": choice.note, "date": choice.date}
        row |= {"html_url": urls.get(HTML), "pdf_url": urls.get(PDF)}
    return row


def _origin(url: str) -> tuple[str, str]:
    parts = urlsplit(moved(url))
    return parts.scheme or "https", (parts.hostname or "").lower()


def _formats(row: dict[str, str | None]) -> str:
    return {(True, True): "html_and_pdf", (True, False): "html_only"}.get(
        (row["html_url"] is not None, row["pdf_url"] is not None), "pdf_only"
    )


def summarize(rows: list[dict[str, str | None]], robots: dict[str, Robots]) -> dict[str, Any]:
    """Counts per state and per host, with each host's robots.txt verdict on its planned links."""
    states: dict[str, Counter[str]] = defaultdict(Counter)
    hosts: dict[str, Counter[str]] = defaultdict(Counter)
    host_states: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        state = str(row["state"])
        need = row["kind"] != "usable"
        group = "without_text" if need else "with_text"
        tally = states[state]
        tally["eligible"] += 1
        tally[group] += 1
        if need:
            tally[f"kind:{row['kind']}"] += 1
            tally[f"outcome:{row['outcome']}"] += 1
        if row["outcome"] != "planned":
            continue
        tally[f"{group}:{_formats(row)}"] += 1
        for media, url in ((HTML, row["html_url"]), (PDF, row["pdf_url"])):
            if url is None:
                continue
            host = _origin(url)[1]
            host_states[host].add(state)
            fmt = "html" if media == HTML else "pdf"
            hosts[host][f"{group}:{fmt}"] += 1
            if robots[host].allows(moved(url)):
                hosts[host][f"{group}:{fmt}:allowed"] += 1
    return {
        "states": {s: _flat(c) for s, c in sorted(states.items())},
        "hosts": {
            h: {
                "states": sorted(host_states[h]),
                "links": _flat(c),
                "robots": {
                    "url": robots[h].url,
                    "status": robots[h].status,
                    "rule": robots[h].rule,
                    "crawl_delay": robots[h].crawl_delay(),
                    "sha256": robots[h].sha256,
                    "read_at": robots[h].read_at,
                },
            }
            for h, c in sorted(hosts.items(), key=lambda kv: -sum(kv[1].values()))
        },
    }


def _flat(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def build_inventory(
    tables_dir: Path,
    *,
    plan_path: Path,
    report_path: Path,
    read: Callable[[str, str], Robots] = read_robots,
) -> dict[str, Any]:
    with queries.connect_quiet() as con:
        con.execute("SET memory_limit = '6GB'")
        records = queries.text_population(
            con,
            bills_glob=str(tables_dir / "bills" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            patterns=PATTERNS,
        )
        rows = [plan_row(r) for r in records]
        del records
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        columns = {c: [r[c] for r in rows] for c in PLAN_COLUMNS}
        queries.write_fetch_plan(con, columns=columns, path=plan_path)
    origins = {
        _origin(url)
        for r in rows
        if r["outcome"] == "planned"
        for url in (r["html_url"], r["pdf_url"])
        if url is not None
    }
    # One request per host; a host served over both schemes is asked over https.
    schemes: dict[str, str] = {}
    for scheme, host in sorted(origins):
        schemes[host] = "https" if "https" in (schemes.get(host), scheme) else scheme
    robots = {host: read(scheme, host) for host, scheme in sorted(schemes.items())}
    doc: dict[str, Any] = {"command": COMMAND, "plan": str(plan_path)}
    totals = Counter(
        f"{'without_text' if r['kind'] != 'usable' else 'with_text'}:{r['outcome']}" for r in rows
    )
    doc["totals"] = {"eligible": len(rows)} | _flat(totals)
    doc |= summarize(rows, robots)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc
