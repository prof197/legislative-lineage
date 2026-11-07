"""`lineage text fetch`: the full run of the text layer.

The jobs come from the fetch plan and rules/text_layer.toml. A record is fetched when its state
is fetched and either its stored text is unusable or its state's fetched text replaces Open
States' (lineage.text_rules.replaced_states). Each record takes its state's preferred link
format, or the next one when the record lacks it, and a PDF is read as the state's rules say.
California's records come from the Legislature's bulk download (lineage.ca_bulk). Everything
else is disclosed, with its reason, in reports/text_fetch.json. See DESIGN.md#dr-6.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lineage import queries
from lineage.ca_bulk import fetch_california
from lineage.disk import DEFAULT_FLOOR_BYTES
from lineage.text_fetch import Job, fetch_all, host_of
from lineage.text_rules import RULES, TextRules, load_rules, replaced_states, usable

COMMAND = "uv run lineage text fetch"
STORE = Path("data/text/full")
REPORT = Path("reports/text_fetch.json")
QUALITY = Path("reports/text_quality.json")
RAW = Path("data/raw/ca-bulk")


def plan_jobs(
    plan: list[dict[str, Any]], rules: TextRules, replace: set[str]
) -> tuple[list[Job], list[str], Counter[tuple[str, str]]]:
    """The legislature jobs, the ids California's bulk download serves, and the records left
    out, counted by (state, reason)."""
    jobs: list[Job] = []
    bulk: list[str] = []
    left: Counter[tuple[str, str]] = Counter()
    for row in plan:
        state = row["state"]
        if row["kind"] == "usable" and state not in replace:
            continue
        rule = rules.states.get(state)
        if row["outcome"] != "planned":
            left[(state, row["outcome"])] += 1
        elif rule is None or not rule.fetch:
            left[(state, rule.reason if rule and rule.reason else "no rule")] += 1
        elif rule.source == "bulk":
            bulk.append(row["id"])
        else:
            fmt = next((f for f in rule.formats if row[f"{f}_url"]), None)
            if fmt is None:
                left[(state, "no link in the state's formats")] += 1
                continue
            url = row[f"{fmt}_url"]
            if host_of(url) in rules.retired:
                left[(state, f"retired host: {rules.retired[host_of(url)]}")] += 1
                continue
            layout = fmt == "pdf" and rule.reading == "layout"
            note, date = row["note"] or "", row["date"] or ""
            jobs.append(Job(row["id"], url, fmt, note, date, layout))
    return jobs, bulk, left


def _stored_outcomes(store: Path) -> dict[str, str]:
    """The latest outcome per record id across the store's hosts."""
    latest: dict[str, str] = {}
    for path in sorted(store.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                latest[row["id"]] = row["outcome"]
    return latest


def run_fetch(
    tables_dir: Path,
    *,
    plan_path: Path,
    store: Path = STORE,
    raw_dir: Path = RAW,
    report_path: Path = REPORT,
    quality_path: Path = QUALITY,
    rules_path: Path = RULES,
    fetch: bool = True,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
) -> dict[str, Any]:
    rules = load_rules(rules_path)
    replace = set(replaced_states(json.loads(quality_path.read_text()), rules))
    with queries.connect_quiet() as con:
        plan = queries.fetch_plan(con, path=plan_path)
    jobs, bulk, left = plan_jobs(plan, rules, replace)
    if fetch:
        fetch_all(
            jobs,
            store,
            containers=rules.containers,
            check=lambda text: usable(text, rules),
            floor_bytes=floor_bytes,
        )
        with queries.connect_quiet() as con:
            records = queries.identifiers(
                con, bills_glob=str(tables_dir / "bills" / "*.parquet"), ids=bulk
            )
        fetch_california(
            records, store, raw_dir, floor_bytes=floor_bytes, check=lambda t: usable(t, rules)
        )
    outcomes = _stored_outcomes(store)
    states: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"jobs": 0, "outcomes": Counter(), "left_out": {}}
    )
    by_id = {row["id"]: row for row in plan}
    for record_id in [j.id for j in jobs] + bulk:
        entry = states[by_id[record_id]["state"]]
        entry["jobs"] += 1
        entry["outcomes"][outcomes.get(record_id, "not_fetched")] += 1
    for (state, reason), n in sorted(left.items()):
        states[state]["left_out"][reason] = n
    total: Counter[str] = Counter()
    left_total: Counter[str] = Counter()
    for entry in states.values():
        total.update(entry["outcomes"])
        left_total.update(entry["left_out"])
    doc: dict[str, Any] = {
        "command": COMMAND,
        "replaced_states": sorted(replace),
        "jobs": len(jobs) + len(bulk),
        "outcomes": dict(sorted(total.items())),
        "left_out": dict(sorted(left_total.items())),
        "states": {
            state: {**entry, "outcomes": dict(sorted(entry["outcomes"].items()))}
            for state, entry in sorted(states.items())
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc
