"""`lineage intake`: one day of new and changed bills from Open States' API v3, with their text.

Each state is asked for its bills updated since its own cursor, oldest update first
(lineage.live.openstates_api). Every page is recorded under `<runs>/<day>/<ST>/` as it arrives, and
the next request starts from the last bill's update time rather than from a page number, so a
bill updated while the run pages is read again at the end instead of shifting a later page past
it. A state's cursor is read from its recorded pages: a state that fails, or that the key's
daily limit stops, resumes on the next run from its last recorded page, and the others are not
held back. A 401 or 403 stops the whole run, since no state would fare better with that key.

Text. The API carries no bill text, so each record's chosen version (lineage.text.versions) is
fetched through the text layer (lineage.text.fetch: robots.txt, pacing, backoff, extraction),
with its state's format, PDF reading and content element (rules/text_layer.toml), into the
intake's own store. A fetched text joins the day in the record itself, as `raw_text` and
`raw_text_url`, the fields an archive's record carries its text in: `lineage.live.updates.fold`
hashes them, and a rebuild's archives (`write_archives`), `lineage ingest` and
`lineage text merge` read them, with no other path to join. A record whose new version could not
be fetched keeps the text it carried before, so a transient error does not erase its matches;
the next run retries the version, and the record joins that day once it has its text.
`<runs>/<day>/texts.json` names the stored text each record of the day took.

The day directory (`day.json` and `<ST>_<session>_bills.json`, the format
`lineage.live.updates.load_day` reads) is assembled from the recorded pages, `texts.json`, the
store and the earlier days alone. A written day is never changed: running the day again sends
no request, assembles it again and checks it is the same, byte for byte. States are reported
in `<reports>/<day>.json`. See DESIGN.md#updates and DESIGN.md#dr-6.
"""

import json
import math
import os
import shutil
import tempfile
import urllib.error
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from typing import Any, Protocol

from lineage.core import db
from lineage.core.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.core.http import moved
from lineage.core.paths import DATA
from lineage.live import queries
from lineage.live.openstates_api import (
    KEY_ENV,
    KEY_HEADER,
    PER_PAGE,
    ROOT,
    STATES,
    Bill,
    BillList,
    bills_url,
    bulk_record,
    jurisdiction,
)
from lineage.live.updates import BILLS_SUFFIX, DAY_FILE, Folded, fold, load_day
from lineage.text.fetch import Job, fetch_all, host_of, stored
from lineage.text.rules import RULES, TextRules, load_rules, usable
from lineage.text.versions import Link, Version, choose_version

COMMAND = "uv run lineage intake"
INTAKE = DATA / "intake"
RUNS = INTAKE / "runs"
REPORTS = INTAKE / "reports"
DAYS = DATA / "days"
STORE = DATA / "text" / "intake"
TEXTS_FILE = "texts.json"
API_SOURCE = "openstates_api_v3"
TEXT_SOURCE = "legislature"
# The API's limiter sends 429 without Retry-After, so open_url backs off by its own schedule:
# twice this, then four times, a minute and then two, past the per-minute window.
API_PAUSE = 30.0
_TICK = timedelta(microseconds=1)
_FORMATS = {"text/html": "html", "application/pdf": "pdf"}


class IntakeError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Tier:
    """A key's request limits. The API's documentation publishes none; its source (api-v3,
    `api/auth.py`) sets the default tier at 10 a minute and 250 a day."""

    per_minute: int
    per_day: int

    @property
    def interval(self) -> float:
        return 60.0 / self.per_minute


DEFAULT_TIER = Tier(per_minute=10, per_day=250)
# The tiers the API's source sets (api/auth.py, api/rate_limiter.py); which one a key gets is
# known only once it is registered.
TIERS = {
    "default": DEFAULT_TIER,
    "bronze": Tier(per_minute=40, per_day=1_000),
    "silver": Tier(per_minute=80, per_day=50_000),
}
ESTIMATE = Path("reports/intake_estimate.json")


class Opener(Protocol):
    """`lineage.core.http.open_url`'s signature, with the request headers the key travels in."""

    def __call__(
        self,
        url: str,
        pause: float,
        *,
        interval: float = ...,
        headers: Mapping[str, str] | None = None,
    ) -> HTTPResponse: ...


TextFetcher = Callable[[list[Job], Path], object]


def api_key() -> str:
    """The key, from the environment only; it is never written, printed or put in a URL."""
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise IntakeError(f"{KEY_ENV} is not set")
    return key


# Recorded pages and the cursor they give.


@dataclass(frozen=True, slots=True)
class Page:
    since: datetime
    number: int
    bills: BillList


def _read_page(path: Path) -> Page:
    raw = json.loads(path.read_text())
    return Page(
        datetime.fromisoformat(raw["updated_since"]),
        raw["page"],
        BillList.model_validate(raw["body"]),
    )


def _page_files(day_dir: Path, state: str) -> list[Path]:
    return sorted((day_dir / state).glob("*.json"))


def _write_page(
    day_dir: Path, state: str, since: datetime, number: int, body: Any, floor: int
) -> None:
    path = day_dir / state / f"{len(_page_files(day_dir, state)) + 1:04d}.json"
    text = json.dumps(
        {"updated_since": since.isoformat(), "page": number, "body": body}, ensure_ascii=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    require_free_space(path.parent, needed=len(text.encode()), floor=floor)
    partial = path.with_suffix(".partial")
    partial.write_text(text)
    partial.replace(path)


def after(page: Page) -> tuple[datetime, int] | None:
    """The request that follows a page, or None when the page ended the list.

    The next request starts at the page's last update time; the bills updated at that instant
    come again and are deduplicated. A full page of bills updated at one instant cannot move the
    cursor, so the next page of the same request is asked instead."""
    results = page.bills.results
    pagination = page.bills.pagination
    if len(results) < PER_PAGE or pagination.page >= pagination.max_page:
        return None
    latest = max(b.updated_at for b in results)
    if latest > page.since:
        return latest, 1
    return page.since, page.number + 1


def _end(page: Page) -> datetime:
    """Where the next run starts once a page ended the list: past its last bill."""
    latest = max((b.updated_at for b in page.bills.results), default=None)
    return page.since if latest is None else latest + _TICK


def cursor(runs: Path, state: str, as_of: date, start: datetime) -> tuple[datetime, int] | None:
    """The next request for `state` on `as_of`: from its last recorded page on or before that
    day, or `start` if it has none. None when its list already ended on `as_of`."""
    days = sorted(
        p
        for p in runs.glob(f"*/{state}")
        if p.parent.name <= as_of.isoformat() and _page_files(p.parent, state)
    )
    if not days:
        return start, 1
    last = _read_page(_page_files(days[-1].parent, state)[-1])
    following = after(last)
    if following is not None:
        return following
    if days[-1].parent.name == as_of.isoformat():
        return None
    return _end(last), 1


# The API's side of a run.


@dataclass(slots=True)
class StateRun:
    outcome: str  # complete, failed, rate_limited, budget, not_run
    since: str | None = None
    pages: int = 0
    error: str | None = None


@dataclass(slots=True)
class _Budget:
    left: int
    used: int = 0


def _run_state(
    state: str,
    following: tuple[datetime, int],
    day_dir: Path,
    *,
    opener: Opener,
    key: str,
    root: str,
    tier: Tier,
    pause: float,
    budget: _Budget,
    floor: int,
) -> StateRun:
    since, number = following
    run = StateRun("complete", since=since.isoformat())
    while True:
        if budget.left <= 0:
            run.outcome = "budget"
            return run
        budget.left -= 1
        budget.used += 1
        url = bills_url(state, since, number, root=root)
        try:
            with opener(url, pause, interval=tier.interval, headers={KEY_HEADER: key}) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            e.close()
            if e.code in {401, 403}:
                raise IntakeError(f"the API refused the key: HTTP {e.code}") from None
            run.outcome = "rate_limited" if e.code == 429 else "failed"
            run.error = f"HTTP {e.code}"
            return run
        except (urllib.error.URLError, TimeoutError, ConnectionError, HTTPException) as e:
            run.outcome, run.error = "failed", type(e).__name__
            return run
        try:
            body = json.loads(raw)
            bills = BillList.model_validate(body)
        except ValueError as e:
            run.outcome, run.error = "failed", f"unreadable response: {type(e).__name__}"
            return run
        if any(b.jurisdiction.id != jurisdiction(state) for b in bills.results):
            run.outcome, run.error = "failed", "the response holds another state's bills"
            return run
        _write_page(day_dir, state, since, number, body, floor)
        run.pages += 1
        following_page = after(Page(since, number, bills))
        if following_page is None:
            return run
        since, number = following_page


# A day's records and their text.


def seen(day_dir: Path) -> dict[str, tuple[str, Bill]]:
    """Every bill a day's pages hold, by id, as last updated, with its state."""
    out: dict[str, tuple[str, Bill]] = {}
    if not day_dir.exists():
        return out
    for state_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
        for path in _page_files(day_dir, state_dir.name):
            for bill in _read_page(path).bills.results:
                was = out.get(bill.id)
                if was is None or bill.updated_at >= was[1].updated_at:
                    out[bill.id] = (state_dir.name, bill)
    return out


def job_for(record: Mapping[str, Any], state: str, rules: TextRules) -> Job | str:
    """The text-layer job for a record's chosen version, or why it has none."""
    rule = rules.states.get(state)
    if rule is None:
        return "no rule"
    if not rule.fetch:
        return rule.reason or "not fetched"
    if rule.source == "bulk":
        return "bulk download only, not fetched daily"
    versions = [
        Version(
            v["note"] or "",
            v["date"] or "",
            tuple(Link(lk["url"], lk["media_type"] or "") for lk in v["links"] if lk["url"]),
        )
        for v in record["versions"]
    ]
    choice = choose_version(versions, state)
    if choice is None:
        return "no bill version"
    urls = {_FORMATS[lk.media_type]: lk.url for lk in reversed(choice.links)}
    fmt = next((f for f in rule.formats if f in urls), None)
    if fmt is None:
        return "no link in the state's formats"
    url = urls[fmt]
    if host_of(url) in rules.retired:
        return f"retired host: {rules.retired[host_of(url)]}"
    layout = fmt == "pdf" and rule.reading == "layout"
    return Job(str(record["id"]), url, fmt, choice.note, choice.date, layout)


def _scan(
    store: Path, wanted: Mapping[str, str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """For each wanted id, the first stored row that kept usable text from the wanted address,
    and the latest outcome stored for that address."""
    ok: dict[str, dict[str, Any]] = {}
    latest: dict[str, str] = {}
    for path in sorted(store.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if wanted.get(row["id"]) != row["fetched_url"]:
                    continue
                latest[row["id"]] = row["outcome"]
                if row["outcome"] == "ok":
                    ok.setdefault(row["id"], row)
    return ok, latest


def _before(days: Path, as_of: date, *, later_allowed: bool) -> Folded | None:
    """The fold of the days before `as_of`."""
    if not days.exists():
        return None
    loaded = [
        load_day(p)
        for p in sorted(days.iterdir())
        if not p.name.startswith(".") and (p / DAY_FILE).exists()
    ]
    later = [d.as_of for d in loaded if d.as_of > as_of]
    if later and not later_allowed:
        raise IntakeError(f"{max(later)} is already written; days are written in order")
    earlier = sorted((d for d in loaded if d.as_of < as_of), key=lambda d: d.as_of)
    return fold(earlier) if earlier else None


def _earlier(before: Folded | None) -> Mapping[str, tuple[str, str, Mapping[str, Any]]]:
    return before.records if before is not None else {}


def assemble(as_of: date, day_dir: Path, store: Path, before: Folded | None) -> dict[str, str]:
    """The day directory's files, by name, from the day's pages, its `texts.json`, the store and
    the earlier days."""
    bills = seen(day_dir)
    texts_path = day_dir / TEXTS_FILE
    texts: dict[str, dict[str, str]] = (
        json.loads(texts_path.read_text()) if texts_path.exists() else {}
    )
    rows, _ = _scan(store, {i: t["url"] for i, t in texts.items()})
    earlier = _earlier(before)

    def text_of(bill_id: str) -> tuple[str, str | None]:
        if bill_id in texts:
            row = rows.get(bill_id)
            if row is None or row["sha256"] != texts[bill_id]["sha256"]:
                raise IntakeError(f"{bill_id}: the text {TEXTS_FILE} names is not in the store")
            return row["text"], row["fetched_url"]
        was = earlier.get(bill_id)
        if was is not None and was[2].get("raw_text"):
            return was[2]["raw_text"], was[2].get("raw_text_url")
        return "", None

    by_file: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for bill_id, (state, bill) in bills.items():
        record = bulk_record(bill)
        record["raw_text"], record["raw_text_url"] = text_of(bill_id)
        by_file[(state, bill.session)].append(record)
    for bill_id in sorted(set(texts) - set(bills)):
        if bill_id not in earlier:
            raise IntakeError(f"{bill_id}: in {TEXTS_FILE} but in no earlier day")
        state, session, row = earlier[bill_id]
        record = dict(row)
        record["raw_text"], record["raw_text_url"] = text_of(bill_id)
        by_file[(state, session)].append(record)
    has_text = any(r["raw_text"] for rs in by_file.values() for r in rs)
    info = {"as_of": as_of.isoformat(), "sources": [API_SOURCE] + [TEXT_SOURCE] * has_text}
    files = {DAY_FILE: json.dumps(info, indent=1) + "\n"}
    for (state, session), records in sorted(by_file.items()):
        records.sort(key=lambda r: r["id"])
        files[f"{state}_{session}{BILLS_SUFFIX}"] = json.dumps(records, ensure_ascii=False)
    return files


def _write_day(days: Path, as_of: date, files: Mapping[str, str], floor: int) -> bool:
    """Write the day once; True if written now. A written day is never changed."""
    target = days / as_of.isoformat()
    if target.exists():
        existing = {p.name: p.read_text() for p in target.iterdir() if p.is_file()}
        if existing != files:
            raise IntakeError(f"{target.name} is written already, and differently")
        return False
    days.mkdir(parents=True, exist_ok=True)
    require_free_space(days, needed=sum(len(c.encode()) for c in files.values()), floor=floor)
    building = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=days))
    try:
        # day.json last: a reader takes a directory with day.json for a whole day.
        for name in sorted(files, key=lambda n: n == DAY_FILE):
            (building / name).write_text(files[name])
        building.rename(target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return True


def default_fetch_texts(rules: TextRules, floor_bytes: int) -> TextFetcher:
    """The text layer's full fetcher, with the rules' content elements and usable-text bar."""

    def fetch(jobs: list[Job], store: Path) -> object:
        return fetch_all(
            jobs,
            store,
            containers=rules.containers,
            check=lambda text: usable(text, rules),
            floor_bytes=floor_bytes,
        )

    return fetch


def _fetch_texts(jobs: Mapping[str, Job], store: Path, fetch_texts: TextFetcher) -> dict[str, str]:
    """Fetch the jobs the store does not settle yet; return the address each id wants."""
    wanted = {bill_id: moved(job.url) for bill_id, job in jobs.items()}
    settled: set[tuple[str, str, str]] = set()
    for host in {host_of(j.url) for j in jobs.values()}:
        settled |= stored(store / f"{host}.jsonl")
    todo = [j for j in jobs.values() if (j.id, j.format, moved(j.url)) not in settled]
    if todo:
        store.mkdir(parents=True, exist_ok=True)
        fetch_texts(sorted(todo, key=lambda j: j.id), store)
    return wanted


def _run_states(
    states: Sequence[str],
    as_of: date,
    day_dir: Path,
    *,
    runs: Path,
    start: datetime,
    opener: Opener,
    key: str,
    root: str,
    tier: Tier,
    pause: float,
    floor: int,
) -> tuple[dict[str, StateRun], _Budget]:
    """Ask each state's updated bills, under the tier's daily budget; returns each state's run
    and the budget as spent."""
    # The states furthest behind go first, so a limit that stops the run starves none for good.
    following = {s: cursor(runs, s, as_of, start) for s in states}
    order = sorted(
        states, key=lambda s: (following[s] or (datetime.max.replace(tzinfo=start.tzinfo), 0), s)
    )
    budget = _Budget(left=tier.per_day)
    results: dict[str, StateRun] = {}
    stopped: str | None = None
    for state in order:
        nxt = following[state]
        if stopped is not None:
            results[state] = StateRun("not_run", error=stopped)
        elif nxt is None:
            results[state] = StateRun("complete")
        else:
            results[state] = _run_state(
                state, nxt, day_dir, opener=opener, key=key, root=root, tier=tier,
                pause=pause, budget=budget, floor=floor,
            )  # fmt: skip
            if results[state].outcome == "rate_limited":
                stopped = "the key's rate limit stopped the run"
    return results, budget


_Earlier = Mapping[str, tuple[str, str, Mapping[str, Any]]]


def _text_jobs(
    records: Mapping[str, tuple[str, Mapping[str, Any]]], earlier: _Earlier, rules: TextRules
) -> tuple[dict[str, Job], dict[str, str]]:
    """The text to fetch for today's records, and earlier ones still without their chosen
    version's text; with the reason a record of today's has none to fetch."""
    jobs: dict[str, Job] = {}
    reasons: dict[str, str] = {}
    for bill_id, (state, record) in records.items():
        job = job_for(record, state, rules)
        if isinstance(job, Job):
            jobs[bill_id] = job
        else:
            reasons[bill_id] = job
    for bill_id, (state, _, row) in earlier.items():
        if bill_id in records:
            continue
        job = job_for(row, state, rules)
        if isinstance(job, Job) and moved(job.url) != row.get("raw_text_url"):
            jobs[bill_id] = job
    return jobs, reasons


def _text_outcomes(
    records: Mapping[str, tuple[str, Mapping[str, Any]]],
    texts: Mapping[str, Any],
    earlier: _Earlier,
    why_none: Callable[[str], str],
) -> dict[str, Counter[str]]:
    """Each state's records by what text they have: fetched, kept from earlier, or none, with
    the reason."""
    per_state: dict[str, Counter[str]] = defaultdict(Counter)
    for bill_id, (state, _) in records.items():
        if bill_id in texts:
            kind = "fetched"
        elif bill_id in earlier and earlier[bill_id][2].get("raw_text"):
            kind = "kept_earlier_text"
        else:
            kind = f"none: {why_none(bill_id)}"
        per_state[state][kind] += 1
    return per_state


def _report(
    as_of: date,
    *,
    tier: Tier,
    budget: _Budget,
    results: Mapping[str, StateRun],
    records: Mapping[str, Any],
    texts: Mapping[str, Any],
    per_state: Mapping[str, Counter[str]],
) -> dict[str, Any]:
    return {
        "command": COMMAND,
        "as_of": as_of.isoformat(),
        "day": "written",
        "tier": {"per_minute": tier.per_minute, "per_day": tier.per_day},
        "requests": budget.used,
        "outcomes": dict(sorted(Counter(r.outcome for r in results.values()).items())),
        "records": len(records),
        "texts": dict(sorted(sum(per_state.values(), Counter[str]()).items())),
        "earlier_records_given_text": len(set(texts) - set(records)),
        "states": {
            state: {
                "outcome": run.outcome,
                "since": run.since,
                "pages": run.pages,
                "error": run.error,
                "records": sum(per_state[state].values()),
                "texts": dict(sorted(per_state[state].items())),
            }
            for state, run in sorted(results.items())
        },
    }


def run_intake(
    as_of: date,
    *,
    opener: Opener,
    key: str,
    start: datetime,
    states: Sequence[str] = STATES,
    root: str = ROOT,
    runs: Path = RUNS,
    days: Path = DAYS,
    store: Path = STORE,
    reports: Path = REPORTS,
    rules_path: Path = RULES,
    tier: Tier = DEFAULT_TIER,
    pause: float = API_PAUSE,
    fetch_texts: TextFetcher | None = None,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
) -> dict[str, Any]:
    """Take in one day: ask each state's updated bills, fetch their text, write the day and its
    report; return the report. `start` is where a state with no recorded page begins. A day
    already written is assembled again and checked, with no request sent."""
    if start.tzinfo is None:
        raise IntakeError("start needs a UTC offset")
    day_dir = runs / as_of.isoformat()
    if (days / as_of.isoformat()).exists():
        before = _before(days, as_of, later_allowed=True)
        _write_day(days, as_of, assemble(as_of, day_dir, store, before), floor_bytes)
        return {"command": COMMAND, "as_of": as_of.isoformat(), "day": "unchanged"}
    before = _before(days, as_of, later_allowed=False)
    rules = load_rules(rules_path)
    results, budget = _run_states(
        states, as_of, day_dir, runs=runs, start=start, opener=opener, key=key, root=root,
        tier=tier, pause=pause, floor=floor_bytes,
    )  # fmt: skip

    # Text: today's records, and earlier ones still without their chosen version's text.
    bills = seen(day_dir)
    records = {bill_id: (state, bulk_record(bill)) for bill_id, (state, bill) in bills.items()}
    earlier = _earlier(before)
    jobs, reasons = _text_jobs(records, earlier, rules)
    wanted = _fetch_texts(jobs, store, fetch_texts or default_fetch_texts(rules, floor_bytes))
    ok, latest = _scan(store, wanted)
    texts = {
        bill_id: {"url": row["fetched_url"], "sha256": row["sha256"]}
        for bill_id, row in sorted(ok.items())
    }
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / TEXTS_FILE).write_text(json.dumps(texts, indent=1) + "\n")
    _write_day(days, as_of, assemble(as_of, day_dir, store, before), floor_bytes)

    per_state = _text_outcomes(
        records,
        texts,
        earlier,
        lambda bill_id: reasons.get(bill_id) or latest.get(bill_id) or "not_fetched",
    )
    report = _report(
        as_of,
        tier=tier,
        budget=budget,
        results=results,
        records=records,
        texts=texts,
        per_state=per_state,
    )
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{as_of.isoformat()}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


# How many requests January needs: the estimate the key's limit is compared with.


def pages_for(bills: int) -> int:
    """Requests to read `bills` updated bills: twenty on the first page, then nineteen new on
    each, since every page after the first starts at the last one's final bill, and one more
    when the last page is full, to see the list end."""
    if bills < PER_PAGE:
        return 1
    return (bills - PER_PAGE) // (PER_PAGE - 1) + 2


def requests_per_day(
    counts: Sequence[tuple[str, str, int]], days: Sequence[str], states: int = len(STATES)
) -> dict[str, int]:
    """The requests each day needs, from (day, state, bills updated) rows: every state is asked
    once a day even when nothing changed."""
    by_day: dict[str, dict[str, int]] = defaultdict(dict)
    for day, state, n in counts:
        by_day[day][state] = n
    return {
        day: sum(pages_for(n) for n in by_day[day].values()) + states - len(by_day[day])
        for day in days
    }


def january_estimate(tables_dir: Path, year: int, tiers: Sequence[Tier] = ()) -> dict[str, Any]:
    """January `year`'s requests a day, from the bills table: a bill counts as updated on each
    day it has an action, a lower bound on what the API reports as updated."""
    days = [f"{year}-01-{d:02}" for d in range(1, 32)]
    with db.connect_quiet() as con:
        counts = queries.bills_acted_per_day(
            con, bills_glob=str(tables_dir / "bills" / "*.parquet"), month=f"{year}-01"
        )
        new_bills = queries.bills_first_acted(
            con, bills_glob=str(tables_dir / "bills" / "*.parquet"), month=f"{year}-01"
        )
    per_day = requests_per_day(counts, days)
    total = sum(per_day.values())
    peak_day = max(per_day, key=lambda d: per_day[d])
    return {
        "year": year,
        "bills_updated": sum(n for _, _, n in counts),
        "bills_new": new_bills,
        "requests": total,
        "mean_per_day": math.ceil(total / len(days)),
        "peak": {"day": peak_day, "requests": per_day[peak_day]},
        "days_over": {
            f"{t.per_day}/day": sum(r > t.per_day for r in per_day.values()) for t in tiers
        },
        "per_day": per_day,
    }
