"""The Atom feed (RFC 4287), `feed.xml` at the export's root: the live track's events, from
every state, derived from the change log (lineage.live.updates) and checked before it is written.

Its events, each an entry dated the day it was logged:

- `new_bill`: a bill of the hosted state, first seen that day, with relatives.
- `relative_gained`: a bill of any state, seen before, that gained relatives that day; one entry
  per bill and day, naming every relative it gained.
- `origin_gained`: a bill of any state that gained a match to a published model document a
  page would list (lineage.origins.matches.FLOOR), its document not withheld
  (lineage.origins.withheld); one entry per bill and day.
- `states_group`: bills from three or more states, linked by relatives, all first seen within
  14 days. Precisely: on day D, take the bills the live track first saw (a `RecordAdded`) from
  D-13 to D, and link two of them when either lists the other as a relative after D's changes.
  A group is a connected set of those links whose bills come from at least three states. It is
  reported on D unless a group of day D-1 that shares a bill with it already held every one of
  its states, so a group is reported when it forms and again only when it reaches a new state.
  A bill in the corpus before the live track began has no first-seen day and is in no group.

Entries state dates and shared text, never who wrote anything, and never carry a model
document's text: an origin names the document's title and the library that holds it, as a
page does. An entry links to the bill's page when the bill is one the site hosts, else to the
text's source; a group links to its first hosted bill, else to its first bill's source.

Each entry's id is a URN, a UUID 5 of its kind, day and bill (a group's of its bills), so an
entry keeps its id however often the feed is written. The feed keeps the 14 days of events
ending at the last day applied, a reader polling weekly loses none of them, and its `updated` is
that day, so the same days always give the same bytes. The feed is checked against RFC 4287's
required elements and cardinalities with the standard library's parser before it is written.

Owned by lane L3 (M12). See DESIGN.md#updates.

See DESIGN.md#dr-13.
"""

import re
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from lineage.core import db
from lineage.export import queries as export_queries
from lineage.export.atom import ATOM, violations
from lineage.export.pages import bill_refs
from lineage.export.schema import Meta
from lineage.live.publish import FEED_FILE, META, changes_on, logged_days
from lineage.live.updates import (
    CHANGES_DIR,
    Change,
    OriginGained,
    RecordAdded,
    RelativeGained,
    RelativeLost,
)
from lineage.origins.collect import PACKAGE as COLLECT_PACKAGE
from lineage.origins.collect import collectors
from lineage.origins.matches import FLOOR
from lineage.origins.modeldocs import MODELDOCS_DIR, read_docs
from lineage.origins.withheld import REPORT as WITHHELD_REPORT
from lineage.origins.withheld import load_withheld

RETENTION_DAYS = 14
GROUP_DAYS = 14
GROUP_STATES = 3
TITLE = "Legislative Lineage: new matches"
SUBTITLE = (
    "Bills in every state newly found to share text with bills in other states or with "
    "published model documents, as of each day's update."
)
AUTHOR = "Legislative Lineage"
# Fixed once: every entry id is a UUID 5 under it, so ids never change.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "urn:legislative-lineage:feed")

EventKind = Literal["new_bill", "relative_gained", "origin_gained", "states_group"]


class FeedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    day: date
    bill_id: str | None  # the bill the entry is about; None for a group
    # relatives gained (with the stages that found each), documents matched (with the bill's
    # share), or a group's bills (with the day each was first seen)
    related: tuple[tuple[str, str], ...]

    @property
    def key(self) -> str:
        if self.bill_id is not None:
            return f"{self.kind}:{self.day.isoformat()}:{self.bill_id}"
        return f"{self.kind}:{self.day.isoformat()}:{','.join(r for r, _ in self.related)}"

    @property
    def id(self) -> str:
        return f"urn:uuid:{uuid.uuid5(NAMESPACE, self.key)}"


# Groups of bills first seen together.

Edge = frozenset[str]


def _components(members: Iterable[str], edges: Iterable[Edge]) -> list[frozenset[str]]:
    parent: dict[str, str] = {m: m for m in members}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in edges:
        a, b = sorted(edge)
        if a in parent and b in parent:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    out: dict[str, set[str]] = defaultdict(set)
    for m in parent:
        out[root(m)].add(m)
    return [frozenset(c) for _, c in sorted(out.items())]


def groups_on(
    day: date,
    first_seen: Mapping[str, tuple[date, str]],
    edges: Iterable[Edge],
    *,
    window: int = GROUP_DAYS,
    min_states: int = GROUP_STATES,
) -> list[frozenset[str]]:
    """The groups of `day`: bills first seen in the `window` days ending on it, linked by
    `edges`, with bills from at least `min_states` states."""
    start = day - timedelta(days=window - 1)
    members = [b for b, (seen, _) in first_seen.items() if start <= seen <= day]
    return [
        c for c in _components(members, edges) if len({first_seen[b][1] for b in c}) >= min_states
    ]


def new_groups(
    day: date,
    first_seen: Mapping[str, tuple[date, str]],
    edges_now: Iterable[Edge],
    edges_before: Iterable[Edge],
) -> list[frozenset[str]]:
    """The groups reported on `day`: those no group of the day before, sharing a bill with
    it, already held every state of."""
    before = groups_on(day - timedelta(days=1), first_seen, edges_before)
    out: list[frozenset[str]] = []
    for group in groups_on(day, first_seen, edges_now):
        states = {first_seen[b][1] for b in group}
        if not any(group & g and states <= {first_seen[b][1] for b in g} for g in before):
            out.append(group)
    return out


def feed_events(
    changes: Sequence[Change],
    *,
    hosted_state: str,
    through: date,
    listed_origin: Callable[[OriginGained], bool],
    retention: int = RETENTION_DAYS,
) -> list[Event]:
    """Every event of the `retention` days ending on `through`, from the change log of every
    day up to it, newest first."""
    by_day: dict[date, list[Change]] = defaultdict(list)
    for c in changes:
        if c.as_of <= through:
            by_day[c.as_of].append(c)
    first_seen: dict[str, tuple[date, str]] = {
        c.bill_id: (c.as_of, c.state) for c in changes if isinstance(c, RecordAdded)
    }
    start = through - timedelta(days=retention - 1)
    directed: set[tuple[str, str]] = set()
    edges_before: frozenset[Edge] = frozenset()
    out: list[Event] = []
    for day in sorted(by_day):
        log = by_day[day]
        for c in log:
            # Only bills the live track first saw can be in a group, so only their links count.
            if isinstance(c, RelativeGained | RelativeLost) and {c.bill_id, c.relative_id} <= set(
                first_seen
            ):
                pair = (c.bill_id, c.relative_id)
                if isinstance(c, RelativeGained):
                    directed.add(pair)
                else:
                    directed.discard(pair)
        edges_now = frozenset(frozenset(p) for p in directed)
        if day >= start:
            out.extend(_day_events(day, log, first_seen, hosted_state, listed_origin))
            for group in new_groups(day, first_seen, edges_now, edges_before):
                members = sorted(group, key=lambda b: (first_seen[b][0], b))
                related = tuple((b, first_seen[b][0].isoformat()) for b in members)
                out.append(Event("states_group", day, None, related))
        edges_before = edges_now
    return sorted(out, key=lambda e: (-e.day.toordinal(), e.kind, e.key))


def _day_events(
    day: date,
    log: Sequence[Change],
    first_seen: Mapping[str, tuple[date, str]],
    hosted_state: str,
    listed_origin: Callable[[OriginGained], bool],
) -> list[Event]:
    gained: dict[str, list[tuple[str, str]]] = defaultdict(list)
    origins: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in log:
        if isinstance(c, RelativeGained):
            gained[c.bill_id].append((c.relative_id, "+".join(c.stages)))
        elif isinstance(c, OriginGained) and listed_origin(c):
            origins[c.bill_id].append((c.doc_id, f"{c.bill_share:.4f} {c.model_share:.4f}"))
    out: list[Event] = []
    for bill_id, related in sorted(gained.items()):
        new = first_seen.get(bill_id, (None, None))[0] == day
        if new and first_seen[bill_id][1] == hosted_state:
            out.append(Event("new_bill", day, bill_id, tuple(sorted(related))))
        elif not new:
            out.append(Event("relative_gained", day, bill_id, tuple(sorted(related))))
    for bill_id, related in sorted(origins.items()):
        out.append(Event("origin_gained", day, bill_id, tuple(sorted(related))))
    return out


# Entries.


@dataclass(frozen=True, slots=True)
class BillInfo:
    state: str
    identifier: str
    title: str
    link: str | None  # the hosted page, or the text's source
    hosted: bool  # the site has its page


@dataclass(frozen=True, slots=True)
class DocInfo:
    title: str
    organization: str  # the library's: for a leaked document, the archive that published it
    leaked: bool
    drafted_by: str | None = None  # a leaked document's drafting organization


def listed_origin(c: OriginGained, withheld: frozenset[str] = frozenset()) -> bool:
    """Whether a page would list the gained match: by its larger share, which the log keeps
    both of, and its document not withheld (lineage.origins.withheld)."""
    return max(c.bill_share, c.model_share) >= FLOOR and c.doc_id not in withheld


def _article(name: str) -> str:
    # By sound for the names a library carries: "an ALEC", "an American", but "a Uniform".
    # A vowel, but not the "you" of "Uni-", or a letter sounded with one, read out as an acronym.
    vowel = re.match(r"[AEIOaeio]|[Uu](?!ni)|[AEFHILMNORSX](?:[A-Z]|$)", name) is not None
    return "an" if vowel else "a"


def model_named(title: str, organization: str, drafted_by: str | None, *, quoted: bool) -> str:
    """A model document as an origin line names it, as the site's does (site/src/lib/origins.ts,
    modelNamed): "[title], a model published by [organization]", or for a leaked document "[title],
    an [drafting organization] model from a leaked document published by [archive]" (ruled at the
    deploy review). The title drops an opening name of whoever drafted it."""
    shown = shown_title(title, drafted_by or organization)
    named = f"“{shown}”" if quoted else shown
    if drafted_by is None:
        return f"{named}, a model published by {organization}"
    return (
        f"{named}, {_article(drafted_by)} {drafted_by} model from a leaked document published by "
        f"{organization}"
    )


def shown_title(title: str, organization: str) -> str:
    """A title as an origin line shows it, as the site's does (site/src/lib/origins.ts,
    shownTitle): an opening name or abbreviation of its own organization dropped, with what
    joins it to the rest, since the line names the organization after it. A title that is
    nothing else stays whole."""
    name = re.sub(r"\s*\(.*\)\s*$", "", organization).strip()
    abbreviation = re.search(r"\(([A-Z][A-Z&.-]{1,})\)", organization)
    found = [organization, name] + ([abbreviation.group(1)] if abbreviation else [])
    names = sorted(
        {n for f in found for n in (f, re.sub(r"(?i)^the\s+", "", f)) if len(n) > 1},
        key=lambda n: -len(n),
    )
    for n in names:
        if not title.lower().startswith(n.lower()):
            continue
        rest = title[len(n) :]
        if rest and not re.match(r"(?:['\u2019]s\b|[\s:\u2013\u2014-])", rest):
            continue
        kept = re.sub(r"^(?:['\u2019]s)?[\s:\u2013\u2014-]*", "", rest).strip()
        return kept or title
    return title


def _named(b: BillInfo) -> str:
    return f"{b.state} {b.identifier}, “{b.title}”"


def _how(stages: str) -> str:
    # The site's two lists (DESIGN.md#export).
    return "shares wording" if "near_duplicate" in stages.split("+") else "similar in meaning"


@dataclass(frozen=True, slots=True)
class Entry:
    id: str
    title: str
    updated: str
    link: str | None
    content: str
    kind: EventKind


def entry(event: Event, bills: Mapping[str, BillInfo], docs: Mapping[str, DocInfo]) -> Entry:
    """An event as a feed entry: dates and shared text, never who wrote anything."""
    day = event.day.isoformat()
    updated = f"{day}T00:00:00Z"
    if event.kind == "states_group":
        members = [bills[b] for b, _ in event.related]
        states = sorted({m.state for m in members})
        seen = sorted(s for _, s in event.related)
        lines = [
            f"{len(members)} bills from {', '.join(states)}, first seen between {seen[0]} and "
            f"{seen[-1]}, are linked by shared text, each with at least one other among them:",
            *(f"- {_named(bills[b])}, first seen {s}" for b, s in event.related),
        ]
        # Members are in the order first seen, so this is the first hosted, else the first.
        linked = [m for m in members if m.hosted] or members
        return Entry(
            id=event.id,
            title=f"Bills in {len(states)} states first seen within {GROUP_DAYS} days share text",
            updated=updated,
            link=linked[0].link,
            content="\n".join(lines),
            kind=event.kind,
        )
    if event.bill_id is None:
        raise FeedError(f"a {event.kind} event names no bill")
    bill = bills[event.bill_id]
    n = len(event.related)
    if event.kind == "origin_gained":
        title = f"{bill.state} {bill.identifier} shares wording with " + (
            "a published model document" if n == 1 else f"{n} published model documents"
        )
        lines = [f"As of {day}, {_named(bill)}, shares wording with:"]
        for doc_id, shares in event.related:
            doc = docs[doc_id]
            bill_share, model_share = (float(x) for x in shares.split())
            lines.append(
                f"- {model_named(doc.title, doc.organization, doc.drafted_by, quoted=True)}: "
                f"{bill_share:.0%} of the bill's wording appears in it, covering "
                f"{model_share:.0%} of the model"
            )
    else:
        others = "a bill in another state" if n == 1 else f"{n} bills in other states"
        named = f"{bill.state} {bill.identifier}"
        if event.kind == "new_bill":
            title = f"{named} is new and shares text with {others}"
            lines = [f"{_named(bill)}, was first seen on {day}. It shares text with:"]
        else:
            title = f"{named} is newly found to share text with {others}"
            lines = [f"As of {day}, {_named(bill)}, shares text with:"]
        lines += [f"- {_named(bills[r])} ({_how(stages)})" for r, stages in event.related]
    return Entry(
        id=event.id,
        title=title,
        updated=updated,
        link=bill.link,
        content="\n".join(lines),
        kind=event.kind,
    )


def _el(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    child = ET.SubElement(parent, f"{{{ATOM}}}{tag}", attrs)
    if text is not None:
        child.text = text
    return child


def render(entries: Sequence[Entry], *, feed_id: str, site_url: str, updated: str) -> bytes:
    """The feed document, entries in the order given."""
    ET.register_namespace("", ATOM)
    root = ET.Element(f"{{{ATOM}}}feed")
    _el(root, "id", feed_id)
    _el(root, "title", TITLE)
    _el(root, "subtitle", SUBTITLE)
    _el(root, "updated", updated)
    author = _el(root, "author")
    _el(author, "name", AUTHOR)
    _el(root, "link", rel="self", type="application/atom+xml", href=f"{site_url}/data/{FEED_FILE}")
    _el(root, "link", rel="alternate", type="text/html", href=f"{site_url}/")
    for e in entries:
        node = _el(root, "entry")
        _el(node, "id", e.id)
        _el(node, "title", e.title)
        _el(node, "updated", e.updated)
        if e.link is not None:
            _el(node, "link", rel="alternate", href=e.link)
        _el(node, "category", term=e.kind)
        _el(node, "content", e.content, type="text")
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def check(data: bytes) -> None:
    problems = violations(data)
    if problems:
        raise FeedError("the feed breaks RFC 4287: " + "; ".join(problems))


# Writing it.


def _site(url: str) -> str:
    if not re.match(r"^https?://[^/\s]+", url):
        raise FeedError(f"the site's URL must be absolute http(s): {url!r}")
    return url.rstrip("/")


def write_feed(
    tables: Path,
    *,
    state: str,
    export: Path,
    site_url: str,
    changes_dir: Path = CHANGES_DIR,
    modeldocs_root: Path = MODELDOCS_DIR,
    collector_package: str = COLLECT_PACKAGE,
    retention: int = RETENTION_DAYS,
    withheld_report: Path = WITHHELD_REPORT,
) -> dict[str, Any]:
    """Write `export`/feed.xml from the change log, through the last day applied to the export
    (its `as_of`, else the last logged day), after checking it; returns the report."""
    site = _site(site_url)
    meta = Meta.model_validate_json((export / META).read_text())
    logged = logged_days(changes_dir)
    if meta.as_of is not None:
        through = date.fromisoformat(meta.as_of)
    elif logged:
        through = logged[-1]
    else:
        through = max(
            (datetime.fromisoformat(a.generated_at).date() for a in meta.archives),
            default=date(1970, 1, 1),
        )
    changes = [c for d in logged if d <= through for c in changes_on(changes_dir, d)]

    withheld = load_withheld(withheld_report).ids
    events = feed_events(
        changes,
        hosted_state=state,
        through=through,
        listed_origin=lambda c: listed_origin(c, withheld),
        retention=retention,
    )
    bills = bill_infos(tables, state=state, site_url=site, ids=_bills_of(events))
    sources = [c.source for c in collectors(collector_package).values()]
    docs = {
        d.id: DocInfo(
            title=d.title, organization=s.organization, leaked=d.leaked, drafted_by=d.drafted_by
        )
        for s in sources
        for d in read_docs(s, modeldocs_root)
    }
    missing = sorted(
        {r for e in events if e.kind == "origin_gained" for r, _ in e.related} - set(docs)
    )
    if missing:
        raise FeedError(f"an origin names {missing[0]}, which no collected source holds")
    entries = [entry(e, bills, docs) for e in events]
    data = render(
        entries,
        feed_id=f"urn:uuid:{uuid.uuid5(NAMESPACE, f'feed:{site}')}",
        site_url=site,
        updated=f"{through.isoformat()}T00:00:00Z",
    )
    check(data)
    path = export / FEED_FILE
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    partial.replace(path)
    kinds: dict[str, int] = defaultdict(int)
    for e in entries:
        kinds[e.kind] += 1
    return {
        "through": through.isoformat(),
        "retention_days": retention,
        "entries": len(entries),
        "by_kind": dict(sorted(kinds.items())),
        "states": sorted({b.state for b in bills.values()}),
        "bytes": len(data),
        "path": str(path),
    }


def _bills_of(events: Iterable[Event]) -> list[str]:
    out: set[str] = set()
    for e in events:
        if e.bill_id is not None:
            out.add(e.bill_id)
        if e.kind in ("new_bill", "relative_gained", "states_group"):
            out |= {r for r, _ in e.related}
    return sorted(out)


def bill_infos(tables: Path, *, state: str, site_url: str, ids: list[str]) -> dict[str, BillInfo]:
    """Each bill as an entry names it, linked to its page where the site hosts it, else to its
    text's source."""
    if not ids:
        return {}
    bills_glob = str(tables / "bills" / "*.parquet")
    with db.connect_quiet() as con:
        refs, texts = bill_refs(con, tables, ids)
        hosted = set(
            export_queries.hosted_bill_ids(
                con, bills_glob=bills_glob, eligible_path=tables / "eligible.parquet", state=state
            )
        )
    out: dict[str, BillInfo] = {}
    for bill_id, ref in refs.items():
        text = texts.get(bill_id)
        link = (
            f"{site_url}/bills/{ref.slug}/"
            if bill_id in hosted
            else (text.url if text is not None else None)
        )
        out[bill_id] = BillInfo(
            state=ref.state,
            identifier=ref.identifier,
            title=ref.title,
            link=link,
            hosted=bill_id in hosted,
        )
    return out
