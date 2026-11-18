"""Updates: the live track's contracts, which `lineage update` (lane L1) builds on.

A day is a directory of the records an intake run saw new or changed, in Open States' bill
shape, one file per state and session, with `day.json` giving its as-of date and sources. The
corpus after day N is the fold of days 1 to N: each record as last seen, every earlier text kept
as a version it replaced. An update over day N must leave the same `Snapshot` as a full rebuild
over days 1 to N, with the same pinned corpus statistics, and write a change log of what the day
added, updated and replaced and of every relative gained or lost. See DESIGN.md#updates, DR-7.
"""

import hashlib
import json
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from lineage import queries
from lineage.listing import shipped_flag
from lineage.origins import read_origin_matches
from lineage.pairs import load_families
from lineage.paths import DATA
from lineage.relatives import bill_relatives
from lineage.semantic import MODEL

DAY_FILE = "day.json"
BILLS_SUFFIX = "_bills.json"
CHANGES_DIR = DATA / "changes"


class UpdateError(Exception):
    pass


class DayInfo(BaseModel):
    """`day.json`: when the day's records were seen, and by which sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of: date
    sources: list[str]


@dataclass(frozen=True, slots=True)
class Day:
    as_of: date
    sources: tuple[str, ...]
    # Open States bill records, by (state, session), each list in the file's order.
    records: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]


def load_day(path: Path) -> Day:
    """One day's directory. A record seen twice in a day is an error: which one stands would
    depend on file order."""
    info = DayInfo.model_validate_json((path / DAY_FILE).read_text())
    records: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    seen: set[str] = set()
    for file in sorted(path.glob(f"*{BILLS_SUFFIX}")):
        state, _, session = file.name.removesuffix(BILLS_SUFFIX).partition("_")
        if not (len(state) == 2 and state.isupper() and session):
            raise UpdateError(f"{file.name}: not <ST>_<session>{BILLS_SUFFIX}")
        rows: list[Mapping[str, Any]] = json.loads(file.read_text())
        for row in rows:
            if row["id"] in seen:
                raise UpdateError(f"{path.name}: {row['id']} appears twice")
            seen.add(row["id"])
        records[(state, session)] = rows
    return Day(as_of=info.as_of, sources=tuple(info.sources), records=records)


def load_days(root: Path) -> list[Day]:
    """Every day under `root`, oldest first; two days with one as-of date are an error."""
    days = sorted(
        (load_day(p) for p in root.iterdir() if (p / DAY_FILE).exists()), key=lambda d: d.as_of
    )
    dates = [d.as_of for d in days]
    if len(set(dates)) != len(dates):
        raise UpdateError(f"two days share an as-of date under {root}")
    return days


def text_hash(record: Mapping[str, Any]) -> str | None:
    """A record's text identity: the SHA-256 of the text it carries, or None without one."""
    text = record.get("raw_text")
    return hashlib.sha256(text.encode()).hexdigest() if text else None


class RecordVersion(BaseModel):
    """One text a record has carried, and the days it stood. A later version replaces it
    without erasing it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    bill_id: str
    text_sha256: str | None
    first_seen: date
    replaced_on: date | None = None


@dataclass(frozen=True, slots=True)
class Folded:
    """The corpus after a run of days: each record as last seen, and every version."""

    as_of: date
    records: Mapping[str, tuple[str, str, Mapping[str, Any]]]  # id -> (state, session, record)
    versions: Sequence[RecordVersion]
    first_seen: Mapping[str, date] = field(default_factory=dict[str, date])


def fold(days: Iterable[Day]) -> Folded:
    """Days oldest first, folded: a record seen again stands as last seen, and a change of text
    closes the version before it."""
    records: dict[str, tuple[str, str, Mapping[str, Any]]] = {}
    open_version: dict[str, RecordVersion] = {}
    closed: list[RecordVersion] = []
    first: dict[str, date] = {}
    as_of: date | None = None
    for day in days:
        if as_of is not None and day.as_of <= as_of:
            raise UpdateError(f"days out of order: {day.as_of} after {as_of}")
        as_of = day.as_of
        for (state, session), rows in day.records.items():
            for row in rows:
                bill_id = str(row["id"])
                was = records.get(bill_id)
                if was is not None and (was[0], was[1]) != (state, session):
                    raise UpdateError(
                        f"{bill_id} moved from {was[0]} {was[1]} to {state} {session}"
                    )
                records[bill_id] = (state, session, row)
                first.setdefault(bill_id, day.as_of)
                digest = text_hash(row)
                current = open_version.get(bill_id)
                if current is None or current.text_sha256 != digest:
                    if current is not None:
                        closed.append(current.model_copy(update={"replaced_on": day.as_of}))
                    open_version[bill_id] = RecordVersion(
                        bill_id=bill_id, text_sha256=digest, first_seen=day.as_of
                    )
    if as_of is None:
        raise UpdateError("no days to fold")
    versions = sorted([*closed, *open_version.values()], key=lambda v: (v.bill_id, v.first_seen))
    return Folded(as_of=as_of, records=records, versions=versions, first_seen=first)


def write_archives(folded: Folded, raw: Path) -> Path:
    """The folded records as Open States archives under `raw`, one per state and session,
    each stamped as of the fold's last day: every record a day did not change still stands as
    of that day, which is what a status like "died when its session ended" is judged against."""
    by_archive: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for bill_id in sorted(folded.records):
        state, session, row = folded.records[bill_id]
        by_archive[(state, session)].append(row)
    stamp = f"{folded.as_of.isoformat()} 00:00:00"
    for (state, session), rows in sorted(by_archive.items()):
        archive = raw / state / f"{state}_{session}_json_days.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        readme = (
            "Open States Data Export\n\n"
            f"State: {state}\nSession: {session}\nGenerated At: {stamp}\n"
            "JSON Format Version: 1.0\n"
        )
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("README", readme)
            z.writestr(f"{state}/{session}/{state}_{session}{BILLS_SUFFIX}", json.dumps(rows))
    return raw


# The change log. Relatives are directed: when two bills match, each gains the other, so a page
# that gains a relative is one entry whichever bill arrived.


class _Change(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of: date
    bill_id: str


class RecordAdded(_Change):
    kind: Literal["added"] = "added"
    state: str
    session: str
    text_sha256: str | None


class RecordUpdated(_Change):
    """Seen again with the same text: its actions, title or status may have changed."""

    kind: Literal["updated"] = "updated"


class RecordReplaced(_Change):
    kind: Literal["replaced"] = "replaced"
    old_sha256: str | None
    new_sha256: str | None


class RelativeGained(_Change):
    kind: Literal["gained"] = "gained"
    relative_id: str
    stages: list[str]


class RelativeLost(_Change):
    kind: Literal["lost"] = "lost"
    relative_id: str


class OriginGained(_Change):
    """A bill found to share wording with a published model document (lineage.origins)."""

    kind: Literal["origin_gained"] = "origin_gained"
    doc_id: str
    stages: list[str]
    bill_share: float
    # Both shares, so the feed applies the listing floor to the larger and states both, as a
    # page does (ruled at M11's review), from the log alone.
    model_share: float


class OriginLost(_Change):
    kind: Literal["origin_lost"] = "origin_lost"
    doc_id: str


Change = Annotated[
    RecordAdded
    | RecordUpdated
    | RecordReplaced
    | RelativeGained
    | RelativeLost
    | OriginGained
    | OriginLost,
    Field(discriminator="kind"),
]
_CHANGE: TypeAdapter[Change] = TypeAdapter(Change)


def write_changes(directory: Path, as_of: date, changes: Sequence[Change]) -> Path:
    """Append a day's changes as `<as_of>.jsonl`. The log is append-only: a day already logged
    may be written again only with the same changes, so re-running a day changes nothing."""
    path = directory / f"{as_of.isoformat()}.jsonl"
    body = "".join(_CHANGE.dump_json(c).decode() + "\n" for c in changes)
    if path.exists():
        if path.read_text() != body:
            raise UpdateError(f"{path} already logs different changes for {as_of}")
        return path
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(body)
    partial.replace(path)
    return path


def read_changes(directory: Path) -> list[Change]:
    """Every logged change, oldest day first, in the order written."""
    out: list[Change] = []
    for path in sorted(directory.glob("*.jsonl")):
        out.extend(_CHANGE.validate_json(line) for line in path.read_text().splitlines() if line)
    return out


def record_changes(before: Folded | None, day: Day) -> list[Change]:
    """What a day did to the records: each one added, updated or replaced."""
    out: list[Change] = []
    for (state, session), rows in sorted(day.records.items()):
        for row in rows:
            bill_id, digest = str(row["id"]), text_hash(row)
            was = before.records.get(bill_id) if before else None
            if was is None:
                out.append(
                    RecordAdded(
                        as_of=day.as_of,
                        bill_id=bill_id,
                        state=state,
                        session=session,
                        text_sha256=digest,
                    )
                )
            elif text_hash(was[2]) != digest:
                out.append(
                    RecordReplaced(
                        as_of=day.as_of,
                        bill_id=bill_id,
                        old_sha256=text_hash(was[2]),
                        new_sha256=digest,
                    )
                )
            else:
                out.append(RecordUpdated(as_of=day.as_of, bill_id=bill_id))
    return sorted(out, key=lambda c: c.bill_id)


# What an update must reproduce.

Relative = tuple[str, tuple[str, ...], int, float]  # (bill, stages, sections matched, share)
Origin = tuple[str, tuple[str, ...], float, float]  # (document, stages, model share, bill share)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What a build shows that an update must reproduce exactly: each bill's status, the
    same-state families as sets of bills (a family's name is a label, not a fact), and each
    matchable bill's relatives in other states with the stages that found them and the share
    of its words they match, rounded below any difference of order of summation."""

    statuses: Mapping[str, str | None]
    families: frozenset[frozenset[str]]
    relatives: Mapping[str, tuple[Relative, ...]]
    origins: Mapping[str, tuple[Origin, ...]] = field(default_factory=dict[str, tuple[Origin, ...]])


SHARE_PLACES = 9


def snapshot(tables_dir: Path, *, model: str = MODEL) -> Snapshot:
    """A build's snapshot, read from its tables the way pages read them (lineage.relatives)."""
    with queries.connect_quiet() as con:
        statuses = queries.bill_statuses(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
        matchable = queries.matchable_bill_ids(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
        )
    members: dict[str, set[str]] = defaultdict(set)
    for bill_id, family in load_families(tables_dir).items():
        members[family].add(bill_id)
    found = bill_relatives(
        tables_dir, matchable, model=model, shown=len(statuses), flag=shipped_flag(tables_dir)
    )
    relatives = {
        bill_id: tuple(
            sorted(
                (
                    m.bill_id,
                    tuple(sorted(m.sections_by_stage)),
                    m.sections_matched,
                    round(m.share, SHARE_PLACES),
                )
                for m in r.matches
            )
        )
        for bill_id, r in found.items()
    }
    origins: dict[str, list[Origin]] = defaultdict(list)
    for o in read_origin_matches(tables_dir):
        origins[o.bill_id].append(
            (
                o.doc_id,
                tuple(sorted(o.stages)),
                round(o.model_share, SHARE_PLACES),
                round(o.bill_share, SHARE_PLACES),
            )
        )
    return Snapshot(
        statuses=statuses,
        families=frozenset(frozenset(m) for m in members.values()),
        relatives=relatives,
        origins={b: tuple(sorted(os)) for b, os in origins.items()},
    )


def relative_changes(before: Snapshot | None, after: Snapshot, as_of: date) -> list[Change]:
    """Every relative a bill gained or lost between two snapshots, by bill then relative."""
    out: list[Change] = []
    was: Mapping[str, tuple[Relative, ...]] = before.relatives if before else {}
    for bill_id in sorted(set(was) | set(after.relatives)):
        old = {r[0]: r for r in was.get(bill_id, ())}
        new = {r[0]: r for r in after.relatives.get(bill_id, ())}
        for rel in sorted(set(new) - set(old)):
            out.append(
                RelativeGained(
                    as_of=as_of, bill_id=bill_id, relative_id=rel, stages=list(new[rel][1])
                )
            )
        for rel in sorted(set(old) - set(new)):
            out.append(RelativeLost(as_of=as_of, bill_id=bill_id, relative_id=rel))
    return out


def origin_changes(before: Snapshot | None, after: Snapshot, as_of: date) -> list[Change]:
    """Every published model a bill gained or lost a match with, by bill then document."""
    out: list[Change] = []
    was: Mapping[str, tuple[Origin, ...]] = before.origins if before else {}
    for bill_id in sorted(set(was) | set(after.origins)):
        old = {o[0]: o for o in was.get(bill_id, ())}
        new = {o[0]: o for o in after.origins.get(bill_id, ())}
        for doc in sorted(set(new) - set(old)):
            out.append(
                OriginGained(
                    as_of=as_of,
                    bill_id=bill_id,
                    doc_id=doc,
                    stages=list(new[doc][1]),
                    bill_share=new[doc][3],
                    model_share=new[doc][2],
                )
            )
        for doc in sorted(set(old) - set(new)):
            out.append(OriginLost(as_of=as_of, bill_id=bill_id, doc_id=doc))
    return out
