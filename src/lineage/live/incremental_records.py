"""The update's records: the day's bills, eligibility, texts and sections written into the
staging view (lineage.live.incremental).

See DESIGN.md#updates, DR-7.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from lineage.core import db
from lineage.data import queries as data_queries
from lineage.data.doctypes import decide, load_rules
from lineage.data.sessions import Session, load_sessions, status_end_date
from lineage.data.status import OUTCOME_CLASSIFICATIONS, STATUS_WORDING, Action, bill_status
from lineage.live import queries
from lineage.live.staging import Staging
from lineage.live.updates import Change, Day, Folded, UpdateError, record_changes
from lineage.text import queries as text_queries
from lineage.text.sectioning import append_sections
from lineage.text.sections import split_sections

# The suffix of the bills file a session the corpus has not seen before is written to, as
# lineage.live.updates.write_archives names its archives.
NEW_ARCHIVE = "json_days"


@dataclass(frozen=True, slots=True)
class Records:
    changes: list[Change]
    fresh: set[str]  # records that take their exported text: added, or their text replaced
    retired: set[str]  # records whose sections before are retired
    resplit: set[str]  # records split again: fresh, or eligible otherwise than before


def _statuses(
    rows: Iterable[tuple[str, list[str], list[dict[str, Any]]]],
    *,
    state: str,
    session_end: date | None,
    as_of: date,
) -> data_queries.StatusColumns:
    columns = data_queries.StatusColumns(id=[], status=[], reason=[], on=[])
    for bill_id, kinds, actions in rows:
        outcome = bill_status(
            [
                Action(
                    date=a["date"],
                    classification=tuple(a["classification"] or ()),
                    order=a["order"],
                    organization=a["organization__name"] or "",
                    description=a["description"] or "",
                )
                for a in actions or []
            ],
            kinds=kinds,
            state=state,
            session_end=session_end,
            as_of=as_of,
        )
        columns.id.append(bill_id)
        columns.status.append(outcome.status.value)
        columns.reason.append(outcome.reason)
        columns.on.append(outcome.on)
    return columns


def _next_part(sections: Path, stem: str) -> str:
    numbers = [
        int(p.stem.rsplit("-", 1)[1])
        for p in sections.glob(f"{stem}-*.parquet")
        if p.stem.rsplit("-", 1)[1].isdigit()
    ]
    return f"{stem}-{max(numbers, default=-1) + 1:03d}"


_Sessions = dict[tuple[str, str], Session]


def _stamp(day: Day) -> str:
    return f"{day.as_of.isoformat()} 00:00:00"


@dataclass(frozen=True, slots=True)
class _Stored:
    """What the tables hold before the day, for the day's records."""

    records: dict[str, tuple[str, str, str | None]]  # id -> (state, session, text)
    files: dict[tuple[str, str], tuple[str, str]]  # (state, session) -> (bills file, archive)
    eligible: dict[str, bool]  # every record's eligibility


def _stored(tables: Path, ids: list[str]) -> _Stored:
    bills_glob = str(tables / "bills" / "*.parquet")
    with db.connect_quiet() as con:
        stored = {
            i: (st, se, text)
            for i, st, se, text in queries.stored_records(con, bills_glob=bills_glob, ids=ids)
        }
        files = {
            (st, se): (name, archive)
            for name, st, se, archive in queries.bills_files(con, bills_glob=bills_glob)
        }
        before_eligible = {
            i: e for i, e, _ in queries.eligibility_rows(con, path=tables / "eligible.parquet")
        }
    return _Stored(stored, files, before_eligible)


def _refuse_moves(day: Day, stored: Mapping[str, tuple[str, str, str | None]]) -> None:
    """A record stays in the session it was filed in."""
    for (state, session), rows in day.records.items():
        for row in rows:
            was = stored.get(str(row["id"]))
            if was is not None and (was[0], was[1]) != (state, session):
                raise UpdateError(f"{row['id']} moved from {was[0]} {was[1]} to {state} {session}")


def _write_day_bills(
    con: duckdb.DuckDBPyConnection,
    tables: Path,
    view: Staging,
    day: Day,
    *,
    files: Mapping[tuple[str, str], tuple[str, str]],
    fresh: set[str],
    sessions: _Sessions,
    scratch: Path,
) -> dict[tuple[str, str], str]:
    """Each archive the day touches: its bills file with the day's records merged in, and its
    texts file; returns each one's bills file name."""
    stamp = _stamp(day)
    names: dict[tuple[str, str], str] = {}
    for (state, session), rows in sorted(day.records.items()):
        name, archive = files.get(
            (state, session),
            (
                f"{state}_{session}_{NEW_ARCHIVE}.parquet",
                f"{state}/{state}_{session}_{NEW_ARCHIVE}.zip",
            ),
        )
        names[(state, session)] = name
        day_json, merged = scratch / "day.json", scratch / "merged.json"
        day_json.write_text(json.dumps(list(rows)))
        old = tables / "bills" / name
        queries.merged_records_json(
            con,
            bills_path=old if old.exists() else None,
            day_path=day_json,
            out_path=merged,
            replaced=[str(r["id"]) for r in rows],
        )
        status = _statuses(
            data_queries.outcome_actions(
                con,
                json_path=merged,
                kinds=sorted(OUTCOME_CLASSIFICATIONS),
                wording=STATUS_WORDING,
            ),
            state=state,
            session_end=status_end_date(sessions, state, session),
            as_of=day.as_of,
        )
        data_queries.copy_bills_to_parquet(
            con,
            json_path=merged,
            parquet_path=view.fresh(f"bills/{name}"),
            state=state,
            generated_at=stamp,
            archive=archive,
            status=status,
        )
        texts = tables / "texts" / name
        queries.merge_texts_file(
            con,
            texts_path=texts if texts.exists() else None,
            bills_path=view.root / "bills" / name,
            fresh=sorted(fresh & {str(r["id"]) for r in rows}),
            out_path=view.fresh(f"texts/{name}"),
        )
    return names


def _restamp_other_bills(
    con: duckdb.DuckDBPyConnection, tables: Path, view: Staging, day: Day, sessions: _Sessions
) -> None:
    """Every bills file the day does not touch, stamped as of the day, its statuses judged on
    it."""
    for name, state, session, _ in queries.bills_files(
        con, bills_glob=str(tables / "bills" / "*.parquet")
    ):
        if (state, session) in day.records:
            continue
        rows_of = queries.stored_outcome_actions(
            con,
            bills_path=tables / "bills" / name,
            kinds=sorted(OUTCOME_CLASSIFICATIONS),
            wording=STATUS_WORDING,
        )
        queries.restamp_bills(
            con,
            bills_path=tables / "bills" / name,
            out_path=view.fresh(f"bills/{name}"),
            generated_at=_stamp(day),
            status=_statuses(
                rows_of,
                state=state,
                session_end=status_end_date(sessions, state, session),
                as_of=day.as_of,
            ),
        )


def _write_eligibility(
    con: duckdb.DuckDBPyConnection, tables: Path, view: Staging, ids: list[str], rules_path: Path
) -> dict[str, tuple[bool, int | None]]:
    """The day's records decided again, every other record's kept; returns every record's."""
    rules = load_rules(rules_path)
    decided = {
        r.id: decide(rules, r.state, r.type, r.title)
        for r in queries.record_types_of(
            con, bills_glob=str(view.root / "bills" / "*.parquet"), ids=ids
        )
    }
    eligibility = {
        i: (e, rule)
        for i, e, rule in queries.eligibility_rows(con, path=tables / "eligible.parquet")
    }
    for bill_id, rule in decided.items():
        eligibility[bill_id] = (rule is not None and rule.match, rule.number if rule else None)
    ordered = sorted(eligibility)
    data_queries.write_eligibility(
        con,
        parquet_path=view.fresh("eligible.parquet"),
        ids=ordered,
        eligible=[eligibility[i][0] for i in ordered],
        rules=[eligibility[i][1] for i in ordered],
    )
    return eligibility


def _write_sections(
    tables: Path,
    view: Staging,
    day: Day,
    names: Mapping[tuple[str, str], str],
    *,
    retired: set[str],
    resplit: set[str],
) -> dict[str, str]:
    """The retired records' rows dropped from the sections files holding them, and the records
    split again written to one new file per archive; returns each new file's name by its
    archive's bills file."""
    new_parts: dict[str, str] = {}
    with db.connect_quiet() as con:
        holding = (
            queries.files_holding(
                con,
                glob=str(tables / "sections" / "*.parquet"),
                column="bill_id",
                ids=sorted(retired),
            )
            if retired
            else []
        )
        for part in holding:
            kept = queries.copy_without(
                con,
                path=tables / "sections" / part,
                out_path=view.fresh(f"sections/{part}"),
                column="bill_id",
                ids=sorted(retired),
            )
            if not kept:
                view.remove(f"sections/{part}")
        for (state, session), name in sorted(names.items()):
            wanted = sorted(resplit & {str(r["id"]) for r in day.records[(state, session)]})
            if not wanted:
                continue
            found = queries.texts_of(con, texts_path=view.root / "texts" / name, ids=wanted)
            columns = text_queries.SectionColumns([], [], [], [], [], [], [])
            for bill_id, text in found:
                split = split_sections(text, state)
                append_sections(columns, bill_id, split)
            if columns.bill_id:
                part = _next_part(tables / "sections", Path(name).stem)
                text_queries.write_sections(
                    con, parquet_path=view.fresh(f"sections/{part}.parquet"), columns=columns
                )
                new_parts[name] = part
    return new_parts


def update_records(
    tables: Path, view: Staging, day: Day, *, sessions_path: Path, rules_path: Path, scratch: Path
) -> tuple[Records, dict[str, str]]:
    """Write the day's bills, eligibility, texts and sections into the view; return what it did
    to the records, and each new sections file's name by the archive it belongs to."""
    sessions = load_sessions(sessions_path)
    ids = sorted(str(r["id"]) for rows in day.records.values() for r in rows)
    stored = _stored(tables, ids)
    _refuse_moves(day, stored.records)
    before = Folded(
        as_of=day.as_of,
        records={i: (st, se, {"raw_text": text}) for i, (st, se, text) in stored.records.items()},
        versions=[],
    )
    changes = record_changes(before, day)
    fresh = {c.bill_id for c in changes if c.kind in ("added", "replaced")}

    view.own("bills")
    view.own("texts")
    with db.connect_quiet() as con:
        names = _write_day_bills(
            con,
            tables,
            view,
            day,
            files=stored.files,
            fresh=fresh,
            sessions=sessions,
            scratch=scratch,
        )
        _restamp_other_bills(con, tables, view, day, sessions)
        eligibility = _write_eligibility(con, tables, view, ids, rules_path)
    flipped = {
        i
        for i in ids
        if i in stored.eligible and i not in fresh and stored.eligible[i] != eligibility[i][0]
    }
    retired = {i for i in ids if i in stored.records} & (fresh | flipped)
    resplit = fresh | flipped
    new_parts = _write_sections(tables, view, day, names, retired=retired, resplit=resplit)
    return Records(changes, fresh, retired, resplit), new_parts
