"""Lane L1's incremental core: one day applied to a built corpus, matched in both directions.

`apply_day` meets `update`, the contract (DESIGN.md#updates, DR-7). It works in three parts:

- **Records.** The day's records join their archives' bills files through the reader ingest
  uses, and every bills file is stamped as of the day, so each status is judged on it. A
  record added or whose text changed takes its exported text, is split, and its sections go
  into a new sections file; a replaced record, or one whose document type now decides its
  eligibility otherwise, has its earlier sections retired.
- **Matching.** Only the new sections are signed and embedded. The band index takes the new
  rows and loses the retired ones, and the section pairs whose candidacy that can change,
  under the bucket cap applied exactly, are recomputed from every pair of their windows. New
  texts are searched exactly against every stored vector, and each stored neighbor list they
  enter, or that loses a retired text, is brought to what a build keeps under the cap of K.
  The families that can merge or split are regrouped, and the semantic stage's section pairs
  around every changed neighbor are expanded again. A replaced text's sections, signatures,
  bucket entries, windows and vectors are retired: a vector no window carries is zeroed.
- **The log.** Each bill whose relatives can have changed is listed before and after, as a
  page lists it, and the differences, with `record_changes` and the origin matches' own,
  are the day's changes (lineage.live.updates.write_changes).

The stoplist and the state boilerplate flag are the base build's, never recounted (DR-7): a
new section takes no flag until the next full build counts it. Where the tables hold origin
matches, the model documents are matched again over the updated tables with `lineage
origins`' own matching (lineage.origins.matching), and the differences logged.

Every table is written into a staging view of the tables, in which an unchanged file is a link
to the one in place. The view replaces the tables only once everything is built and the day's
log is written, so the bills listed before are read from the tables as they stood, and an
update that stops before then leaves them as they were. One that stops while the view replaces
them leaves a mark, and the next update refuses the tables. A day applied is journaled with a
digest of its records: the same day again changes nothing, and other records under its date
are refused.

Owned by lane L1 (M10). See DESIGN.md#updates, DR-7.
"""

import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lineage.core import db
from lineage.core.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.data import queries as data_queries
from lineage.data.doctypes import RULES, decide, load_rules
from lineage.data.sessions import load_sessions, status_end_date
from lineage.data.status import OUTCOME_CLASSIFICATIONS, STATUS_WORDING, Action, bill_status
from lineage.live import queries
from lineage.live.updates import (
    CHANGES_DIR,
    SHARE_PLACES,
    Change,
    Day,
    Folded,
    Origin,
    Relative,
    Snapshot,
    UpdateError,
    origin_changes,
    record_changes,
    relative_changes,
    write_changes,
)
from lineage.match.listing import shipped_flag
from lineage.match.near import queries as near_queries
from lineage.match.near.bands import load_band_index, update_bands
from lineage.match.near.families import update_families
from lineage.match.near.minhash import NUM_PERM
from lineage.match.near.pairs import (
    SectionPairs,
    load_cross_state_pairs,
    load_families,
    load_same_state_pairs,
    pair_codes,
    recompute_section_pairs,
    renumbered,
    replaced_pairs,
    section_bounds,
    sections_of_window_pairs,
    write_section_pairs,
)
from lineage.match.near.signatures import (
    open_signatures,
    row_labels,
    sign_part,
)
from lineage.match.near.stoplist import load_stoplist
from lineage.match.relatives import bill_relatives
from lineage.match.semantic import queries as semantic_queries
from lineage.match.semantic.models import MODELS, MODELS_DIR, require_model
from lineage.match.semantic.neighbors import update_neighbors
from lineage.match.semantic.pairs import (
    SemanticPairs,
    changed_texts,
    load_built_semantic_pairs,
    pairs_dir,
    renumbered_semantic,
    update_semantic_pairs,
    write_semantic_pairs,
)
from lineage.match.semantic.stage import BUILD_THRESHOLD, MODEL, Embedder, K
from lineage.match.semantic.vectors import (
    SEMANTIC_DIR,
    extend_store,
    load_plan,
    load_vectors,
    semantic_dir,
    text_count,
    vectors_dir,
    write_plan,
)
from lineage.origins.matches import (
    OriginMatch,
    origins_path,
    read_origin_matches,
    write_origin_matches,
)
from lineage.text import queries as text_queries
from lineage.text.boilerplate import boilerplate_kind
from lineage.text.sections import normalize, split_sections
from lineage.text.windows import WINDOW_WORDS, display_windows

SESSIONS = Path(__file__).parents[3] / "snapshot" / "sessions.csv"
UPDATES_DIR = "updates"
APPLIED = "applied.jsonl"
PENDING = "pending.json"
# The suffix of the bills file a session the corpus has not seen before is written to, as
# lineage.live.updates.write_archives names its archives.
NEW_ARCHIVE = "json_days"


class NoEmbedderError(UpdateError, NotImplementedError):
    """No embedder for the tables' model: the day's new windows cannot be embedded. It is also
    a NotImplementedError, the error `lineage update` reports for a step not built yet."""


_EMBEDDERS: dict[str, Embedder] = {}


def register_embedder(model: str, embed: Embedder) -> None:
    """Embed `model`'s windows with `embed`: a model `lineage.match.semantic.models` does not
    pin, such as the tests' stand-in, which the tables were embedded with."""
    _EMBEDDERS[model] = embed


def unregister_embedder(model: str) -> None:
    _EMBEDDERS.pop(model, None)


def embedder_for(model: str, models_dir: Path = MODELS_DIR) -> Callable[[], Embedder]:
    """A loader of `model`'s embedder, refused now if there is none, so an update stops before
    it writes anything; the model itself is loaded only when a day brings a new text."""
    if model in _EMBEDDERS:
        registered = _EMBEDDERS[model]
        return lambda: registered
    if model not in MODELS:
        raise NoEmbedderError(f"no embedder for model {model!r}; known: {', '.join(MODELS)}")
    require_model(model, models_dir)

    def load() -> Embedder:
        from lineage.match.semantic.embed import embed, load_model  # fastembed is slow to import

        loaded = load_model(model, models_dir)
        return lambda texts: embed(loaded, texts)

    return load


OriginMatcher = Callable[[Path], list[OriginMatch]]


def _default_origins(model: str, floor_bytes: int) -> OriginMatcher:
    def match(tables_dir: Path) -> list[OriginMatch]:
        from lineage.origins.collect import collectors
        from lineage.origins.matching import match_models
        from lineage.origins.modeldocs import MODELDOCS_DIR, read_docs

        docs = [d for c in collectors().values() for d in read_docs(c.source, MODELDOCS_DIR)]
        return match_models(
            tables_dir, docs, modeldocs_root=MODELDOCS_DIR, model=model, floor_bytes=floor_bytes
        )

    return match


# ---- The journal ----------------------------------------------------------------------------


def day_digest(day: Day) -> str:
    """What a day holds, as one SHA-256: its date, sources and records."""
    doc = {
        "as_of": day.as_of.isoformat(),
        "sources": list(day.sources),
        "records": {f"{st}_{se}": list(rows) for (st, se), rows in sorted(day.records.items())},
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def applied_days(tables_dir: Path) -> dict[date, str]:
    """Each day applied to the tables, with its digest."""
    path = tables_dir / UPDATES_DIR / APPLIED
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {date.fromisoformat(r["as_of"]): r["sha256"] for r in rows}


def _journal(tables_dir: Path, day: Day, digest: str) -> None:
    path = tables_dir / UPDATES_DIR / APPLIED
    with path.open("a") as f:
        f.write(json.dumps({"as_of": day.as_of.isoformat(), "sha256": digest}) + "\n")


# ---- The staging view -----------------------------------------------------------------------


class Staging:
    """The tables after an update, built beside them: every entry starts as a link to the one
    in place, a directory written into is made real with links to its entries, and a file
    written replaces its link. `commit` moves what was written into place and removes what was
    dropped, so the tables change only at the end."""

    def __init__(self, tables_dir: Path, root: Path) -> None:
        self.tables = tables_dir
        self.root = root
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        for entry in sorted(tables_dir.iterdir()):
            if entry.name != UPDATES_DIR:
                (root / entry.name).symlink_to(entry.resolve())

    def own(self, rel: str) -> Path:
        """`rel` as a real directory in the view, its entries links to those in place. Its
        parents are made real first, so nothing below a link is ever written into."""
        parent = rel.rpartition("/")[0]
        if parent:
            self.own(parent)
        path = self.root / rel
        if path.is_symlink():
            target = path.resolve()
            path.unlink()
            path.mkdir()
            for entry in sorted(target.iterdir()):
                (path / entry.name).symlink_to(entry.resolve())
        elif not path.exists():
            path.mkdir()
        return path

    def fresh(self, rel: str) -> Path:
        """A path to write the file `rel` at, never through a link to the file in place."""
        parent, _, _ = rel.rpartition("/")
        if parent:
            self.own(parent)
        path = self.root / rel
        if path.is_symlink():
            path.unlink()
        return path

    def fresh_dir(self, rel: str) -> Path:
        """An empty real directory at `rel`, replacing the table in place on commit."""
        path = self.fresh(rel)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        return path

    def remove(self, rel: str) -> None:
        path = self.fresh(rel)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def commit(self) -> None:
        self._commit(self.root, self.tables)
        shutil.rmtree(self.root)

    def _commit(self, view: Path, place: Path) -> None:
        if place.is_symlink():
            # A table shared with another build by a link: what is unchanged stays shared.
            place.unlink()
        place.mkdir(parents=True, exist_ok=True)
        names = {p.name for p in view.iterdir()}
        for old in place.iterdir():
            if old.name not in names and not (place == self.tables and old.name == UPDATES_DIR):
                if old.is_dir() and not old.is_symlink():
                    shutil.rmtree(old)
                else:
                    old.unlink()
        for entry in sorted(view.iterdir()):
            target = place / entry.name
            if entry.is_symlink():
                if not target.exists() or target.resolve() != entry.resolve():
                    if target.is_symlink() or target.is_file():
                        target.unlink()
                    target.symlink_to(entry.resolve())
            elif entry.is_dir():
                if target.exists() and not target.is_dir():
                    target.unlink()
                self._commit(entry, target)
            else:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                entry.replace(target)


# ---- Records --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Records:
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


def _records(
    tables: Path, view: Staging, day: Day, *, sessions_path: Path, rules_path: Path, scratch: Path
) -> tuple[_Records, dict[str, str]]:
    """Write the day's bills, eligibility, texts and sections into the view; return what it did
    to the records, and each new sections file's name by the archive it belongs to."""
    stamp = f"{day.as_of.isoformat()} 00:00:00"
    sessions = load_sessions(sessions_path)
    bills_glob = str(tables / "bills" / "*.parquet")
    ids = sorted(str(r["id"]) for rows in day.records.values() for r in rows)
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
    for (state, session), rows in day.records.items():
        for row in rows:
            was = stored.get(str(row["id"]))
            if was is not None and (was[0], was[1]) != (state, session):
                raise UpdateError(f"{row['id']} moved from {was[0]} {was[1]} to {state} {session}")
    before = Folded(
        as_of=day.as_of,
        records={i: (st, se, {"raw_text": text}) for i, (st, se, text) in stored.items()},
        versions=[],
    )
    changes = record_changes(before, day)
    fresh = {c.bill_id for c in changes if c.kind in ("added", "replaced")}

    view.own("bills")
    view.own("texts")
    by_archive: dict[tuple[str, str], Sequence[Mapping[str, Any]]] = dict(day.records)
    names: dict[tuple[str, str], str] = {}
    with db.connect_quiet() as con:
        for (state, session), rows in sorted(by_archive.items()):
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
        for name, state, session, _ in queries.bills_files(con, bills_glob=bills_glob):
            if (state, session) in by_archive:
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
                generated_at=stamp,
                status=_statuses(
                    rows_of,
                    state=state,
                    session_end=status_end_date(sessions, state, session),
                    as_of=day.as_of,
                ),
            )

        # Eligibility: the day's records decided again, every other record's kept.
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
    flipped = {
        i
        for i in ids
        if i in before_eligible and i not in fresh and before_eligible[i] != eligibility[i][0]
    }
    retired = {i for i in ids if i in stored} & (fresh | flipped)
    resplit = fresh | flipped

    # Sections: the retired records' rows dropped from the files holding them, and the records
    # split again written to one new file per archive.
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
            wanted = sorted(resplit & {str(r["id"]) for r in by_archive[(state, session)]})
            if not wanted:
                continue
            found = queries.texts_of(con, texts_path=view.root / "texts" / name, ids=wanted)
            columns = text_queries.SectionColumns([], [], [], [], [], [], [])
            for bill_id, text in found:
                split = split_sections(text, state)
                for s in split.sections:
                    normalized = normalize(s.text)
                    columns.bill_id.append(bill_id)
                    columns.ordinal.append(s.ordinal)
                    columns.heading.append(s.heading)
                    columns.text.append(s.text)
                    columns.normalized.append(normalized)
                    columns.by_heading.append(split.by_heading)
                    columns.boilerplate.append(boilerplate_kind(normalized))
            if columns.bill_id:
                part = _next_part(tables / "sections", Path(name).stem)
                text_queries.write_sections(
                    con, parquet_path=view.fresh(f"sections/{part}.parquet"), columns=columns
                )
                new_parts[name] = part
    return _Records(changes, fresh, retired, resplit), new_parts


# ---- The near-duplicate stage ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Renumbering:
    row_map: NDArray[np.int64]  # each old signature row's new number, -1 where retired
    section_map: NDArray[np.int64]  # each old section's new number, -1 where retired
    fresh_rows: NDArray[np.int64]  # the new rows' numbers
    fresh_values: NDArray[np.uint32]


def _signatures(
    tables: Path, view: Staging, retired: set[str], new_parts: Iterable[str]
) -> _Renumbering:
    """Drop the retired records' signature rows, sign the new sections files, and number the
    rows and sections afresh in file order, as a build numbers them."""
    dest = view.own("signatures")
    old_files = sorted((tables / "signatures").glob("*.npy"))
    kept = {p.stem: np.ones(int(np.load(p, mmap_mode="r").shape[0]), dtype=bool) for p in old_files}
    with db.connect_quiet() as con:
        # Only the files that hold a retired record's rows are read and written again.
        holding = (
            queries.files_holding(
                con,
                glob=str(tables / "signatures" / "*.parquet"),
                column="bill_id",
                ids=sorted(retired),
            )
            if retired
            else []
        )
        for name in holding:
            path = (tables / "signatures" / name).with_suffix(".npy")
            ids, ordinals, windows = near_queries.read_signature_keys(
                con, parquet_path=path.with_suffix(".parquet")
            )
            keep = np.fromiter((b not in retired for b in ids), dtype=bool, count=len(ids))
            kept[path.stem] = keep
            view.remove(f"signatures/{path.name}")
            view.remove(f"signatures/{path.stem}.parquet")
            if keep.any():
                np.save(dest / path.name, np.asarray(np.load(path, mmap_mode="r"))[keep])
                at = np.flatnonzero(keep).tolist()
                near_queries.write_signature_keys(
                    con,
                    parquet_path=dest / f"{path.stem}.parquet",
                    bill_ids=[ids[i] for i in at],
                    ordinals=[ordinals[i] for i in at],
                    windows=[windows[i] for i in at],
                )
    stoplist = load_stoplist(tables)
    added: set[str] = set()
    for part in new_parts:
        sign_part(
            view.root / "sections" / f"{part}.parquet",
            view.root / "eligible.parquet",
            stoplist,
            dest,
        )
        if (dest / f"{part}.npy").exists():
            added.add(part)

    new_files = sorted(dest.glob("*.npy"))
    old_rows = [kept[p.stem].size for p in old_files]
    row_map = np.full(int(sum(old_rows)), -1, dtype=np.int64)
    starts = np.cumsum([0, *old_rows]).tolist()
    old_start = {p.stem: int(starts[i]) for i, p in enumerate(old_files)}
    fresh_rows: list[NDArray[np.int64]] = []
    fresh_values: list[NDArray[np.uint32]] = []
    at = 0
    for path in new_files:
        rows = int(np.load(path, mmap_mode="r").shape[0])
        if path.stem in added:
            fresh_rows.append(np.arange(at, at + rows, dtype=np.int64))
            fresh_values.append(cast("NDArray[np.uint32]", np.load(path)))
        else:
            keep = kept[path.stem]
            start = old_start[path.stem]
            row_map[start + np.flatnonzero(keep)] = at + np.arange(rows, dtype=np.int64)
        at += rows
    before = row_labels(open_signatures(tables))
    after = row_labels(open_signatures(view.root))
    first_rows = (
        np.flatnonzero(np.r_[True, np.diff(before.section) != 0])
        if before.section.size
        else np.empty(0, dtype=np.int64)
    )
    mapped = row_map[first_rows]
    section_map = np.where(mapped >= 0, after.section[np.maximum(mapped, 0)], -1).astype(np.int64)
    values = (
        np.concatenate(fresh_values) if fresh_values else np.empty((0, NUM_PERM), dtype=np.uint32)
    )
    rows_new = np.concatenate(fresh_rows) if fresh_rows else np.empty(0, dtype=np.int64)
    return _Renumbering(row_map, section_map, rows_new, values)


@dataclass(frozen=True, slots=True)
class _NearDuplicates:
    cross_before: SectionPairs
    cross_after: SectionPairs
    families_before: dict[str, str]
    families_after: dict[str, str]


def _near_duplicates(
    tables: Path, view: Staging, numbers: _Renumbering, resplit: set[str]
) -> _NearDuplicates:
    """The band index, both pairs tables and the families after the day, into the view."""
    bands = view.fresh_dir("bands")
    codes = update_bands(
        load_band_index(tables), numbers.row_map, numbers.fresh_values, numbers.fresh_rows, bands
    )
    after = row_labels(open_signatures(view.root))
    before = row_labels(open_signatures(tables))
    left, right = sections_of_window_pairs(codes, after.section)
    found = recompute_section_pairs(
        open_signatures(view.root),
        load_band_index(view.root),
        section_bounds(after.section),
        left,
        right,
    )
    with db.connect_quiet() as con:
        state_of = near_queries.bill_states(con, bills_glob=str(view.root / "bills" / "*.parquet"))
    states = np.array([state_of[b] for b in after.section_bills], dtype=object)
    cross = states[found.left] != states[found.right] if found.left.size else np.empty(0, bool)
    recomputed = pair_codes(left, right)
    pairs: dict[str, tuple[SectionPairs, SectionPairs]] = {}
    for name, old, mask in (
        ("pairs", load_cross_state_pairs(tables), cross),
        ("same_state_pairs", load_same_state_pairs(tables), ~cross),
    ):
        part = SectionPairs(found.left[mask], found.right[mask], found.agreements[mask])
        new = replaced_pairs(renumbered(old, numbers.section_map), recomputed, part)
        write_section_pairs(view.fresh_dir(name), new)
        pairs[name] = (old, new)

    old_same, new_same = pairs["same_state_pairs"]
    carried = renumbered(old_same, numbers.section_map)
    changed = np.setxor1d(
        pair_codes(carried.left, carried.right), pair_codes(new_same.left, new_same.right)
    )
    affected = set(resplit)
    for code in changed.tolist():
        affected |= {after.section_bills[code >> 32], after.section_bills[code & 0xFFFF_FFFF]}
    gone = numbers.section_map < 0
    lost = gone[np.asarray(old_same.left)] | gone[np.asarray(old_same.right)]
    for s in np.concatenate(
        [np.asarray(old_same.left)[lost], np.asarray(old_same.right)[lost]]
    ).tolist():
        affected.add(before.section_bills[int(s)])
    bills = np.unique(np.array(after.section_bills, dtype=str))
    bill_of_section = np.searchsorted(bills, np.array(after.section_bills, dtype=str)).astype(
        np.int32
    )
    families_before = load_families(tables)
    families_after = update_families(
        families_before, bills, bill_of_section, new_same.left, new_same.right, affected
    )
    view.fresh_dir("families")
    with db.connect_quiet() as con:
        near_queries.write_families(
            con,
            parquet_path=view.root / "families" / "families.parquet",
            bill_ids=list(families_after),
            families=list(families_after.values()),
        )
    _carry_state_boilerplate(tables, view, numbers.section_map, len(after.section_bills))
    cross_old, cross_new = pairs["pairs"]
    return _NearDuplicates(cross_old, cross_new, families_before, families_after)


def _carry_state_boilerplate(
    tables: Path, view: Staging, section_map: NDArray[np.int64], sections: int
) -> None:
    """The base build's state boilerplate counts under the new section numbers (DR-7): pinned,
    so a new section counts no families until the next full build counts it."""
    path = tables / "state_boilerplate" / "families_reached.npy"
    if not path.exists():
        return
    old = np.load(path)
    counts = np.zeros(sections, dtype=old.dtype)
    kept = section_map >= 0
    counts[section_map[kept]] = old[kept]
    np.save(view.fresh("state_boilerplate/families_reached.npy"), counts)


# ---- The semantic stage ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Semantic:
    before: SemanticPairs
    carried: SemanticPairs
    after: SemanticPairs


def _semantic(
    tables: Path,
    view: Staging,
    numbers: _Renumbering,
    retired: set[str],
    new_parts: Iterable[str],
    *,
    model: str,
    embedder: Callable[[], Embedder],
    floor_bytes: int,
) -> _Semantic:
    """The windows, vectors, neighbors and section pairs of the semantic stage after the day."""
    plan = load_plan(tables)
    if plan is None:
        raise UpdateError(f"no semantic plan in {semantic_dir(tables)}; run `lineage embed`")
    units_rel = f"{SEMANTIC_DIR}/units"
    view.own(units_rel)
    view.own(f"{SEMANTIC_DIR}/texts")
    view.own(f"{SEMANTIC_DIR}/{model}")
    units_glob_before = str(semantic_dir(tables) / "units" / "*.parquet")
    candidates: set[int] = set()
    with db.connect_quiet() as con:
        holding = (
            queries.files_holding(
                con, glob=units_glob_before, column="bill_id", ids=sorted(retired)
            )
            if retired
            else []
        )
        for part in holding:
            path = semantic_dir(tables) / "units" / part
            candidates |= set(queries.unit_text_ids(con, units_path=path, ids=sorted(retired)))
            kept = queries.copy_without(
                con,
                path=path,
                out_path=view.fresh(f"{units_rel}/{part}"),
                column="bill_id",
                ids=sorted(retired),
            )
            if not kept:
                view.remove(f"{units_rel}/{part}")

        # The new sections' windows, as the plan cuts them, each text with the id it has.
        cut: list[tuple[str, str, int, int, str]] = []  # (part, bill, ordinal, window, text)
        for part in new_parts:
            scan = near_queries.matchable_sections(
                con,
                sections_path=view.root / "sections" / f"{part}.parquet",
                eligible_path=view.root / "eligible.parquet",
                display=True,
            )
            for bill_id, ordinal, text, _ in scan.fetchall():
                for number, window in enumerate(
                    display_windows(text, size=WINDOW_WORDS, stride=WINDOW_WORDS)
                ):
                    cut.append((part, bill_id, ordinal, number, window))
        known = (
            queries.semantic_text_ids(
                con,
                texts_glob=str(semantic_dir(tables) / "texts" / "*.parquet"),
                texts=sorted({w for *_, w in cut}),
            )
            if cut
            else {}
        )
        first = text_count(tables, model, plan)
        new_texts: list[str] = []
        ids: dict[str, int] = dict(known)
        for *_, window in cut:
            if window not in ids:
                ids[window] = first + len(new_texts)
                new_texts.append(window)
        by_part: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
        for part, bill_id, ordinal, number, window in cut:
            by_part[part].append((bill_id, ordinal, number, ids[window]))
        for part, rows in by_part.items():
            semantic_queries.write_semantic_units(
                con,
                parquet_path=view.fresh(f"{units_rel}/{part}.parquet"),
                bill_ids=[r[0] for r in rows],
                ordinals=[r[1] for r in rows],
                windows=[r[2] for r in rows],
                text_ids=[r[3] for r in rows],
            )
        units_glob = str(view.root / units_rel / "*.parquet")
        still = (
            set(queries.carried_text_ids(con, units_glob=units_glob, text_ids=sorted(candidates)))
            if candidates
            else set[int]()
        )
        retired_texts = sorted(candidates - still)
        windows = queries.row_count(con, glob=units_glob)

    shards = load_vectors(tables, model, plan)
    dim = int(shards[0].shape[1]) if shards else 0
    if new_texts:
        vectors = embedder()(new_texts).astype(np.float16)
    else:
        vectors = np.empty((0, dim), dtype=np.float16)
    require_free_space(view.root, needed=int(vectors.nbytes) * 2 + (1 << 20), floor=floor_bytes)
    _, shard_count = extend_store(view.root, model, plan, new_texts, vectors, retired_texts)
    total = first + len(new_texts)
    after_plan = write_plan(view.root, windows=windows, distinct=total, shards=shard_count)

    where = vectors_dir(tables, model) / "neighbors"
    ids_before = np.asarray(np.load(where / "ids.npy"))
    scores_before = np.asarray(np.load(where / "scores.npy"))
    blocks = load_vectors(view.root, model, after_plan)
    ids_after, scores_after, changed = update_neighbors(
        blocks,
        ids_before,
        scores_before,
        fresh=np.arange(first, total, dtype=np.int64),
        retired=np.array(retired_texts, dtype=np.int64),
        k=int(ids_before.shape[1]) if ids_before.ndim == 2 else K,
        threshold=BUILD_THRESHOLD,
    )
    rel = f"{SEMANTIC_DIR}/{model}/neighbors"
    view.own(rel)
    np.save(view.fresh(f"{rel}/ids.npy"), ids_after)
    np.save(view.fresh(f"{rel}/scores.npy"), scores_after)

    before, built = load_built_semantic_pairs(tables, model)
    carried = renumbered_semantic(before, numbers.section_map)
    unit_texts = np.array(sorted({ids[w] for *_, w in cut}), dtype=np.int64)
    texts = changed_texts(ids_before, ids_after, changed, unit_texts)
    after = update_semantic_pairs(view.root, carried, texts, ids_after, scores_after)
    write_semantic_pairs(
        view.fresh_dir(str(pairs_dir(view.root, model).relative_to(view.root))), after, built
    )
    return _Semantic(before, carried, after)


# ---- Relatives and origins ------------------------------------------------------------------


def _bills_of(sections: Iterable[int], section_bills: Sequence[str]) -> set[str]:
    return {section_bills[s] for s in sections}


def _changed_sections(
    old_left: NDArray[Any],
    old_right: NDArray[Any],
    old_value: NDArray[Any],
    new_left: NDArray[Any],
    new_right: NDArray[Any],
    new_value: NDArray[Any],
) -> set[int]:
    """Sections (new numbers) of every pair added, dropped or changed in value."""
    old = set(
        zip(
            np.asarray(old_left).tolist(),
            np.asarray(old_right).tolist(),
            np.asarray(old_value).tolist(),
            strict=True,
        )
    )
    new = set(
        zip(
            np.asarray(new_left).tolist(),
            np.asarray(new_right).tolist(),
            np.asarray(new_value).tolist(),
            strict=True,
        )
    )
    return {s for a, b, _ in old ^ new for s in (a, b)}


def _relatives(tables: Path, bills: set[str], model: str) -> dict[str, tuple[Relative, ...]]:
    """The listed relatives of the wanted matchable bills, as `lineage.live.updates.snapshot` reads
    them."""
    with db.connect_quiet() as con:
        matchable = set(
            queries.matchable_bill_ids(
                con,
                sections_glob=str(tables / "sections" / "*.parquet"),
                eligible_path=tables / "eligible.parquet",
            )
        )
    wanted = sorted(bills & matchable)
    if not wanted:
        return {}
    found = bill_relatives(
        tables, wanted, model=model, shown=sys.maxsize, flag=shipped_flag(tables)
    )
    return {
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


def _origins(tables: Path) -> dict[str, tuple[Origin, ...]]:
    """Each bill's origin matches, as `lineage.live.updates.snapshot` reads them."""
    out: dict[str, list[Origin]] = defaultdict(list)
    for o in read_origin_matches(tables):
        out[o.bill_id].append(
            (
                o.doc_id,
                tuple(sorted(o.stages)),
                round(o.model_share, SHARE_PLACES),
                round(o.bill_share, SHARE_PLACES),
            )
        )
    return {b: tuple(sorted(os)) for b, os in out.items()}


def _members(families: Mapping[str, str]) -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for bill, family in families.items():
        groups[family].add(bill)
    return {bill: frozenset(groups[family]) for bill, family in families.items()}


def _related_bills(
    tables: Path,
    view: Path,
    day_ids: set[str],
    near: _NearDuplicates,
    semantic: _Semantic,
    section_map: NDArray[np.int64],
) -> set[str]:
    """Every bill whose relatives the day can have changed: its own records; the bills of each
    section pair, of either stage, the day added, dropped or changed; and the bills paired with
    a bill whose family changed, since relatives are grouped by family."""
    before = row_labels(open_signatures(tables)).section_bills
    after = row_labels(open_signatures(view)).section_bills
    out = set(day_ids)
    gone = section_map < 0
    old_pairs: list[tuple[NDArray[Any], NDArray[Any]]] = [
        (np.asarray(near.cross_before.left), np.asarray(near.cross_before.right)),
        (np.asarray(semantic.before.left), np.asarray(semantic.before.right)),
    ]
    for left, right in old_pairs:
        lost = gone[left] | gone[right]
        out |= _bills_of(np.concatenate([left[lost], right[lost]]).tolist(), before)
    carried_near = renumbered(near.cross_before, section_map)
    out |= _bills_of(
        _changed_sections(
            carried_near.left,
            carried_near.right,
            carried_near.agreements,
            near.cross_after.left,
            near.cross_after.right,
            near.cross_after.agreements,
        ),
        after,
    )
    out |= _bills_of(
        _changed_sections(
            semantic.carried.left,
            semantic.carried.right,
            semantic.carried.cosine,
            semantic.after.left,
            semantic.after.right,
            semantic.after.cosine,
        ),
        after,
    )
    was, now = _members(near.families_before), _members(near.families_after)
    regrouped = {b for b in set(was) | set(now) if was.get(b) != now.get(b)}
    out |= regrouped
    for pairs, bills in (
        ((near.cross_before.left, near.cross_before.right), before),
        ((semantic.before.left, semantic.before.right), before),
        ((near.cross_after.left, near.cross_after.right), after),
        ((semantic.after.left, semantic.after.right), after),
    ):
        left, right = np.asarray(pairs[0]), np.asarray(pairs[1])
        mine = np.fromiter((b in regrouped for b in bills), dtype=bool, count=len(bills))
        if not mine.any() or not left.size:
            continue
        touch = mine[left] | mine[right]
        out |= _bills_of(np.concatenate([left[touch], right[touch]]).tolist(), bills)
    return out


# ---- The update -----------------------------------------------------------------------------


# What an update writes whole: every bills file is stamped again, and the band index, both pairs
# tables, the families, the eligibility table and the pinned counts are written again.
_WRITTEN_WHOLE = (
    "bills",
    "bands",
    "pairs",
    "same_state_pairs",
    "families",
    "eligible.parquet",
    "state_boilerplate",
)


def _bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def _staged_bytes(tables_dir: Path, day: Day, model: str) -> int:
    """An upper bound on what an update stages beside the tables: the tables it writes whole,
    the neighbor lists and semantic pairs, every file of an archive the day touches, two vector
    shards, and room for the day's records several times over (their JSON, texts, sections,
    windows and vectors)."""
    whole = sum(_bytes(tables_dir / name) for name in _WRITTEN_WHOLE)
    whole += _bytes(vectors_dir(tables_dir, model) / "neighbors")
    whole += _bytes(pairs_dir(tables_dir, model))
    with db.connect_quiet() as con:
        files = queries.bills_files(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    stems = {Path(n).stem for n, st, se, _ in files if (st, se) in day.records}
    archives = sum(
        p.stat().st_size
        for d in ("texts", "sections", "signatures", f"{SEMANTIC_DIR}/units")
        if (tables_dir / d).is_dir()
        for p in (tables_dir / d).iterdir()
        if p.is_file() and any(p.name.startswith(stem) for stem in stems)
    )
    shards = sorted(
        (p.stat().st_size for p in vectors_dir(tables_dir, model).glob("*.npy")), reverse=True
    )
    records = len(json.dumps({f"{k[0]}_{k[1]}": v for k, v in day.records.items()}))
    return whole + archives + sum(shards[:2]) + 8 * records + (1 << 20)


def apply_day(
    tables_dir: Path,
    day: Day,
    *,
    model: str,
    changes_dir: Path,
    floor_bytes: int,
    sessions_path: Path = SESSIONS,
    rules_path: Path = RULES,
    embed: Embedder | None = None,
    match_origins: OriginMatcher | None = None,
) -> list[Change]:
    """See `update`. `embed` embeds the day's new windows, by default with the
    model's pinned files or the embedder registered for it (`register_embedder`);
    `match_origins` matches the model documents against the tables, by default with
    `lineage origins`' own matching, run only when the tables hold origin matches."""
    digest = day_digest(day)
    applied = applied_days(tables_dir)
    if day.as_of in applied:
        if applied[day.as_of] != digest:
            raise UpdateError(f"{day.as_of} was applied with other records")
        return []
    if applied and day.as_of < max(applied):
        raise UpdateError(f"days out of order: {day.as_of} after {max(applied)}")
    updates = tables_dir / UPDATES_DIR
    if (updates / PENDING).exists():
        raise UpdateError(
            f"an update stopped partway in {tables_dir} ({updates / PENDING}); rebuild the tables"
        )
    with db.connect_quiet() as con:
        stamp = queries.latest_stamp(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    if stamp is not None and day.as_of <= stamp.date():
        raise UpdateError(f"the corpus already stands as of {stamp.date()}, not before {day.as_of}")
    semantic_root = semantic_dir(tables_dir)
    others = (
        sorted(
            p.name
            for p in semantic_root.iterdir()
            if p.is_dir() and p.name not in ("units", "texts", model)
        )
        if semantic_root.exists()
        else []
    )
    if others:
        raise UpdateError(
            f"the tables hold vectors of {', '.join(others)} too; an update embeds {model}"
        )
    embedder = (lambda: embed) if embed is not None else embedder_for(model)
    # Origin matches are recomputed only for tables that hold them (lineage origins).
    has_origins = origins_path(tables_dir).exists()
    matcher = (match_origins or _default_origins(model, floor_bytes)) if has_origins else None
    require_free_space(tables_dir, needed=_staged_bytes(tables_dir, day, model), floor=floor_bytes)

    updates.mkdir(parents=True, exist_ok=True)
    view = Staging(tables_dir, updates / "staging")
    try:
        changes = _staged_changes(
            tables_dir,
            view,
            day,
            model=model,
            embedder=embedder,
            matcher=matcher,
            sessions_path=sessions_path,
            rules_path=rules_path,
            floor_bytes=floor_bytes,
        )
        # The log first: it refuses a day logged with other changes before the tables change.
        write_changes(changes_dir, day.as_of, changes)
    except BaseException:
        shutil.rmtree(view.root, ignore_errors=True)
        raise
    # From here the tables change; a stop partway leaves the mark, and the next update refuses.
    (updates / PENDING).write_text(json.dumps({"as_of": day.as_of.isoformat(), "sha256": digest}))
    view.commit()
    _journal(tables_dir, day, digest)
    (updates / PENDING).unlink()
    return changes


def _staged_changes(
    tables_dir: Path,
    view: Staging,
    day: Day,
    *,
    model: str,
    embedder: Callable[[], Embedder],
    matcher: OriginMatcher | None,
    sessions_path: Path,
    rules_path: Path,
    floor_bytes: int,
) -> list[Change]:
    """Build the tables after `day` in `view` and return the day's changes."""
    with tempfile.TemporaryDirectory(dir=view.root.parent) as scratch:
        records, new_parts = _records(
            tables_dir,
            view,
            day,
            sessions_path=sessions_path,
            rules_path=rules_path,
            scratch=Path(scratch),
        )
    numbers = _signatures(tables_dir, view, records.retired, new_parts.values())
    near = _near_duplicates(tables_dir, view, numbers, records.resplit)
    semantic = _semantic(
        tables_dir,
        view,
        numbers,
        records.retired,
        new_parts.values(),
        model=model,
        embedder=embedder,
        floor_bytes=floor_bytes,
    )
    if matcher is not None:
        view.own("origins")
        write_origin_matches(view.root, matcher(view.root))

    day_ids = {str(r["id"]) for rows in day.records.values() for r in rows}
    related = _related_bills(tables_dir, view.root, day_ids, near, semantic, numbers.section_map)
    before = Snapshot({}, frozenset(), _relatives(tables_dir, related, model), _origins(tables_dir))
    after = Snapshot({}, frozenset(), _relatives(view.root, related, model), _origins(view.root))
    return [
        *records.changes,
        *relative_changes(before, after, day.as_of),
        *origin_changes(before, after, day.as_of),
    ]


def update(
    tables_dir: Path,
    day: Day,
    *,
    model: str = MODEL,
    changes_dir: Path = CHANGES_DIR,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
) -> list[Change]:
    """Apply one day to a built corpus in place and log its changes; returns them.

    The contract lane L1 builds to (DESIGN.md#updates): the tables afterwards give the same
    `snapshot` as a full rebuild over every day so far with the same pinned corpus statistics;
    the log holds `record_changes` for the day, then `relative_changes` and `origin_changes`
    between the snapshots before and after it; each new text is matched against the whole corpus
    and the model documents, and every existing bill it matches gains it as a relative; a day
    already applied changes nothing and returns no changes; every large write first checks
    `floor_bytes` of free space."""
    return apply_day(tables_dir, day, model=model, changes_dir=changes_dir, floor_bytes=floor_bytes)
