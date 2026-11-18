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

The parts are modules of their own: the records (lineage.live.incremental_records), the
near-duplicate stage (incremental_near), the semantic stage (incremental_semantic), the bills
listed before and after (incremental_relatives), and the staging view (lineage.live.staging).
This module holds the journal, the embedder and the order the parts run in.

Owned by lane L1 (M10). See DESIGN.md#updates, DR-7.
"""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path

from lineage.core import db
from lineage.core.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.data.doctypes import RULES
from lineage.live import queries
from lineage.live.incremental_near import update_near_duplicates, update_signatures
from lineage.live.incremental_records import update_records
from lineage.live.incremental_relatives import bill_origins, listed_relatives, related_bills
from lineage.live.incremental_semantic import update_semantic
from lineage.live.staging import UPDATES_DIR, Staging
from lineage.live.updates import (
    CHANGES_DIR,
    Change,
    Day,
    Snapshot,
    UpdateError,
    origin_changes,
    relative_changes,
    write_changes,
)
from lineage.match.semantic.models import MODELS, MODELS_DIR, require_model
from lineage.match.semantic.pairs import pairs_dir
from lineage.match.semantic.stage import MODEL, Embedder
from lineage.match.semantic.vectors import SEMANTIC_DIR, semantic_dir, vectors_dir
from lineage.origins.matches import OriginMatch, origins_path, write_origin_matches

SESSIONS = Path(__file__).parents[3] / "snapshot" / "sessions.csv"
APPLIED = "applied.jsonl"
PENDING = "pending.json"


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
        records, new_parts = update_records(
            tables_dir,
            view,
            day,
            sessions_path=sessions_path,
            rules_path=rules_path,
            scratch=Path(scratch),
        )
    numbers = update_signatures(tables_dir, view, records.retired, new_parts.values())
    near = update_near_duplicates(tables_dir, view, numbers, records.resplit)
    semantic = update_semantic(
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
    related = related_bills(tables_dir, view.root, day_ids, near, semantic, numbers.section_map)
    before = Snapshot(
        {}, frozenset(), listed_relatives(tables_dir, related, model), bill_origins(tables_dir)
    )
    after = Snapshot(
        {}, frozenset(), listed_relatives(view.root, related, model), bill_origins(view.root)
    )
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
