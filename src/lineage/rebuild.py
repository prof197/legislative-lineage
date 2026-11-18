"""`lineage rebuild`: the folded days laid over the snapshot's archives and the whole pipeline
run again into a tables directory of its own, the reference an update is held to (DR-7).

The weekly job runs it twice over the days the updated tables have had, and compares
(lineage.schedule): once with the base build's corpus statistics, pinned, which must agree with
the updated tables, and once with them counted afresh, whose difference is the drift.

- **Archives.** The days up to `through` are folded (lineage.updates.fold), each record as last
  seen. An archive of the snapshot no day touches is copied with its README restamped; one a day
  touches takes the day's records in place of its own and the new ones beside them, sorted by
  id as an update sorts it; a session the snapshot lacks gets an archive of its own, named as
  lineage.updates.write_archives names it. Every archive is stamped as of the fold's last day,
  since an update restamps every bills file, and a status is judged on that day.
- **Texts.** The text layer's choice is an input, as the archives are: each record takes the
  text `lineage text merge` gives it, but a record a day added or whose text a day replaced,
  which takes the text it carries, as an update gives it (lineage.queries.merge_texts_file).
- **Statistics.** With `pinned_from`, the updated tables at the same day, the stoplist is copied
  from them and the state boilerplate counts are carried over to the rebuild's sections by bill
  and ordinal; those tables carry the base build's statistics, never recounted, with no flag on
  a section a day brought. Without it the stoplist is counted at `max_share` and the counts are
  built afresh, as a build counts them.
- **Vectors.** A window's vector is a function of its text and the model alone, and the
  embedding is the build's longest stage by far (reports/cost.json), so with `vectors_from` a
  shard whose every text those tables already embedded is filled from them, and only the rest
  is embedded. The neighbors, the semantic pairs and everything after them are built afresh.

The tables are built beside `out`, under `<out>.building`, and replace it only once complete;
the merged archives, their manifest and the stages' reports go to `<out>.build/`.

See DESIGN.md#updates, DR-7.
"""

import json
import shutil
import time
import zipfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from lineage import queries
from lineage.core import db
from lineage.core.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.core.hashing import file_sha256
from lineage.core.paths import DATA
from lineage.data.doctypes import RULES as DOCUMENT_TYPES
from lineage.data.doctypes import build_eligibility
from lineage.data.ingest import ingest
from lineage.data.snapshot import bills_member, load_manifest, scan, verify, write_manifest
from lineage.incremental import applied_days, embedder_for
from lineage.match.near.index import build_index
from lineage.match.near.stoplist import build_stoplist
from lineage.match.semantic import queries as semantic_queries
from lineage.match.semantic.models import MODELS, MODELS_DIR
from lineage.match.semantic.neighbors import EF_SEARCH as NEIGHBORS_EF_SEARCH
from lineage.match.semantic.neighbors import RECALL_FLOOR, SPEC, build_neighbors
from lineage.match.semantic.pairs import build_semantic_pairs, section_keys
from lineage.match.semantic.stage import MODEL
from lineage.match.semantic.vectors import (
    SHARD_TEXTS,
    build_plan,
    digest,
    embed_corpus,
    load_plan,
    load_vectors,
    missing_shards,
    rows_of,
    semantic_dir,
    shard_name,
    vectors_dir,
)
from lineage.match.state_boilerplate import (
    build_state_boilerplate,
    has_counts,
    load_families_reached,
)
from lineage.origins.matches import origins_path, write_origin_matches
from lineage.text.inventory import PLAN as TEXT_PLAN
from lineage.text.merge import BULK, FETCHED, merge_texts
from lineage.text.rules import RULES as TEXT_RULES
from lineage.text.run import QUALITY as TEXT_QUALITY
from lineage.text.run import STORE as TEXT_STORE
from lineage.text.sectioning import build_sections
from lineage.updates import BILLS_SUFFIX, Folded, fold, load_days, text_hash

RAW = DATA / "raw"
MANIFEST = Path("snapshot/manifest.json")
SESSIONS = Path("snapshot/sessions.csv")
MAX_SHARE = 0.001  # the stoplist's share, as `lineage boilerplate` counts it
REPORT = "rebuild.json"
# The suffix of an archive for a session the snapshot lacks, as lineage.updates.write_archives
# and lineage.incremental name it.
NEW_ARCHIVE = "json_days"


class RebuildError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Base:
    """The snapshot's archives the days are laid over."""

    raw: Path = RAW
    manifest: Path = MANIFEST


@dataclass(frozen=True, slots=True)
class Laid:
    """The archives written, and the records whose text a day brought."""

    archives: int
    fresh: Mapping[str, frozenset[str]]  # bills file name -> ids added or with their text replaced
    touched: int  # archives of the snapshot a day touched
    new_sessions: int


def work_dir(out: Path) -> Path:
    """Where a rebuild into `out` keeps its archives, manifest and reports."""
    return out.with_name(out.name + ".build")


def _stamp(as_of: date) -> str:
    return f"{as_of.isoformat()} 00:00:00"


def _readme(text: str, stamp: str) -> str:
    lines = [
        f"Generated At: {stamp}" if line.startswith("Generated At:") else line
        for line in text.splitlines()
    ]
    return "\n".join(lines) + "\n"


def _write_archive(path: Path, readme: str, member: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stored, as Open States stores them: ingest copies the member out without inflating it.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("README", readme)
        z.writestr(member, json.dumps(list(rows)))


def lay_days(folded: Folded | None, base: Base | None, raw: Path, *, as_of: date) -> Laid:
    """The snapshot's archives with the folded days laid over them, under `raw`, every one
    stamped as of `as_of`."""
    stamp = _stamp(as_of)
    by_archive: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    if folded is not None:
        for bill_id, (state, session, row) in folded.records.items():
            by_archive[(state, session)][bill_id] = row
    archives = load_manifest(base.manifest) if base is not None else []
    if base is not None:
        verify(base.raw, archives)
    seen: dict[tuple[str, str], str] = {}
    fresh: dict[str, frozenset[str]] = {}
    touched = 0
    for a in archives:
        key = (a.state, a.session)
        source = base.raw / a.path if base is not None else Path()
        target = raw / a.path
        if key in by_archive and key in seen:
            raise RebuildError(
                f"{a.state} {a.session} has two archives, {seen[key]} and {a.path}; a day's "
                "records would belong to either"
            )
        seen[key] = a.path
        with zipfile.ZipFile(source) as z:
            member = bills_member(z)
            readme = _readme(z.read("README").decode(), stamp)
            if key not in by_archive:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as out:
                    out.writestr("README", readme)
                    with z.open(member) as src, out.open(member, "w") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                continue
            rows: list[Mapping[str, Any]] = json.loads(z.read(member))
        day_rows = by_archive.pop(key)
        before = {str(r["id"]): r for r in rows}
        new = {
            bill_id
            for bill_id, row in day_rows.items()
            if bill_id not in before or text_hash(before[bill_id]) != text_hash(row)
        }
        merged = {**before, **day_rows}
        _write_archive(target, readme, member, [merged[i] for i in sorted(merged)])
        fresh[Path(a.path).stem + ".parquet"] = frozenset(new)
        touched += 1
    for (state, session), day_rows in sorted(by_archive.items()):
        name = f"{state}_{session}_{NEW_ARCHIVE}"
        readme = (
            "Open States Data Export\n\n"
            f"State: {state}\nSession: {session}\nGenerated At: {stamp}\n"
            "JSON Format Version: 1.0\n"
        )
        _write_archive(
            raw / state / f"{name}.zip",
            readme,
            f"{state}/{session}/{state}_{session}{BILLS_SUFFIX}",
            [day_rows[i] for i in sorted(day_rows)],
        )
        fresh[f"{name}.parquet"] = frozenset(day_rows)
    return Laid(
        archives=len(scan(raw)),
        fresh=fresh,
        touched=touched,
        new_sessions=len(by_archive),
    )


def _bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def _exported_for_fresh(tables: Path, fresh: Mapping[str, frozenset[str]]) -> int:
    """Give each record a day brought the text it carries, as an update does; the rest keep
    the merge's choice. Returns the files rewritten."""
    rewritten = 0
    with db.connect_quiet() as con:
        for name, ids in sorted(fresh.items()):
            texts, bills = tables / "texts" / name, tables / "bills" / name
            if not ids or not texts.exists():
                continue
            partial = texts.with_name(texts.name + ".partial")
            queries.merge_texts_file(
                con, texts_path=texts, bills_path=bills, fresh=sorted(ids), out_path=partial
            )
            partial.replace(texts)
            rewritten += 1
    return rewritten


def _carry_counts(tables: Path, pinned_from: Path) -> dict[str, int]:
    """The pinned tables' state boilerplate counts on the rebuild's sections, by bill and
    ordinal; a section they do not have counts no families, as a new section in an update."""
    counts = load_families_reached(pinned_from)
    bills, ordinals = section_keys(pinned_from)
    by_key = {
        (b, int(o)): int(c)
        for b, o, c in zip(bills, ordinals.tolist(), counts.tolist(), strict=True)
    }
    now_bills, now_ordinals = section_keys(tables)
    carried = np.zeros(len(now_bills), dtype=counts.dtype)
    found = 0
    for i, key in enumerate(zip(now_bills, now_ordinals.tolist(), strict=True)):
        c = by_key.get((key[0], int(key[1])))
        if c is not None:
            carried[i] = c
            found += 1
    where = tables / "state_boilerplate"
    where.mkdir(parents=True, exist_ok=True)
    np.save(where / "families_reached.npy", carried)
    return {"sections": len(now_bills), "carried": found, "not_in_pinned": len(now_bills) - found}


def _vector_index(
    source: Path, model: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], Any]:
    """Every text `source` embedded with `model`, by digest: (digests sorted, their text ids,
    the vector shards). A retired text, blanked, is left out."""
    plan = load_plan(source)
    if plan is None or missing_shards(source, model, plan):
        raise RebuildError(f"{source} holds no complete set of {model} vectors to reuse")
    blocks = load_vectors(source, model, plan)
    found: list[bytes] = []
    ids: list[int] = []
    with db.connect_quiet() as con:
        for s in range(plan.shards):
            texts = semantic_queries.read_semantic_texts(
                con, parquet_path=semantic_dir(source) / "texts" / f"{shard_name(s)}.parquet"
            )
            for n, text in enumerate(texts):
                if text:
                    found.append(digest(text))
                    ids.append(s * SHARD_TEXTS + n)
    keys = np.array(found, dtype="S16")
    order = np.argsort(keys, kind="stable")
    return keys[order], np.array(ids, dtype=np.int64)[order], blocks


def _reuse_vectors(tables: Path, source: Path, model: str) -> dict[str, int]:
    """Fill each shard of `tables` whose every text `source` has embedded; returns the counts."""
    plan = load_plan(tables)
    assert plan is not None
    keys, ids, blocks = _vector_index(source, model)
    out = vectors_dir(tables, model)
    out.mkdir(parents=True, exist_ok=True)
    filled = texts_reused = 0
    with db.connect_quiet() as con:
        for s in range(plan.shards):
            texts = semantic_queries.read_semantic_texts(
                con, parquet_path=semantic_dir(tables) / "texts" / f"{shard_name(s)}.parquet"
            )
            wanted = np.array([digest(t) for t in texts], dtype="S16")
            at = np.searchsorted(keys, wanted)
            hit = (at < keys.size) & (keys[np.minimum(at, keys.size - 1)] == wanted)
            if not keys.size or not bool(hit.all()):
                continue
            vectors = rows_of(blocks, ids[at]).astype(np.float16)
            partial = out / f"{shard_name(s)}.npy.partial"
            with partial.open("wb") as f:
                np.save(f, vectors)
            partial.replace(out / f"{shard_name(s)}.npy")
            filled += 1
            texts_reused += len(texts)
    return {"shards": plan.shards, "shards_reused": filled, "texts_reused": texts_reused}


def _embed_rest(
    tables: Path, model: str, *, models_dir: Path, floor_bytes: int, progress: Callable[[str], None]
) -> int:
    """Embed the shards not filled yet: with `lineage embed`'s worker processes for a pinned
    model, in this process for one registered (lineage.incremental.register_embedder)."""
    plan = load_plan(tables)
    assert plan is not None
    todo = missing_shards(tables, model, plan)
    if not todo:
        return 0
    if model in MODELS:
        embed_corpus(
            tables, model, models_dir=models_dir, floor_bytes=floor_bytes, progress=progress
        )
        return len(todo)
    embed = embedder_for(model, models_dir)()
    out = vectors_dir(tables, model)
    out.mkdir(parents=True, exist_ok=True)
    with db.connect_quiet() as con:
        for s in todo:
            texts = semantic_queries.read_semantic_texts(
                con, parquet_path=semantic_dir(tables) / "texts" / f"{shard_name(s)}.parquet"
            )
            np.save(out / f"{shard_name(s)}.npy", embed(texts).astype(np.float16))
    return len(todo)


def _origins(tables: Path, model: str, floor_bytes: int) -> int:
    from lineage.origins.collect import collectors
    from lineage.origins.matching import match_models
    from lineage.origins.modeldocs import MODELDOCS_DIR, read_docs

    docs = [d for c in collectors().values() for d in read_docs(c.source, MODELDOCS_DIR)]
    matches = match_models(
        tables, docs, modeldocs_root=MODELDOCS_DIR, model=model, floor_bytes=floor_bytes
    )
    write_origin_matches(tables, matches)
    return len(matches)


@dataclass
class _Timer:
    progress: Callable[[str], None]
    seconds: dict[str, float] = field(default_factory=dict[str, float])

    def run[T](self, name: str, step: Callable[[], T]) -> T:
        started = time.monotonic()
        self.progress(f"rebuild: {name}")
        value = step()
        self.seconds[name] = round(time.monotonic() - started, 1)
        return value


def rebuild(
    out: Path,
    *,
    days_dir: Path,
    through: date | None = None,
    base: Base | None = Base(),  # noqa: B008 - frozen, so one default is safe
    sessions: Path = SESSIONS,
    pinned_from: Path | None = None,
    vectors_from: Path | None = None,
    max_share: float = MAX_SHARE,
    origins: bool | None = None,
    model: str = MODEL,
    models_dir: Path = MODELS_DIR,
    document_types: Path = DOCUMENT_TYPES,
    text_plan: Path = TEXT_PLAN,
    text_store: Path = TEXT_STORE,
    text_rules: Path = TEXT_RULES,
    text_quality: Path = TEXT_QUALITY,
    neighbors_spec: str = SPEC,
    ef_search: int | None = NEIGHBORS_EF_SEARCH,
    workers: int = 1,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """A full build into `out` over `base`'s archives and the days under `days_dir` up to
    `through` (every day when None); returns the report, also written to `<out>.build/`.

    With `pinned_from`, tables that have had exactly the days folded here, the corpus
    statistics are theirs; without it they are counted afresh, the stoplist at `max_share`.
    Origin matches are made when `origins` is True, or, when None, when the tables the
    statistics or vectors come from hold them."""
    started = time.monotonic()
    days = (
        [d for d in load_days(days_dir) if through is None or d.as_of <= through]
        if (days_dir.exists())
        else []
    )
    if not days and base is None:
        raise RebuildError(
            f"no days under {days_dir} up to {through} and no archives to lay them on"
        )
    folded = fold(days) if days else None
    as_of = folded.as_of if folded is not None else None
    if pinned_from is not None:
        applied = applied_days(pinned_from)
        last = max(applied, default=None)
        if last != as_of:
            raise RebuildError(
                f"{pinned_from} stands as of {last or 'its build'}, the days folded here as of "
                f"{as_of or 'none'}: pinned statistics come from tables at the same day"
            )
        if not (pinned_from / "stoplist.parquet").exists():
            raise RebuildError(f"{pinned_from} holds no stoplist to pin")
    if as_of is None:
        archives = load_manifest(base.manifest) if base is not None else []
        stamps = sorted(a.generated_at[:10] for a in archives)
        as_of = date.fromisoformat(stamps[-1]) if stamps else date.today()
    reuse = vectors_from if vectors_from is not None else pinned_from
    match_origins = (
        origins if origins is not None else (reuse is not None and origins_path(reuse).exists())
    )

    work = work_dir(out)
    building = out.with_name(out.name + ".building")
    for stale in (work, building):
        if stale.exists():
            shutil.rmtree(stale)
    raw, manifest, reports = work / "raw", work / "manifest.json", work / "reports"
    reports.mkdir(parents=True)
    base_bytes = sum(a.bytes for a in load_manifest(base.manifest)) if base is not None else 0
    # The archives are written once and read by ingest; the tables of a build like this one
    # are the best bound on its own, and without one the archives several times over.
    tables_bound = _bytes(reuse) if reuse is not None else 4 * base_bytes
    require_free_space(out.parent, needed=base_bytes + tables_bound + (1 << 26), floor=floor_bytes)

    timer = _Timer(progress)
    doc: dict[str, Any] = {
        "out": out.name,
        "days": [d.as_of.isoformat() for d in days],
        "as_of": as_of.isoformat(),
        "statistics": "pinned" if pinned_from is not None else "refreshed",
        "max_share": None if pinned_from is not None else max_share,
        "model": model,
    }
    try:
        laid = timer.run("archives", lambda: lay_days(folded, base, raw, as_of=as_of))
        write_manifest(scan(raw), manifest)
        doc["archives"] = {
            "written": laid.archives,
            "touched_by_days": laid.touched,
            "new_sessions": laid.new_sessions,
            "records_with_day_text": sum(len(v) for v in laid.fresh.values()),
        }
        building.mkdir(parents=True)
        result = timer.run(
            "ingest",
            lambda: ingest(raw, manifest, sessions, building, floor_bytes=floor_bytes),
        )
        doc["records"] = result.rows
        shutil.rmtree(raw)  # as large as the snapshot, and derived from it and the days
        counts = timer.run(
            "texts",
            lambda: merge_texts(
                building,
                plan_path=text_plan,
                store=text_store,
                rules_path=text_rules,
                quality_path=text_quality,
                floor_bytes=floor_bytes,
            ),
        )
        doc["text_sources"] = counts
        if counts.get(FETCHED) or counts.get(BULK):
            doc["texts_files_with_day_text"] = _exported_for_fresh(building, laid.fresh)
        timer.run(
            "sections", lambda: build_sections(building, workers=workers, floor_bytes=floor_bytes)
        )
        timer.run(
            "eligible",
            lambda: build_eligibility(
                building,
                rules_path=document_types,
                report_path=reports / "document_types.json",
            ),
        )
        if pinned_from is not None:
            shutil.copy2(pinned_from / "stoplist.parquet", building / "stoplist.parquet")
        else:
            timer.run(
                "stoplist",
                lambda: build_stoplist(
                    building,
                    max_share=max_share,
                    workers=workers,
                    report_path=reports / "boilerplate.json",
                    manifest_sha256=file_sha256(manifest),
                    rules_sha256=file_sha256(document_types),
                ),
            )
        timer.run(
            "index",
            lambda: build_index(
                building,
                workers=workers,
                floor_bytes=floor_bytes,
                report_path=reports / "index.json",
            ),
        )
        plan = timer.run("plan", lambda: build_plan(building, floor_bytes=floor_bytes))
        doc["vectors"] = (
            timer.run("reuse", lambda: _reuse_vectors(building, reuse, model))
            if reuse is not None
            else {"shards": plan.shards, "shards_reused": 0, "texts_reused": 0}
        )
        doc["vectors"]["shards_embedded"] = timer.run(
            "embed",
            lambda: _embed_rest(
                building, model, models_dir=models_dir, floor_bytes=floor_bytes, progress=progress
            ),
        )
        neighbors = timer.run(
            "neighbors",
            lambda: build_neighbors(
                building,
                model=model,
                spec=neighbors_spec,
                ef_search=ef_search,
                floor_bytes=floor_bytes,
                progress=progress,
            ),
        )
        (reports / "neighbors.json").write_text(json.dumps(neighbors, indent=1, default=str) + "\n")
        recall = neighbors["exact_check"]["recall"]
        if not neighbors["within_memory_bound"]:
            raise RebuildError("the neighbors' peak memory crossed the bound")
        if recall < RECALL_FLOOR:
            raise RebuildError(f"the neighbors' recall {recall} is below {RECALL_FLOOR}")
        timer.run(
            "semantic-pairs",
            lambda: build_semantic_pairs(building, model=model, floor_bytes=floor_bytes),
        )
        if pinned_from is not None:
            if has_counts(pinned_from):
                doc["state_boilerplate"] = timer.run(
                    "state-boilerplate", lambda: _carry_counts(building, pinned_from)
                )
        else:
            timer.run(
                "state-boilerplate",
                lambda: build_state_boilerplate(building, model=model, floor_bytes=floor_bytes),
            )
        if match_origins:
            doc["origin_matches"] = timer.run(
                "origins", lambda: _origins(building, model, floor_bytes)
            )
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    if out.exists():
        shutil.rmtree(out)
    building.rename(out)
    doc["seconds"] = timer.seconds | {"total": round(time.monotonic() - started, 1)}
    (work / REPORT).write_text(json.dumps(doc, indent=1) + "\n")
    return doc
