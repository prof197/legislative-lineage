"""`lineage export --state ST`: the static site's JSON for one state's bills (DESIGN.md#export).

Every file is an instance of a model in lineage.export.schema, validated as it is written, and
the JSON Schema those models emit is the site's contract (DR-1). One build writes:

- `meta.json`: the data stamp (each hosted archive's Generated At), the stories, and the
  validation page's figures from the committed results.
- `bills.json`: every hosted bill, for the browse page's filter.
- `bills/<slug>.json`: one page per hosted bill: its record, its text's source, its sections,
  and its relatives as a page lists them (lineage.match.relatives), each with the section pairs it
  shares with the hosted bill.
- `compare/<a>--<b>.json`: each listed relative's, and each featured story pair's, matched
  sections aligned (lineage.align), with a featured pair's grounded explanation from the cache
  (lineage.explain.explain); a pair is never sent to a model here.
- `texts/<slug>.json`: the display text of each section some comparison shows, once per bill.
  A section is shown beside many relatives, so keeping its text out of the comparisons keeps
  the site's size to its distinct text (measured in reports/export.json).
- `stories/<slug>.json`: each featured story's anchor and relatives (reports/featured_pairs.json).

The hosted bills are the state's records the document-type rules match. The first bill of a
comparison is the hosted bill, or a story's anchor, whose text the comparison starts from. The
output is written to a fresh directory and swapped in whole, so a failed export never leaves a
mix of two builds.
"""

import json
import multiprocessing
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from lineage.align import Alignment, Embedder, align
from lineage.core.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.core.paths import DATA
from lineage.explain.explain import CACHE_DIR
from lineage.export.origin_story import FILE as ORIGIN_STORY_FILE
from lineage.export.origin_story import story_file as origin_story_file
from lineage.export.pages import (
    FEATURED_PAIRS,
    FEATURED_REPORT,
    RESULTS,
    STORIES_RULES,
    BillPair,
    OrdinalPair,
    bill_list,
    bill_page,
    carries_origin_story,
    comparison,
    comparison_name,
    plan_export,
    texts_by_slug,
)
from lineage.export.release import RELEASE_MANIFEST as RELEASE_MANIFEST
from lineage.export.release import ReleaseError as ReleaseError
from lineage.export.release import package as package
from lineage.export.release import write_release_manifest as write_release_manifest
from lineage.export.schema import (
    TOP_LEVEL,
)
from lineage.match.semantic.embed import embed as embed_texts
from lineage.match.semantic.embed import load_model
from lineage.match.semantic.stage import MODEL
from lineage.origins.collect import PACKAGE as COLLECT_PACKAGE
from lineage.origins.labels import LABELS as ORIGIN_LABELS
from lineage.origins.modeldocs import MODELDOCS_DIR
from lineage.origins.withheld import REPORT as WITHHELD_REPORT

EXPORT_DIR = DATA / "export"
REPORT = Path("reports/export.json")
SCHEMA = Path("schema/export.schema.json")
COMMAND = "uv run lineage export --state {state}"
# GitHub Pages' limit on a published site; DR-2 reverses at half of it.
SITE_LIMIT = 1_000_000_000
_SPAWN = multiprocessing.get_context("spawn")
CHUNK = 64  # section pairs a worker aligns per task
# One process per performance core of the reference machine, each embedding on one thread.
WORKERS = 8
_embed: Embedder | None = None  # each worker's, set by _start


def _start(models_dir: Path | None) -> None:
    """Load the paraphrase embedder once per process: the semantic stage's model, one thread
    each, since the workers already fill the cores. None marks no paraphrase."""
    global _embed
    if models_dir is None:
        _embed = None
        return
    model = load_model(MODEL, models_dir, threads=1)

    def embed(texts: Sequence[str]) -> NDArray[np.float32]:
        return embed_texts(model, texts)

    _embed = embed


def _align_chunk(texts: list[tuple[str, str]]) -> list[Alignment]:
    return [align(a, b, embed=_embed) for a, b in texts]


def align_all(
    texts: Sequence[tuple[str, str]],
    *,
    models_dir: Path | None,
    workers: int,
    progress: Callable[[str], None] = lambda _: None,
) -> list[Alignment]:
    """Each pair of section texts aligned, in order, by `workers` processes (in this one when
    1), paraphrase marked with the semantic stage's model from `models_dir` unless None."""
    chunks = [list(texts[i : i + CHUNK]) for i in range(0, len(texts), CHUNK)]
    out: list[Alignment] = []
    if workers <= 1:
        _start(models_dir)
        results: Iterable[list[Alignment]] = map(_align_chunk, chunks)
        for done, got in enumerate(results, start=1):
            out.extend(got)
            if done % 50 == 0:
                progress(f"aligned {len(out)} of {len(texts)} section pairs")
        return out
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=_SPAWN, initializer=_start, initargs=(models_dir,)
    ) as pool:
        for done, got in enumerate(pool.map(_align_chunk, chunks), start=1):
            out.extend(got)
            if done % 50 == 0:
                progress(f"aligned {len(out)} of {len(texts)} section pairs")
    return out


def export_schema() -> dict[str, Any]:
    """The JSON Schema of every file the export writes, as the site reads them."""
    _, schema = models_json_schema(
        [(m, "serialization") for m in TOP_LEVEL], title="Legislative Lineage export"
    )
    return schema


def schema_text() -> str:
    return json.dumps(export_schema(), indent=1, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _Written:
    files: dict[str, int]  # per kind, how many
    bytes: dict[str, int]  # per kind, how many bytes
    largest: tuple[str, int]


def _write(root: Path, files: Mapping[str, BaseModel]) -> _Written:
    count: dict[str, int] = defaultdict(int)
    size: dict[str, int] = defaultdict(int)
    largest = ("", 0)
    for name, model in sorted(files.items()):
        data = model.model_dump_json().encode()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        kind = name.split("/")[0] if "/" in name else name
        count[kind] += 1
        size[kind] += len(data)
        if len(data) > largest[1]:
            largest = (name, len(data))
    return _Written(dict(sorted(count.items())), dict(sorted(size.items())), largest)


def build_export(
    tables: Path,
    *,
    state: str,
    out: Path,
    models_dir: Path | None,
    workers: int = 1,
    cache: Path = CACHE_DIR,
    featured_pairs: Path = FEATURED_PAIRS,
    featured_report: Path = FEATURED_REPORT,
    stories_rules: Path = STORIES_RULES,
    results: Path = RESULTS,
    modeldocs_root: Path = MODELDOCS_DIR,
    collector_package: str = COLLECT_PACKAGE,
    model: str = MODEL,
    require_flag: bool = True,
    floor: int = DEFAULT_FLOOR_BYTES,
    origin_labels: Path = ORIGIN_LABELS,
    withheld_report: Path = WITHHELD_REPORT,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Write `state`'s site data to `out`, replacing what was there; returns the report. Every
    file is rendered from the plan as the live track renders the ones a day touches
    (lineage.export.pages); only the alignments are made here."""
    started = time.monotonic()
    # The plan reads every committed input before the alignment, which takes hours for a large
    # state: a checkout whose reports are older than this code fails there, not at the last step.
    plan = plan_export(
        tables,
        state=state,
        cache=cache,
        featured_pairs=featured_pairs,
        featured_report=featured_report,
        stories_rules=stories_rules,
        results=results,
        modeldocs_root=modeldocs_root,
        collector_package=collector_package,
        model=model,
        require_flag=require_flag,
        origin_labels=origin_labels,
        withheld_report=withheld_report,
        progress=progress,
    )
    text_bytes = sum(len(t.encode()) for _, t, _ in plan.shown.values())
    n_pairs = sum(len(p) for p in plan.found.values())
    # Each text written once, spans at most a few hundred bytes a pair, and the pages.
    require_free_space(out.parent, needed=2 * text_bytes + 4096 * n_pairs + (64 << 20), floor=floor)
    progress(f"{len(plan.found)} comparisons, {n_pairs} section pairs to align")

    ordered = [
        (pair, ordinals) for pair, ps in sorted(plan.found.items()) for ordinals in sorted(ps)
    ]
    alignments = align_all(
        [(plan.shown[(a, oa)][1], plan.shown[(b, ob)][1]) for (a, b), (oa, ob) in ordered],
        models_dir=models_dir,
        workers=workers,
        progress=progress,
    )
    aligned: dict[BillPair, dict[OrdinalPair, Alignment]] = defaultdict(dict)
    for (pair, ordinals), alignment in zip(ordered, alignments, strict=True):
        aligned[pair][ordinals] = alignment

    files: dict[str, BaseModel] = {}
    featured_explained = 0
    withheld = 0
    for pair in sorted(plan.found):
        compared = comparison(plan, pair, aligned[pair])
        files[comparison_name(plan, pair)] = compared
        for p in compared.pairs:
            if p.explanation is not None:
                featured_explained += 1
                withheld += p.explanation.withheld
    for name, texts in texts_by_slug(plan).items():
        files[f"texts/{name}.json"] = texts
    pages = [bill_page(plan, bill_id) for bill_id in plan.hosted]
    for page in pages:
        files[f"bills/{page.bill.slug}.json"] = page
    files["bills.json"] = bill_list(plan)
    for key, story in plan.stories.items():
        files[f"stories/{key}.json"] = story
    if carries_origin_story(state, stories_rules):
        chosen = origin_story_file(pages, plan.judged)
        if chosen is not None:
            files[ORIGIN_STORY_FILE] = chosen
    files["sources.json"] = plan.sources
    files["meta.json"] = plan.meta

    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    written = _write(partial, files)
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    total = sum(written.bytes.values())
    relatives = plan.relatives.values()
    return {
        "command": COMMAND.format(state=state),
        "state": state,
        "schema": str(SCHEMA),
        "hosted_bills": len(plan.hosted),
        "bills_with_relatives": sum(1 for r in relatives if r.matches),
        "relatives_listed": sum(len(r.matches) for r in relatives),
        "comparisons": len(plan.found),
        "section_pairs_aligned": n_pairs,
        "featured_pairs_explained": featured_explained,
        "featured_pairs_withheld": withheld,
        "files": written.files,
        "bytes": written.bytes,
        "total_bytes": total,
        "share_of_site_limit": round(total / SITE_LIMIT, 4),
        "largest_file": {"path": written.largest[0], "bytes": written.largest[1]},
        "seconds": round(time.monotonic() - started),
    }
