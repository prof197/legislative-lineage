"""Matching's commands: the stoplist, the index, embedding, neighbors, the semantic
pairs, the state boilerplate flag, a query, and the embedding models' fetch.
"""

import json
import os
import time
from pathlib import Path
from typing import Annotated

import typer

from lineage.cli.app import (
    ManifestOption,
    MinFreeOption,
    RulesOption,
    SemanticOption,
    TablesOption,
    app,
    model_app,
    semantic_search,
)
from lineage.core.disk import GiB, InsufficientDiskSpaceError
from lineage.core.hashing import file_sha256
from lineage.core.http import FetchError
from lineage.core.memory import MemoryBoundError
from lineage.core.paths import DATA
from lineage.data.doctypes import RULES
from lineage.match.listing import flagged_sections
from lineage.match.near.index import build_index
from lineage.match.near.stoplist import StoplistError, build_stoplist
from lineage.match.query import IndexMissingError, query_text, require_index
from lineage.match.semantic.models import MODELS, MODELS_DIR, ModelError, fetch_model
from lineage.match.semantic.neighbors import EF_SEARCH, RECALL_FLOOR, build_neighbors
from lineage.match.semantic.neighbors import REPORT as NEIGHBORS_REPORT
from lineage.match.semantic.neighbors import check_at as check_neighbors_at
from lineage.match.semantic.neighbors import write_report as write_neighbors_report
from lineage.match.semantic.pairs import REPORT as SEMANTIC_PAIRS_REPORT
from lineage.match.semantic.pairs import SemanticPairsError, build_semantic_pairs
from lineage.match.semantic.pairs import write_report as write_semantic_pairs_report
from lineage.match.semantic.stage import MODEL as SEMANTIC_MODEL
from lineage.match.semantic.stage import THRESHOLD as THRESHOLD_STAGE
from lineage.match.semantic.vectors import PROCESSES as EMBED_PROCESSES
from lineage.match.semantic.vectors import REPORT as EMBED_REPORT
from lineage.match.semantic.vectors import THREADS as EMBED_THREADS
from lineage.match.semantic.vectors import EmbedError, embed_corpus, vectors_dir
from lineage.match.semantic.vectors import write_report as write_embed_report
from lineage.match.state_boilerplate import (
    MIN_FAMILIES,
    StateBoilerplateError,
    build_state_boilerplate,
)
from lineage.match.state_boilerplate import REPORT as STATE_BOILERPLATE_REPORT
from lineage.match.state_boilerplate import write_report as write_state_boilerplate_report


@app.command()
def boilerplate(
    tables: TablesOption = DATA / "tables",
    manifest: ManifestOption = Path("snapshot/manifest.json"),
    rules: RulesOption = RULES,
    max_share: Annotated[
        float,
        typer.Option(
            "--max-share", help="Stoplist shingles found in more than this share of sections."
        ),
    ] = 0.001,
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = Path(
        "reports/boilerplate.json"
    ),
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = (
        os.cpu_count() or 1
    ),
) -> None:
    """Build the shingle stoplist from the sections table and report the boilerplate found."""
    try:
        result = build_stoplist(
            tables,
            max_share=max_share,
            workers=workers,
            report_path=report,
            manifest_sha256=file_sha256(manifest),
            rules_sha256=file_sha256(rules),
        )
    except StoplistError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"{result.stoplisted:,} of {result.distinct_shingles:,} shingles stoplisted, removing "
        f"{result.occurrences_removed / max(result.occurrences, 1):.1%} of occurrences across "
        f"{result.sections:,} sections ({result.seconds:.0f}s) -> {tables / 'stoplist.parquet'}, "
        f"{report}"
    )


@app.command("embed")
def embed_command(
    tables: TablesOption = DATA / "tables",
    model: Annotated[
        str, typer.Option("--model", help=f"Embedding model: {', '.join(MODELS)}.")
    ] = SEMANTIC_MODEL,
    models_dir: Annotated[
        Path, typer.Option("--models", help="Directory of model files (local).")
    ] = MODELS_DIR,
    processes: Annotated[int, typer.Option("--processes", help="Worker processes.")] = (
        EMBED_PROCESSES
    ),
    threads: Annotated[int, typer.Option("--threads", help="Threads per worker.")] = (
        EMBED_THREADS
    ),
    min_free_gib: MinFreeOption = 20.0,
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = (
        EMBED_REPORT
    ),
) -> None:
    """Embed every distinct semantic window, float16, resumable -> <tables>/semantic/."""
    try:
        doc = embed_corpus(
            tables,
            model,
            models_dir=models_dir,
            processes=processes,
            threads=threads,
            floor_bytes=int(min_free_gib * GiB),
            progress=typer.echo,
        )
    except (EmbedError, ModelError, InsufficientDiskSpaceError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_embed_report(doc, report)
    typer.echo(
        f"{doc['distinct_texts']:,} distinct texts of {doc['windows']:,} windows, "
        f"{doc['shards'] - doc['shards_left']} of {doc['shards']} shards embedded "
        f"({doc['vector_bytes'] / GiB:.2f} GiB) -> {vectors_dir(tables, model)}; {report}"
    )


@app.command("neighbors")
def neighbors_command(
    tables: TablesOption = DATA / "tables",
    min_free_gib: MinFreeOption = 20.0,
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = (
        NEIGHBORS_REPORT
    ),
    ef_search: Annotated[
        int, typer.Option("--ef-search", help="HNSW's search width; wider finds more.")
    ] = EF_SEARCH,
    check_at_stage: Annotated[
        bool,
        typer.Option(
            "--check-at-stage",
            help="Build nothing: check the built neighbors' recall at the stage's threshold.",
        ),
    ] = False,
) -> None:
    """Every distinct window's nearest neighbors by meaning -> <tables>/semantic/<model>/."""
    if check_at_stage:
        try:
            checked = check_neighbors_at(tables, THRESHOLD_STAGE)
        except EmbedError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
        doc = json.loads(report.read_text()) | {"exact_check_at_stage_threshold": checked}
        write_neighbors_report(doc, report)
        typer.echo(
            f"recall {checked['recall']} at {checked['threshold']} on {checked['texts']:,} texts "
            f"searched exactly ({checked['true_neighbors']:,} true neighbors) -> {report}"
        )
        if checked["recall"] < RECALL_FLOOR:
            typer.echo(f"error: recall {checked['recall']} is below {RECALL_FLOOR}", err=True)
            raise typer.Exit(1)
        return
    try:
        doc = build_neighbors(
            tables,
            ef_search=ef_search,
            floor_bytes=int(min_free_gib * GiB),
            progress=typer.echo,
        )
    except (EmbedError, InsufficientDiskSpaceError, MemoryBoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_neighbors_report(doc, report)
    check = doc["exact_check"]
    typer.echo(
        f"{doc['texts_with_a_neighbor']:,} of {doc['texts']:,} texts have a neighbor at a cosine "
        f"of {doc['threshold']} or more; {doc['neighbors_kept']:,} kept; recall "
        f"{check['recall']} on {check['texts']:,} texts searched exactly; peak "
        f"{doc['peak_memory_bytes'] / GiB:.2f} GiB; {doc['seconds']}s -> {report}"
    )
    if not doc["within_memory_bound"]:
        typer.echo("error: the build's peak memory crossed the bound", err=True)
        raise typer.Exit(1)
    if check["recall"] < RECALL_FLOOR:
        typer.echo(
            f"error: recall {check['recall']} is below {RECALL_FLOOR}; rerun with a larger "
            f"--ef-search than {ef_search}",
            err=True,
        )
        raise typer.Exit(1)


@app.command("semantic-pairs")
def semantic_pairs_command(
    tables: TablesOption = DATA / "tables",
    min_free_gib: MinFreeOption = 20.0,
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = (
        SEMANTIC_PAIRS_REPORT
    ),
) -> None:
    """The neighbors as cross-state section pairs, best cosine and window pair each ->
    <tables>/semantic/<model>/pairs/."""
    try:
        doc = build_semantic_pairs(tables, floor_bytes=int(min_free_gib * GiB))
    except (SemanticPairsError, InsufficientDiskSpaceError, MemoryBoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_semantic_pairs_report(doc, report)
    typer.echo(
        f"{doc['pairs']:,} cross-state section pairs from {doc['neighbors_read']:,} neighbors; "
        f"{doc['windows_left_out']:,} windows of unsigned sections left out; peak "
        f"{doc['peak_memory_bytes'] / GiB:.2f} GiB; {doc['seconds']}s -> {report}"
    )


@app.command("state-boilerplate")
def state_boilerplate_command(
    tables: TablesOption = DATA / "tables",
    min_free_gib: MinFreeOption = 20.0,
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = (
        STATE_BOILERPLATE_REPORT
    ),
) -> None:
    """Count the families of its own state each section's near-duplicates reach ->
    <tables>/state_boilerplate/; flagged at MIN_FAMILIES or more."""
    try:
        doc = build_state_boilerplate(tables, floor_bytes=int(min_free_gib * GiB))
    except (
        StateBoilerplateError,
        SemanticPairsError,
        FileNotFoundError,
        InsufficientDiskSpaceError,
        MemoryBoundError,
    ) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_state_boilerplate_report(doc, report)
    at = next(r for r in doc["sweep"] if r["min_families"] == MIN_FAMILIES)
    typer.echo(
        f"{at['sections']:,} sections flagged at {MIN_FAMILIES} families or more, of "
        f"{doc['sections']:,}; peak {doc['peak_memory_bytes'] / GiB:.2f} GiB; "
        f"{doc['seconds']}s -> {report}"
    )


@app.command()
def index(
    tables: TablesOption = DATA / "tables",
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = (
        os.cpu_count() or 1
    ),
    min_free_gib: MinFreeOption = 20.0,
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = Path(
        "reports/index.json"
    ),
) -> None:
    """Build the search index: signatures, LSH bands, verified pairs, families."""
    started = time.monotonic()
    try:
        doc = build_index(
            tables, workers=workers, floor_bytes=int(min_free_gib * GiB), report_path=report
        )
    except (StoplistError, InsufficientDiskSpaceError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    counts, lsh = doc["sections"], doc["lsh"]
    pairs, capped = lsh["candidate_pairs"], lsh["capped"]
    verified, families = doc["pairs"]["verified"], doc["families"]
    typer.echo(
        f"{counts['signed']:,} of {counts['matchable']:,} matchable sections signed "
        f"({counts['too_short']:,} too short, {counts['all_stoplisted']:,} all stoplisted) -> "
        f"{tables / 'signatures'} ({doc['signature_bytes'] / GiB:.2f} GiB)\n"
        f"{lsh['bands']} bands x {lsh['rows']} rows -> {tables / 'bands'} "
        f"({lsh['index_bytes'] / GiB:.2f} GiB): {pairs['all']:,} candidate pairs, "
        f"{pairs['cross_state']:,} across states; {capped['buckets']:,} buckets over the cap "
        f"of {lsh['bucket_cap']} skipped, dropping {capped['cross_state_pairs_lost']:,} "
        f"cross-state pairs\n"
        f"{verified['all']:,} verified at an estimate of {doc['pairs']['threshold']} or more: "
        f"{verified['cross_state']:,} cross-state pairs -> {tables / 'pairs'}; "
        f"{families['families_of_two_or_more']:,} same-state families of two or more bills "
        f"({families['bills_in_them']:,} bills) -> {tables / 'families'}\n"
        f"{time.monotonic() - started:.0f}s, {report}"
    )


@app.command()
def query(
    file: Annotated[
        Path,
        typer.Argument(help="The bill's text, as a plain-text file.", exists=True, dir_okay=False),
    ],
    tables: TablesOption = DATA / "tables",
    limit: Annotated[int, typer.Option("--limit", help="Families to show.")] = 20,
    semantic: SemanticOption = True,
) -> None:
    """Find a bill's relatives in the index, offline: bills that share its text, by section."""
    started = time.monotonic()
    try:
        require_index(tables)
        search = semantic_search(tables) if semantic else None
        result = query_text(
            file.read_text(), tables, semantic=search, flagged=flagged_sections(tables)
        )
    except (IndexMissingError, StoplistError, EmbedError, ModelError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(f"{file}: {result.sections} sections, {result.searched} searched")
    if not result.matches:
        typer.echo("No relatives found in the index.")
        return
    hidden = 0
    for found_by, heading in (("wording", "Sharing wording"), ("meaning", "By meaning alone")):
        matches = [m for m in result.matches if m.found_by == found_by]
        if not matches:
            continue
        typer.echo(f"{heading}: {len(matches):,}")
        typer.echo(
            f"{'share':>5}  {'best':>4}  {'cos':>4}  {'by':<4}  {'state':<5}  {'session':<10}  "
            f"{'bill':<12}  {'family':>6}  {'status':<10}  title"
        )
        for m in matches[:limit]:
            by = "".join(_STAGE_MARKS[s] for s in sorted(m.sections_by_stage))
            cos = f"{m.best_cosine:.2f}" if m.best_cosine is not None else ""
            typer.echo(
                f"{m.share:>5.0%}  {m.best_estimate:>4.2f}  {cos:>4}  {by:<4}  {m.state:<5}  "
                f"{m.session:<10}  {m.identifier:<12}  {m.family_bills:>6}  "
                f"{m.status or '':<10}  {m.title[:70]}"
            )
        hidden += max(len(matches) - limit, 0)
    more = f"; {hidden:,} more not shown (--limit, per list)" if hidden > 0 else ""
    noun = "family matches" if len(result.matches) == 1 else "families match"
    typer.echo(
        f"{len(result.matches):,} {noun}{more}. Share: of the words in the query's searched "
        f"sections, the share in sections the family matches. Best: the highest estimated "
        f"similarity of a matched section. Cos: the semantic stage's best cosine. By: n, the "
        f"near-duplicate stage; s, the semantic stage. Family: the state's near-identical "
        f"bills, counted "
        f"once. {time.monotonic() - started:.1f}s"
    )


_STAGE_MARKS = {"near_duplicate": "n", "semantic": "s"}


@model_app.command("fetch")
def model_fetch(
    names: Annotated[
        list[str] | None,
        typer.Argument(help=f"Models to fetch; all when none: {', '.join(MODELS)}."),
    ] = None,
    models_dir: Annotated[
        Path, typer.Option("--models", help="Directory of model files (local).")
    ] = MODELS_DIR,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Download each model's pinned files that are missing or changed -> data/models/<name>/."""
    wanted = names or list(MODELS)
    unknown = [n for n in wanted if n not in MODELS]
    if unknown:
        typer.echo(
            f"error: unknown model {', '.join(unknown)}; known: {', '.join(MODELS)}", err=True
        )
        raise typer.Exit(1)
    for name in wanted:
        try:
            fetched, present = fetch_model(name, models_dir, floor=int(min_free_gib * GiB))
        except (FetchError, InsufficientDiskSpaceError) as e:
            typer.echo(f"error: {name}: {e}", err=True)
            raise typer.Exit(1) from e
        typer.echo(f"{name}: {fetched} fetched, {present} already present -> {models_dir / name}")
