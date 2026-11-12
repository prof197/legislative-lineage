"""The `lineage` command line. Each subcommand is a thin wrapper over a library function."""

import datetime
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from numpy.typing import NDArray

from lineage import __version__
from lineage.acts import ACTS, ACTS_DIR, HELD_OUT_ACTS, HELD_OUT_DIR, ActTextError
from lineage.agreement import PAIRS as AGREEMENT_PAIRS
from lineage.agreement import SAMPLE as AGREEMENT_SAMPLE
from lineage.agreement import SAMPLE_REPORT as AGREEMENT_SAMPLE_REPORT
from lineage.agreement import SEED as AGREEMENT_SEED
from lineage.agreement import (
    AgreementError,
    agreement,
    draw_subsample,
    subsample_report,
    subsample_rows,
    write_subsample,
)
from lineage.align import Embedder
from lineage.answer_key import ANSWER_KEY, HELD_OUT_KEY, build_answer_key, render_answer_key
from lineage.commission import RESULTS, CommissionError, commission_recall, write_results
from lineage.cost import REPORT as COST_REPORT
from lineage.cost import cost_report
from lineage.cost import write_report as write_cost_report
from lineage.disk import GiB, InsufficientDiskSpaceError
from lineage.doctypes import RULES, DocTypesError, build_eligibility, rule_samples
from lineage.embed import embed as embed_texts
from lineage.embed import load_model
from lineage.fetch import SOURCE, FetchError, fetch_raw
from lineage.index import build_index
from lineage.ingest import ingest as run_ingest
from lineage.label_page import DATA as LABEL_PAGE_DATA
from lineage.label_page import OUT as LABEL_PAGE_OUT
from lineage.label_page import append_rows as append_label_rows
from lineage.label_page import imported_rows as imported_label_rows
from lineage.label_page import load_saved as load_saved_labels
from lineage.label_page import page_data as label_page_data
from lineage.label_page import write_page as write_label_page
from lineage.labels import (
    LABELS_PATH,
    Item,
    LabelError,
    current_labels,
    load_labels,
    load_sample,
    pending,
    person,
    read_items,
    section_hashes,
    session,
)
from lineage.labels import PERSON as PERSON_PREFIX
from lineage.labels import model_labeler as model_labeler_name
from lineage.memory import MemoryBoundError
from lineage.model_labels import BUDGET as LABEL_BUDGET
from lineage.model_labels import MAX_COST as LABEL_MAX_COST
from lineage.model_labels import MODEL_ID as LABEL_MODEL
from lineage.model_labels import PROMPT_VERSION as LABEL_PROMPT_VERSION
from lineage.model_labels import PROMPTS as LABEL_PROMPTS
from lineage.model_labels import SPEND as LABEL_SPEND
from lineage.model_labels import anthropic_ask as anthropic_label_ask
from lineage.model_labels import (
    anthropic_batches,
    batch_usage,
    label_in_batch,
    label_with_model,
    model_pending,
    projected_cost,
)
from lineage.model_labels import record_spend as record_label_spend
from lineage.model_labels import spent as label_spent
from lineage.models import MODELS, MODELS_DIR, ModelError, fetch_model
from lineage.neighbors import EF_SEARCH, RECALL_FLOOR, build_neighbors
from lineage.neighbors import REPORT as NEIGHBORS_REPORT
from lineage.neighbors import write_report as write_neighbors_report
from lineage.pair_data import PAIR_DATA_DIR, PairDataError, fetch_pair_data
from lineage.pair_eval import REPORT as PAIR_REPORT
from lineage.pair_eval import evaluate as evaluate_pairs
from lineage.pair_eval import write_report as write_pair_report
from lineage.precision import SAMPLE as PRECISION_SAMPLE
from lineage.precision import SAMPLE_REPORT as PRECISION_SAMPLE_REPORT
from lineage.precision import PrecisionError, build_sample, estimate, load_sizes, write_sample
from lineage.query import IndexMissingError, query_text, require_index
from lineage.relatives_sample import LABELS as RELATIVES_LABELS
from lineage.relatives_sample import PARTS as RELATIVES_PARTS
from lineage.relatives_sample import SAMPLE as RELATIVES_SAMPLE
from lineage.relatives_sample import (
    RelativesSampleError,
    build_relatives_sample,
    load_pages,
    with_passages,
)
from lineage.relatives_sample import estimate as relatives_estimate
from lineage.relatives_sample import write_sample as write_relatives_sample
from lineage.results_table import EXPORTED_ONLY, README, ResultsTableError, rendered_readme
from lineage.sectioning import SectioningError, build_sections
from lineage.semantic import MODEL as SEMANTIC_MODEL
from lineage.semantic import SemanticSearch, open_search
from lineage.semantic_pairs import REPORT as SEMANTIC_PAIRS_REPORT
from lineage.semantic_pairs import SemanticPairsError, build_semantic_pairs
from lineage.semantic_pairs import write_report as write_semantic_pairs_report
from lineage.sessions import (
    CARRYOVER,
    SCRAPERS_REPO,
    SessionsError,
    fetch_scraper_sessions,
    session_rows,
    write_sessions,
)
from lineage.snapshot import SnapshotError, load_manifest, scan, write_manifest
from lineage.state_boilerplate import (
    FLAGGED_SAMPLE,
    FLAGGED_SAMPLE_REPORT,
    MIN_FAMILIES,
    StateBoilerplateError,
    build_flagged_sample,
    build_state_boilerplate,
    has_counts,
    with_flag,
)
from lineage.state_boilerplate import REPORT as STATE_BOILERPLATE_REPORT
from lineage.state_boilerplate import write_report as write_state_boilerplate_report
from lineage.stats import write_coverage
from lineage.stoplist import StoplistError, build_stoplist, file_sha256
from lineage.text_inventory import PLAN, REPORT, StalePlanError, build_inventory
from lineage.text_merge import merge_texts
from lineage.text_quality import REPORT as QUALITY
from lineage.text_quality import STORE, run_sample
from lineage.text_run import REPORT as FETCH_REPORT
from lineage.text_run import run_fetch
from lineage.ulc import ENACTMENTS, HELD_OUT_ENACTMENTS, UlcError
from lineage.vectors import PROCESSES as EMBED_PROCESSES
from lineage.vectors import REPORT as EMBED_REPORT
from lineage.vectors import THREADS as EMBED_THREADS
from lineage.vectors import EmbedError, embed_corpus, vectors_dir
from lineage.vectors import write_report as write_embed_report

app = typer.Typer(no_args_is_help=True, add_completion=False)
eval_app = typer.Typer(no_args_is_help=True, help="Run an eval and write its results.")
app.add_typer(eval_app, name="eval")
text_app = typer.Typer(
    no_args_is_help=True, help="The text layer: bill text fetched from the source."
)
app.add_typer(text_app, name="text")
model_app = typer.Typer(
    no_args_is_help=True, help="The embedding models the semantic stage runs, pinned by hash."
)
app.add_typer(model_app, name="model")
pair_data_app = typer.Typer(
    no_args_is_help=True,
    help="The labeled pair dataset (Kim et al., EMNLP 2021), local and for evaluation only.",
)
app.add_typer(pair_data_app, name="pair-data")

RawOption = Annotated[Path, typer.Option("--raw", help="Directory of Open States archives.")]
ManifestOption = Annotated[Path, typer.Option("--manifest", help="Snapshot manifest (committed).")]
TablesOption = Annotated[Path, typer.Option("--tables", help="Directory of Parquet tables.")]
RulesOption = Annotated[Path, typer.Option("--rules", help="Document-type rules (committed).")]
MinFreeOption = Annotated[
    float, typer.Option("--min-free-gib", help="Refuse to run if disk would drop below this.")
]
SemanticOption = Annotated[
    bool,
    typer.Option(
        "--semantic/--no-semantic",
        help="Search by meaning too, over the embedded corpus (`lineage embed`).",
    ),
]


def _model_embedder(models_dir: Path = MODELS_DIR) -> Embedder:
    model = load_model(SEMANTIC_MODEL, models_dir)

    def embed(texts: Sequence[str]) -> NDArray[np.float32]:
        return embed_texts(model, texts)

    return embed


def _semantic_search(tables: Path, models_dir: Path = MODELS_DIR) -> SemanticSearch:
    model = load_model(SEMANTIC_MODEL, models_dir)
    return open_search(tables, lambda texts: embed_texts(model, texts))


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"lineage {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            # Eager so `--version` answers before Typer demands a subcommand.
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Trace provisions shared between state bills."""


@app.command()
def snapshot(
    raw: RawOption = Path("data/raw"),
    manifest: ManifestOption = Path("snapshot/manifest.json"),
) -> None:
    """Pin the archives in RAW: write each one's size, SHA-256 and Generated At to the manifest."""
    try:
        archives = scan(raw)
    except SnapshotError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    write_manifest(archives, manifest)
    dates = sorted(a.generated_at for a in archives)
    span = f", generated {dates[0][:10]} to {dates[-1][:10]}" if dates else ""
    total = sum(a.bytes for a in archives) / GiB
    typer.echo(f"{len(archives)} archives, {total:.2f} GiB{span} -> {manifest}")


@app.command("fetch-raw")
def fetch_raw_command(
    raw: RawOption = Path("data/raw"),
    manifest: ManifestOption = Path("snapshot/manifest.json"),
    repin: Annotated[
        bool,
        typer.Option(
            "--repin", help="Keep archives whose SHA-256 changed and record them in the manifest."
        ),
    ] = False,
    source: Annotated[str, typer.Option("--source", help="Base URL of the archives.")] = SOURCE,
    pause: Annotated[float, typer.Option("--pause", help="Seconds between requests.")] = 1.0,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Download the archives the manifest pins that are missing from RAW."""
    try:
        result = fetch_raw(
            raw,
            manifest,
            source=source,
            floor_bytes=int(min_free_gib * GiB),
            repin=repin,
            pause=pause,
        )
    except (SnapshotError, InsufficientDiskSpaceError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"{result.fetched} fetched ({result.bytes / GiB:.2f} GiB), {result.present} already "
        f"present, {len(result.repinned)} re-pinned -> {raw}"
    )
    if result.problems:
        typer.echo("not fetched:\n  " + "\n  ".join(result.problems), err=True)
        raise typer.Exit(1)


@app.command()
def sessions(
    commit: Annotated[
        str, typer.Option("--commit", help="Full openstates-scrapers commit SHA to read from.")
    ],
    manifest: ManifestOption = Path("snapshot/manifest.json"),
    out: Annotated[Path, typer.Option("--out", help="Session dates CSV (committed).")] = Path(
        "snapshot/sessions.csv"
    ),
) -> None:
    """Pin start and end dates for every session in the manifest, from Open States' scrapers."""
    # A branch name would make the CSV unreproducible, so only a full SHA is accepted.
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        typer.echo("--commit must be a full 40-character commit SHA", err=True)
        raise typer.Exit(1)
    wanted = sorted({(a.state, a.session) for a in load_manifest(manifest)})
    try:
        scraped = fetch_scraper_sessions(sorted({s for s, _ in wanted}), commit=commit)
    except SessionsError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    rows = session_rows(scraped, wanted, carryover=CARRYOVER)
    write_sessions(rows, out, source=SCRAPERS_REPO, commit=commit)
    unknown = sum(r.end_date is None for r in rows)
    typer.echo(f"{len(rows)} sessions, {unknown} without an end date -> {out}")


@app.command()
def ingest(
    raw: RawOption = Path("data/raw"),
    manifest: ManifestOption = Path("snapshot/manifest.json"),
    out: Annotated[Path, typer.Option("--out", help="Directory for Parquet tables.")] = Path(
        "data/tables"
    ),
    sessions: Annotated[
        Path, typer.Option("--sessions", help="Session dates CSV (committed).")
    ] = Path("snapshot/sessions.csv"),
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Verify the raw archives against the manifest and load them into the `bills` table."""
    try:
        result = run_ingest(raw, manifest, sessions, out, floor_bytes=int(min_free_gib * GiB))
    except (SnapshotError, InsufficientDiskSpaceError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"{result.rows:,} records from {result.archives} archives -> {out / 'bills'} "
        f"({result.output_bytes / GiB:.2f} GiB, {result.seconds:.0f}s)"
    )


@app.command()
def split(
    tables: TablesOption = Path("data/tables"),
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = (
        os.cpu_count() or 1
    ),
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Split every bill's text into sections and write the `sections` table."""
    try:
        result = build_sections(tables, workers=workers, floor_bytes=int(min_free_gib * GiB))
    except (SectioningError, InsufficientDiskSpaceError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"{result.sections:,} sections from {result.with_text:,} of {result.bills:,} bills with "
        f"text; {result.by_heading:,} split at headings -> {tables / 'sections'} "
        f"({result.output_bytes / GiB:.2f} GiB, {result.seconds:.0f}s)"
    )


@app.command()
def eligible(
    tables: TablesOption = Path("data/tables"),
    rules: RulesOption = RULES,
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = Path(
        "reports/document_types.json"
    ),
) -> None:
    """Decide which records are matched, by the document-type rules, and count each rule."""
    try:
        doc = build_eligibility(tables, rules_path=rules, report_path=report)
    except DocTypesError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    totals = doc["totals"]
    typer.echo(
        f"{totals['eligible']:,} of {totals['records']:,} records eligible, "
        f"{totals['eligible_with_sections']:,} with sections; "
        f"no text: {', '.join(doc['states_without_text']) or 'none'} -> {report}"
    )


@app.command()
def boilerplate(
    tables: TablesOption = Path("data/tables"),
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
    tables: TablesOption = Path("data/tables"),
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
    tables: TablesOption = Path("data/tables"),
    min_free_gib: MinFreeOption = 20.0,
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = (
        NEIGHBORS_REPORT
    ),
    ef_search: Annotated[
        int, typer.Option("--ef-search", help="HNSW's search width; wider finds more.")
    ] = EF_SEARCH,
) -> None:
    """Every distinct window's nearest neighbors by meaning -> <tables>/semantic/<model>/."""
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
    tables: TablesOption = Path("data/tables"),
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
    tables: TablesOption = Path("data/tables"),
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
    tables: TablesOption = Path("data/tables"),
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
    tables: TablesOption = Path("data/tables"),
    limit: Annotated[int, typer.Option("--limit", help="Families to show.")] = 20,
    semantic: SemanticOption = True,
) -> None:
    """Find a bill's relatives in the index, offline: bills that share its text, by section."""
    started = time.monotonic()
    try:
        require_index(tables)
        search = _semantic_search(tables) if semantic else None
        result = query_text(file.read_text(), tables, semantic=search)
    except (IndexMissingError, StoplistError, EmbedError, ModelError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(f"{file}: {result.sections} sections, {result.searched} searched")
    if not result.matches:
        typer.echo("No relatives found in the index.")
        return
    typer.echo(
        f"{'share':>5}  {'best':>4}  {'cos':>4}  {'by':<4}  {'state':<5}  {'session':<10}  "
        f"{'bill':<12}  {'family':>6}  {'status':<10}  title"
    )
    for m in result.matches[:limit]:
        by = "".join(_STAGE_MARKS[s] for s in sorted(m.sections_by_stage))
        cos = f"{m.best_cosine:.2f}" if m.best_cosine is not None else ""
        typer.echo(
            f"{m.share:>5.0%}  {m.best_estimate:>4.2f}  {cos:>4}  {by:<4}  {m.state:<5}  "
            f"{m.session:<10}  {m.identifier:<12}  {m.family_bills:>6}  {m.status or '':<10}  "
            f"{m.title[:70]}"
        )
    hidden = len(result.matches) - limit
    more = f"; {hidden:,} more not shown (--limit)" if hidden > 0 else ""
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


@app.command()
def label(
    sample: Annotated[
        Path,
        typer.Argument(
            help="The precision sample, one pair per line.", exists=True, dir_okay=False
        ),
    ],
    labeler: Annotated[
        str, typer.Option("--labeler", help="Your GitHub handle; labels are never kept by name.")
    ],
    tables: TablesOption = Path("data/tables"),
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file, appended to (committed).")
    ] = LABELS_PATH,
    color: Annotated[bool, typer.Option("--color/--no-color", help="Mark spans in color.")] = True,
    paraphrase: Annotated[
        bool,
        typer.Option(
            "--paraphrase/--no-paraphrase",
            help="Mark paraphrased sentences, with the semantic stage's model (local).",
        ),
    ] = True,
    models_dir: Annotated[
        Path, typer.Option("--models", help="Directory of model files (local).")
    ] = MODELS_DIR,
) -> None:
    """Label sampled matches by hand, unrelated (0) to identical (4); stop and resume at will."""
    try:
        who = person(labeler)
        embedder = _model_embedder(models_dir) if paraphrase else None
        items, kept = _current_items(sample, tables, labels)
    except (LabelError, ModelError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    todo = pending(items, kept)
    if not todo:
        typer.echo(f"Every pair in {sample} has a label in {labels}.")
        return
    labels.parent.mkdir(parents=True, exist_ok=True)
    done = session(
        todo,
        ask=lambda prompt: str(
            typer.prompt(prompt, default="", show_default=False, prompt_suffix="")
        ),
        show=typer.echo,
        labels_path=labels,
        labeler=who,
        today=datetime.date.today().isoformat(),
        embed=embedder,
        color=color,
    )
    typer.echo(f"{done} labeled, {len(todo) - done} left -> {labels}")


@app.command("label-page")
def label_page(
    sample: Annotated[
        Path,
        typer.Argument(
            help="The precision sample, one pair per line.", exists=True, dir_okay=False
        ),
    ],
    labeler: Annotated[
        str, typer.Option("--labeler", help="Your GitHub handle; labels are never kept by name.")
    ],
    tables: TablesOption = Path("data/tables"),
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file; pairs a person labeled are left out.")
    ] = LABELS_PATH,
    out: Annotated[
        Path, typer.Option("--out", help="Directory for the page and its data (local).")
    ] = LABEL_PAGE_OUT,
    paraphrase: Annotated[
        bool,
        typer.Option(
            "--paraphrase/--no-paraphrase",
            help="Mark paraphrased sentences, with the semantic stage's model (local).",
        ),
    ] = True,
    models_dir: Annotated[
        Path, typer.Option("--models", help="Directory of model files (local).")
    ] = MODELS_DIR,
) -> None:
    """Write a labeling page for a phone: the pending pairs, blind to stages, for an artifact."""
    try:
        embedder = _model_embedder(models_dir) if paraphrase else None
        items, kept = _current_items(sample, tables, labels)
        data = label_page_data(pending(items, kept), handle=labeler, sample=sample, embed=embedder)
    except (LabelError, ModelError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    page = write_label_page(data, out)
    typer.echo(f"{len(data['pairs'])} pending pairs -> {page} and {out / LABEL_PAGE_DATA}")


@app.command("label-import")
def label_import(
    saved: Annotated[
        Path,
        typer.Argument(
            help="The page's saved labels, one JSON object per line.", exists=True, dir_okay=False
        ),
    ],
    sample: Annotated[
        Path,
        typer.Option("--sample", help="The sample the page was written for.", exists=True),
    ],
    labeler: Annotated[
        str, typer.Option("--labeler", help="The GitHub handle the page was written for.")
    ],
    tables: TablesOption = Path("data/tables"),
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file, appended to (committed).")
    ] = LABELS_PATH,
) -> None:
    """Append a labeling page's saved labels to the labels file, each checked against the sample."""
    try:
        items, kept = _current_items(sample, tables, labels)
        rows = imported_label_rows(load_saved_labels(saved), items, kept, handle=labeler)
    except LabelError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    append_label_rows(labels, rows)
    left = len(pending(items, [*kept, *rows]))
    typer.echo(f"{len(rows)} labels imported, {left} pairs of {sample} left -> {labels}")


def _current_items(
    sample: Path, tables: Path, labels: Path
) -> tuple[list[Item], list[dict[str, Any]]]:
    """The sample's pairs read from the tables, and the labels whose sections still exist,
    saying how many were dropped."""
    items = read_items(tables, load_sample(sample))
    existing = load_labels(labels)
    kept, dropped = current_labels(existing, section_hashes(tables, existing))
    if dropped:
        noun = "label no longer matches" if dropped == 1 else "labels no longer match"
        typer.echo(f"{dropped} {noun} its section's text and is not counted.")
    return items, kept


@app.command("precision-sample")
def precision_sample(
    tables: TablesOption = Path("data/tables"),
    sample: Annotated[
        Path, typer.Option("--sample", help="The sample file, one pair per line (committed).")
    ] = PRECISION_SAMPLE,
    report: Annotated[
        Path | None, typer.Option("--report", help="The draw's report JSON (committed).")
    ] = None,
    flagged: Annotated[
        bool,
        typer.Option(
            "--flagged", help="Draw from the pairs the state boilerplate flag drops instead."
        ),
    ] = False,
) -> None:
    """Draw the precision sample from both stages' cross-state pairs, stratified by stage."""
    if flagged:
        sample = FLAGGED_SAMPLE if sample == PRECISION_SAMPLE else sample
        report = report or FLAGGED_SAMPLE_REPORT
    report = report or PRECISION_SAMPLE_REPORT
    try:
        rows, doc = build_flagged_sample(tables) if flagged else build_sample(tables)
    except (PrecisionError, SemanticPairsError, StateBoilerplateError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_sample(rows, doc, sample_path=sample, report_path=report)
    sizes = ", ".join(f"{name} {s['drawn']} of {s['size']:,}" for name, s in doc["strata"].items())
    typer.echo(f"{len(rows)} pairs drawn: {sizes} -> {sample}, {report}")


@app.command("agreement-sample")
def agreement_sample(
    source: Annotated[
        Path,
        typer.Argument(
            help="The precision sample, one pair per line.", exists=True, dir_okay=False
        ),
    ] = PRECISION_SAMPLE,
    sample: Annotated[
        Path, typer.Option("--sample", help="The subsample file, one pair per line (committed).")
    ] = AGREEMENT_SAMPLE,
    report: Annotated[
        Path, typer.Option("--report", help="The draw's report JSON (committed).")
    ] = AGREEMENT_SAMPLE_REPORT,
) -> None:
    """Draw the pairs a person labels to check the model's labels, stratified by stage."""
    try:
        drawn = draw_subsample(load_sample(source))
    except (LabelError, PrecisionError, AgreementError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    doc = subsample_report(drawn, source=source, size=AGREEMENT_PAIRS, seed=AGREEMENT_SEED)
    write_subsample(subsample_rows(drawn), doc, sample_path=sample, report_path=report)
    counts = ", ".join(f"{name} {n}" for name, n in doc["strata"].items())
    typer.echo(f"{len(drawn)} pairs drawn: {counts} -> {sample}, {report}")


@app.command("relatives-sample")
def relatives_sample(
    tables: TablesOption = Path("data/tables"),
    part: Annotated[
        str,
        typer.Option(
            "--part", help="tuning (the draw's first bills) or reporting (the next ones)."
        ),
    ] = "tuning",
    sample: Annotated[
        Path | None,
        typer.Option("--sample", help="The sample file, one relative per line (committed)."),
    ] = None,
    report: Annotated[
        Path | None, typer.Option("--report", help="The draw's report JSON (committed).")
    ] = None,
) -> None:
    """Draw listed relatives of random bills of one state, to measure what a page lists."""
    if part not in RELATIVES_PARTS:
        typer.echo(f"error: --part is one of {', '.join(RELATIVES_PARTS)}, not {part!r}", err=True)
        raise typer.Exit(1)
    skip, default_sample, default_report = RELATIVES_PARTS[part]
    sample = sample or default_sample
    report = report or default_report
    try:
        rows, doc = build_relatives_sample(tables, skip=skip)
    except (RelativesSampleError, SemanticPairsError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_relatives_sample(rows, doc, sample_path=sample, report_path=report)
    typer.echo(
        f"{len(rows)} relatives drawn from {doc['state_bills_with_a_relative']:,} "
        f"{doc['state']} bills with a relative -> {sample}, {report}"
    )


@app.command("label-model")
def label_model(
    sample: Annotated[
        Path,
        typer.Argument(
            help="The precision sample, one pair per line.", exists=True, dir_okay=False
        ),
    ] = PRECISION_SAMPLE,
    tables: TablesOption = Path("data/tables"),
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file, appended to (committed).")
    ] = LABELS_PATH,
    model: Annotated[str, typer.Option("--model", help="The model asked.")] = LABEL_MODEL,
    prompt_version: Annotated[
        str,
        typer.Option("--prompt", help=f"The prompt and effort asked: {', '.join(LABEL_PROMPTS)}."),
    ] = LABEL_PROMPT_VERSION,
    live: Annotated[
        bool, typer.Option("--live", help="Ask the model; without it nothing is sent.")
    ] = False,
    max_cost: Annotated[
        float, typer.Option("--max-cost", help="Dollars the run may be projected to cost.")
    ] = LABEL_MAX_COST,
    budget: Annotated[
        float,
        typer.Option("--budget", help="Dollars every batch may cost, spent and projected."),
    ] = LABEL_BUDGET,
    spend: Annotated[
        Path, typer.Option("--spend", help="The batches' spend ledger (committed).")
    ] = LABEL_SPEND,
    batch: Annotated[
        bool, typer.Option("--batch", help="Ask through the Batches API, at half price.")
    ] = False,
    batch_id: Annotated[
        str | None, typer.Option("--batch-id", help="Collect a batch already sent.")
    ] = None,
    models_dir: Annotated[
        Path, typer.Option("--models", help="Directory of model files (local), for alignment.")
    ] = MODELS_DIR,
    also: Annotated[
        list[str] | None,
        typer.Option(
            "--also",
            help="Another SAMPLE=LABELS asked in the same batch under the same prompt; repeatable.",
        ),
    ] = None,
) -> None:
    """Provisional labels from a model, kept apart from people's; asks only with --live."""
    if prompt_version not in LABEL_PROMPTS:
        typer.echo(
            f"error: no prompt {prompt_version!r}; one of {', '.join(LABEL_PROMPTS)}", err=True
        )
        raise typer.Exit(1)
    prompt = LABEL_PROMPTS[prompt_version]
    targets = [(sample, labels)]
    for extra in also or []:
        extra_sample, sep, extra_labels = extra.partition("=")
        if not sep or not extra_sample or not extra_labels:
            typer.echo(f"error: --also takes SAMPLE=LABELS, not {extra!r}", err=True)
            raise typer.Exit(1)
        targets.append((Path(extra_sample), Path(extra_labels)))
    if len(targets) > 1 and not (batch or batch_id is not None):
        typer.echo("error: --also asks several samples in one batch; add --batch", err=True)
        raise typer.Exit(1)
    todo: list[Item] = []
    paths: dict[str, Path] = {}
    try:
        for target_sample, target_labels in targets:
            items, kept = _current_items(target_sample, tables, target_labels)
            for item in model_pending(items, kept, prompt):
                if item.pair.pair in paths:
                    raise LabelError(f"pair id {item.pair.pair} is in two of the samples asked")
                paths[item.pair.pair] = target_labels
                todo.append(item)
    except LabelError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    if prompt.passages:
        try:
            todo = with_passages(todo, _model_embedder(models_dir))
        except ModelError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
    noun = "pair" if len(todo) == 1 else "pairs"
    batched = batch or batch_id is not None
    try:
        cost = projected_cost(todo, model=model, batch=batched, prompt=prompt)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    projected = f"projected at most ${cost:,.2f}"
    already = label_spent(spend)
    if not live:
        typer.echo(
            f"{len(todo)} {noun} would be asked of {model} under {prompt.version}, {projected}; "
            f"${already:,.2f} of the ${budget:,.2f} budget spent; run with --live to ask."
        )
        return
    if cost > max_cost:
        typer.echo(
            f"error: {len(todo)} {noun} {projected}, over the ${max_cost:,.2f} cap; nothing asked",
            err=True,
        )
        raise typer.Exit(1)
    if batch_id is None and already + cost > budget:
        typer.echo(
            f"error: {len(todo)} {noun} {projected}, with ${already:,.2f} spent, over the "
            f"${budget:,.2f} budget; nothing asked",
            err=True,
        )
        raise typer.Exit(1)
    for _, target_labels in targets:
        target_labels.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    if batched:
        batches = anthropic_batches()
        done, unusable, sent = label_in_batch(
            todo,
            batches=batches,
            model=model,
            labels_path=paths,
            today=today,
            wait=lambda: time.sleep(60),
            batch_id=batch_id,
            submitted=lambda b: typer.echo(f"sent as batch {b}; collect with --batch-id {b}"),
            prompt=prompt,
        )
        ledger = record_label_spend(
            spend, batch_id=sent, model=model, prompt=prompt.version,
            usage=batch_usage(batches, sent), today=today,
        )  # fmt: skip
        files = ", ".join(sorted({str(p) for _, p in targets}))
        typer.echo(f"{done} labeled from batch {sent}, {unusable} unusable replies -> {files}")
        typer.echo(f"${ledger['spent']:,.2f} of the ${budget:,.2f} budget spent -> {spend}")
        return
    done, unusable = label_with_model(
        todo, ask=anthropic_label_ask(model, prompt), labels_path=labels, today=today,
        prompt=prompt,
    )  # fmt: skip
    typer.echo(f"{done} labeled by {model}, {unusable} unusable replies -> {labels}")


@app.command("label-spend")
def label_spend(
    batch_id: Annotated[str, typer.Option("--batch-id", help="An ended batch to record.")],
    prompt_version: Annotated[
        str, typer.Option("--prompt", help="The prompt the batch was asked under.")
    ] = LABEL_PROMPT_VERSION,
    model: Annotated[str, typer.Option("--model", help="The model asked.")] = LABEL_MODEL,
    spend: Annotated[
        Path, typer.Option("--spend", help="The batches' spend ledger (committed).")
    ] = LABEL_SPEND,
) -> None:
    """Record what an ended batch cost, from its results' token counts; labels nothing."""
    batches = anthropic_batches()
    usage = batch_usage(batches, batch_id)
    ledger = record_label_spend(
        spend, batch_id=batch_id, model=model, prompt=prompt_version, usage=usage,
        today=datetime.date.today().isoformat(),
    )  # fmt: skip
    replies, input_tokens, output_tokens = usage
    typer.echo(
        f"batch {batch_id}: {replies} replies, {input_tokens:,} input and {output_tokens:,} "
        f"output tokens; ${ledger['spent']:,.2f} of the ${ledger['budget']:,.2f} budget spent "
        f"-> {spend}"
    )


@app.command()
def stats(
    tables: TablesOption = Path("data/tables"),
    manifest: ManifestOption = Path("snapshot/manifest.json"),
    sessions: Annotated[
        Path, typer.Option("--sessions", help="Session dates CSV (committed).")
    ] = Path("snapshot/sessions.csv"),
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = Path(
        "reports/coverage.json"
    ),
    rules: RulesOption = RULES,
    samples: Annotated[
        bool,
        typer.Option(
            "--samples",
            help="Also write sample titles for each document-type rule to a local file.",
        ),
    ] = False,
    samples_report: Annotated[
        Path, typer.Option("--samples-report", help="Sample titles JSON (local, gitignored).")
    ] = Path("data/reports/document_type_samples.json"),
) -> None:
    """Measure text coverage, sections, document types and status per state and session."""
    doc = write_coverage(tables, sessions_path=sessions, manifest_path=manifest, report_path=report)
    t = doc["totals"]
    typer.echo(
        f"{t['records']:,} records: {t['missing_text']:,} without text, {t['blank_text']:,} "
        f"blank, {t['with_sections']:,} with sections ({t['split_at_headings']:,} at headings), "
        f"{t['sections']:,} sections -> {report}"
    )
    if samples:
        try:
            found = rule_samples(tables, rules_path=rules)
        except DocTypesError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from None
        samples_report.parent.mkdir(parents=True, exist_ok=True)
        samples_report.write_text(json.dumps(found, indent=1, ensure_ascii=False) + "\n")
        typer.echo(f"sample titles for {len(found['rules'])} rules -> {samples_report}")


@eval_app.command("commission")
def eval_commission(
    tables: TablesOption = Path("data/tables"),
    acts: Annotated[
        Path | None,
        typer.Option(
            "--acts",
            help="Directory of uniform act texts (local, gitignored) [default: data/acts/, "
            "or data/acts/held-out/ with --held-out].",
        ),
    ] = None,
    enactments: Annotated[
        Path | None,
        typer.Option(
            "--enactments",
            help="The commission's enactment lists (committed) [default: the development "
            "lists, or the held-out lists with --held-out].",
        ),
    ] = None,
    sessions: Annotated[
        Path, typer.Option("--sessions", help="Session dates CSV (committed).")
    ] = Path("snapshot/sessions.csv"),
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
    held_out: Annotated[
        bool, typer.Option("--held-out", help="Evaluate the held-out acts (with --final only).")
    ] = False,
    final: Annotated[
        bool, typer.Option("--final", help="The one final evaluation of the held-out acts.")
    ] = False,
    semantic: SemanticOption = True,
) -> None:
    """Commission recall: what share of each uniform act's enactments its text finds."""
    if held_out and not final:
        typer.echo(
            "The held-out acts are evaluated once, at the end of M5 (DESIGN.md#commission-recall). "
            "Pass --final for that run.",
            err=True,
        )
        raise typer.Exit(1)
    if final and not held_out:
        typer.echo("--final applies only with --held-out.", err=True)
        raise typer.Exit(1)
    try:
        search = _semantic_search(tables) if semantic else None
        section = commission_recall(
            tables,
            acts_dir=acts or (HELD_OUT_DIR if held_out else ACTS_DIR),
            enactments_path=enactments or (HELD_OUT_ENACTMENTS if held_out else ENACTMENTS),
            sessions_path=sessions,
            act_ids=HELD_OUT_ACTS if held_out else ACTS,
            development_path=ENACTMENTS if held_out else None,
            semantic=search,
        )
    except (
        CommissionError,
        IndexMissingError,
        StoplistError,
        UlcError,
        EmbedError,
        ModelError,
    ) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    write_results(results, "commission_held_out" if held_out else "commission", section)
    pooled = section["recall"]["pooled"]
    for label, key in (("all enactments", "all"), ("with text", "with_text")):
        r = pooled[key]
        share = r["found"] / r["enactments"] if r["enactments"] else 0.0
        typer.echo(f"recall, {label}: {r['found']}/{r['enactments']} = {share:.1%}")
    if section["stage_ablation"]:
        for stage, r in section["stage_ablation"].items():
            typer.echo(
                f"  {stage}: {r['pooled']['all']['found']}/{r['pooled']['all']['enactments']}"
            )
    misses = ", ".join(f"{k} {v}" for k, v in section["miss_reasons"].items())
    typer.echo(f"misses: {misses or 'none'} -> {results}")


@eval_app.command("answer-key")
def eval_answer_key(
    tables: TablesOption = Path("data/tables"),
    enactments: Annotated[
        Path | None,
        typer.Option(
            "--enactments",
            help="The commission's enactment lists (committed) [default: the development "
            "lists, or the held-out lists with --held-out].",
        ),
    ] = None,
    sessions: Annotated[
        Path, typer.Option("--sessions", help="Session dates CSV (committed).")
    ] = Path("snapshot/sessions.csv"),
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Answer key CSV (committed) [default: reports/answer_key.csv, or "
            "reports/answer_key_heldout.csv with --held-out].",
        ),
    ] = None,
    held_out: Annotated[
        bool,
        typer.Option(
            "--held-out", help="Resolve the held-out lists, skipping the development set's bills."
        ),
    ] = False,
) -> None:
    """The bills each commission list row resolved to, for checking the mapping by eye.

    Resolving the held-out lists runs no matching, so it may run before the final evaluation.
    """
    out = out or (HELD_OUT_KEY if held_out else ANSWER_KEY)
    try:
        note, rows = build_answer_key(
            tables,
            enactments_path=enactments or (HELD_OUT_ENACTMENTS if held_out else ENACTMENTS),
            sessions_path=sessions,
            development_path=ENACTMENTS if held_out else None,
            acts=frozenset(HELD_OUT_ACTS) if held_out else None,
        )
    except UlcError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_answer_key(rows, note))
    counted = {(r["act"], r["jurisdiction"], r["year"]) for r in rows if r["counted"] == "yes"}
    resolved = {(r["act"], r["jurisdiction"], r["year"]) for r in rows if r["bill_id"]}
    typer.echo(f"{len(resolved)} of {len(counted)} counted rows resolved to a bill -> {out}")


@eval_app.command("readme")
def eval_readme(
    readme: Annotated[Path, typer.Option("--readme", help="The README to update.")] = README,
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
    exported_only: Annotated[
        Path,
        typer.Option("--exported-only", help="Results on Open States' text alone (committed)."),
    ] = EXPORTED_ONLY,
) -> None:
    """Regenerate the README's results table from the results files."""
    try:
        text = rendered_readme(
            readme.read_text(),
            json.loads(results.read_text()),
            json.loads(exported_only.read_text()),
        )
    except ResultsTableError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    readme.write_text(text)
    typer.echo(f"results table <- {results}, {exported_only} -> {readme}")


@text_app.command("inventory")
def text_inventory(
    tables: TablesOption = Path("data/tables"),
    plan: Annotated[Path, typer.Option("--plan", help="Fetch plan Parquet (local).")] = PLAN,
    out: Annotated[Path, typer.Option("--out", help="Inventory JSON (committed).")] = REPORT,
) -> None:
    """Choose each eligible record's version to fetch, count by host, and read robots.txt."""
    doc = build_inventory(tables, plan_path=plan, report_path=out)
    totals = doc["totals"]
    need = sum(n for k, n in totals.items() if k.startswith("without_text:"))
    typer.echo(
        f"{totals['eligible']:,} eligible records, {need:,} without usable text, "
        f"{totals.get('without_text:planned', 0):,} with a bill version to fetch, "
        f"on {len(doc['hosts'])} hosts -> {out}"
    )


@text_app.command("sample")
def text_sample(
    tables: TablesOption = Path("data/tables"),
    plan: Annotated[Path, typer.Option("--plan", help="Fetch plan Parquet (local).")] = PLAN,
    store: Annotated[Path, typer.Option("--store", help="Fetched sample (local).")] = STORE,
    out: Annotated[Path, typer.Option("--out", help="Quality report JSON (committed).")] = QUALITY,
    fetch: Annotated[bool, typer.Option(help="Fetch what the store lacks before judging.")] = True,
) -> None:
    """Fetch a seeded sample per state and format and judge its text against Open States'."""
    try:
        doc = run_sample(tables, plan_path=plan, store=store, report_path=out, fetch=fetch)
    except StalePlanError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    cells = [c for formats in doc["states"].values() for c in formats.values()]
    sampled = sum(sum(c["sampled"].values()) for c in cells)
    ok = sum(c["outcomes"].get("ok", 0) for c in cells)
    typer.echo(f"{ok:,} of {sampled:,} sampled documents extracted, {len(cells)} cells -> {out}")


@text_app.command("fetch")
def text_fetch(
    tables: TablesOption = Path("data/tables"),
    plan: Annotated[Path, typer.Option("--plan", help="Fetch plan Parquet (local).")] = PLAN,
    out: Annotated[Path, typer.Option("--out", help="Fetch report JSON (committed).")] = (
        FETCH_REPORT
    ),
    fetch: Annotated[bool, typer.Option(help="Fetch what the store lacks before counting.")] = True,
) -> None:
    """Fetch every planned record's text, as rules/text_layer.toml says; resumable."""
    try:
        doc = run_fetch(tables, plan_path=plan, report_path=out, fetch=fetch)
    except (InsufficientDiskSpaceError, StalePlanError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    ok = doc["outcomes"].get("ok", 0)
    left = sum(doc["left_out"].values())
    typer.echo(
        f"{ok:,} of {doc['jobs']:,} records fetched with usable text, {left:,} left out -> {out}"
    )


@text_app.command("merge")
def text_merge(
    tables: TablesOption = Path("data/tables"),
    plan: Annotated[Path, typer.Option("--plan", help="Fetch plan Parquet (local).")] = PLAN,
    exported_only: Annotated[
        bool, typer.Option(help="Use only the exported text, to measure without the fetch.")
    ] = False,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Choose each record's text, fetched or exported -> data/tables/texts/."""
    try:
        counts = merge_texts(
            tables, plan_path=plan, exported_only=exported_only,
            floor_bytes=int(min_free_gib * GiB),
        )  # fmt: skip
    except (InsufficientDiskSpaceError, StalePlanError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    shown = ", ".join(f"{source} {n:,}" for source, n in counts.items())
    typer.echo(f"{sum(counts.values()):,} records by text source: {shown} -> {tables / 'texts'}")


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


@pair_data_app.command("fetch")
def pair_data_fetch(
    directory: Annotated[
        Path, typer.Option("--dir", help="Directory of the dataset's files (local).")
    ] = PAIR_DATA_DIR,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Download the human-annotated pair files at their pinned commit -> data/pair_data/."""
    try:
        fetched, present = fetch_pair_data(directory, floor=int(min_free_gib * GiB))
    except (FetchError, InsufficientDiskSpaceError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{fetched} fetched, {present} already present -> {directory}")


@eval_app.command("agreement")
def eval_agreement(
    sample: Annotated[
        Path,
        typer.Argument(
            help="The agreement subsample, one pair per line.", exists=True, dir_okay=False
        ),
    ] = AGREEMENT_SAMPLE,
    tables: TablesOption = Path("data/tables"),
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file (committed).")
    ] = LABELS_PATH,
    model_labeler: Annotated[
        str, typer.Option("--model-labeler", help="The model labeler compared, model id@prompt.")
    ] = model_labeler_name(LABEL_MODEL, LABEL_PROMPT_VERSION),
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
) -> None:
    """A person's labels against a model's on the subsample: weighted kappa, cutoff agreement."""
    try:
        items, kept = _current_items(sample, tables, labels)
        doc = agreement(items, kept, model_labeler=model_labeler)
    except (LabelError, PrecisionError, AgreementError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    # One comparison per model labeler, so a re-label at another effort keeps the first.
    previous: dict[str, Any] = (
        json.loads(results.read_text()).get("label_agreement", {}) if results.exists() else {}
    )
    by_labeler = {k: v for k, v in previous.items() if k != "command"}
    by_labeler[model_labeler] = doc
    write_results(
        results,
        "label_agreement",
        {"command": "uv run lineage eval agreement", **dict(sorted(by_labeler.items()))},
    )
    kappa = doc["weighted_kappa"]
    share = doc["cutoff_agreement"]
    verdict = "passes" if doc["passes"] else "does not pass"
    typer.echo(
        f"{doc['pairs_labeled_by_both']} of {doc['pairs_in_subsample']} pairs labeled by both; "
        f"weighted kappa {'-' if kappa is None else f'{kappa:.3f}'} (bar {doc['kappa_from']}), "
        f"cutoff agreement {'-' if share is None else f'{share:.1%}'} "
        f"(bar {doc['cutoff_agreement_from']:.0%}): {verdict}"
    )
    typer.echo(f"-> {results}")


@eval_app.command("relatives")
def eval_relatives(
    sample: Annotated[
        Path,
        typer.Argument(help="The relatives sample, one per line.", exists=True, dir_okay=False),
    ] = RELATIVES_SAMPLE,
    tables: TablesOption = Path("data/tables"),
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file (committed).")
    ] = RELATIVES_LABELS,
    model_labeler: Annotated[
        str, typer.Option("--model-labeler", help="The model labeler counted, model id@prompt.")
    ] = model_labeler_name(LABEL_MODEL, "relatives-v1"),
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
) -> None:
    """The precision of what a page lists, from the sampled relatives' labels."""
    try:
        items, kept = _current_items(sample, tables, labels)
        pages = load_pages(sample)
    except LabelError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    section: dict[str, Any] = {
        "command": "uv run lineage eval relatives",
        "person": relatives_estimate(items, kept, pages, labeler=PERSON_PREFIX),
        "provisional_model": relatives_estimate(
            items, kept, pages, labeler=model_labeler, exact=True
        ),
    }
    write_results(results, "relatives_precision", section)
    for who in ("person", "provisional_model"):
        overall = section[who]["overall"]
        shown = (
            "no labels"
            if overall["precision"] is None
            else (
                f"{overall['correct']}/{overall['labeled']} = {overall['precision']:.3f} "
                f"(95% {overall['interval_95'][0]:.3f}-{overall['interval_95'][1]:.3f})"
            )
        )
        typer.echo(f"{who.replace('_', ' ')}: {shown}")
    typer.echo(f"-> {results}")


@eval_app.command("precision")
def eval_precision(
    sample: Annotated[
        Path,
        typer.Argument(
            help="The precision sample, one pair per line.", exists=True, dir_okay=False
        ),
    ] = PRECISION_SAMPLE,
    tables: TablesOption = Path("data/tables"),
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file (committed).")
    ] = LABELS_PATH,
    strata: Annotated[
        Path, typer.Option("--strata", help="The draw's report, with each stratum's size.")
    ] = PRECISION_SAMPLE_REPORT,
    flag_report: Annotated[
        Path, typer.Option("--flag-report", help="The state boilerplate flag's report.")
    ] = STATE_BOILERPLATE_REPORT,
    flagged: Annotated[
        Path, typer.Option("--flagged", help="The sample of the pairs the flag drops.")
    ] = FLAGGED_SAMPLE,
    flagged_report: Annotated[
        Path, typer.Option("--flagged-report", help="That sample's draw report.")
    ] = FLAGGED_SAMPLE_REPORT,
    model_labeler: Annotated[
        str, typer.Option("--model-labeler", help="The model labeler counted, model id@prompt.")
    ] = model_labeler_name(LABEL_MODEL, LABEL_PROMPT_VERSION),
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
) -> None:
    """Precision per stratum, stage and overall, from people's labels and, apart, a model's,
    with the state boilerplate flag off and on."""
    try:
        items, kept = _current_items(sample, tables, labels)
        sizes = load_sizes(strata)
        measured = estimate(items, kept, sizes, model_labeler=model_labeler)
        dropped = (
            read_items(tables, load_sample(flagged))
            if flagged.exists() and has_counts(tables)
            else None
        )
        measured["state_boilerplate_flag"] = with_flag(
            tables,
            items,
            kept,
            sizes,
            report_path=flag_report,
            dropped_items=dropped,
            dropped_report=flagged_report,
            model_labeler=model_labeler,
        )
    except (LabelError, PrecisionError, StateBoilerplateError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    section: dict[str, Any] = {"command": "uv run lineage eval precision", **measured}
    write_results(results, "precision", section)
    for who in ("person", "provisional_model"):
        overall = section[who]["overall"]
        shown = (
            "not every stratum labeled"
            if overall is None
            else (
                f"{overall['precision']:.3f} (95% {overall['interval_95'][0]:.3f}"
                f"-{overall['interval_95'][1]:.3f})"
            )
        )
        typer.echo(f"{who.replace('_', ' ')}: {shown}")
        flag = section["state_boilerplate_flag"]
        if flag is not None and flag["flag_on"][who]["overall"] is not None:
            on = flag["flag_on"][who]["overall"]
            typer.echo(
                f"  with the state boilerplate flag: {on['precision']:.3f} "
                f"(95% {on['interval_95'][0]:.3f}-{on['interval_95'][1]:.3f})"
            )
    typer.echo(f"-> {results}")


@eval_app.command("cost")
def eval_cost(
    tables: TablesOption = Path("data/tables"),
    acts: Annotated[
        Path, typer.Option("--acts", help="Directory of the development act texts (local).")
    ] = ACTS_DIR,
    reports: Annotated[
        Path, typer.Option("--reports", help="The committed reports the build wrote.")
    ] = Path("reports"),
    report: Annotated[Path, typer.Option("--report", help="Report JSON (committed).")] = (
        COST_REPORT
    ),
    semantic: SemanticOption = True,
) -> None:
    """Cost and runtime per stage: the build's from its reports, each query's timed here."""
    try:
        require_index(tables)
        search = _semantic_search(tables) if semantic else None
        doc = cost_report(tables, acts, reports, search)
    except (IndexMissingError, EmbedError, ModelError, ActTextError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_cost_report(doc, report)
    for act, timed in doc["query"].items():
        warm = ", ".join(f"{k} {v:.2f}s" for k, v in timed["warm"].items())
        typer.echo(f"{act}: {warm} warm")
    typer.echo(f"-> {report}")


@eval_app.command("pair-data")
def eval_pair_data(
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model", help=f"Embedding models to score; all when none: {', '.join(MODELS)}."
        ),
    ] = None,
    tables: TablesOption = Path("data/tables"),
    directory: Annotated[
        Path, typer.Option("--dir", help="Directory of the dataset's files (local).")
    ] = PAIR_DATA_DIR,
    models_dir: Annotated[
        Path, typer.Option("--models", help="Directory of model files (local).")
    ] = MODELS_DIR,
    report: Annotated[
        Path, typer.Option("--report", help="Metrics JSON (committed).")
    ] = PAIR_REPORT,
) -> None:
    """Score both matching stages on the labeled pair dataset -> reports/pair_data.json."""
    names = model or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        typer.echo(f"error: unknown model {', '.join(unknown)}", err=True)
        raise typer.Exit(1)
    try:
        doc = evaluate_pairs(tables, names, data_dir=directory, models_dir=models_dir)
    except (PairDataError, ModelError, StoplistError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_pair_report(doc, report)
    near = doc["near_duplicate"]
    typer.echo(f"near-duplicate: precision {near['precision']}, recall {near['recall']} (test)")
    semantic = doc["semantic"]
    target = str(semantic["target_precision"])
    for name, entry in semantic["models"].items():
        at = entry["at_precision"][target]
        typer.echo(
            f"{name}: ROC AUC {entry['test_roc_auc']}; threshold {at['threshold']} at precision "
            f"{target} on train and val gives test precision {at.get('test_precision')}, "
            f"recall {at.get('test_recall')}"
        )
    typer.echo(f"-> {report}")
