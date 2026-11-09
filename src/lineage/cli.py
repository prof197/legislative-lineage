"""The `lineage` command line. Each subcommand is a thin wrapper over a library function."""

import json
import os
import time
from pathlib import Path
from typing import Annotated

import typer

from lineage import __version__
from lineage.acts import ACTS, ACTS_DIR, HELD_OUT_ACTS, HELD_OUT_DIR
from lineage.answer_key import ANSWER_KEY, HELD_OUT_KEY, build_answer_key, render_answer_key
from lineage.commission import RESULTS, CommissionError, commission_recall, write_results
from lineage.disk import GiB, InsufficientDiskSpaceError
from lineage.doctypes import RULES, DocTypesError, build_eligibility, rule_samples
from lineage.fetch import SOURCE, FetchError, fetch_raw
from lineage.index import build_index
from lineage.ingest import ingest as run_ingest
from lineage.models import MODELS, MODELS_DIR, ModelError, fetch_model
from lineage.pair_data import PAIR_DATA_DIR, PairDataError, fetch_pair_data
from lineage.pair_eval import REPORT as PAIR_REPORT
from lineage.pair_eval import evaluate as evaluate_pairs
from lineage.pair_eval import write_report as write_pair_report
from lineage.query import IndexMissingError, query_text
from lineage.results_table import EXPORTED_ONLY, README, ResultsTableError, rendered_readme
from lineage.sectioning import SectioningError, build_sections
from lineage.semantic import MODEL as SEMANTIC_MODEL
from lineage.sessions import (
    CARRYOVER,
    SCRAPERS_REPO,
    SessionsError,
    fetch_scraper_sessions,
    session_rows,
    write_sessions,
)
from lineage.snapshot import SnapshotError, load_manifest, scan, write_manifest
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
    """Build the search index: signatures, LSH bands, verified cross-state pairs, families."""
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
) -> None:
    """Find a bill's relatives in the index, offline: bills that share its text, by section."""
    started = time.monotonic()
    try:
        result = query_text(file.read_text(), tables)
    except (IndexMissingError, StoplistError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(f"{file}: {result.sections} sections, {result.signed} signed")
    if not result.matches:
        typer.echo("No near-duplicates found in the index.")
        return
    typer.echo(
        f"{'share':>5}  {'best':>4}  {'state':<5}  {'session':<10}  {'bill':<12}  "
        f"{'family':>6}  {'status':<10}  title"
    )
    for m in result.matches[:limit]:
        typer.echo(
            f"{m.share:>5.0%}  {m.best_estimate:>4.2f}  {m.state:<5}  {m.session:<10}  "
            f"{m.identifier:<12}  {m.family_bills:>6}  {m.status or '':<10}  {m.title[:70]}"
        )
    hidden = len(result.matches) - limit
    more = f"; {hidden:,} more not shown (--limit)" if hidden > 0 else ""
    noun = "family matches" if len(result.matches) == 1 else "families match"
    typer.echo(
        f"{len(result.matches):,} {noun}{more}. Share: of the query's signed sections, "
        f"the share with a match in the family. Best: the highest estimated similarity of a "
        f"matched section. Family: the state's near-identical bills, counted once. "
        f"{time.monotonic() - started:.1f}s"
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
        section = commission_recall(
            tables,
            acts_dir=acts or (HELD_OUT_DIR if held_out else ACTS_DIR),
            enactments_path=enactments or (HELD_OUT_ENACTMENTS if held_out else ENACTMENTS),
            sessions_path=sessions,
            act_ids=HELD_OUT_ACTS if held_out else ACTS,
            development_path=ENACTMENTS if held_out else None,
        )
    except (CommissionError, IndexMissingError, StoplistError, UlcError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    write_results(results, "commission_held_out" if held_out else "commission", section)
    pooled = section["recall"]["pooled"]
    for label, key in (("all enactments", "all"), ("with text", "with_text")):
        r = pooled[key]
        share = r["found"] / r["enactments"] if r["enactments"] else 0.0
        typer.echo(f"recall, {label}: {r['found']}/{r['enactments']} = {share:.1%}")
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
