"""The data layer's commands: the snapshot, sessions, ingest, document types and stats."""

import json
from pathlib import Path
from typing import Annotated

import typer

from lineage.cli.app import (
    ManifestOption,
    MinFreeOption,
    RawOption,
    RulesOption,
    TablesOption,
    app,
)
from lineage.core.disk import GiB, InsufficientDiskSpaceError
from lineage.core.paths import DATA
from lineage.data.archives import SOURCE, fetch_raw
from lineage.data.doctypes import RULES, DocTypesError, build_eligibility, rule_samples
from lineage.data.ingest import ingest as run_ingest
from lineage.data.sessions import (
    CARRYOVER,
    SCRAPERS_REPO,
    SessionsError,
    fetch_scraper_sessions,
    session_rows,
    write_sessions,
)
from lineage.data.snapshot import SnapshotError, load_manifest, scan, write_manifest
from lineage.data.stats import write_coverage


@app.command()
def snapshot(
    raw: RawOption = DATA / "raw",
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
    raw: RawOption = DATA / "raw",
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
    raw: RawOption = DATA / "raw",
    manifest: ManifestOption = Path("snapshot/manifest.json"),
    out: Annotated[Path, typer.Option("--out", help="Directory for Parquet tables.")] = DATA
    / "tables",
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
def eligible(
    tables: TablesOption = DATA / "tables",
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
def stats(
    tables: TablesOption = DATA / "tables",
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
    ] = DATA / "reports/document_type_samples.json",
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
