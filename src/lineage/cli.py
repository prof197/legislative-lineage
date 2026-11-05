"""The `lineage` command line. Each subcommand is a thin wrapper over a library function."""

from pathlib import Path
from typing import Annotated

import typer

from lineage import __version__
from lineage.disk import GiB, InsufficientDiskSpaceError
from lineage.fetch import SOURCE, fetch_raw
from lineage.ingest import ingest as run_ingest
from lineage.snapshot import SnapshotError, scan, write_manifest

app = typer.Typer(no_args_is_help=True, add_completion=False)

RawOption = Annotated[Path, typer.Option("--raw", help="Directory of Open States archives.")]
ManifestOption = Annotated[Path, typer.Option("--manifest", help="Snapshot manifest (committed).")]
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
def ingest(
    raw: RawOption = Path("data/raw"),
    manifest: ManifestOption = Path("snapshot/manifest.json"),
    out: Annotated[Path, typer.Option("--out", help="Directory for Parquet tables.")] = Path(
        "data/tables"
    ),
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Verify the raw archives against the manifest and load them into the `bills` table."""
    try:
        result = run_ingest(raw, manifest, out, floor_bytes=int(min_free_gib * GiB))
    except (SnapshotError, InsufficientDiskSpaceError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"{result.rows:,} records from {result.archives} archives -> {out / 'bills'} "
        f"({result.output_bytes / GiB:.2f} GiB, {result.seconds:.0f}s)"
    )
