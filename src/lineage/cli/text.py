"""The text layer's commands: sections, and the text inventory, sample, fetch and merge."""

import os
from pathlib import Path
from typing import Annotated

import typer

from lineage.cli.app import (
    MinFreeOption,
    PlanOption,
    TablesOption,
    app,
    exits_on,
    fail,
    text_app,
)
from lineage.core.disk import GiB, InsufficientDiskSpaceError
from lineage.core.paths import DATA
from lineage.text.inventory import PLAN, REPORT, StalePlanError, build_inventory, check_plan
from lineage.text.merge import merge_texts
from lineage.text.quality import REPORT as QUALITY
from lineage.text.quality import STORE, run_sample
from lineage.text.run import REPORT as FETCH_REPORT
from lineage.text.run import run_fetch
from lineage.text.sectioning import SectioningError, build_sections


@app.command()
def split(
    tables: TablesOption = DATA / "tables",
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = (
        os.cpu_count() or 1
    ),
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Split every bill's text into sections and write the `sections` table."""
    with exits_on(SectioningError, InsufficientDiskSpaceError, bare=True):
        result = build_sections(tables, workers=workers, floor_bytes=int(min_free_gib * GiB))
    typer.echo(
        f"{result.sections:,} sections from {result.with_text:,} of {result.bills:,} bills with "
        f"text; {result.by_heading:,} split at headings -> {tables / 'sections'} "
        f"({result.output_bytes / GiB:.2f} GiB, {result.seconds:.0f}s)"
    )


@text_app.command("check-plan")
def text_check_plan(plan: PlanOption = PLAN) -> None:
    """Fail unless the stored fetch plan was built from the code checked out (or none is
    stored); the weekly job's first step."""
    problem = check_plan(plan)
    if problem is not None:
        fail(f"error: {problem}")
    typer.echo(f"{plan}: built from this code" if plan.exists() else f"{plan}: none stored")


@text_app.command("inventory")
def text_inventory(
    tables: TablesOption = DATA / "tables",
    plan: PlanOption = PLAN,
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
    tables: TablesOption = DATA / "tables",
    plan: PlanOption = PLAN,
    store: Annotated[Path, typer.Option("--store", help="Fetched sample (local).")] = STORE,
    out: Annotated[Path, typer.Option("--out", help="Quality report JSON (committed).")] = QUALITY,
    fetch: Annotated[bool, typer.Option(help="Fetch what the store lacks before judging.")] = True,
) -> None:
    """Fetch a seeded sample per state and format and judge its text against Open States'."""
    with exits_on(StalePlanError):
        doc = run_sample(tables, plan_path=plan, store=store, report_path=out, fetch=fetch)
    cells = [c for formats in doc["states"].values() for c in formats.values()]
    sampled = sum(sum(c["sampled"].values()) for c in cells)
    ok = sum(c["outcomes"].get("ok", 0) for c in cells)
    typer.echo(f"{ok:,} of {sampled:,} sampled documents extracted, {len(cells)} cells -> {out}")


@text_app.command("fetch")
def text_fetch(
    tables: TablesOption = DATA / "tables",
    plan: PlanOption = PLAN,
    out: Annotated[Path, typer.Option("--out", help="Fetch report JSON (committed).")] = (
        FETCH_REPORT
    ),
    fetch: Annotated[bool, typer.Option(help="Fetch what the store lacks before counting.")] = True,
) -> None:
    """Fetch every planned record's text, as rules/text_layer.toml says; resumable."""
    with exits_on(InsufficientDiskSpaceError, StalePlanError):
        doc = run_fetch(tables, plan_path=plan, report_path=out, fetch=fetch)
    ok = doc["outcomes"].get("ok", 0)
    left = sum(doc["left_out"].values())
    typer.echo(
        f"{ok:,} of {doc['jobs']:,} records fetched with usable text, {left:,} left out -> {out}"
    )


@text_app.command("merge")
def text_merge(
    tables: TablesOption = DATA / "tables",
    plan: PlanOption = PLAN,
    exported_only: Annotated[
        bool, typer.Option(help="Use only the exported text, to measure without the fetch.")
    ] = False,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Choose each record's text, fetched or exported -> data/tables/texts/."""
    with exits_on(InsufficientDiskSpaceError, StalePlanError):
        counts = merge_texts(
            tables, plan_path=plan, exported_only=exported_only,
            floor_bytes=int(min_free_gib * GiB),
        )  # fmt: skip
    shown = ", ".join(f"{source} {n:,}" for source, n in counts.items())
    typer.echo(f"{sum(counts.values()):,} records by text source: {shown} -> {tables / 'texts'}")
