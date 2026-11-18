"""The export's commands: the featured stories, the site's export and the README's
tables.
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from lineage.cli.app import (
    TablesOption,
    app,
    eval_app,
)
from lineage.core.paths import DATA
from lineage.evals.answer_key import (
    ANSWER_KEY,
)
from lineage.evals.commission import (
    RESULTS,
)
from lineage.explain.explain import CACHE_DIR as EXPLAIN_CACHE
from lineage.explain.explain import (
    NotCachedError,
)
from lineage.export.build import (
    EXPORT_DIR,
    RELEASE_MANIFEST,
    STORIES_RULES,
    ExportError,
    build_export,
    write_release_manifest,
)
from lineage.export.build import REPORT as EXPORT_REPORT
from lineage.export.build import SCHEMA as EXPORT_SCHEMA
from lineage.export.build import WORKERS as EXPORT_WORKERS
from lineage.export.build import package as package_export
from lineage.export.build import schema_text as export_schema_text
from lineage.export.featured import PAIRS as FEATURED_PAIRS
from lineage.export.featured import REPORT as FEATURED_REPORT
from lineage.export.featured import FeaturedError, build_featured, write_featured
from lineage.export.release import ReleaseError
from lineage.export.results_table import (
    EXPORTED_ONLY,
    README,
    SAMPLED_STATE_REPORT,
    ResultsTableError,
    hosted_state,
    render_stories,
    rendered_readme,
)
from lineage.match.semantic.models import MODELS_DIR
from lineage.match.semantic.pairs import SemanticPairsError
from lineage.match.state_boilerplate import (
    StateBoilerplateError,
)
from lineage.origins.withheld import REPORT as WITHHELD_REPORT


@app.command("featured")
def featured_command(
    tables: TablesOption = DATA / "tables",
    answer_key: Annotated[
        Path, typer.Option("--answer-key", help="The commission's lists resolved (committed).")
    ] = ANSWER_KEY,
    out: Annotated[
        Path, typer.Option("--out", help="The featured pairs, one per line (committed).")
    ] = FEATURED_PAIRS,
    report: Annotated[
        Path, typer.Option("--report", help="Each story's relatives (committed).")
    ] = FEATURED_REPORT,
) -> None:
    """The featured stories' section pairs: each anchor against its listed relatives."""
    try:
        rows, doc = build_featured(tables, answer_key=answer_key)
    except (FeaturedError, SemanticPairsError, StateBoilerplateError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    write_featured(rows, doc, pairs_path=out, report_path=report)
    for name, story in doc["stories"].items():
        typer.echo(
            f"{name}: {story['section_pairs']} section pairs with {len(story['relatives'])} "
            f"relatives, {story['relatives_without_a_pair']} without a pair, "
            f"{story['section_pairs_out_of_scope']} out of the story's sections"
        )
    typer.echo(f"{len(rows)} pairs -> {out}, {report}")


@app.command("export")
def export_command(
    state: Annotated[
        str, typer.Option("--state", help="The state to host, by its postal code.")
    ] = "GA",
    tables: TablesOption = DATA / "tables",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where the site data goes (local); default data/export/<ST>."),
    ] = None,
    cache: Annotated[
        Path, typer.Option("--cache", help="Cached explanations (committed).")
    ] = EXPLAIN_CACHE,
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
    report: Annotated[
        Path, typer.Option("--report", help="The export's sizes and counts (committed).")
    ] = EXPORT_REPORT,
    featured_pairs: Annotated[
        Path, typer.Option("--featured-pairs", help="The featured pairs (committed).")
    ] = FEATURED_PAIRS,
    featured_report: Annotated[
        Path, typer.Option("--featured-report", help="Each story's relatives (committed).")
    ] = FEATURED_REPORT,
    stories_rules: Annotated[
        Path, typer.Option("--stories", help="Which stories each state's site carries (committed).")
    ] = STORIES_RULES,
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
    schema_only: Annotated[
        bool,
        typer.Option("--schema", help="Write the export's JSON Schema (committed) and stop."),
    ] = False,
    workers: Annotated[
        int, typer.Option("--workers", help="Processes aligning section pairs.")
    ] = EXPORT_WORKERS,
    package_tag: Annotated[
        str | None,
        typer.Option(
            "--package",
            help="Package the existing export as a release asset under this tag "
            "(data-<st>-YYYY-MM-DD), refused unless every file is one the export contract names "
            "and valid against its model, with checksums.json inside and its SHA-256 beside it; "
            "pin it in the release manifest, and stop.",
        ),
    ] = None,
    release_manifest: Annotated[
        Path,
        typer.Option(
            "--release-manifest",
            help="The data release the deploy builds from (committed), written by --package.",
        ),
    ] = RELEASE_MANIFEST,
) -> None:
    """The static site's JSON for one state's bills, with the stories and the data stamp."""
    if schema_only:
        EXPORT_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
        EXPORT_SCHEMA.write_text(export_schema_text())
        typer.echo(f"-> {EXPORT_SCHEMA}")
        return
    state = state.upper()
    target = out if out is not None else EXPORT_DIR / state
    if package_tag is not None:
        try:
            asset, checksum = package_export(
                target, state=state, tag=package_tag, out_dir=target.parent
            )
            write_release_manifest(asset, tag=package_tag, path=release_manifest)
        except (ExportError, ReleaseError) as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
        typer.echo(
            f"{checksum.read_text().strip()} ({asset.stat().st_size / 1e6:.1f} MB) -> {asset}"
        )
        typer.echo(f"pinned -> {release_manifest}; commit it to deploy this release")
        return
    try:
        doc = build_export(
            tables,
            state=state,
            out=target,
            models_dir=models_dir if paraphrase else None,
            workers=workers,
            cache=cache,
            featured_pairs=featured_pairs,
            featured_report=featured_report,
            stories_rules=stories_rules,
            results=results,
            progress=typer.echo,
        )
    except (ExportError, NotCachedError, SemanticPairsError, StateBoilerplateError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    report.write_text(json.dumps(doc, indent=1) + "\n")
    typer.echo(
        f"{doc['hosted_bills']} bills, {doc['comparisons']} comparisons, "
        f"{doc['total_bytes'] / 1e6:.1f} MB ({doc['share_of_site_limit']:.1%} of the site limit) "
        f"-> {target}, {report}"
    )


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
            json.loads(SAMPLED_STATE_REPORT.read_text())["state"],
            stories=render_stories(hosted_state()),
            withheld=json.loads(WITHHELD_REPORT.read_text()) if WITHHELD_REPORT.exists() else None,
        )
    except ResultsTableError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    readme.write_text(text)
    typer.echo(f"results table <- {results}, {exported_only} -> {readme}")
