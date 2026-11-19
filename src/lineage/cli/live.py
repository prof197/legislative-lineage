"""The live track's commands: update, rebuild, intake, `lineage publish` and
`lineage schedule`.
"""

import datetime
import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from lineage.cli.app import (
    AlignWorkersOption,
    CacheOption,
    DayOption,
    FeaturedPairsOption,
    FeaturedReportOption,
    LiveExportOption,
    ManifestOption,
    MinFreeOption,
    ModelsOption,
    RawOption,
    ResultsOption,
    SessionsOption,
    StoriesOption,
    TablesOption,
    app,
    exits_on,
    fail,
    publish_app,
    schedule_app,
)
from lineage.core.disk import GiB, InsufficientDiskSpaceError
from lineage.core.http import open_url
from lineage.core.memory import MemoryBoundError
from lineage.core.paths import DATA
from lineage.data.doctypes import DocTypesError
from lineage.data.snapshot import SnapshotError
from lineage.evals.commission import (
    RESULTS,
)
from lineage.explain.explain import CACHE_DIR as EXPLAIN_CACHE
from lineage.explain.explain import (
    NotCachedError,
)
from lineage.export.build import EXPORT_DIR
from lineage.export.build import WORKERS as EXPORT_WORKERS
from lineage.export.featured import PAIRS as FEATURED_PAIRS
from lineage.export.featured import REPORT as FEATURED_REPORT
from lineage.export.pages import STORIES_RULES, ExportError, ExportInputs
from lineage.export.release import ReleaseError
from lineage.live import deploy
from lineage.live import rebuild as rebuilding
from lineage.live.deploy_ledger import DEPLOY_DIR
from lineage.live.feed import FeedError, write_feed
from lineage.live.incremental import update
from lineage.live.intake import (
    DAYS,
    ESTIMATE,
    TIERS,
    IntakeError,
    Tier,
    api_key,
    january_estimate,
    run_intake,
)
from lineage.live.publish import PublishError, export_changes, promote, staging_dir
from lineage.live.schedule import (
    SCHEDULE,
    TEMPLATES,
    WEEKLY_REPORT,
    ScheduleError,
    When,
    calendar_intervals,
    cli_runner,
    daily_steps,
    osascript_notifier,
    render_template,
    run_job,
    weekly_job_steps,
    weekly_report,
)
from lineage.live.updates import CHANGES_DIR, UpdateError, load_day
from lineage.match.near.stoplist import StoplistError
from lineage.match.query import IndexMissingError
from lineage.match.semantic.models import MODELS_DIR, ModelError
from lineage.match.semantic.neighbors import EF_SEARCH, SPEC
from lineage.match.semantic.pairs import SemanticPairsError
from lineage.match.semantic.stage import MODEL as SEMANTIC_MODEL
from lineage.match.semantic.vectors import EmbedError
from lineage.match.state_boilerplate import (
    StateBoilerplateError,
)
from lineage.origins.modeldocs import ModelDocError
from lineage.text.inventory import PLAN
from lineage.text.sectioning import SectioningError

PUBLISH_REPORTS = DATA / "publish"

StateOption = Annotated[str, typer.Option("--state", help="The hosted state, by postal code.")]
ChangesOption = Annotated[
    Path, typer.Option("--changes", help="The change log, one file per day (local).")
]
SiteOption = Annotated[
    str, typer.Option("--site-url", help="The site's address, absolute, for the feed's links.")
]
EmbeddingModelOption = Annotated[str, typer.Option("--model", help="The embedding model.")]
HeldDayOption = Annotated[str, typer.Option("--day", help="The held day, YYYY-MM-DD.")]


@app.command("update")
def update_command(
    day: Annotated[
        Path, typer.Option("--day", help="A day's directory: day.json and its records (local).")
    ],
    tables: TablesOption = DATA / "tables",
    changes: ChangesOption = CHANGES_DIR,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Apply one day's new and changed records to the built corpus, matching in both
    directions, and log what changed -> <changes>/<as_of>.jsonl (DESIGN.md#updates)."""
    try:
        logged = update(
            tables, load_day(day), changes_dir=changes, floor_bytes=int(min_free_gib * GiB)
        )
    except (UpdateError, InsufficientDiskSpaceError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    except NotImplementedError as e:
        typer.echo(f"error: not built yet: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{len(logged)} changes -> {changes}")


# What the whole pipeline, run again by a rebuild, may raise.
_REBUILD_ERRORS: tuple[type[Exception], ...] = (
    rebuilding.RebuildError,
    UpdateError,
    SnapshotError,
    InsufficientDiskSpaceError,
    StoplistError,
    EmbedError,
    ModelError,
    SemanticPairsError,
    StateBoilerplateError,
    MemoryBoundError,
    SectioningError,
    DocTypesError,
    ModelDocError,
    IndexMissingError,
)


@app.command("rebuild")
def rebuild_command(
    out: Annotated[Path, typer.Option("--out", help="The rebuild's tables directory (local).")],
    days: Annotated[
        Path, typer.Option("--days", help="The days taken in, one directory each (local).")
    ] = DATA / "days",
    through: Annotated[
        str | None,
        typer.Option("--through", help="Fold the days up to this one, YYYY-MM-DD [default: all]."),
    ] = None,
    pinned_from: Annotated[
        Path | None,
        typer.Option(
            "--pinned-from",
            help="Tables at the same day whose stoplist and state boilerplate counts to keep; "
            "without it they are counted afresh.",
        ),
    ] = None,
    vectors_from: Annotated[
        Path | None,
        typer.Option(
            "--vectors-from",
            help="Tables whose vectors to reuse for the same texts [default: --pinned-from].",
        ),
    ] = None,
    raw: RawOption = rebuilding.RAW,
    manifest: ManifestOption = rebuilding.MANIFEST,
    days_only: Annotated[
        bool,
        typer.Option("--days-only", help="Lay the days over no archives: the days are the corpus."),
    ] = False,
    sessions: SessionsOption = rebuilding.SESSIONS,
    text_plan: Annotated[
        Path, typer.Option("--text-plan", help="The text layer's fetch plan, as `text merge`.")
    ] = PLAN,
    max_share: Annotated[
        float,
        typer.Option("--max-share", help="A refreshed stoplist's share, as `lineage boilerplate`."),
    ] = rebuilding.MAX_SHARE,
    origins: Annotated[
        bool | None,
        typer.Option(
            "--origins/--no-origins",
            help="Match the model documents [default: when the tables reused hold matches].",
        ),
    ] = None,
    model: EmbeddingModelOption = SEMANTIC_MODEL,
    exact: Annotated[
        bool,
        typer.Option("--exact", help="Search neighbors exactly, not by HNSW (small corpora)."),
    ] = False,
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = (
        os.cpu_count() or 1
    ),
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """The folded days laid over the snapshot's archives and the whole pipeline run into OUT,
    with pinned or refreshed corpus statistics -> OUT, OUT.build/rebuild.json (DR-7)."""
    with exits_on(*_REBUILD_ERRORS):
        doc = rebuilding.rebuild(
            out,
            days_dir=days,
            through=datetime.date.fromisoformat(through) if through else None,
            base=None if days_only else rebuilding.Base(raw=raw, manifest=manifest),
            sessions=sessions,
            text_plan=text_plan,
            pinned_from=pinned_from,
            vectors_from=vectors_from,
            max_share=max_share,
            origins=origins,
            model=model,
            neighbors_spec="Flat" if exact else SPEC,
            ef_search=None if exact else EF_SEARCH,
            workers=workers,
            floor_bytes=int(min_free_gib * GiB),
            progress=typer.echo,
        )
    typer.echo(_rebuilt(doc, out))


def _rebuilt(doc: dict[str, Any], out: Path) -> str:
    """The rebuild's summary: records, days, statistics, vector shards reused, time."""
    v = doc["vectors"]
    return (
        f"{doc['records']:,} records as of {doc['as_of']}, {len(doc['days'])} days over "
        f"{doc['archives']['written']} archives, statistics {doc['statistics']}; "
        f"{v['shards_reused']} of {v['shards']} vector shards reused, {v['shards_embedded']} "
        f"embedded; {doc['seconds']['total']:.0f}s -> {out}, "
        f"{rebuilding.work_dir(out) / rebuilding.REPORT}"
    )


@app.command("intake")
def intake_command(
    as_of: Annotated[
        str | None, typer.Option("--day", help="The day's date, YYYY-MM-DD [default: today, UTC].")
    ] = None,
    start: Annotated[
        str | None,
        typer.Option(
            "--start",
            help="Where a state with no recorded page starts, ISO 8601 with an offset "
            "[default: the start of the day before].",
        ),
    ] = None,
    tier: Annotated[
        str, typer.Option("--tier", help=f"The key's limits: {', '.join(TIERS)}.")
    ] = "default",
    days: Annotated[Path, typer.Option("--days", help="Days directory (local).")] = DAYS,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Take in one day of new and changed bills from Open States' API v3, with their text
    -> <days>/<day>/ and <data>/intake/reports/<day>.json (DESIGN.md#updates). The key comes
    from OPENSTATES_API_KEY."""
    if tier not in TIERS:
        fail(f"error: no tier {tier!r}; one of {', '.join(TIERS)}")
    day = (
        datetime.date.fromisoformat(as_of) if as_of else datetime.datetime.now(datetime.UTC).date()
    )
    begin = (
        datetime.datetime.fromisoformat(start)
        if start
        else datetime.datetime.combine(
            day - datetime.timedelta(days=1), datetime.time.min, datetime.UTC
        )
    )
    with exits_on(IntakeError, InsufficientDiskSpaceError):
        report = run_intake(
            day,
            opener=open_url,
            key=api_key(),
            start=begin,
            days=days,
            tier=TIERS[tier],
            floor_bytes=int(min_free_gib * GiB),
        )
    outcomes = ", ".join(f"{k} {v}" for k, v in report.get("outcomes", {}).items())
    typer.echo(f"{report['as_of']}: day {report['day']}" + (f"; {outcomes}" if outcomes else ""))


@app.command("intake-estimate")
def intake_estimate_command(
    tables: TablesOption = DATA / "tables",
    years: Annotated[
        list[int] | None, typer.Option("--year", help="A January to estimate; repeatable.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Report (committed).")] = ESTIMATE,
) -> None:
    """The API requests a January day needs, from the snapshot's actions, against each tier's
    limit; no network (DESIGN.md#updates)."""
    chosen = years or [2023, 2024, 2025, 2026]
    tiers: list[Tier] = list(TIERS.values())
    estimates = [january_estimate(tables, y, tiers) for y in chosen]
    report = {
        "command": "uv run lineage intake-estimate",
        "method": "a bill counts as updated on each day it has an action, a lower bound on what "
        "the API reports as updated; a state with nothing new still costs one request",
        "tiers": {k: {"per_minute": t.per_minute, "per_day": t.per_day} for k, t in TIERS.items()},
        "januaries": estimates,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    for e in estimates:
        typer.echo(
            f"January {e['year']}: {e['requests']:,} requests, mean {e['mean_per_day']}/day, "
            f"peak {e['peak']['requests']} on {e['peak']['day']}; days over {e['days_over']}"
        )
    typer.echo(f"-> {out}")


@publish_app.command("pages")
def publish_pages_command(
    state: StateOption = "GA",
    tables: TablesOption = DATA / "tables",
    export: LiveExportOption = None,
    changes: ChangesOption = CHANGES_DIR,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Apply the logged days from this one, YYYY-MM-DD."),
    ] = None,
    cache: CacheOption = EXPLAIN_CACHE,
    paraphrase: Annotated[
        bool,
        typer.Option(
            "--paraphrase/--no-paraphrase",
            help="Mark paraphrase as the export in place was built (local model).",
        ),
    ] = True,
    models_dir: ModelsOption = MODELS_DIR,
    workers: AlignWorkersOption = EXPORT_WORKERS,
    featured_pairs: FeaturedPairsOption = FEATURED_PAIRS,
    featured_report: FeaturedReportOption = FEATURED_REPORT,
    stories_rules: StoriesOption = STORIES_RULES,
    results: ResultsOption = RESULTS,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Apply the logged days the export lacks to a staging copy of it, rewriting only the files
    they touch -> <export>.staging/, data/publish/<ST>.json (DESIGN.md#updates)."""
    state = state.upper()
    target = export if export is not None else EXPORT_DIR / state
    with exits_on(
        PublishError,
        ExportError,
        NotCachedError,
        SemanticPairsError,
        StateBoilerplateError,
        UpdateError,
        InsufficientDiskSpaceError,
    ):
        doc = export_changes(
            tables,
            state=state,
            export=target,
            changes_dir=changes,
            since=datetime.date.fromisoformat(since) if since else None,
            models_dir=models_dir if paraphrase else None,
            workers=workers,
            inputs=ExportInputs(
                cache=cache,
                featured_pairs=featured_pairs,
                featured_report=featured_report,
                stories_rules=stories_rules,
                results=results,
            ),
            floor=int(min_free_gib * GiB),
            progress=typer.echo,
        )
    PUBLISH_REPORTS.mkdir(parents=True, exist_ok=True)
    report = PUBLISH_REPORTS / f"{state}.json"
    report.write_text(json.dumps(doc, indent=1) + "\n")
    written = sum(doc["files_written"].values())
    typer.echo(
        f"{len(doc['days'])} days applied, as of {doc['as_of']}: {written} files written, "
        f"{doc['files_removed']} removed, {doc['section_pairs_aligned']} section pairs aligned; "
        f"{len(doc['beyond_log'])} beyond the log -> {doc['staging']}, {report}"
    )


@publish_app.command("feed")
def publish_feed_command(
    site_url: SiteOption,
    state: StateOption = "GA",
    tables: TablesOption = DATA / "tables",
    export: Annotated[
        Path | None,
        typer.Option(
            "--export", help="The export to write feed.xml into; default the staging copy."
        ),
    ] = None,
    changes: ChangesOption = CHANGES_DIR,
) -> None:
    """The Atom feed of the live track's events from every state, checked against RFC 4287
    -> <export>/feed.xml (DESIGN.md#updates)."""
    state = state.upper()
    target = export if export is not None else staging_dir(EXPORT_DIR / state)
    with exits_on(FeedError, PublishError, ExportError, FileNotFoundError):
        doc = write_feed(tables, state=state, export=target, site_url=site_url, changes_dir=changes)
    kinds = ", ".join(f"{k} {v}" for k, v in doc["by_kind"].items())
    typer.echo(
        f"{doc['entries']} entries through {doc['through']} ({kinds or 'none'}), "
        f"{len(doc['states'])} states, {doc['bytes']:,} bytes -> {doc['path']}"
    )


@publish_app.command("promote")
def publish_promote_command(
    state: StateOption = "GA",
    export: LiveExportOption = None,
) -> None:
    """Swap the staging copy in as the live export. Nothing is deployed or published."""
    target = export if export is not None else EXPORT_DIR / state.upper()
    with exits_on(PublishError):
        promote(target)
    typer.echo(f"-> {target}")


@publish_app.command("weekly")
def publish_weekly_command(
    pinned: Annotated[
        Path, typer.Option("--pinned", help="A full rebuild with the base build's statistics.")
    ],
    updated: Annotated[Path, typer.Option("--updated", help="The updated tables (local).")] = DATA
    / "tables",
    refreshed: Annotated[
        Path | None,
        typer.Option("--refreshed", help="A full rebuild with the statistics counted afresh."),
    ] = None,
    report: Annotated[
        Path, typer.Option("--report", help="The weekly report (local).")
    ] = WEEKLY_REPORT,
    model: EmbeddingModelOption = SEMANTIC_MODEL,
) -> None:
    """The weekly rebuild's check: the share of relatives the updated tables and the rebuild
    agree on, failing below 98%, and the drift when the statistics are refreshed (DR-7)."""
    with exits_on(ScheduleError):
        doc = weekly_report(updated, pinned, refreshed, model=model, report=report)
    a, d = doc["agreement"], doc["drift"]
    drifted = "" if d is None else f"; {d['relatives_changed']} relatives drift when refreshed"
    typer.echo(
        f"{a['share']:.2%} of relatives agree ({a['relatives_agreed']} of "
        f"{a['relatives_updated']}){drifted} -> {report}"
    )


DeployRootOption = Annotated[
    Path, typer.Option("--deploy-dir", help="The deploy's records and releases (local).")
]


def _deploy_paths(root: Path) -> deploy.DeployPaths:
    return deploy.DeployPaths(root=root, checkout=root / "checkout")


@publish_app.command("deploy")
def publish_deploy_command(
    state: StateOption = "GA",
    export: LiveExportOption = None,
    changes: ChangesOption = CHANGES_DIR,
    root: DeployRootOption = DEPLOY_DIR,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Deploy the live export's day: pack the release, the gates, the circuit breaker, then
    the GitHub release, the pin's commit, the push and the Pages workflow (DESIGN.md#updates).
    Exits 3 when the day is held for review, printing its counts and baseline last."""
    state = state.upper()
    target = export if export is not None else EXPORT_DIR / state
    with exits_on(deploy.DeployError, ReleaseError):
        record = deploy.deploy_day(
            state=state,
            export=target,
            changes_dir=changes,
            paths=_deploy_paths(root),
            floor=int(min_free_gib * GiB),
        )
    if record.state == "held":
        typer.echo(
            json.dumps(
                {
                    "day": record.day.isoformat(),
                    "tag": record.tag,
                    "held_on_counts": record.held_on_counts,
                    "held_behind": None if record.held_behind is None else str(record.held_behind),
                    "reasons": record.reasons,
                    "counts": record.counts,
                    "baseline": record.baseline,
                }
            )
        )
        raise typer.Exit(deploy.HELD)
    typer.echo(f"{record.day}: {record.state} as {record.tag}; {record.counts}")


@publish_app.command("release")
def publish_release_command(
    day: HeldDayOption,
    root: DeployRootOption = DEPLOY_DIR,
    repack: Annotated[
        bool,
        typer.Option(
            "--repack",
            help="Pack the live export as it stands now, after withholding the documents behind "
            "bad claims and running publish pages, feed and promote.",
        ),
    ] = False,
    state: StateOption = "GA",
    export: LiveExportOption = None,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Approve a day the circuit breaker held, once reviewed: its release, or with --repack the
    live export as it now stands; the site's gates run again, the pin committed and
    leak-checked, pushed, and the Pages workflow dispatched."""
    state = state.upper()
    with exits_on(deploy.DeployError, ReleaseError):
        record = deploy.release_day(
            datetime.date.fromisoformat(day),
            paths=_deploy_paths(root),
            repack=repack,
            export=export if export is not None else EXPORT_DIR / state,
            state=state,
            floor=int(min_free_gib * GiB),
        )
    typer.echo(f"{record.day}: {record.state} as {record.tag}")


@publish_app.command("reject")
def publish_reject_command(
    day: HeldDayOption,
    root: DeployRootOption = DEPLOY_DIR,
) -> None:
    """Reject a day the circuit breaker held, once reviewed: its release is never published and
    nothing leaves this machine. The live track keeps its data, and the next day's run publishes
    as usual, with it."""
    with exits_on(deploy.DeployError):
        record = deploy.reject_day(datetime.date.fromisoformat(day), paths=_deploy_paths(root))
    typer.echo(
        f"{record.day}: rejected; {record.tag} stays unpublished, and the next day's run "
        "publishes as usual"
    )


def _review_names(tables: Path, state: str, site_url: str) -> tuple[Any, dict[str, dict[str, Any]]]:
    return (
        deploy.describe_bills(tables, state=state, site_url=site_url),
        deploy.document_names(),
    )


@publish_app.command("candidate")
def publish_candidate_command(
    site_url: SiteOption,
    state: StateOption = "GA",
    day: DayOption = None,
    tables: TablesOption = DATA / "tables",
    export: LiveExportOption = None,
    changes: ChangesOption = CHANGES_DIR,
    root: DeployRootOption = DEPLOY_DIR,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Before session: the live export packed and gated as a release candidate, with a review
    summary against the pinned release -> <deploy>/candidates/<tag>/review.txt. Publishes
    nothing."""
    state = state.upper()
    as_of = datetime.date.fromisoformat(day) if day else datetime.datetime.now(datetime.UTC).date()
    describe, documents = _review_names(tables, state, site_url)
    with exits_on(deploy.DeployError, ReleaseError):
        doc = deploy.prepare_candidate(
            as_of,
            state=state,
            export=export if export is not None else EXPORT_DIR / state,
            changes_dir=changes,
            paths=_deploy_paths(root),
            describe=describe,
            documents=documents,
            floor=int(min_free_gib * GiB),
        )
    files = doc["files"]
    typer.echo(
        f"{doc['tag']} against {doc['pinned']}: {len(files['added'])} added, "
        f"{len(files['changed'])} changed, {len(files['removed'])} removed; "
        f"{len(doc['origin_claims'])} new origin claims -> "
        f"{root / 'candidates' / doc['tag'] / 'review.txt'}"
    )


@publish_app.command("review")
def publish_review_command(
    site_url: SiteOption,
    state: StateOption = "GA",
    through: Annotated[
        str | None, typer.Option("--through", help="The week's last day [default: today, UTC].")
    ] = None,
    tables: TablesOption = DATA / "tables",
    changes: ChangesOption = CHANGES_DIR,
    root: DeployRootOption = DEPLOY_DIR,
) -> None:
    """During session: the week's new origin claims, each with its bill, document, shares,
    stages and the day first listed, for the maintainer's weekly review ->
    <deploy>/reviews/<day>.txt."""
    day = (
        datetime.date.fromisoformat(through)
        if through
        else datetime.datetime.now(datetime.UTC).date()
    )
    describe, documents = _review_names(tables, state.upper(), site_url)
    out = deploy.weekly_review(
        day, changes_dir=changes, paths=_deploy_paths(root), describe=describe, documents=documents
    )
    n = len(json.loads(out.read_text())["origin_claims"])
    typer.echo(f"{n} new origin claims in the week through {day} -> {out.with_suffix('.txt')}")


@schedule_app.command("daily")
def schedule_daily_command(
    site_url: SiteOption,
    state: StateOption = "GA",
    day: DayOption = None,
    tier: Annotated[str, typer.Option("--tier", help="The intake key's limits.")] = "default",
) -> None:
    """The daily job: intake, update, the incremental export and the feed, swapped in only when
    every step succeeds, and from the first day of session the deploy (rules/deploy.toml); a
    failure or a held day is recorded in data/schedule/status.json and alerted. Exits 3 when
    the day is held for review."""
    as_of = datetime.date.fromisoformat(day) if day else datetime.datetime.now(datetime.UTC).date()
    steps = daily_steps(as_of, state=state.upper(), site_url=site_url, tier=tier)
    status = run_job("daily", steps, run=cli_runner, notify=osascript_notifier)
    if status.state == "held":
        typer.echo(f"daily {as_of}: held for review; see data/schedule/status.json")
        raise typer.Exit(deploy.HELD)
    typer.echo(f"daily {as_of}: " + ("ok" if status.ok else f"stopped at {status.failed_step}"))
    raise typer.Exit(0 if status.ok else 1)


@schedule_app.command("weekly")
def schedule_weekly_command(
    site_url: SiteOption,
    state: StateOption = "GA",
    day: DayOption = None,
) -> None:
    """The weekly job: two full rebuilds over the days the tables have had, then the check,
    which fails the job below 98% agreement (DR-7); then, before session, a release candidate
    for review, and during it the week's new origin claims (DESIGN.md#updates)."""
    as_of = datetime.date.fromisoformat(day) if day else datetime.datetime.now(datetime.UTC).date()
    steps = weekly_job_steps(state=state.upper(), site_url=site_url, day=as_of)
    status = run_job("weekly", steps, run=cli_runner, notify=osascript_notifier)
    typer.echo(f"weekly {as_of}: " + ("ok" if status.ok else f"stopped at {status.failed_step}"))
    raise typer.Exit(0 if status.ok else 1)


@schedule_app.command("plist")
def schedule_plist_command(
    job: Annotated[str, typer.Argument(help="daily or weekly.")],
    out: Annotated[Path, typer.Option("--out", help="Where the filled plist goes (local).")],
    site_url: SiteOption,
    state: StateOption = "GA",
    hour: Annotated[
        int | None, typer.Option("--hour", help="The hour it runs, local time; else as ruled.")
    ] = None,
    minute: Annotated[
        int | None, typer.Option("--minute", help="The minute it runs; else as ruled.")
    ] = None,
    weekdays: Annotated[
        str | None,
        typer.Option("--weekdays", help="Days it runs, 0 Sunday to 6 Saturday, comma-separated."),
    ] = None,
    label: Annotated[str, typer.Option("--label", help="launchd's label prefix.")] = (
        "local.legislative-lineage"
    ),
) -> None:
    """Fill a launchd template with this checkout, the data directory and the PATH it runs
    under -> OUT. Loading it with launchctl is the operator's step."""
    if job not in ("daily", "weekly"):
        fail("error: the job is daily or weekly")
    ruled = SCHEDULE[job]
    when = When(
        hour=ruled.hour if hour is None else hour,
        minute=ruled.minute if minute is None else minute,
        weekdays=ruled.weekdays
        if weekdays is None
        else tuple(int(d) for d in weekdays.split(",") if d.strip()),
    )
    values = {
        "REPO": str(Path.cwd().resolve()),
        "DATA": str(DATA.resolve()),
        "STATE": state.upper(),
        "SITE_URL": site_url,
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "INTERVALS": calendar_intervals(when),
        "LABEL": label,
    }
    with exits_on(ScheduleError):
        text = render_template((TEMPLATES / f"lineage-{job}.plist.template").read_text(), values)
    out.write_text(text)
    typer.echo(f"-> {out}; load it with launchctl when ready")
