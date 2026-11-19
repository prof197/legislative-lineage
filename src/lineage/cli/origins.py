"""Documented origins' commands: collecting model legislation, matching it, its
samples and labels, withholding, and `lineage eval`'s origin measures.
"""

import datetime
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer

from lineage.cli.app import (
    AlignModelsOption,
    BatchIdOption,
    LabelsAppendOption,
    MinFreeOption,
    ModelDocsOption,
    OriginLabelsOption,
    ResultsOption,
    TablesOption,
    app,
    eval_app,
    exits_on,
    fail,
    model_embedder,
)
from lineage.core.disk import GiB, InsufficientDiskSpaceError
from lineage.core.paths import DATA
from lineage.evals.commission import (
    RESULTS,
    run_configuration,
    write_results,
)
from lineage.evals.model_labels import MAX_COST as LABEL_MAX_COST
from lineage.evals.model_labels import MODEL_ID as LABEL_MODEL
from lineage.evals.model_labels import (
    anthropic_batches,
)
from lineage.evals.relatives_sample import (
    with_passages,
)
from lineage.match.listing import flagged_sections
from lineage.match.near.stoplist import StoplistError
from lineage.match.query import IndexMissingError
from lineage.match.semantic.models import MODELS_DIR, ModelError
from lineage.match.semantic.stage import MODEL as SEMANTIC_MODEL
from lineage.match.semantic.vectors import EmbedError
from lineage.origins.collect import REPORTS_DIR as COLLECT_REPORTS
from lineage.origins.collect import CollectContext, CollectError, collectors, run_collector
from lineage.origins.collect import write_report as write_collect_report
from lineage.origins.labels import LABELS as ORIGIN_LABELS
from lineage.origins.labels import PROMPT as ORIGIN_PROMPT
from lineage.origins.labels import answers_by_match
from lineage.origins.labels import build_items as build_origin_items
from lineage.origins.labels import correct_by_sections as origin_correct_by_sections
from lineage.origins.labels import label_in_batch as origin_label_in_batch
from lineage.origins.labels import labelers as origin_labelers
from lineage.origins.labels import load_labels as load_origin_labels
from lineage.origins.labels import pair_sections as origin_pair_sections
from lineage.origins.labels import pending as origin_pending
from lineage.origins.labels import plan as plan_origin_labels
from lineage.origins.labels import record as record_origin_spend
from lineage.origins.matches import FLOOR as ORIGIN_FLOOR
from lineage.origins.matches import REPORT as ORIGINS_REPORT
from lineage.origins.matches import (
    origins_report,
    read_origin_matches,
    write_origin_matches,
    write_origins_report,
)
from lineage.origins.matching import match_models
from lineage.origins.modeldocs import MODELDOCS_DIR, ModelDocError, read_docs
from lineage.origins.precision import PARTS as ORIGIN_PARTS
from lineage.origins.precision import REPORT as ORIGIN_SAMPLE_REPORT
from lineage.origins.precision import RESULTS_SECTIONS as ORIGIN_RESULTS_SECTIONS
from lineage.origins.precision import SPEND as ORIGIN_SPEND
from lineage.origins.precision import (
    OriginPrecisionError,
    doc_chars,
    draw_part,
    excluded_keys,
    plan_labels,
    prompt_chars,
    refuse_over_cap,
    section_chars,
    write_part,
)
from lineage.origins.precision import census_reading as origin_census_reading
from lineage.origins.precision import load_sample as load_origin_sample
from lineage.origins.precision import reading as origin_precision_reading
from lineage.origins.precision import retired as origin_retired
from lineage.origins.precision import without_intervals as without_origin_intervals
from lineage.origins.reach import REPORT as ORIGIN_REACH_REPORT
from lineage.origins.reach import proposal as origin_reach_proposal
from lineage.origins.recall import KEYS as ORIGIN_KEYS
from lineage.origins.recall import OriginRecallError, load_key, origin_recall, same_slug
from lineage.origins.threshold import REPORT as ORIGIN_THRESHOLD_REPORT
from lineage.origins.threshold import correct_by_match as origin_correct_by_match
from lineage.origins.threshold import correct_by_pair as origin_correct_by_pair
from lineage.origins.threshold import proposal as origin_threshold_proposal
from lineage.origins.withheld import REPORT as WITHHELD_REPORT
from lineage.origins.withheld import Reviewed as WithheldReviewed
from lineage.origins.withheld import listed_count as withheld_listed_count
from lineage.origins.withheld import read_reviewed as read_withheld_reviewed
from lineage.origins.withheld import tallies as withheld_tallies
from lineage.origins.withheld import write_report as write_withheld


@app.command("collect-models")
def collect_models_command(
    source: Annotated[str, typer.Argument(help="The source's key, as its collector declares it.")],
    root: ModelDocsOption = MODELDOCS_DIR,
    reports: Annotated[
        Path, typer.Option("--reports", help="Each source's coverage report (committed).")
    ] = COLLECT_REPORTS,
    min_free_gib: MinFreeOption = 20.0,
) -> None:
    """Collect one source's model legislation through its collector, honoring robots.txt ->
    <modeldocs>/<source>/, <reports>/<source>.json (DESIGN.md#model-documents)."""
    context = CollectContext(root=root, floor_bytes=int(min_free_gib * GiB))
    with exits_on(CollectError, ModelDocError, InsufficientDiskSpaceError):
        report = run_collector(source, context)
    path = write_collect_report(report, reports)
    withdrawn = f", {len(report.withdrawn)} withdrawn" if report.withdrawn else ""
    typer.echo(
        f"{report.found} found: {report.collected} collected, {report.unchanged} unchanged, "
        f"{len(report.blocked)} blocked by robots.txt, {len(report.failed)} failed{withdrawn} "
        f"-> {path}"
    )


@app.command("origins")
def origins_command(
    tables: TablesOption = DATA / "tables",
    root: ModelDocsOption = MODELDOCS_DIR,
    min_free_gib: MinFreeOption = 20.0,
    report: Annotated[Path, typer.Option("--report", help="Counts per source and stage.")] = (
        ORIGINS_REPORT
    ),
    report_only: Annotated[
        bool,
        typer.Option("--report-only", help="Count the table as it stands; match nothing."),
    ] = False,
    patterns: Annotated[
        bool,
        typer.Option(
            "--patterns/--no-patterns",
            help="The hand-written generic patterns; off only to measure what they still add.",
        ),
    ] = True,
) -> None:
    """Match every collected model document against the corpus -> <tables>/origins/matches.jsonl,
    reports/origins.json (DESIGN.md#origin-matches)."""
    docs = [d for c in collectors().values() for d in read_docs(c.source, root)]
    if report_only:
        counted = origins_report(read_origin_matches(tables), docs, seconds=None)
        typer.echo(f"{counted.matches} matches counted -> {write_origins_report(counted, report)}")
        return
    started = time.perf_counter()
    with exits_on(
        ModelDocError,
        InsufficientDiskSpaceError,
        IndexMissingError,
        StoplistError,
        EmbedError,
        ModelError,
    ):
        matches = match_models(
            tables,
            docs,
            modeldocs_root=root,
            model=SEMANTIC_MODEL,
            floor_bytes=int(min_free_gib * GiB),
            patterns=patterns,
        )
    path = write_origin_matches(tables, matches)
    counted = origins_report(matches, docs, seconds=time.perf_counter() - started)
    typer.echo(
        f"{len(matches)} matches from {len(docs)} documents -> {path}, "
        f"{write_origins_report(counted, report)}"
    )


@app.command("origin-sample")
def origin_sample_command(
    part: Annotated[
        str,
        typer.Option(
            "--part",
            help="tuning, reporting (once matching is final), or census_eppc (every EPPC "
            "match listed).",
        ),
    ] = "tuning",
    tables: TablesOption = DATA / "tables",
    root: ModelDocsOption = MODELDOCS_DIR,
    final: Annotated[
        bool, typer.Option("--final", help="Matching is final: the reporting sample may be drawn.")
    ] = False,
    batch: Annotated[
        bool, typer.Option("--batch", help="Project at the Batches API's half price.")
    ] = False,
) -> None:
    """Draw an origin precision sample and project what labeling it would cost; asks nothing
    (DESIGN.md#origin-matches)."""
    if part not in ORIGIN_PARTS:
        fail(f"error: no part {part!r}; one of {', '.join(ORIGIN_PARTS)}")
    chosen = ORIGIN_PARTS[part]
    with exits_on(OriginPrecisionError, ModelDocError):
        rows, report = draw_part(
            read_origin_matches(tables), chosen, excluded=excluded_keys(chosen), final=final
        )
        if not rows:
            raise OriginPrecisionError(
                f"no origin matches in {tables}; run `lineage origins` first"
            )
        wanted = {r["doc_id"] for r in rows}
        docs = [
            d for c in collectors().values() for d in read_docs(c.source, root) if d.id in wanted
        ]
        chars = prompt_chars(
            rows,
            sections=section_chars(tables, [r["bill_id"] for r in rows]),
            docs=doc_chars(docs, root),
        )
        plan = plan_labels(chars, batch=batch)
        path = write_part(rows, report | {"label_plan": plan.as_dict()}, chosen)
        refuse_over_cap(plan)
    typer.echo(
        f"{len(rows)} matches -> {path}; labels projected at most ${plan.projected:,.2f}, "
        f"${plan.spent:,.2f} of ${plan.budget:,.2f} spent"
    )


@app.command("withhold")
def withhold_command(
    tables: TablesOption = DATA / "tables",
    root: ModelDocsOption = MODELDOCS_DIR,
    labels: Annotated[
        Path, typer.Option("--labels", help="Origin precision labels (committed).")
    ] = ORIGIN_LABELS,
    out: Annotated[
        Path, typer.Option("--out", help="The withheld documents' report (committed).")
    ] = WITHHELD_REPORT,
    doc: Annotated[
        str | None,
        typer.Option("--doc", help="Withhold this document after reviewing a held day's claims."),
    ] = None,
    note: Annotated[
        str | None, typer.Option("--note", help="Why, for the record: what the review found.")
    ] = None,
) -> None:
    """The model documents withheld from pages, the feed and stories: at least 3 labeled listed
    matches, fewer than half judged correct; and those withheld by hand with --doc after a held
    day's review, which are kept when the report is written again (DESIGN.md#origin-matches)."""
    if (doc is None) != (note is None):
        fail("error: --doc and --note go together")
    with exits_on(OriginPrecisionError, ModelDocError):
        docs = [d for c in collectors().values() for d in read_docs(c.source, root)]
        matches = read_origin_matches(tables)
        found = withheld_tallies(matches, docs, tables, load_origin_labels(labels))
        reviewed = read_withheld_reviewed(out)
        if doc is not None and note is not None:
            titles = {d.id: d.title for d in docs}
            if doc not in titles:
                raise ModelDocError(f"no collected document {doc}")
            reviewed = [r for r in reviewed if r.doc_id != doc] + [
                WithheldReviewed(
                    doc_id=doc,
                    title=titles[doc],
                    listed=withheld_listed_count(matches, doc),
                    note=note,
                    on=datetime.date.today().isoformat(),
                )
            ]
    path = write_withheld(found, out, reviewed)
    held = [t for t in found if t.withheld]
    for t in held:
        typer.echo(f"withheld {t.doc_id}: {t.correct} of {t.labeled} labeled matches correct")
    for r in reviewed:
        typer.echo(f"withheld {r.doc_id} after review on {r.on}: {r.note}")
    typer.echo(
        f"{len(held)} of {len(found)} documents with labeled matches withheld, "
        f"{len(reviewed)} by review -> {path}"
    )


@app.command("origin-label")
def origin_label_command(
    part: Annotated[str, typer.Option("--part", help="The sample labeled.")] = "tuning",
    tables: TablesOption = DATA / "tables",
    root: ModelDocsOption = MODELDOCS_DIR,
    labels: LabelsAppendOption = ORIGIN_LABELS,
    spend: Annotated[
        Path, typer.Option("--spend", help="v2's spend ledger (committed).")
    ] = ORIGIN_SPEND,
    live: Annotated[
        bool, typer.Option("--live", help="Ask, through the Batches API; else nothing is sent.")
    ] = False,
    batch_id: BatchIdOption = None,
    models_dir: AlignModelsOption = MODELS_DIR,
    max_cost: Annotated[
        float,
        typer.Option(
            "--max-cost",
            help="Dollars the run may be projected to cost; v2's budget still applies.",
        ),
    ] = LABEL_MAX_COST,
) -> None:
    """Label an origin sample's matches by a model on each one's strongest section pair; asks
    only with --live (DESIGN.md#origin-matches)."""
    if part not in ORIGIN_PARTS:
        fail(f"error: no part {part!r}; one of {', '.join(ORIGIN_PARTS)}")
    chosen = ORIGIN_PARTS[part]
    with exits_on(OriginPrecisionError, ModelDocError, ModelError):
        if not chosen.sample.exists():
            raise OriginPrecisionError(f"no {chosen.sample}; run `lineage origin-sample` first")
        rows = load_origin_sample(chosen.sample)
        docs = [d for c in collectors().values() for d in read_docs(c.source, root)]
        items = origin_pending(
            build_origin_items(rows, read_origin_matches(tables), docs, tables, root=root),
            load_origin_labels(labels),
        )
        items = [
            replace(i, item=it)
            for i, it in zip(
                items,
                with_passages([i.item for i in items], model_embedder(models_dir)),
                strict=True,
            )
        ]
        planned = plan_origin_labels(items, spend_path=spend, max_cost=max_cost)
    head = (
        f"{planned.items} matches under {ORIGIN_PROMPT.version}, projected at most "
        f"${planned.projected:,.2f}; ${planned.spent:,.2f} of v2's ${planned.budget:,.2f} spent"
    )
    if not live:
        typer.echo(f"{head}; run with --live to ask.")
        return
    refusal = planned.refusal() if batch_id is None else None
    if refusal:
        fail(f"error: {refusal}")
    typer.echo(head)
    today = datetime.date.today().isoformat()
    batches = anthropic_batches()
    done, unusable, sent = origin_label_in_batch(
        items,
        batches=batches,
        model=LABEL_MODEL,
        labels_path=labels,
        today=today,
        wait=lambda: time.sleep(60),
        batch_id=batch_id,
        submitted=lambda b: typer.echo(f"sent as batch {b}; collect with --batch-id {b}"),
    )
    ledger = record_origin_spend(
        batches, sent, model=LABEL_MODEL, today=today, spend_path=spend, planned=planned
    )
    typer.echo(f"{done} labeled from batch {sent}, {unusable} unusable replies -> {labels}")
    typer.echo(f"${ledger['spent']:,.2f} of v2's ${ledger['budget']:,.2f} spent -> {spend}")


@eval_app.command("origins")
def eval_origins(
    tables: TablesOption = DATA / "tables",
    key: Annotated[
        str,
        typer.Option(
            "--key",
            help="The answer key: development, v1_held_out (v1's held-out acts, opened in "
            "v1), or held_out (sealed, with --final).",
        ),
    ] = "development",
    source: Annotated[
        str, typer.Option("--source", help="The source whose documents are named by act slug.")
    ] = "commission",
    results: ResultsOption = RESULTS,
    final: Annotated[
        bool, typer.Option("--final", help="The one run of a sealed key per configuration.")
    ] = False,
) -> None:
    """Origin recall: what share of each model's enactments its origin matches name
    (DESIGN.md#origin-matches)."""
    if key not in ORIGIN_KEYS:
        fail(f"error: no answer key {key!r}; one of {', '.join(ORIGIN_KEYS)}")
    answer_key = ORIGIN_KEYS[key]
    name = f"origins_{answer_key.name}"
    previous = json.loads(results.read_text()).get(name) if results.exists() else None
    with exits_on(OriginRecallError):
        acts = {e.act for e in load_key(answer_key.path).enactments}
        section = origin_recall(
            read_origin_matches(tables),
            answer_key,
            same_slug(acts, source),
            final=final,
            # v1's listing configuration, and the floor a page lists an origin at
            configuration=run_configuration(True, flagged_sections(tables))
            | {"origin_floor": ORIGIN_FLOOR}
            if answer_key.sealed and final
            else None,
            previous=previous,
        )
    write_results(results, name, section)
    r = section["recall"]["pooled"]
    status = f" ({section['status']})" if section["status"] else ""
    typer.echo(
        f"origin recall: {r['found']}/{r['enactments']} {r['interval_95']}{status} -> {results}"
    )


@eval_app.command("origin-threshold")
def eval_origin_threshold(
    tables: TablesOption = DATA / "tables",
    labels: OriginLabelsOption = ORIGIN_LABELS,
    report: Annotated[
        Path, typer.Option("--report", help="The proposal (committed).")
    ] = ORIGIN_THRESHOLD_REPORT,
) -> None:
    """Each candidate floor's precision on the origin tuning sample, beside its development
    recall (DESIGN.md#origin-matches)."""
    tuning = ORIGIN_PARTS["tuning"]
    with exits_on(OriginPrecisionError, OriginRecallError, ValueError):
        if not tuning.sample.exists():
            raise OriginPrecisionError(f"no {tuning.sample}; run `lineage origin-sample` first")
        sizes = json.loads(ORIGIN_SAMPLE_REPORT.read_text())["tuning"]["strata"]
        answer_key = ORIGIN_KEYS["development"]
        acts = {e.act for e in load_key(answer_key.path).enactments}
        doc = origin_threshold_proposal(
            load_origin_sample(tuning.sample),
            load_origin_labels(labels),
            sizes,
            read_origin_matches(tables),
            load_key(answer_key.path),
            same_slug(acts, "commission"),
        )
    report.write_text(json.dumps(doc, indent=1) + "\n")
    for f in doc["floors"]:
        p = f["precision"]
        r = f["development_recall"]
        shown = "unlabeled" if p is None else f"{p['precision']:.2f} {p['interval_95']}"
        typer.echo(
            f"floor {f['floor']:.2f}: precision {shown}, {f['matches_listed']:,} listed, "
            f"recall {r['found']}/{r['enactments']}"
        )
    typer.echo(f"-> {report}")


@eval_app.command("origin-reach")
def eval_origin_reach(
    tables: TablesOption = DATA / "tables",
    labels: OriginLabelsOption = ORIGIN_LABELS,
    root: ModelDocsOption = MODELDOCS_DIR,
    report: Annotated[
        Path, typer.Option("--report", help="The proposal (committed).")
    ] = ORIGIN_REACH_REPORT,
) -> None:
    """Each candidate reach scale's precision on the labeled origin matches, beside its
    development recall, and the scale the rule picks (DESIGN.md#origin-matches)."""
    with exits_on(OriginPrecisionError, OriginRecallError, ValueError):
        matches = read_origin_matches(tables)
        docs = [d for c in collectors().values() for d in read_docs(c.source, root)]
        answers = answers_by_match(matches, docs, tables, load_origin_labels(labels), root=root)
        answer_key = ORIGIN_KEYS["development"]
        acts = {e.act for e in load_key(answer_key.path).enactments}
        doc = origin_reach_proposal(
            matches, answers, load_key(answer_key.path), same_slug(acts, "commission")
        )
    report.write_text(json.dumps(doc, indent=1) + "\n")
    for r in doc["scales"]:
        p, d = r["labeled"], r["development_recall"]
        shown = "unlabeled" if p["precision"] is None else f"{p['precision']:.2f} of {p['labeled']}"
        typer.echo(
            f"scale {r['scale'] or 'none'}: precision {shown}, {r['matches_listed']:,} listed, "
            f"recall {d['found']}/{d['enactments']}"
        )
    typer.echo(f"chosen: {doc['chosen']} -> {report}")


@eval_app.command("origin-precision")
def eval_origin_precision(
    part: Annotated[
        str,
        typer.Option(
            "--part",
            help="reporting_weighted; reporting_after_compact_fix, reporting_after_fix and "
            "reporting are the readings before each change to matching.",
        ),
    ] = "reporting_weighted",
    labels: OriginLabelsOption = ORIGIN_LABELS,
    tables: TablesOption = DATA / "tables",
    root: ModelDocsOption = MODELDOCS_DIR,
    results: ResultsOption = RESULTS,
    intervals_only: Annotated[
        bool,
        typer.Option(
            "--intervals-only",
            help="Regenerate the stored reading's intervals; refused if anything else would move.",
        ),
    ] = False,
) -> None:
    """Origin precision on a reporting sample, read once: overall, by stage, by source, and
    ALEC's against the rest (DESIGN.md#origin-matches)."""
    if part not in ORIGIN_RESULTS_SECTIONS:
        fail(f"error: no reporting part {part!r}; one of {', '.join(ORIGIN_RESULTS_SECTIONS)}")
    reporting = ORIGIN_PARTS[part]
    name = ORIGIN_RESULTS_SECTIONS[part]
    with exits_on(OriginPrecisionError, ModelDocError, ValueError):
        doc = _stored_results(results)
        _refuse_reading_again(
            doc, part=part, name=name, results=results, intervals_only=intervals_only
        )
        if not reporting.sample.exists():
            raise OriginPrecisionError(
                f"no {reporting.sample}; run `lineage origin-sample --part {part} --final`"
            )
        rows = load_origin_sample(reporting.sample)
        correct, labelers = _origin_correct(
            rows, labels, by_sections=reporting.by_sections, tables=tables, root=root
        )
        read = origin_precision_reading(
            rows, correct, json.loads(ORIGIN_SAMPLE_REPORT.read_text())[part]
        )
    section = (
        {
            "command": f"uv run lineage eval origin-precision --part {part}",
            "sample": str(reporting.sample),
            "labels": str(labels),
            "prompt": ORIGIN_PROMPT.version,
        }
        | ({"labeler": labelers} if labelers else {})
        | read
    )
    if intervals_only and without_origin_intervals(section) != without_origin_intervals(doc[name]):
        fail(
            f"error: the reading would change other than its intervals ({results}, {name}); "
            "it is read once"
        )
    write_results(results, name, section)
    typer.echo(_precision_line("overall", read["overall"]))
    for group, g in read["compared"].items():
        typer.echo(_precision_line(group, g))
    typer.echo(f"-> {results}")


def _stored_results(results: Path) -> dict[str, Any]:
    """The results file as it stands, a reading published before its part existed moved to
    its own part's name first (and written back)."""
    doc: dict[str, Any] = json.loads(results.read_text()) if results.exists() else {}
    moved = origin_retired(doc)
    if moved != doc:
        doc = moved
        results.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def _refuse_reading_again(
    doc: Mapping[str, Any], *, part: str, name: str, results: Path, intervals_only: bool
) -> None:
    """A reporting sample is read once: refuse a second reading, and regenerating the
    intervals of a reading never stored."""
    if intervals_only and name not in doc:
        raise OriginPrecisionError(f"no stored reading in {results} to regenerate")
    if name in doc and not intervals_only:
        raise OriginPrecisionError(
            f"the {part} sample was read once already ({results}, {name}); a second "
            "reading could be chosen"
        )


def _origin_correct(
    rows: list[dict[str, Any]], labels: Path, *, by_sections: bool, tables: Path, root: Path
) -> tuple[dict[str, bool], list[str]]:
    """Each sampled match's answer, from its strongest section pair's label or its pair's,
    and the labelers counted (named only when read by sections)."""
    if not by_sections:
        return origin_correct_by_pair(load_origin_labels(labels)), []
    docs = [d for c in collectors().values() for d in read_docs(c.source, root)]
    items = build_origin_items(rows, read_origin_matches(tables), docs, tables, root=root)
    correct = origin_correct_by_sections(items, load_origin_labels(labels))
    return correct, origin_labelers(items, load_origin_labels(labels))


def _precision_line(group: str, g: Mapping[str, Any]) -> str:
    """One group's precision, interval and counts, as the reading prints it."""
    p = g["precision"]
    shown = "too few labels" if p is None else f"{p['precision']:.2f} {p['interval_95']}"
    return f"{group}: {shown}, {g['correct']}/{g['labeled']} of {g['matches']:,} listed"


@eval_app.command("origin-census")
def eval_origin_census(
    part: Annotated[str, typer.Option("--part", help="The census read.")] = "census_eppc",
    tables: TablesOption = DATA / "tables",
    root: ModelDocsOption = MODELDOCS_DIR,
    labels: OriginLabelsOption = ORIGIN_LABELS,
    results: ResultsOption = RESULTS,
) -> None:
    """A census of one source's listed matches: its precision as a count, and the section pairs
    each wrong match rests on, by kind (DESIGN.md#origin-matches)."""
    chosen = ORIGIN_PARTS.get(part)
    if chosen is None or not chosen.census or chosen.source is None:
        censuses = ", ".join(n for n, c in ORIGIN_PARTS.items() if c.census)
        fail(f"error: no census {part!r}; one of {censuses}")
    with exits_on(OriginPrecisionError, ModelDocError, ValueError):
        if not chosen.sample.exists():
            raise OriginPrecisionError(f"no {chosen.sample}; run `lineage origin-sample` first")
        rows = load_origin_sample(chosen.sample)
        correct = origin_correct_by_match(load_origin_labels(labels))
        wrong = [
            (str(r["bill_id"]), str(r["doc_id"]))
            for r in rows
            if not correct.get((str(r["bill_id"]), str(r["doc_id"])), True)
        ]
        docs = [d for c in collectors().values() for d in read_docs(c.source, root)]
        sections = origin_pair_sections(wrong, read_origin_matches(tables), docs, tables, root=root)
        read = origin_census_reading(rows, correct, sections, source=chosen.source)
    section = {
        "command": f"uv run lineage eval origin-census --part {part}",
        "sample": str(chosen.sample),
        "labels": str(labels),
        "prompt": ORIGIN_PROMPT.version,
    } | read
    write_results(results, f"origins_{part}", section)
    typer.echo(
        f"{chosen.source}: {read['correct']}/{read['matches']} listed matches correct "
        f"({read['precision']}), {len(read['wrong'])} wrong -> {results}"
    )
