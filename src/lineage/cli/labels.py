"""The labeling commands: a person's labels at the terminal or on a phone page, the
precision and agreement samples, the relatives sample, and a model's provisional labels and
their spend.
"""

import datetime
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from lineage.cli.app import (
    AlignModelsOption,
    LabelsAppendOption,
    ModelsOption,
    ParaphraseOption,
    TablesOption,
    app,
    model_embedder,
)
from lineage.core.paths import DATA
from lineage.evals.agreement import PAIRS as AGREEMENT_PAIRS
from lineage.evals.agreement import SAMPLE as AGREEMENT_SAMPLE
from lineage.evals.agreement import SAMPLE_REPORT as AGREEMENT_SAMPLE_REPORT
from lineage.evals.agreement import SEED as AGREEMENT_SEED
from lineage.evals.agreement import (
    AgreementError,
    draw_subsample,
    subsample_report,
    subsample_rows,
    write_subsample,
)
from lineage.evals.flag_precision import (
    FLAGGED_SAMPLE,
    FLAGGED_SAMPLE_REPORT,
    build_flagged_sample,
)
from lineage.evals.label_page import OUT as LABEL_PAGE_OUT
from lineage.evals.label_page import PAIRS_FILE as LABEL_PAGE_DATA
from lineage.evals.label_page import append_rows as append_label_rows
from lineage.evals.label_page import imported_rows as imported_label_rows
from lineage.evals.label_page import load_saved as load_saved_labels
from lineage.evals.label_page import page_data as label_page_data
from lineage.evals.label_page import write_page as write_label_page
from lineage.evals.labels import (
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
from lineage.evals.model_labels import BUDGET as LABEL_BUDGET
from lineage.evals.model_labels import MAX_COST as LABEL_MAX_COST
from lineage.evals.model_labels import MODEL_ID as LABEL_MODEL
from lineage.evals.model_labels import PROMPT_VERSION as LABEL_PROMPT_VERSION
from lineage.evals.model_labels import PROMPTS as LABEL_PROMPTS
from lineage.evals.model_labels import SPEND as LABEL_SPEND
from lineage.evals.model_labels import (
    Prompt,
    anthropic_batches,
    batch_usage,
    label_in_batch,
    label_with_model,
    model_pending,
    projected_cost,
)
from lineage.evals.model_labels import anthropic_ask as anthropic_label_ask
from lineage.evals.model_labels import record_spend as record_label_spend
from lineage.evals.model_labels import spent as label_spent
from lineage.evals.precision import SAMPLE as PRECISION_SAMPLE
from lineage.evals.precision import SAMPLE_REPORT as PRECISION_SAMPLE_REPORT
from lineage.evals.precision import build_sample, write_sample
from lineage.evals.relatives_sample import PARTS as RELATIVES_PARTS
from lineage.evals.relatives_sample import (
    RelativesSampleError,
    build_relatives_sample,
    load_pages,
    with_passages,
)
from lineage.evals.relatives_sample import write_sample as write_relatives_sample
from lineage.match.listing import shipped_flag
from lineage.match.semantic.models import MODELS_DIR, ModelError
from lineage.match.semantic.pairs import SemanticPairsError
from lineage.match.state_boilerplate import (
    StateBoilerplateError,
)
from lineage.match.strata import PrecisionError


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
    tables: TablesOption = DATA / "tables",
    labels: LabelsAppendOption = LABELS_PATH,
    color: Annotated[bool, typer.Option("--color/--no-color", help="Mark spans in color.")] = True,
    paraphrase: ParaphraseOption = True,
    models_dir: ModelsOption = MODELS_DIR,
) -> None:
    """Label sampled matches by hand, unrelated (0) to identical (4); stop and resume at will."""
    try:
        who = person(labeler)
        embedder = model_embedder(models_dir) if paraphrase else None
        items, kept = current_items(sample, tables, labels)
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
    tables: TablesOption = DATA / "tables",
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file; pairs a person labeled are left out.")
    ] = LABELS_PATH,
    out: Annotated[
        Path, typer.Option("--out", help="Directory for the page and its data (local).")
    ] = LABEL_PAGE_OUT,
    paraphrase: ParaphraseOption = True,
    models_dir: ModelsOption = MODELS_DIR,
) -> None:
    """Write a labeling page for a phone: the pending pairs, blind to stages, for an artifact."""
    try:
        embedder = model_embedder(models_dir) if paraphrase else None
        items, kept = current_items(sample, tables, labels)
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
    tables: TablesOption = DATA / "tables",
    labels: LabelsAppendOption = LABELS_PATH,
) -> None:
    """Append a labeling page's saved labels to the labels file, each checked against the sample."""
    try:
        items, kept = current_items(sample, tables, labels)
        rows = imported_label_rows(load_saved_labels(saved), items, kept, handle=labeler)
    except LabelError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    append_label_rows(labels, rows)
    left = len(pending(items, [*kept, *rows]))
    typer.echo(f"{len(rows)} labels imported, {left} pairs of {sample} left -> {labels}")


@app.command("precision-sample")
def precision_sample(
    tables: TablesOption = DATA / "tables",
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
    tables: TablesOption = DATA / "tables",
    part: Annotated[
        str,
        typer.Option(
            "--part", help="tuning, tuning_093 or reporting (lineage.evals.relatives_sample.PARTS)."
        ),
    ] = "tuning",
    sample: Annotated[
        Path | None,
        typer.Option("--sample", help="The sample file, one relative per line (committed)."),
    ] = None,
    report: Annotated[
        Path | None, typer.Option("--report", help="The draw's report JSON (committed).")
    ] = None,
    excluded_samples: Annotated[
        list[Path] | None,
        typer.Option("--excluded", help="A sample whose bills this part never draws (repeat)."),
    ] = None,
) -> None:
    """Draw listed relatives of random bills of one state, to measure what a page lists."""
    if part not in RELATIVES_PARTS:
        typer.echo(f"error: --part is one of {', '.join(RELATIVES_PARTS)}, not {part!r}", err=True)
        raise typer.Exit(1)
    chosen = RELATIVES_PARTS[part]
    sample = sample or chosen.sample
    report = report or chosen.report
    try:
        excluded = {
            str(p["bill_id"])
            for other in (excluded_samples or chosen.excludes)
            for p in load_pages(other).values()
        }
        flag = shipped_flag(tables) if chosen.flag else None
        if chosen.flag and flag is None:
            raise RelativesSampleError(
                "no state boilerplate counts; run `lineage state-boilerplate`"
            )
        rows, doc = build_relatives_sample(
            tables,
            seed=chosen.seed,
            threshold=chosen.threshold,
            flag=flag,
            excluded=excluded,
            first=chosen.first,
            exempt_at=chosen.exempt_at,
        )
    except (RelativesSampleError, SemanticPairsError, StateBoilerplateError) as e:
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
    tables: TablesOption = DATA / "tables",
    labels: LabelsAppendOption = LABELS_PATH,
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
    models_dir: AlignModelsOption = MODELS_DIR,
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
    batched = batch or batch_id is not None
    targets = _label_targets(sample, labels, also, batched=batched)
    todo, paths = _model_todo(targets, tables, prompt, models_dir)
    noun = "pair" if len(todo) == 1 else "pairs"
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
    asked = f"{len(todo)} {noun} {projected}"
    _check_label_cost(
        asked, cost, already=already, max_cost=max_cost, budget=budget, batch_id=batch_id
    )
    for _, target_labels in targets:
        target_labels.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    if batched:
        _label_batched(
            todo, paths, targets, model=model, prompt=prompt, batch_id=batch_id, spend=spend,
            budget=budget, today=today,
        )  # fmt: skip
        return
    done, unusable = label_with_model(
        todo, ask=anthropic_label_ask(model, prompt), labels_path=labels, today=today,
        prompt=prompt,
    )  # fmt: skip
    typer.echo(f"{done} labeled by {model}, {unusable} unusable replies -> {labels}")


def _label_targets(
    sample: Path, labels: Path, also: list[str] | None, *, batched: bool
) -> list[tuple[Path, Path]]:
    """Each sample asked and the labels file it appends to: SAMPLE first, then each --also."""
    targets = [(sample, labels)]
    for extra in also or []:
        extra_sample, sep, extra_labels = extra.partition("=")
        if not sep or not extra_sample or not extra_labels:
            typer.echo(f"error: --also takes SAMPLE=LABELS, not {extra!r}", err=True)
            raise typer.Exit(1)
        targets.append((Path(extra_sample), Path(extra_labels)))
    if len(targets) > 1 and not batched:
        typer.echo("error: --also asks several samples in one batch; add --batch", err=True)
        raise typer.Exit(1)
    return targets


def _model_todo(
    targets: list[tuple[Path, Path]], tables: Path, prompt: Prompt, models_dir: Path
) -> tuple[list[Item], dict[str, Path]]:
    """The pairs no model has labeled under the prompt, across every sample asked, with their
    passages when the prompt asks about passages, and each pair's labels file by pair id."""
    todo: list[Item] = []
    paths: dict[str, Path] = {}
    try:
        for target_sample, target_labels in targets:
            items, kept = current_items(target_sample, tables, target_labels)
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
            todo = with_passages(todo, model_embedder(models_dir))
        except ModelError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(1) from e
    return todo, paths


def _check_label_cost(
    asked: str,
    cost: float,
    *,
    already: float,
    max_cost: float,
    budget: float,
    batch_id: str | None,
) -> None:
    """Refuse a run projected over the cap, or a new one that would take the spend over the
    budget; collecting a batch already sent spends nothing new."""
    if cost > max_cost:
        typer.echo(f"error: {asked}, over the ${max_cost:,.2f} cap; nothing asked", err=True)
        raise typer.Exit(1)
    if batch_id is None and already + cost > budget:
        typer.echo(
            f"error: {asked}, with ${already:,.2f} spent, over the "
            f"${budget:,.2f} budget; nothing asked",
            err=True,
        )
        raise typer.Exit(1)


def _label_batched(
    todo: list[Item],
    paths: dict[str, Path],
    targets: list[tuple[Path, Path]],
    *,
    model: str,
    prompt: Prompt,
    batch_id: str | None,
    spend: Path,
    budget: float,
    today: str,
) -> None:
    """Ask every pair in one batch, or collect `batch_id`, and record the batch's spend."""
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


def current_items(
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
