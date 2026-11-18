"""The evaluations' commands: labeling, the samples, and `lineage eval`'s measures of
matching.
"""

import datetime
import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from lineage.cli.app import (
    MinFreeOption,
    SemanticOption,
    TablesOption,
    app,
    eval_app,
    model_embedder,
    pair_alignments,
    pair_data_app,
    semantic_search,
)
from lineage.core.disk import GiB, InsufficientDiskSpaceError
from lineage.core.http import FetchError
from lineage.core.paths import DATA
from lineage.evals.acts import (
    ACTS,
    ACTS_DIR,
    HELD_OUT_ACTS,
    HELD_OUT_DIR,
    ORIGIN_HELD_OUT_ACTS,
    ActTextError,
)
from lineage.evals.agreement import PAIRS as AGREEMENT_PAIRS
from lineage.evals.agreement import SAMPLE as AGREEMENT_SAMPLE
from lineage.evals.agreement import SAMPLE_REPORT as AGREEMENT_SAMPLE_REPORT
from lineage.evals.agreement import SEED as AGREEMENT_SEED
from lineage.evals.agreement import (
    AgreementError,
    agreement,
    draw_subsample,
    subsample_report,
    subsample_rows,
    write_subsample,
)
from lineage.evals.answer_key import (
    ANSWER_KEY,
    HELD_OUT_KEY,
    ORIGINS_HELD_OUT_KEY,
    build_answer_key,
    render_answer_key,
)
from lineage.evals.commission import (
    RESULTS,
    CommissionError,
    check_held_out_configuration,
    commission_recall,
    run_configuration,
    with_earlier_runs,
    write_results,
)
from lineage.evals.cost import REPORT as COST_REPORT
from lineage.evals.cost import cost_report
from lineage.evals.cost import write_report as write_cost_report
from lineage.evals.faithfulness import (
    JUDGE_MODEL,
    JUDGE_PROMPT,
    JUDGMENTS,
    anthropic_judge,
    claims_to_judge,
    faithfulness,
    judge_claims,
    judge_name,
    load_judgments,
)
from lineage.evals.faithfulness import pending as pending_claims
from lineage.evals.faithfulness import projected_cost as projected_judge_cost
from lineage.evals.faithfulness import run_name as faithfulness_run
from lineage.evals.faithfulness import with_earlier_runs as with_earlier_faithfulness
from lineage.evals.flag_precision import (
    FLAGGED_SAMPLE,
    FLAGGED_SAMPLE_REPORT,
    build_flagged_sample,
    with_flag,
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
from lineage.evals.labels import PERSON as PERSON_PREFIX
from lineage.evals.labels import model_labeler as model_labeler_name
from lineage.evals.model_labels import BUDGET as LABEL_BUDGET
from lineage.evals.model_labels import MAX_COST as LABEL_MAX_COST
from lineage.evals.model_labels import MODEL_ID as LABEL_MODEL
from lineage.evals.model_labels import PROMPT_VERSION as LABEL_PROMPT_VERSION
from lineage.evals.model_labels import PROMPTS as LABEL_PROMPTS
from lineage.evals.model_labels import SPEND as LABEL_SPEND
from lineage.evals.model_labels import anthropic_ask as anthropic_label_ask
from lineage.evals.model_labels import (
    anthropic_batches,
    batch_usage,
    label_in_batch,
    label_with_model,
    model_pending,
    projected_cost,
)
from lineage.evals.model_labels import record_spend as record_label_spend
from lineage.evals.model_labels import spent as label_spent
from lineage.evals.pair_data import PAIR_DATA_DIR, PairDataError, fetch_pair_data
from lineage.evals.pair_eval import REPORT as PAIR_REPORT
from lineage.evals.pair_eval import evaluate as evaluate_pairs
from lineage.evals.pair_eval import write_report as write_pair_report
from lineage.evals.precision import SAMPLE as PRECISION_SAMPLE
from lineage.evals.precision import SAMPLE_REPORT as PRECISION_SAMPLE_REPORT
from lineage.evals.precision import build_sample, estimate, load_sizes, write_sample
from lineage.evals.relatives_sample import LABELS as RELATIVES_LABELS
from lineage.evals.relatives_sample import PARTS as RELATIVES_PARTS
from lineage.evals.relatives_sample import SAMPLE as RELATIVES_SAMPLE
from lineage.evals.relatives_sample import (
    RelativesSampleError,
    build_relatives_sample,
    load_pages,
    with_passages,
)
from lineage.evals.relatives_sample import estimate as relatives_estimate
from lineage.evals.relatives_sample import write_sample as write_relatives_sample
from lineage.evals.threshold import CANDIDATES as THRESHOLD_CANDIDATES
from lineage.evals.threshold import FIRST_CHOICE as THRESHOLD_FIRST_CHOICE
from lineage.evals.threshold import REPORT as THRESHOLD_REPORT
from lineage.evals.threshold import (
    ThresholdError,
    at_threshold,
    flag_at_threshold,
    page_precision,
    pair_level_at,
)
from lineage.evals.threshold import by_band as threshold_by_band
from lineage.evals.threshold import proposal as threshold_proposal
from lineage.evals.threshold import sample_inputs as threshold_sample_inputs
from lineage.evals.ulc import ENACTMENTS, HELD_OUT_ENACTMENTS, UlcError
from lineage.explain.explain import CACHE_DIR as EXPLAIN_CACHE
from lineage.explain.explain import (
    GROUNDING_RULES,
    NotCachedError,
    check_cache,
    load_pairs,
)
from lineage.explain.spend import BUDGET as CALL_BUDGET
from lineage.explain.spend import SPEND as CALL_SPEND
from lineage.explain.spend import BudgetError, Meter, check_projection
from lineage.explain.spend import spent as spent_on_calls
from lineage.match.listing import flagged_sections, shipped_flag
from lineage.match.near.stoplist import StoplistError
from lineage.match.query import IndexMissingError, require_index
from lineage.match.semantic.models import MODELS, MODELS_DIR, ModelError
from lineage.match.semantic.pairs import SemanticPairsError
from lineage.match.semantic.vectors import EmbedError
from lineage.match.state_boilerplate import REPORT as STATE_BOILERPLATE_REPORT
from lineage.match.state_boilerplate import (
    StateBoilerplateError,
    has_counts,
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
        embedder = model_embedder(models_dir) if paraphrase else None
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
    tables: TablesOption = DATA / "tables",
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
        embedder = model_embedder(models_dir) if paraphrase else None
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
    tables: TablesOption = DATA / "tables",
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
            todo = with_passages(todo, model_embedder(models_dir))
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


@eval_app.command("commission")
def eval_commission(
    tables: TablesOption = DATA / "tables",
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
        bool,
        typer.Option(
            "--final",
            help="The final evaluation of the held-out acts, once per settled configuration.",
        ),
    ] = False,
    semantic: SemanticOption = True,
) -> None:
    """Commission recall: what share of each uniform act's enactments its text finds."""
    if held_out and not final:
        typer.echo(
            "The held-out acts are evaluated only at the end of M5, once per configuration it "
            "settled (DESIGN.md#commission-recall). Pass --final for that run.",
            err=True,
        )
        raise typer.Exit(1)
    if final and not held_out:
        typer.echo("--final applies only with --held-out.", err=True)
        raise typer.Exit(1)
    flagged = flagged_sections(tables)
    previous: dict[str, Any] | None = (
        json.loads(results.read_text()).get("commission_held_out")
        if held_out and results.exists()
        else None
    )
    try:
        if held_out:
            check_held_out_configuration(previous, run_configuration(semantic, flagged))
        search = semantic_search(tables) if semantic else None
        section = commission_recall(
            tables,
            acts_dir=acts or (HELD_OUT_DIR if held_out else ACTS_DIR),
            enactments_path=enactments or (HELD_OUT_ENACTMENTS if held_out else ENACTMENTS),
            sessions_path=sessions,
            act_ids=HELD_OUT_ACTS if held_out else ACTS,
            development_path=ENACTMENTS if held_out else None,
            semantic=search,
            # Development recall at each threshold option, for the proposal; never held out.
            semantic_thresholds=THRESHOLD_CANDIDATES if search is not None and not held_out else (),
            flagged=flagged,
        )
        if held_out:
            section = with_earlier_runs(section, previous)
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
    tables: TablesOption = DATA / "tables",
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
    origins: Annotated[
        bool,
        typer.Option(
            "--origins",
            help="With --held-out: resolve the reserve acts, origin recall's sealed key "
            "[default out: reports/answer_key_origins_heldout.csv].",
        ),
    ] = False,
) -> None:
    """The bills each commission list row resolved to, for checking the mapping by eye.

    Resolving the held-out lists runs no matching, so it may run before the final evaluation.
    """
    if origins and not held_out:
        typer.echo("error: --origins resolves the held-out lists; pass --held-out", err=True)
        raise typer.Exit(1)
    held_acts = ORIGIN_HELD_OUT_ACTS if origins else HELD_OUT_ACTS
    out = out or ((ORIGINS_HELD_OUT_KEY if origins else HELD_OUT_KEY) if held_out else ANSWER_KEY)
    try:
        note, rows = build_answer_key(
            tables,
            enactments_path=enactments or (HELD_OUT_ENACTMENTS if held_out else ENACTMENTS),
            sessions_path=sessions,
            development_path=ENACTMENTS if held_out else None,
            acts=frozenset(held_acts) if held_out else None,
        )
    except UlcError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_answer_key(rows, note))
    counted = {(r["act"], r["jurisdiction"], r["year"]) for r in rows if r["counted"] == "yes"}
    resolved = {(r["act"], r["jurisdiction"], r["year"]) for r in rows if r["bill_id"]}
    typer.echo(f"{len(resolved)} of {len(counted)} counted rows resolved to a bill -> {out}")


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
    tables: TablesOption = DATA / "tables",
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file (committed).")
    ] = LABELS_PATH,
    model_labeler: Annotated[
        str, typer.Option("--model-labeler", help="The model labeler compared, model id@prompt.")
    ] = model_labeler_name(LABEL_MODEL, LABEL_PROMPT_VERSION),
    strata: Annotated[
        Path, typer.Option("--strata", help="The precision sample's draw report, its strata.")
    ] = PRECISION_SAMPLE_REPORT,
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
) -> None:
    """A person's labels against a model's on the subsample: weighted kappa, cutoff agreement,
    and the precision each gives on the same pairs."""
    try:
        items, kept = _current_items(sample, tables, labels)
        doc = agreement(items, kept, model_labeler=model_labeler, sizes=load_sizes(strata))
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
    part: Annotated[
        str,
        typer.Option(
            "--part",
            help="The sample read: reporting, or a tuning part "
            "(lineage.evals.relatives_sample.PARTS).",
        ),
    ] = "reporting",
    tables: TablesOption = DATA / "tables",
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file (committed).")
    ] = RELATIVES_LABELS,
    model_labeler: Annotated[
        str, typer.Option("--model-labeler", help="The model labeler counted, model id@prompt.")
    ] = model_labeler_name(LABEL_MODEL, "passages-v1-high"),
    models_dir: Annotated[
        Path, typer.Option("--models", help="Directory of model files (local), for alignment.")
    ] = MODELS_DIR,
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
) -> None:
    """The precision of what a page lists, from the reporting sample's labels; with --part, a
    tuning sample's, recorded apart."""
    if part not in RELATIVES_PARTS:
        typer.echo(f"error: --part is one of {', '.join(RELATIVES_PARTS)}, not {part!r}", err=True)
        raise typer.Exit(1)
    sample, draw = RELATIVES_PARTS[part].sample, RELATIVES_PARTS[part].report
    try:
        items, kept = _current_items(sample, tables, labels)
        # The passages the labels were asked on, so the shares with and without them are real.
        items = with_passages(items, model_embedder(models_dir))
        pages = load_pages(sample)
        drawn = json.loads(draw.read_text())
    except (LabelError, ModelError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    section: dict[str, Any] = {
        "command": "uv run lineage eval relatives",
        "sample": str(sample),
        "threshold": drawn.get("threshold"),
        "state_boilerplate_flag": drawn.get("state_boilerplate_flag"),
        "person": relatives_estimate(items, kept, pages, labeler=PERSON_PREFIX),
        "provisional_model": relatives_estimate(
            items, kept, pages, labeler=model_labeler, exact=True
        ),
    }
    # Where along the cosine the model's labels change, as the threshold proposal counted it:
    # of the reporting sample, a description that tunes nothing.
    listed, correct = threshold_sample_inputs(items, kept, pages, model_labeler)
    section["provisional_model"]["by_band"] = threshold_by_band(listed, correct)
    if part != "reporting":
        # What a tuning sample says of each stricter option, estimated from cosines as the
        # proposal estimated them.
        section["provisional_model"]["at_stricter_thresholds"] = [
            {"threshold": t, **page_precision(listed, correct, at_threshold(t))}
            for t in THRESHOLD_CANDIDATES
            if t > float(drawn["threshold"])
        ]
    write_results(
        results, "relatives_precision" if part == "reporting" else f"relatives_precision_{part}",
        section,
    )  # fmt: skip
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


@eval_app.command("threshold")
def eval_threshold(
    sample: Annotated[
        Path,
        typer.Argument(help="The tuning sample of relatives.", exists=True, dir_okay=False),
    ] = RELATIVES_SAMPLE,
    tables: TablesOption = DATA / "tables",
    labels: Annotated[
        Path, typer.Option("--labels", help="Labels file (committed).")
    ] = RELATIVES_LABELS,
    model_labeler: Annotated[
        str, typer.Option("--model-labeler", help="The model labeler counted, model id@prompt.")
    ] = model_labeler_name(LABEL_MODEL, "passages-v1-high"),
    results: Annotated[
        Path, typer.Option("--results", help="Results JSON, read for development recall.")
    ] = RESULTS,
    report: Annotated[
        Path, typer.Option("--report", help="The proposal's report JSON (committed).")
    ] = THRESHOLD_REPORT,
) -> None:
    """Each semantic threshold option's page precision on the tuning sample, with its recall."""
    if (
        sample.resolve() == RELATIVES_PARTS["reporting"].sample.resolve()
        or "reporting" in sample.name
    ):
        typer.echo("error: the reporting sample is read only once the threshold is final", err=True)
        raise typer.Exit(1)
    try:
        items, kept = _current_items(sample, tables, labels)
        pages = load_pages(sample)
        commission = json.loads(results.read_text())["commission"]
        doc = threshold_proposal(items, kept, pages, commission, labeler=model_labeler)
        doc["state_boilerplate_flag"] = flag_at_threshold(
            tables, items, kept, pages, labeler=model_labeler
        )
        doc["state_boilerplate_flag_at_first_choice"] = flag_at_threshold(
            tables, items, kept, pages, labeler=model_labeler, threshold=THRESHOLD_FIRST_CHOICE
        )
    except (LabelError, ThresholdError, KeyError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    report.write_text(json.dumps(doc, indent=1) + "\n")
    for option in doc["options"]:
        name = option["option"] + (f" {option['threshold']}" if "threshold" in option else "")
        page = option.get("page") or option["by_meaning_alone"]
        recall = option["development_recall"]
        recall = recall.get("on_the_page", recall)
        typer.echo(
            f"{name}: page precision {page['precision']}, "
            f"recall {recall['found']}/{recall['enactments']}"
        )
    typer.echo(f"-> {report}")


@eval_app.command("precision")
def eval_precision(
    sample: Annotated[
        Path,
        typer.Argument(
            help="The precision sample, one pair per line.", exists=True, dir_okay=False
        ),
    ] = PRECISION_SAMPLE,
    tables: TablesOption = DATA / "tables",
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
    flagged_labeler: Annotated[
        str,
        typer.Option(
            "--flagged-labeler", help="The model labeler counted on the flag's dropped pairs."
        ),
    ] = model_labeler_name(LABEL_MODEL, "passages-v1-high"),
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
            dropped_labeler=flagged_labeler,
        )
        if has_counts(tables):
            measured["at_stage_threshold"] = pair_level_at(
                tables,
                items,
                kept,
                sizes,
                labeler=model_labeler,
                dropped_items=dropped,
                dropped_labeler=flagged_labeler,
            )
    except (LabelError, PrecisionError, StateBoilerplateError, ThresholdError) as e:
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


@eval_app.command("faithfulness")
def eval_faithfulness(
    pairs: Annotated[
        Path,
        typer.Argument(
            help="The pairs explained, one per line: the featured scope.",
            exists=True,
            dir_okay=False,
        ),
    ],
    tables: TablesOption = DATA / "tables",
    live: Annotated[
        bool,
        typer.Option("--live", help="Ask the judge about unjudged claims (costs money)."),
    ] = False,
    cache: Annotated[
        Path, typer.Option("--cache", help="Cached explanations (committed).")
    ] = EXPLAIN_CACHE,
    judgments: Annotated[
        Path, typer.Option("--judgments", help="The judge's verdicts, appended to (committed).")
    ] = JUDGMENTS,
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
    spend: Annotated[
        Path, typer.Option("--spend", help="M6's model-call ledger (committed).")
    ] = CALL_SPEND,
    results: Annotated[Path, typer.Option("--results", help="Results JSON (committed).")] = RESULTS,
) -> None:
    """Faithfulness, model-judged: the share of kept claims that describe their cited spans
    correctly, beside the hard check that every kept claim cites spans that exist and differ."""
    try:
        aligned = pair_alignments(load_pairs(pairs), tables, models_dir if paraphrase else None)
        prompts = [prompt for prompt, _ in aligned]
        claims, counts = claims_to_judge(prompts, [view for _, view in aligned], cache=cache)
        todo = pending_claims(claims, load_judgments(judgments), judge_name())
        projected = projected_judge_cost(todo)
        typer.echo(
            f"{len(claims)} kept claims, {len(todo)} not yet judged, projected at most "
            f"${projected:.2f}; ${spent_on_calls(spend):.2f} of ${CALL_BUDGET:.2f} spent"
        )
        if todo and live:
            check_projection(projected, spend)
            today = datetime.date.today().isoformat()
            meter = Meter(spend, "judge", JUDGE_MODEL, JUDGE_PROMPT, today, projected=projected)
            judge_claims(todo, ask=anthropic_judge(), path=judgments, meter=meter, today=today)
    except (ValueError, NotCachedError, ModelError, BudgetError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    doc = faithfulness(claims, load_judgments(judgments), counts)
    problems = check_cache(cache)
    section: dict[str, Any] = {
        "command": "uv run lineage eval faithfulness",
        "pairs": str(pairs),
        "pairs_explained": len(prompts),
        "grounding_rules": GROUNDING_RULES,
        "hard_check": {
            "cache_files": len(list(cache.glob("*.json"))) if cache.exists() else 0,
            "problems": problems,
        },
        **doc,
    }
    previous = json.loads(results.read_text()).get("faithfulness") if results.exists() else None
    section = with_earlier_faithfulness(section, previous, run=faithfulness_run(GROUNDING_RULES))
    write_results(results, "faithfulness", section)
    overall = doc["overall"]
    shown = (
        "nothing judged"
        if overall["share"] is None
        else (
            f"{overall['correct']}/{overall['judged']} = {overall['share']:.3f} "
            f"(95% {overall['interval_95'][0]:.3f}-{overall['interval_95'][1]:.3f})"
        )
    )
    typer.echo(f"model-judged faithful: {shown}; hard check: {len(problems)} problems -> {results}")


@eval_app.command("cost")
def eval_cost(
    tables: TablesOption = DATA / "tables",
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
        search = semantic_search(tables) if semantic else None
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
    tables: TablesOption = DATA / "tables",
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
