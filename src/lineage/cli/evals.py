"""The evaluations' commands: labeling, the samples, and `lineage eval`'s measures of
matching.
"""

import datetime
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from lineage.cli.app import (
    AlignModelsOption,
    CacheOption,
    LabelsOption,
    MinFreeOption,
    ModelsOption,
    ParaphraseOption,
    ReportOption,
    ResultsOption,
    SemanticOption,
    SessionsOption,
    TablesOption,
    eval_app,
    model_embedder,
    pair_alignments,
    pair_data_app,
    semantic_search,
)
from lineage.cli.labels import current_items
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
from lineage.evals.agreement import SAMPLE as AGREEMENT_SAMPLE
from lineage.evals.agreement import (
    AgreementError,
    agreement,
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
    with_flag,
)
from lineage.evals.labels import (
    LABELS_PATH,
    LabelError,
    load_sample,
    read_items,
)
from lineage.evals.labels import PERSON as PERSON_PREFIX
from lineage.evals.labels import model_labeler as model_labeler_name
from lineage.evals.model_labels import MODEL_ID as LABEL_MODEL
from lineage.evals.model_labels import PROMPT_VERSION as LABEL_PROMPT_VERSION
from lineage.evals.pair_data import PAIR_DATA_DIR, PairDataError, fetch_pair_data
from lineage.evals.pair_eval import REPORT as PAIR_REPORT
from lineage.evals.pair_eval import evaluate as evaluate_pairs
from lineage.evals.pair_eval import write_report as write_pair_report
from lineage.evals.precision import SAMPLE as PRECISION_SAMPLE
from lineage.evals.precision import SAMPLE_REPORT as PRECISION_SAMPLE_REPORT
from lineage.evals.precision import estimate, load_sizes
from lineage.evals.relatives_sample import LABELS as RELATIVES_LABELS
from lineage.evals.relatives_sample import PARTS as RELATIVES_PARTS
from lineage.evals.relatives_sample import SAMPLE as RELATIVES_SAMPLE
from lineage.evals.relatives_sample import estimate as relatives_estimate
from lineage.evals.relatives_sample import (
    load_pages,
    with_passages,
)
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
from lineage.match.listing import flagged_sections
from lineage.match.near.stoplist import StoplistError
from lineage.match.query import IndexMissingError, require_index
from lineage.match.semantic.models import MODELS, MODELS_DIR, ModelError
from lineage.match.semantic.vectors import EmbedError
from lineage.match.state_boilerplate import REPORT as STATE_BOILERPLATE_REPORT
from lineage.match.state_boilerplate import (
    StateBoilerplateError,
    has_counts,
)
from lineage.match.strata import PrecisionError


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
    sessions: SessionsOption = Path("snapshot/sessions.csv"),
    results: ResultsOption = RESULTS,
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
    sessions: SessionsOption = Path("snapshot/sessions.csv"),
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
    labels: LabelsOption = LABELS_PATH,
    model_labeler: Annotated[
        str, typer.Option("--model-labeler", help="The model labeler compared, model id@prompt.")
    ] = model_labeler_name(LABEL_MODEL, LABEL_PROMPT_VERSION),
    strata: Annotated[
        Path, typer.Option("--strata", help="The precision sample's draw report, its strata.")
    ] = PRECISION_SAMPLE_REPORT,
    results: ResultsOption = RESULTS,
) -> None:
    """A person's labels against a model's on the subsample: weighted kappa, cutoff agreement,
    and the precision each gives on the same pairs."""
    try:
        items, kept = current_items(sample, tables, labels)
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
    labels: LabelsOption = RELATIVES_LABELS,
    model_labeler: Annotated[
        str, typer.Option("--model-labeler", help="The model labeler counted, model id@prompt.")
    ] = model_labeler_name(LABEL_MODEL, "passages-v1-high"),
    models_dir: AlignModelsOption = MODELS_DIR,
    results: ResultsOption = RESULTS,
) -> None:
    """The precision of what a page lists, from the reporting sample's labels; with --part, a
    tuning sample's, recorded apart."""
    if part not in RELATIVES_PARTS:
        typer.echo(f"error: --part is one of {', '.join(RELATIVES_PARTS)}, not {part!r}", err=True)
        raise typer.Exit(1)
    sample, draw = RELATIVES_PARTS[part].sample, RELATIVES_PARTS[part].report
    try:
        items, kept = current_items(sample, tables, labels)
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
    labels: LabelsOption = RELATIVES_LABELS,
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
        items, kept = current_items(sample, tables, labels)
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
    labels: LabelsOption = LABELS_PATH,
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
    results: ResultsOption = RESULTS,
) -> None:
    """Precision per stratum, stage and overall, from people's labels and, apart, a model's,
    with the state boilerplate flag off and on."""
    try:
        items, kept = current_items(sample, tables, labels)
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
    cache: CacheOption = EXPLAIN_CACHE,
    judgments: Annotated[
        Path, typer.Option("--judgments", help="The judge's verdicts, appended to (committed).")
    ] = JUDGMENTS,
    paraphrase: ParaphraseOption = True,
    models_dir: ModelsOption = MODELS_DIR,
    spend: Annotated[
        Path, typer.Option("--spend", help="M6's model-call ledger (committed).")
    ] = CALL_SPEND,
    results: ResultsOption = RESULTS,
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
    report: ReportOption = (COST_REPORT),
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
    models_dir: ModelsOption = MODELS_DIR,
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
