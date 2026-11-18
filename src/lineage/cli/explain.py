"""The explanation step's command."""

import datetime
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from lineage.cli.app import (
    TablesOption,
    app,
    pair_alignments,
)
from lineage.core.paths import DATA
from lineage.explain.explain import CACHE_DIR as EXPLAIN_CACHE
from lineage.explain.explain import MODEL as EXPLAIN_MODEL
from lineage.explain.explain import PROMPT_VERSION as EXPLAIN_PROMPT
from lineage.explain.explain import (
    NotCachedError,
    PairToExplain,
    anthropic_ask,
    asks_needed,
    explain,
    load_pairs,
    reground_cache,
)
from lineage.explain.explain import cache_path as explanation_cache_path
from lineage.explain.explain import projected_cost as projected_explain_cost
from lineage.explain.spend import BUDGET as CALL_BUDGET
from lineage.explain.spend import SPEND as CALL_SPEND
from lineage.explain.spend import BudgetError, Meter, check_projection
from lineage.explain.spend import spent as spent_on_calls
from lineage.match.semantic.models import MODELS_DIR, ModelError


def _pair_prompts(
    todo: Sequence[PairToExplain], tables: Path, models_dir: Path | None
) -> list[list[dict[str, Any]]]:
    """Each pair's aligned spans as the explanation prompt gives them."""
    return [prompt for prompt, _ in pair_alignments(todo, tables, models_dir)]


@app.command("explain")
def explain_command(
    pairs: Annotated[
        Path,
        typer.Argument(
            help="Pairs to explain, one per line: origin and descendant sections.",
            exists=True,
            dir_okay=False,
        ),
    ],
    tables: TablesOption = DATA / "tables",
    live: Annotated[
        bool, typer.Option("--live", help="Ask the model for pairs not in the cache (costs money).")
    ] = False,
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
    spend: Annotated[
        Path, typer.Option("--spend", help="M6's model-call ledger (committed).")
    ] = CALL_SPEND,
    again: Annotated[
        bool,
        typer.Option(
            "--again",
            help="With --live, ask once more for cached explanations written under earlier "
            "grounding rules: withheld ones, and ones keeping a claim the rules now drop.",
        ),
    ] = False,
    afresh: Annotated[
        bool,
        typer.Option(
            "--afresh",
            help="With --live --again, give every cached explanation written under earlier "
            "grounding rules a new round of its own, whatever it held.",
        ),
    ] = False,
    reground: Annotated[
        bool,
        typer.Option(
            "--reground",
            help="Rewrite each cache file's record of what grounding keeps under the current "
            "rules, with no call, and stop.",
        ),
    ] = False,
) -> None:
    """Explain what each descendant changed, grounded in aligned spans; cached, or --live."""
    if reground:
        typer.echo(f"{reground_cache(cache)} cache files regrounded under the current rules")
        return
    try:
        todo = load_pairs(pairs)
        prompts = _pair_prompts(todo, tables, models_dir if paraphrase else None)
        uncached = [p for p in prompts if not explanation_cache_path(p, cache).exists()]
        asks = [asks_needed(p, cache, again=again, afresh=afresh) for p in prompts]
        # Pairs with the same spans share one cache file, asked for once.
        due = list(
            {
                explanation_cache_path(p, cache): (p, n)
                for p, n in zip(prompts, asks, strict=True)
                if n
            }.values()
        )
        projected = projected_explain_cost([p for p, _ in due], [n for _, n in due])
        again_files = sum(explanation_cache_path(p, cache).exists() for p, _ in due)
        typer.echo(
            f"{len(prompts)} pairs, {len(uncached)} not cached, "
            f"{again_files} cache files to ask again, projected at most "
            f"${projected:.2f}; ${spent_on_calls(spend):.2f} of ${CALL_BUDGET:.2f} spent"
        )
        if uncached and not live:
            raise NotCachedError(f"{len(uncached)} pairs not cached; run with --live to ask")
        if again and not live:
            raise NotCachedError("--again asks the model; run with --live")
        if afresh and not again:
            raise NotCachedError("--afresh is a way of asking again; run with --live --again")
        meter = None
        ask = None
        if due:
            check_projection(projected, spend)
            meter = Meter(
                spend,
                "explain",
                EXPLAIN_MODEL,
                EXPLAIN_PROMPT,
                datetime.date.today().isoformat(),
                projected=projected,
            )
            ask = anthropic_ask()
        for pair, for_prompt in zip(todo, prompts, strict=True):
            outcome = explain(
                for_prompt, ask=ask, cache=cache, meter=meter, again=again, afresh=afresh
            )
            (ob, oo), (db, do) = pair.origin, pair.descendant
            typer.echo(f"\n{ob} section {oo} -> {db} section {do}")
            if outcome.withheld:
                typer.echo("  withheld: no grounded account after one retry")
            for claim in outcome.claims:
                cited = ", ".join([*claim.origin_span_ids, *claim.descendant_span_ids])
                typer.echo(f"  {claim.kind}: {claim.summary} [{cited}]")
            for claim, why in outcome.dropped:
                typer.echo(f"  dropped ({why}): {claim.summary}")
    except (ValueError, NotCachedError, ModelError, BudgetError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
