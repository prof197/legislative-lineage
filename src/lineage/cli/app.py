"""The CLI's Typer apps (`lineage` and its groups), the version callback, and the options
and helpers more than one layer's commands use. Each layer's commands live in their own
module beside this one; lineage.cli imports them all. Subcommands stay thin wrappers over
library functions.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from numpy.typing import NDArray

from lineage import __version__
from lineage.align import Embedder, align
from lineage.evals.faithfulness import (
    PairView,
    pair_view,
)
from lineage.explain.explain import (
    PairToExplain,
    pair_texts,
    prompt_spans,
)
from lineage.match.semantic.embed import embed as embed_texts
from lineage.match.semantic.embed import load_model
from lineage.match.semantic.models import MODELS_DIR
from lineage.match.semantic.stage import MODEL as SEMANTIC_MODEL
from lineage.match.semantic.stage import SemanticSearch, open_search

app = typer.Typer(no_args_is_help=True, add_completion=False)
eval_app = typer.Typer(no_args_is_help=True, help="Run an eval and write its results.")
app.add_typer(eval_app, name="eval")
text_app = typer.Typer(
    no_args_is_help=True, help="The text layer: bill text fetched from the source."
)
app.add_typer(text_app, name="text")
model_app = typer.Typer(
    no_args_is_help=True, help="The embedding models the semantic stage runs, pinned by hash."
)
app.add_typer(model_app, name="model")
pair_data_app = typer.Typer(
    no_args_is_help=True,
    help="The labeled pair dataset (Kim et al., EMNLP 2021), local and for evaluation only.",
)
app.add_typer(pair_data_app, name="pair-data")

RawOption = Annotated[Path, typer.Option("--raw", help="Directory of Open States archives.")]
ManifestOption = Annotated[Path, typer.Option("--manifest", help="Snapshot manifest (committed).")]
TablesOption = Annotated[Path, typer.Option("--tables", help="Directory of Parquet tables.")]
RulesOption = Annotated[Path, typer.Option("--rules", help="Document-type rules (committed).")]
MinFreeOption = Annotated[
    float, typer.Option("--min-free-gib", help="Refuse to run if disk would drop below this.")
]
SemanticOption = Annotated[
    bool,
    typer.Option(
        "--semantic/--no-semantic",
        help="Search by meaning too, over the embedded corpus (`lineage embed`).",
    ),
]

# Options more than one command declares, word for word.
ResultsOption = Annotated[Path, typer.Option("--results", help="Results JSON (committed).")]
ModelsOption = Annotated[Path, typer.Option("--models", help="Directory of model files (local).")]
ReportOption = Annotated[Path, typer.Option("--report", help="Report JSON (committed).")]
ModelDocsOption = Annotated[
    Path, typer.Option("--modeldocs", help="Where documents are kept (local).")
]
ParaphraseOption = Annotated[
    bool,
    typer.Option(
        "--paraphrase/--no-paraphrase",
        help="Mark paraphrased sentences, with the semantic stage's model (local).",
    ),
]
SessionsOption = Annotated[Path, typer.Option("--sessions", help="Session dates CSV (committed).")]
LiveExportOption = Annotated[
    Path | None, typer.Option("--export", help="The live export (local); default data/export/<ST>.")
]
LabelsAppendOption = Annotated[
    Path, typer.Option("--labels", help="Labels file, appended to (committed).")
]
LabelsOption = Annotated[Path, typer.Option("--labels", help="Labels file (committed).")]
CacheOption = Annotated[Path, typer.Option("--cache", help="Cached explanations (committed).")]
PlanOption = Annotated[Path, typer.Option("--plan", help="Fetch plan Parquet (local).")]
AlignModelsOption = Annotated[
    Path, typer.Option("--models", help="Directory of model files (local), for alignment.")
]
OriginLabelsOption = Annotated[
    Path, typer.Option("--labels", help="The origin labels (committed).")
]
DayOption = Annotated[
    str | None, typer.Option("--day", help="The day, YYYY-MM-DD [default: today, UTC].")
]


def model_embedder(models_dir: Path = MODELS_DIR) -> Embedder:
    model = load_model(SEMANTIC_MODEL, models_dir)

    def embed(texts: Sequence[str]) -> NDArray[np.float32]:
        return embed_texts(model, texts)

    return embed


def semantic_search(tables: Path, models_dir: Path = MODELS_DIR) -> SemanticSearch:
    model = load_model(SEMANTIC_MODEL, models_dir)
    return open_search(tables, lambda texts: embed_texts(model, texts))


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


def pair_alignments(
    todo: Sequence[PairToExplain], tables: Path, models_dir: Path | None
) -> list[tuple[list[dict[str, Any]], PairView]]:
    """Each pair's aligned spans as the explanation prompt gives them, and its texts as a
    reader sees them, paraphrase marked with the semantic stage's model unless `models_dir`
    is None."""
    embedder = model_embedder(models_dir) if models_dir is not None else None
    out: list[tuple[list[dict[str, Any]], PairView]] = []
    for o, d in pair_texts(tables, todo):
        spans = align(o, d, embed=embedder).spans
        out.append(
            (
                prompt_spans(spans, o, d, origin_side="a"),
                pair_view(spans, o, d, origin_side="a"),
            )
        )
    return out


publish_app = typer.Typer(
    no_args_is_help=True,
    help="The live track's publishing: the incremental export, the feed and the swap.",
)
app.add_typer(publish_app, name="publish")
schedule_app = typer.Typer(
    no_args_is_help=True, help="The live track's scheduled jobs, as launchd runs them."
)
app.add_typer(schedule_app, name="schedule")
