"""The semantic stage: its model, its cosine threshold, and its search for a query's windows.

The model was chosen on the labeled pair dataset before development recall was measured, so
the 36 development enactments did not tune it (`uv run lineage eval pair-data` ->
reports/pair_data.json). bge-small-en-v1.5 scored better than all-MiniLM-L6-v2 at every
precision measured and in ROC AUC. The pair dataset's threshold, the lowest cosine whose
precision on its training and validation pairs reaches 0.9, is BUILD_THRESHOLD: the build keeps
every match down to it. On this corpus a page listed 0.47 right at it, so the threshold pages
list at, THRESHOLD, was set higher on a sample of Georgia's pages (lineage.threshold,
DESIGN.md#choosing-the-threshold). Each window keeps its K nearest before a threshold applies,
so the matches at THRESHOLD are exactly the stored ones at or above it.

A query's matchable sections are cut into the stage's windows, 94 words without overlap, and
each window keeps its K nearest corpus texts at or above the threshold, found by exact search
over the memory-mapped float16 vectors (`lineage embed`); no index is needed. A corpus text
stands for every window that has it, so a match names each section with that text. See
DESIGN.md#semantic-stage and DESIGN.md#pair-dataset.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.ann import exact_search
from lineage.vectors import EmbedError, load_plan, load_vectors, rows_of, semantic_dir
from lineage.windows import WINDOW_WORDS, display_windows

MODEL = "bge-small-en-v1.5"
# reports/pair_data.json: semantic.models[MODEL].at_precision["0.9"].threshold
BUILD_THRESHOLD = 0.8905
# reports/threshold_proposal.json, the option the M5 review chose (DESIGN.md#choosing-the-threshold)
THRESHOLD = 0.93
K = 20  # nearest corpus texts each query window keeps

Embedder = Callable[[Sequence[str]], NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class SemanticSearch:
    """What a query needs to search the embedded corpus."""

    embed: Embedder  # unit-length rows, one per text
    vectors: Sequence[NDArray[Any]]  # the corpus's vectors, in text id order
    units_glob: str  # the windows table, to find the sections behind a text


def open_search(tables_dir: Path, embed: Embedder, model: str = MODEL) -> SemanticSearch:
    plan = load_plan(tables_dir)
    if plan is None:
        raise EmbedError(f"no semantic plan in {semantic_dir(tables_dir)}; run `lineage embed`")
    return SemanticSearch(
        embed,
        load_vectors(tables_dir, model, plan),
        str(semantic_dir(tables_dir) / "units" / "*.parquet"),
    )


@dataclass(frozen=True, slots=True)
class SectionHit:
    query_section: int  # the query's matchable section, numbered from 0
    bill_id: str
    ordinal: int
    cosine: float


def query_windows(sections: Sequence[str]) -> tuple[list[str], NDArray[np.int64]]:
    """Each section's windows, and the section each window comes from."""
    windows: list[str] = []
    owner: list[int] = []
    for number, text in enumerate(sections):
        cut = display_windows(text, size=WINDOW_WORDS, stride=WINDOW_WORDS)
        windows += cut
        owner += [number] * len(cut)
    return windows, np.array(owner, dtype=np.int64)


def search_sections(
    sections: Sequence[str],
    search: SemanticSearch,
    *,
    k: int = K,
    threshold: float = THRESHOLD,
) -> list[SectionHit]:
    """Every (query section, corpus section) pair the stage finds, at the best cosine between
    their windows."""
    windows, owner = query_windows(sections)
    if not windows:
        return []
    found = exact_search(search.vectors, search.embed(windows), k)
    keep = (found.ids >= 0) & (found.scores >= threshold)
    rows, cols = np.nonzero(keep)
    if not rows.size:
        return []
    text_ids = found.ids[rows, cols]
    with queries.connect_quiet() as con:
        units = queries.semantic_units_of(
            con, units_glob=search.units_glob, text_ids=sorted(set(text_ids.tolist()))
        )
    sections_of: dict[int, list[tuple[str, int]]] = {}
    for text_id, bill_id, ordinal in units:
        sections_of.setdefault(text_id, []).append((bill_id, ordinal))
    best: dict[tuple[int, str, int], float] = {}
    for q, text_id, cosine in zip(
        owner[rows].tolist(), text_ids.tolist(), found.scores[rows, cols].tolist(), strict=True
    ):
        for bill_id, ordinal in sections_of.get(text_id, []):
            key = (q, bill_id, ordinal)
            best[key] = max(best.get(key, -1.0), cosine)
    return [SectionHit(q, b, o, c) for (q, b, o), c in sorted(best.items())]


def best_cosines_to_bills(
    sections: Sequence[str], bill_ids: Sequence[str], search: SemanticSearch
) -> dict[str, float]:
    """Each bill's best cosine between any of its windows and any of the query's, whether or
    not the search kept it: why a bill the stage missed was missed."""
    windows, _ = query_windows(sections)
    if not windows or not bill_ids:
        return {}
    with queries.connect_quiet() as con:
        rows = queries.semantic_text_ids_of(
            con, units_glob=search.units_glob, bill_ids=sorted(set(bill_ids))
        )
    if not rows:
        return {}
    ids = np.array([t for _, t in rows], dtype=np.int64)
    scores = (rows_of(search.vectors, ids) @ search.embed(windows).T).max(axis=1)
    best: dict[str, float] = {}
    for (bill_id, _), score in zip(rows, scores.tolist(), strict=True):
        best[bill_id] = max(best.get(bill_id, -1.0), score)
    return best
