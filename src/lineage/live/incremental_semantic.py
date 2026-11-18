"""The update's semantic stage: windows, vectors, neighbors and section pairs after the day
(lineage.live.incremental).

See DESIGN.md#updates, DR-7.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from numpy.typing import NDArray

from lineage.core import db
from lineage.core.disk import require_free_space
from lineage.live import queries
from lineage.live.incremental_near import Renumbering
from lineage.live.staging import Staging
from lineage.live.updates import UpdateError
from lineage.match.near import queries as near_queries
from lineage.match.semantic import queries as semantic_queries
from lineage.match.semantic.neighbors import update_neighbors
from lineage.match.semantic.pairs import (
    SemanticPairs,
    changed_texts,
    load_built_semantic_pairs,
    pairs_dir,
    renumbered_semantic,
    update_semantic_pairs,
    write_semantic_pairs,
)
from lineage.match.semantic.stage import BUILD_THRESHOLD, Embedder, K
from lineage.match.semantic.vectors import (
    SEMANTIC_DIR,
    extend_store,
    load_plan,
    load_vectors,
    semantic_dir,
    text_count,
    vectors_dir,
    write_plan,
)
from lineage.text.windows import WINDOW_WORDS, display_windows


@dataclass(frozen=True, slots=True)
class SemanticUpdate:
    before: SemanticPairs
    carried: SemanticPairs
    after: SemanticPairs


_UNITS = f"{SEMANTIC_DIR}/units"
# A new section's window: (sections file, bill, ordinal, window number, text).
_Cut = tuple[str, str, int, int, str]


def _drop_units(
    con: duckdb.DuckDBPyConnection, tables: Path, view: Staging, retired: set[str]
) -> set[int]:
    """The retired records' windows dropped from the units files holding them; returns the
    texts those windows carried, which may now be carried by none."""
    holding = (
        queries.files_holding(
            con,
            glob=str(semantic_dir(tables) / "units" / "*.parquet"),
            column="bill_id",
            ids=sorted(retired),
        )
        if retired
        else []
    )
    candidates: set[int] = set()
    for part in holding:
        path = semantic_dir(tables) / "units" / part
        candidates |= set(queries.unit_text_ids(con, units_path=path, ids=sorted(retired)))
        kept = queries.copy_without(
            con,
            path=path,
            out_path=view.fresh(f"{_UNITS}/{part}"),
            column="bill_id",
            ids=sorted(retired),
        )
        if not kept:
            view.remove(f"{_UNITS}/{part}")
    return candidates


def _cut_windows(
    con: duckdb.DuckDBPyConnection, view: Staging, new_parts: Iterable[str]
) -> list[_Cut]:
    """The new sections' windows, as the plan cuts them."""
    cut: list[_Cut] = []
    for part in new_parts:
        scan = near_queries.matchable_sections(
            con,
            sections_path=view.root / "sections" / f"{part}.parquet",
            eligible_path=view.root / "eligible.parquet",
            display=True,
        )
        for bill_id, ordinal, text, _ in scan.fetchall():
            for number, window in enumerate(
                display_windows(text, size=WINDOW_WORDS, stride=WINDOW_WORDS)
            ):
                cut.append((part, bill_id, ordinal, number, window))
    return cut


def _text_ids(
    con: duckdb.DuckDBPyConnection, tables: Path, cut: Sequence[_Cut], first: int
) -> tuple[dict[str, int], list[str]]:
    """Each window's text with the id it has, a text not stored before numbered from `first`;
    and the new texts, in that order."""
    known = (
        queries.semantic_text_ids(
            con,
            texts_glob=str(semantic_dir(tables) / "texts" / "*.parquet"),
            texts=sorted({w for *_, w in cut}),
        )
        if cut
        else {}
    )
    new_texts: list[str] = []
    ids: dict[str, int] = dict(known)
    for *_, window in cut:
        if window not in ids:
            ids[window] = first + len(new_texts)
            new_texts.append(window)
    return ids, new_texts


def _write_units(
    con: duckdb.DuckDBPyConnection, view: Staging, cut: Sequence[_Cut], ids: Mapping[str, int]
) -> None:
    """The new windows, one units file per new sections file."""
    by_part: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
    for part, bill_id, ordinal, number, window in cut:
        by_part[part].append((bill_id, ordinal, number, ids[window]))
    for part, rows in by_part.items():
        semantic_queries.write_semantic_units(
            con,
            parquet_path=view.fresh(f"{_UNITS}/{part}.parquet"),
            bill_ids=[r[0] for r in rows],
            ordinals=[r[1] for r in rows],
            windows=[r[2] for r in rows],
            text_ids=[r[3] for r in rows],
        )


def _uncarried(con: duckdb.DuckDBPyConnection, view: Staging, candidates: set[int]) -> list[int]:
    """Of the candidate texts, those no window in the view carries any more."""
    still = (
        set(
            queries.carried_text_ids(
                con, units_glob=str(view.root / _UNITS / "*.parquet"), text_ids=sorted(candidates)
            )
        )
        if candidates
        else set[int]()
    )
    return sorted(candidates - still)


@dataclass(frozen=True, slots=True)
class _Neighbors:
    ids_before: NDArray[Any]
    ids_after: NDArray[np.int32]
    scores_after: NDArray[np.float16]
    changed: NDArray[np.int64]


def _update_neighbor_lists(
    tables: Path,
    view: Staging,
    model: str,
    blocks: Sequence[NDArray[Any]],
    *,
    fresh: NDArray[np.int64],
    retired_texts: Sequence[int],
) -> _Neighbors:
    """Each stored neighbor list a new text enters, or that loses a retired one, brought to
    what a build keeps, into the view."""
    where = vectors_dir(tables, model) / "neighbors"
    ids_before = np.asarray(np.load(where / "ids.npy"))
    scores_before = np.asarray(np.load(where / "scores.npy"))
    ids_after, scores_after, changed = update_neighbors(
        blocks,
        ids_before,
        scores_before,
        fresh=fresh,
        retired=np.array(retired_texts, dtype=np.int64),
        k=int(ids_before.shape[1]) if ids_before.ndim == 2 else K,
        threshold=BUILD_THRESHOLD,
    )
    rel = f"{SEMANTIC_DIR}/{model}/neighbors"
    view.own(rel)
    np.save(view.fresh(f"{rel}/ids.npy"), ids_after)
    np.save(view.fresh(f"{rel}/scores.npy"), scores_after)
    return _Neighbors(ids_before, ids_after, scores_after, changed)


def update_semantic(
    tables: Path,
    view: Staging,
    numbers: Renumbering,
    retired: set[str],
    new_parts: Iterable[str],
    *,
    model: str,
    embedder: Callable[[], Embedder],
    floor_bytes: int,
) -> SemanticUpdate:
    """The windows, vectors, neighbors and section pairs of the semantic stage after the day."""
    plan = load_plan(tables)
    if plan is None:
        raise UpdateError(f"no semantic plan in {semantic_dir(tables)}; run `lineage embed`")
    view.own(_UNITS)
    view.own(f"{SEMANTIC_DIR}/texts")
    view.own(f"{SEMANTIC_DIR}/{model}")
    with db.connect_quiet() as con:
        candidates = _drop_units(con, tables, view, retired)
        cut = _cut_windows(con, view, new_parts)
        first = text_count(tables, model, plan)
        ids, new_texts = _text_ids(con, tables, cut, first)
        _write_units(con, view, cut, ids)
        retired_texts = _uncarried(con, view, candidates)
        windows = queries.row_count(con, glob=str(view.root / _UNITS / "*.parquet"))

    shards = load_vectors(tables, model, plan)
    dim = int(shards[0].shape[1]) if shards else 0
    if new_texts:
        vectors = embedder()(new_texts).astype(np.float16)
    else:
        vectors = np.empty((0, dim), dtype=np.float16)
    require_free_space(view.root, needed=int(vectors.nbytes) * 2 + (1 << 20), floor=floor_bytes)
    _, shard_count = extend_store(view.root, model, plan, new_texts, vectors, retired_texts)
    total = first + len(new_texts)
    after_plan = write_plan(view.root, windows=windows, distinct=total, shards=shard_count)

    found = _update_neighbor_lists(
        tables,
        view,
        model,
        load_vectors(view.root, model, after_plan),
        fresh=np.arange(first, total, dtype=np.int64),
        retired_texts=retired_texts,
    )

    before, built = load_built_semantic_pairs(tables, model)
    carried = renumbered_semantic(before, numbers.section_map)
    unit_texts = np.array(sorted({ids[w] for *_, w in cut}), dtype=np.int64)
    texts = changed_texts(found.ids_before, found.ids_after, found.changed, unit_texts)
    after = update_semantic_pairs(view.root, carried, texts, found.ids_after, found.scores_after)
    write_semantic_pairs(
        view.fresh_dir(str(pairs_dir(view.root, model).relative_to(view.root))), after, built
    )
    return SemanticUpdate(before, carried, after)
