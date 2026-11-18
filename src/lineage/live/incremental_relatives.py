"""The bills whose relatives a day can have changed, and their relatives and origins before
and after it, as `lineage.live.updates.snapshot` reads them (lineage.live.incremental).

See DESIGN.md#updates, DR-7.
"""

import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage.core import db
from lineage.live import queries
from lineage.live.incremental_near import NearDuplicates
from lineage.live.incremental_semantic import SemanticUpdate
from lineage.live.updates import Origin, Relative, origins_of, relatives_of
from lineage.match.listing import shipped_flag
from lineage.match.near.pairs import renumbered
from lineage.match.near.signatures import open_signatures, row_labels
from lineage.match.relatives import bill_relatives
from lineage.origins.matches import read_origin_matches


def _bills_of(sections: Iterable[int], section_bills: Sequence[str]) -> set[str]:
    return {section_bills[s] for s in sections}


def _changed_sections(
    old_left: NDArray[Any],
    old_right: NDArray[Any],
    old_value: NDArray[Any],
    new_left: NDArray[Any],
    new_right: NDArray[Any],
    new_value: NDArray[Any],
) -> set[int]:
    """Sections (new numbers) of every pair added, dropped or changed in value."""
    old = set(
        zip(
            np.asarray(old_left).tolist(),
            np.asarray(old_right).tolist(),
            np.asarray(old_value).tolist(),
            strict=True,
        )
    )
    new = set(
        zip(
            np.asarray(new_left).tolist(),
            np.asarray(new_right).tolist(),
            np.asarray(new_value).tolist(),
            strict=True,
        )
    )
    return {s for a, b, _ in old ^ new for s in (a, b)}


def listed_relatives(tables: Path, bills: set[str], model: str) -> dict[str, tuple[Relative, ...]]:
    """The listed relatives of the wanted matchable bills, as `lineage.live.updates.snapshot` reads
    them."""
    with db.connect_quiet() as con:
        matchable = set(
            queries.matchable_bill_ids(
                con,
                sections_glob=str(tables / "sections" / "*.parquet"),
                eligible_path=tables / "eligible.parquet",
            )
        )
    wanted = sorted(bills & matchable)
    if not wanted:
        return {}
    found = bill_relatives(
        tables, wanted, model=model, shown=sys.maxsize, flag=shipped_flag(tables)
    )
    return relatives_of(found)


def bill_origins(tables: Path) -> dict[str, tuple[Origin, ...]]:
    """Each bill's origin matches, as `lineage.live.updates.snapshot` reads them."""
    return origins_of(read_origin_matches(tables))


def _members(families: Mapping[str, str]) -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for bill, family in families.items():
        groups[family].add(bill)
    return {bill: frozenset(groups[family]) for bill, family in families.items()}


def related_bills(
    tables: Path,
    view: Path,
    day_ids: set[str],
    near: NearDuplicates,
    semantic: SemanticUpdate,
    section_map: NDArray[np.int64],
) -> set[str]:
    """Every bill whose relatives the day can have changed: its own records; the bills of each
    section pair, of either stage, the day added, dropped or changed; and the bills paired with
    a bill whose family changed, since relatives are grouped by family."""
    before = row_labels(open_signatures(tables)).section_bills
    after = row_labels(open_signatures(view)).section_bills
    out = set(day_ids)
    gone = section_map < 0
    old_pairs: list[tuple[NDArray[Any], NDArray[Any]]] = [
        (np.asarray(near.cross_before.left), np.asarray(near.cross_before.right)),
        (np.asarray(semantic.before.left), np.asarray(semantic.before.right)),
    ]
    for left, right in old_pairs:
        lost = gone[left] | gone[right]
        out |= _bills_of(np.concatenate([left[lost], right[lost]]).tolist(), before)
    carried_near = renumbered(near.cross_before, section_map)
    out |= _bills_of(
        _changed_sections(
            carried_near.left,
            carried_near.right,
            carried_near.agreements,
            near.cross_after.left,
            near.cross_after.right,
            near.cross_after.agreements,
        ),
        after,
    )
    out |= _bills_of(
        _changed_sections(
            semantic.carried.left,
            semantic.carried.right,
            semantic.carried.cosine,
            semantic.after.left,
            semantic.after.right,
            semantic.after.cosine,
        ),
        after,
    )
    was, now = _members(near.families_before), _members(near.families_after)
    regrouped = {b for b in set(was) | set(now) if was.get(b) != now.get(b)}
    out |= regrouped
    for pairs, bills in (
        ((near.cross_before.left, near.cross_before.right), before),
        ((semantic.before.left, semantic.before.right), before),
        ((near.cross_after.left, near.cross_after.right), after),
        ((semantic.after.left, semantic.after.right), after),
    ):
        left, right = np.asarray(pairs[0]), np.asarray(pairs[1])
        mine = np.fromiter((b in regrouped for b in bills), dtype=bool, count=len(bills))
        if not mine.any() or not left.size:
            continue
        touch = mine[left] | mine[right]
        out |= _bills_of(np.concatenate([left[touch], right[touch]]).tolist(), bills)
    return out
