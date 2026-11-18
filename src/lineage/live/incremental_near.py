"""The update's near-duplicate stage: signatures, the band index, both pairs tables, the
families and the pinned state boilerplate counts after the day (lineage.live.incremental).

See DESIGN.md#updates, DR-7.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from lineage.core import db
from lineage.live import queries
from lineage.live.staging import Staging
from lineage.match.near import queries as near_queries
from lineage.match.near.bands import load_band_index, update_bands
from lineage.match.near.families import update_families
from lineage.match.near.minhash import NUM_PERM
from lineage.match.near.pairs import (
    SectionPairs,
    load_cross_state_pairs,
    load_families,
    load_same_state_pairs,
    pair_codes,
    recompute_section_pairs,
    renumbered,
    replaced_pairs,
    section_bounds,
    sections_of_window_pairs,
    write_section_pairs,
)
from lineage.match.near.signatures import open_signatures, row_labels, sign_part
from lineage.match.near.stoplist import load_stoplist
from lineage.match.state_boilerplate import COUNT_FILES


@dataclass(frozen=True, slots=True)
class Renumbering:
    row_map: NDArray[np.int64]  # each old signature row's new number, -1 where retired
    section_map: NDArray[np.int64]  # each old section's new number, -1 where retired
    fresh_rows: NDArray[np.int64]  # the new rows' numbers
    fresh_values: NDArray[np.uint32]


def update_signatures(
    tables: Path, view: Staging, retired: set[str], new_parts: Iterable[str]
) -> Renumbering:
    """Drop the retired records' signature rows, sign the new sections files, and number the
    rows and sections afresh in file order, as a build numbers them."""
    dest = view.own("signatures")
    old_files = sorted((tables / "signatures").glob("*.npy"))
    kept = {p.stem: np.ones(int(np.load(p, mmap_mode="r").shape[0]), dtype=bool) for p in old_files}
    with db.connect_quiet() as con:
        # Only the files that hold a retired record's rows are read and written again.
        holding = (
            queries.files_holding(
                con,
                glob=str(tables / "signatures" / "*.parquet"),
                column="bill_id",
                ids=sorted(retired),
            )
            if retired
            else []
        )
        for name in holding:
            path = (tables / "signatures" / name).with_suffix(".npy")
            ids, ordinals, windows = near_queries.read_signature_keys(
                con, parquet_path=path.with_suffix(".parquet")
            )
            keep = np.fromiter((b not in retired for b in ids), dtype=bool, count=len(ids))
            kept[path.stem] = keep
            view.remove(f"signatures/{path.name}")
            view.remove(f"signatures/{path.stem}.parquet")
            if keep.any():
                np.save(dest / path.name, np.asarray(np.load(path, mmap_mode="r"))[keep])
                at = np.flatnonzero(keep).tolist()
                near_queries.write_signature_keys(
                    con,
                    parquet_path=dest / f"{path.stem}.parquet",
                    bill_ids=[ids[i] for i in at],
                    ordinals=[ordinals[i] for i in at],
                    windows=[windows[i] for i in at],
                )
    stoplist = load_stoplist(tables)
    added: set[str] = set()
    for part in new_parts:
        sign_part(
            view.root / "sections" / f"{part}.parquet",
            view.root / "eligible.parquet",
            stoplist,
            dest,
        )
        if (dest / f"{part}.npy").exists():
            added.add(part)

    new_files = sorted(dest.glob("*.npy"))
    old_rows = [kept[p.stem].size for p in old_files]
    row_map = np.full(int(sum(old_rows)), -1, dtype=np.int64)
    starts = np.cumsum([0, *old_rows]).tolist()
    old_start = {p.stem: int(starts[i]) for i, p in enumerate(old_files)}
    fresh_rows: list[NDArray[np.int64]] = []
    fresh_values: list[NDArray[np.uint32]] = []
    at = 0
    for path in new_files:
        rows = int(np.load(path, mmap_mode="r").shape[0])
        if path.stem in added:
            fresh_rows.append(np.arange(at, at + rows, dtype=np.int64))
            fresh_values.append(cast("NDArray[np.uint32]", np.load(path)))
        else:
            keep = kept[path.stem]
            start = old_start[path.stem]
            row_map[start + np.flatnonzero(keep)] = at + np.arange(rows, dtype=np.int64)
        at += rows
    before = row_labels(open_signatures(tables))
    after = row_labels(open_signatures(view.root))
    first_rows = (
        np.flatnonzero(np.r_[True, np.diff(before.section) != 0])
        if before.section.size
        else np.empty(0, dtype=np.int64)
    )
    mapped = row_map[first_rows]
    section_map = np.where(mapped >= 0, after.section[np.maximum(mapped, 0)], -1).astype(np.int64)
    values = (
        np.concatenate(fresh_values) if fresh_values else np.empty((0, NUM_PERM), dtype=np.uint32)
    )
    rows_new = np.concatenate(fresh_rows) if fresh_rows else np.empty(0, dtype=np.int64)
    return Renumbering(row_map, section_map, rows_new, values)


@dataclass(frozen=True, slots=True)
class NearDuplicates:
    cross_before: SectionPairs
    cross_after: SectionPairs
    families_before: dict[str, str]
    families_after: dict[str, str]


def update_near_duplicates(
    tables: Path, view: Staging, numbers: Renumbering, resplit: set[str]
) -> NearDuplicates:
    """The band index, both pairs tables and the families after the day, into the view."""
    bands = view.fresh_dir("bands")
    codes = update_bands(
        load_band_index(tables), numbers.row_map, numbers.fresh_values, numbers.fresh_rows, bands
    )
    after = row_labels(open_signatures(view.root))
    before = row_labels(open_signatures(tables))
    left, right = sections_of_window_pairs(codes, after.section)
    found = recompute_section_pairs(
        open_signatures(view.root),
        load_band_index(view.root),
        section_bounds(after.section),
        left,
        right,
    )
    with db.connect_quiet() as con:
        state_of = near_queries.bill_states(con, bills_glob=str(view.root / "bills" / "*.parquet"))
    states = np.array([state_of[b] for b in after.section_bills], dtype=object)
    cross = states[found.left] != states[found.right] if found.left.size else np.empty(0, bool)
    recomputed = pair_codes(left, right)
    pairs: dict[str, tuple[SectionPairs, SectionPairs]] = {}
    for name, old, mask in (
        ("pairs", load_cross_state_pairs(tables), cross),
        ("same_state_pairs", load_same_state_pairs(tables), ~cross),
    ):
        part = SectionPairs(found.left[mask], found.right[mask], found.agreements[mask])
        new = replaced_pairs(renumbered(old, numbers.section_map), recomputed, part)
        write_section_pairs(view.fresh_dir(name), new)
        pairs[name] = (old, new)

    old_same, new_same = pairs["same_state_pairs"]
    carried = renumbered(old_same, numbers.section_map)
    changed = np.setxor1d(
        pair_codes(carried.left, carried.right), pair_codes(new_same.left, new_same.right)
    )
    affected = set(resplit)
    for code in changed.tolist():
        affected |= {after.section_bills[code >> 32], after.section_bills[code & 0xFFFF_FFFF]}
    gone = numbers.section_map < 0
    lost = gone[np.asarray(old_same.left)] | gone[np.asarray(old_same.right)]
    for s in np.concatenate(
        [np.asarray(old_same.left)[lost], np.asarray(old_same.right)[lost]]
    ).tolist():
        affected.add(before.section_bills[int(s)])
    bills = np.unique(np.array(after.section_bills, dtype=str))
    bill_of_section = np.searchsorted(bills, np.array(after.section_bills, dtype=str)).astype(
        np.int32
    )
    families_before = load_families(tables)
    families_after = update_families(
        families_before, bills, bill_of_section, new_same.left, new_same.right, affected
    )
    view.fresh_dir("families")
    with db.connect_quiet() as con:
        near_queries.write_families(
            con,
            parquet_path=view.root / "families" / "families.parquet",
            bill_ids=list(families_after),
            families=list(families_after.values()),
        )
    _carry_state_boilerplate(tables, view, numbers.section_map, len(after.section_bills))
    cross_old, cross_new = pairs["pairs"]
    return NearDuplicates(cross_old, cross_new, families_before, families_after)


def _carry_state_boilerplate(
    tables: Path, view: Staging, section_map: NDArray[np.int64], sections: int
) -> None:
    """The base build's state boilerplate counts and passage reach under the new section numbers
    (DR-7): pinned, so a new section counts no families until the next full build counts it."""
    for name in COUNT_FILES:
        path = tables / "state_boilerplate" / name
        if not path.exists():
            continue
        old = np.load(path)
        counts = np.zeros(sections, dtype=old.dtype)
        kept = section_map >= 0
        counts[section_map[kept]] = old[kept]
        np.save(view.fresh(f"state_boilerplate/{name}"), counts)
