"""The state boilerplate flag's precision: the precision sample with the flag on, and the pairs
the flag drops, sampled apart under their own seed and read on their own
(lineage.match.state_boilerplate computes the flag; DESIGN.md#boilerplate).
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage.labels import Item, SectionKey
from lineage.match.semantic.pairs import section_keys
from lineage.match.semantic.stage import MODEL
from lineage.match.state_boilerplate import (
    COMMAND,
    MIN_FAMILIES,
    REPORT,
    StateBoilerplateError,
    flagged,
    has_counts,
    load_families_reached,
)
from lineage.match.strata import STRATA, stored_strata, stratum_of
from lineage.precision import SEED, draw, estimate, sample_rows, without_sections

# The pairs the flag drops, sampled apart so their precision is measured directly: a third of
# the main sample's size, under its own seed. Pair ids start with "f".
FLAGGED_SAMPLE = Path("evals/precision_sample_flagged.jsonl")
FLAGGED_SAMPLE_REPORT = Path("reports/precision_sample_flagged.json")
FLAGGED_PER_STRATUM = 34
FLAGGED_SEED = SEED + 1


def dropped_strata(
    tables_dir: Path, *, model: str = MODEL, min_families: int = MIN_FAMILIES
) -> dict[str, NDArray[np.uint64]]:
    """The pairs the flag drops in the build, by stratum, as the dropped pairs were drawn."""
    flag = flagged(load_families_reached(tables_dir), min_families)
    every = stored_strata(tables_dir, model=model)
    kept = stored_strata(tables_dir, model=model, dropped=flag)
    return {n: np.setdiff1d(every[n], kept[n], assume_unique=True) for n in STRATA}


def build_flagged_sample(
    tables_dir: Path,
    *,
    model: str = MODEL,
    per_stratum: int = FLAGGED_PER_STRATUM,
    seed: int = FLAGGED_SEED,
    min_families: int = MIN_FAMILIES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A seeded draw, per stratum, from the cross-state pairs the flag drops, and its report."""
    population = dropped_strata(tables_dir, model=model, min_families=min_families)
    drawn = draw(population, per_stratum=per_stratum, seed=seed)
    bills, ordinals = section_keys(tables_dir)
    rows = sample_rows(drawn, lambda s: SectionKey(bills[s], int(ordinals[s])), prefix="f")
    report = {
        "command": "uv run lineage precision-sample --flagged",
        "seed": seed,
        "per_stratum": per_stratum,
        "model": model,
        "min_families": min_families,
        "strata": {
            name: {
                "stages": list(STRATA[name]),
                "size": int(population[name].size),
                "drawn": sum(1 for n, _ in drawn if n == name),
            }
            for name in STRATA
        },
    }
    return rows, report


def with_flag(
    tables_dir: Path,
    items: Sequence[Item],
    labels: Sequence[Mapping[str, Any]],
    sizes: Mapping[str, int],
    *,
    report_path: Path = REPORT,
    min_families: int = MIN_FAMILIES,
    dropped_items: Sequence[Item] | None = None,
    dropped_report: Path = FLAGGED_SAMPLE_REPORT,
    model_labeler: str | None = None,
    dropped_labeler: str | None = None,
) -> dict[str, Any] | None:
    """Precision with the flag on: the sample's pairs with neither side flagged, weighted by
    the strata's sizes without the flagged sections' pairs. With `dropped_items`, the sample of
    the pairs the flag drops (build_flagged_sample), also their own precision, weighted by the
    dropped pairs' strata, by `dropped_labeler` when they were asked another question than the
    sample (by default `model_labeler`). None before the counts are built; refused when they
    were built from another build than the sample's, or at another threshold."""
    if not has_counts(tables_dir):
        return None
    if not report_path.exists():
        raise StateBoilerplateError(f"{report_path} is missing; run `{COMMAND}`")
    doc = json.loads(report_path.read_text())
    if doc["min_families"] != min_families:
        raise StateBoilerplateError(
            f"{report_path} flags at {doc['min_families']} families, not {min_families}; "
            f"run `{COMMAND}`"
        )
    if doc["precision_strata"]["flag_off"] != dict(sizes):
        raise StateBoilerplateError(
            f"{report_path} counts other strata than the sample was drawn from; run `{COMMAND}`"
        )
    flag = flagged(load_families_reached(tables_dir), min_families)
    bills, ordinals = section_keys(tables_dir)
    dropped = {SectionKey(bills[s], int(ordinals[s])) for s in np.flatnonzero(flag).tolist()}
    kept = without_sections(items, dropped)
    kept_pairs = {i.pair.pair for i in kept}
    dropped_estimate: dict[str, Any] | None = None
    if dropped_items is not None:
        if not dropped_report.exists():
            raise StateBoilerplateError(
                f"{dropped_report} is missing; run `uv run lineage precision-sample --flagged`"
            )
        drawn = json.loads(dropped_report.read_text())
        sizes_dropped = {
            n: doc["precision_strata"]["flag_off"][n] - doc["precision_strata"]["flag_on"][n]
            for n in STRATA
        }
        recorded = {n: int(drawn["strata"][n]["size"]) for n in STRATA}
        if drawn["min_families"] != min_families or recorded != sizes_dropped:
            raise StateBoilerplateError(
                f"{dropped_report} was drawn from other dropped strata than {report_path} "
                "counts; run `uv run lineage precision-sample --flagged`"
            )
        dropped_estimate = estimate(
            dropped_items, labels, sizes_dropped, model_labeler=dropped_labeler or model_labeler
        )
    return {
        "min_families": min_families,
        "dropped_pairs": dropped_estimate,
        "sample_pairs_dropped": {
            name: sum(
                1
                for i in items
                if stratum_of(i.pair.stages) == name and i.pair.pair not in kept_pairs
            )
            for name in STRATA
        },
        "flag_on": estimate(
            kept, labels, doc["precision_strata"]["flag_on"], model_labeler=model_labeler
        ),
    }
