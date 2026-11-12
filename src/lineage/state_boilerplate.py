"""`lineage state-boilerplate`: a section is its state's boilerplate by the families it reaches.

A state's standard clause, such as a request for a federal waiver, can stay under the bucket cap
and the corpus-wide stoplist, since one state's bills are a small share of the corpus, and then
matches hundreds of that state's bills from any other state's copy. Its sign is that the same
text recurs across many unrelated bills of its own state. Each section is counted by the
families of its state, other than its own, that hold a near-duplicate of it
(`<tables>/same_state_pairs/`). Bills that copy one another whole form one family, so a bill's
versions count once and several competing bills on one topic count as a few families, while a
clause that most of a state's bills carry reaches many. Being a section's, the flag drops the
clause from both stages: none of its cross-state pairs counts as a match, by wording or by
meaning.

The count is stored per section; the flag is a count of MIN_FAMILIES or more, set from the
measured distribution before any precision label was counted. It is a precision feature that
recall cannot measure, so it is measured on the precision sample with the flag on and off
(lineage.precision). See DESIGN.md#boilerplate.
"""

import json
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.disk import require_free_space
from lineage.labels import Item, SectionKey
from lineage.memory import BOUND_BYTES, peak_bytes, require_within
from lineage.pairs import load_cross_state_pairs, load_families, load_same_state_pairs
from lineage.precision import (
    SEED,
    STRATA,
    draw,
    estimate,
    sample_rows,
    stored_strata,
    stratum_of,
    without_sections,
)
from lineage.semantic import MODEL
from lineage.semantic_pairs import load_semantic_pairs, section_keys

REPORT = Path("reports/state_boilerplate.json")
COMMAND = "uv run lineage state-boilerplate"
# Set from reports/state_boilerplate.json's distribution; DESIGN.md#boilerplate.
MIN_FAMILIES = 50
SWEEP = (3, 5, 10, 20, 50, 100, 200)
# Sections by the families they reach: 0, 1, 2, 3, 4, 5-9, 10-19, ... and 1,000 or more.
BINS = (0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000)
LARGEST_LISTED = 10
# Interstate compacts are model legislation, and different compacts share a template, so their
# sections reach many of their state's families; the report shows where they fall.
COMPACT_PHRASES = ["licensure compact", "interstate compact"]
# The pairs the flag drops, sampled apart so their precision is measured directly: a third of
# the main sample's size, under its own seed. Pair ids start with "f".
FLAGGED_SAMPLE = Path("evals/precision_sample_flagged.jsonl")
FLAGGED_SAMPLE_REPORT = Path("reports/precision_sample_flagged.json")
FLAGGED_PER_STRATUM = 34
FLAGGED_SEED = SEED + 1


class StateBoilerplateError(Exception):
    pass


def families_reached(
    section_family: NDArray[np.int64], left: NDArray[np.uint32], right: NDArray[np.uint32]
) -> NDArray[np.int64]:
    """Per section, the distinct families other than its own that hold a near-duplicate of it,
    given each section's family and the same-state section pairs."""
    a = np.concatenate([np.asarray(left), np.asarray(right)]).astype(np.int64)
    b = np.concatenate([np.asarray(right), np.asarray(left)]).astype(np.int64)
    reached = section_family[b]
    other = section_family[a] != reached
    codes = np.unique((a[other] << np.int64(32)) | reached[other])
    return np.bincount(codes >> np.int64(32), minlength=section_family.size).astype(np.int64)


def section_families(bills: Sequence[str], family_of_bill: dict[str, str]) -> NDArray[np.int64]:
    """Each section's family as a number, from its bill's."""
    names = sorted(set(family_of_bill.values()))
    number = {name: i for i, name in enumerate(names)}
    return np.fromiter((number[family_of_bill[b]] for b in bills), dtype=np.int64, count=len(bills))


def _where(tables_dir: Path) -> Path:
    return tables_dir / "state_boilerplate"


def has_counts(tables_dir: Path) -> bool:
    return (_where(tables_dir) / "families_reached.npy").exists()


def load_families_reached(tables_dir: Path) -> NDArray[np.int64]:
    """The stored count per section, in the signatures' order; refused when the sections have
    changed since it was built."""
    path = _where(tables_dir) / "families_reached.npy"
    if not path.exists():
        raise StateBoilerplateError(f"no counts in {path.parent}; run `{COMMAND}`")
    counts = cast("NDArray[np.int64]", np.load(path))
    bills, _ = section_keys(tables_dir)
    if counts.size != len(bills):
        raise StateBoilerplateError(
            f"{path} counts {counts.size} sections, the signatures {len(bills)}; run `{COMMAND}`"
        )
    return counts


def flagged(counts: NDArray[np.int64], min_families: int = MIN_FAMILIES) -> NDArray[np.bool_]:
    return counts >= min_families


def _touched(
    left: NDArray[np.uint32], right: NDArray[np.uint32], flag: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    return flag[np.asarray(left)] | flag[np.asarray(right)]


def _histogram(counts: NDArray[np.int64]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for low, high in zip(BINS, [*BINS[1:], None], strict=True):
        if high is None:
            label, within = f"{low}+", counts >= low
        else:
            label = str(low) if high == low + 1 else f"{low}-{high - 1}"
            within = (counts >= low) & (counts < high)
        out.append({"families": label, "sections": int(within.sum())})
    return out


def build_state_boilerplate(
    tables_dir: Path,
    *,
    model: str = MODEL,
    floor_bytes: int,
    min_families: int = MIN_FAMILIES,
    bound_bytes: int = BOUND_BYTES,
) -> dict[str, Any]:
    """Count the families each section reaches, store the counts, and return the report: the
    distribution, what each candidate threshold would flag, and the precision sample's strata
    sizes with the flag on."""
    started = time.monotonic()
    bills, ordinals = section_keys(tables_dir)
    family = section_families(bills, load_families(tables_dir))
    same = load_same_state_pairs(tables_dir)
    counts = families_reached(family, same.left, same.right)
    del same
    require_within(bound_bytes, "counting the families each section reaches")
    where = _where(tables_dir)
    require_free_space(where.parent, needed=counts.size * 8 + (1 << 20), floor=floor_bytes)
    building = where.with_name(where.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    np.save(building / "families_reached.npy", counts)
    if where.exists():
        shutil.rmtree(where)
    building.rename(where)

    near = load_cross_state_pairs(tables_dir)
    meaning = load_semantic_pairs(tables_dir, model)
    with queries.connect_quiet() as con:
        state_of_bill = queries.bill_states(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    states = [state_of_bill[b] for b in bills]

    def at(threshold: int) -> dict[str, Any]:
        flag = flagged(counts, threshold)
        return {
            "min_families": threshold,
            "sections": int(flag.sum()),
            "cross_state_pairs_dropped": {
                "near_duplicate": int(_touched(near.left, near.right, flag).sum()),
                "semantic": int(_touched(meaning.left, meaning.right, flag).sum()),
            },
        }

    with queries.connect_quiet() as con:
        named = queries.sections_naming(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            phrases=COMPACT_PHRASES,
        )
    number = {(b, int(o)): s for s, (b, o) in enumerate(zip(bills, ordinals.tolist(), strict=True))}
    compact = counts[[number[k] for k in named if k in number]]
    flag = flagged(counts, min_families)
    sizes = {
        "flag_off": {n: int(k.size) for n, k in stored_strata(tables_dir, model=model).items()},
        "flag_on": {
            n: int(k.size) for n, k in stored_strata(tables_dir, model=model, dropped=flag).items()
        },
    }
    at_flag: list[int] = np.flatnonzero(flag).tolist()
    by_state = Counter(states[i] for i in at_flag).most_common()
    return {
        "command": COMMAND,
        "min_families": min_families,
        "sections": int(counts.size),
        "sections_reaching_another_family": int((counts > 0).sum()),
        "histogram": _histogram(counts),
        "quantiles_of_sections_reaching_another_family": {
            q: int(np.quantile(counts[counts > 0], float(q))) if (counts > 0).any() else 0
            for q in ("0.5", "0.9", "0.99", "0.999")
        },
        "sweep": [at(t) for t in sorted({*SWEEP, min_families})],
        "compact_sections": {
            "phrases": COMPACT_PHRASES,
            "sections": int(compact.size),
            "quantiles": {
                q: int(np.quantile(compact, float(q))) if compact.size else 0
                for q in ("0.5", "0.9", "0.99")
            },
            "flagged": {str(t): int((compact >= t).sum()) for t in sorted({*SWEEP, min_families})},
        },
        # The precision sample's strata in the build, as the estimate weights them.
        "precision_strata": sizes,
        "flagged_by_state": [{"state": s, "sections": n} for s, n in by_state[:LARGEST_LISTED]],
        "states_with_a_flagged_section": len(by_state),
        "bytes": int(counts.nbytes),
        "seconds": round(time.monotonic() - started),
        "peak_memory_bytes": peak_bytes(),
        "memory_bound_bytes": bound_bytes,
    }


def _dropped(tables_dir: Path, *, model: str, min_families: int) -> dict[str, NDArray[np.uint64]]:
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
    population = _dropped(tables_dir, model=model, min_families=min_families)
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


def write_report(doc: dict[str, Any], path: Path = REPORT) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")
