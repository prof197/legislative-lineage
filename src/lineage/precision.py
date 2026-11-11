"""The precision sample (step 25): which of the build's matches are right, measured per stage.

Recall has an answer key; precision needs labels, so it is measured on a sample. The build's
cross-state section pairs fall into three strata by the stages that found them: the
near-duplicate stage alone, the semantic stage alone, or both. Each stratum is sampled apart,
PER_STRATUM pairs drawn without replacement under a fixed seed, so a stage that finds few pairs
is still measured, and the draw is the same on every run over the same build. The drawn pairs
are shuffled together into one labeling order, so a session stopped part way has labeled every
stratum about evenly.

A pair is correct when labeled CORRECT_FROM or higher (lineage.labels). Each stratum's
precision gets the Wilson interval (lineage.intervals). A stage's precision, and the overall
one, weight the strata it covers by their sizes in the build: a sample with as many pairs from a
small stratum as from a large one would otherwise over-count it. The weighted interval is
Agresti and Coull's: each stratum's share with two successes and two failures added, at 95%,
the variance summed by weight, so a stratum labeled all correct still leaves an interval of
some width. People's labels and a model's provisional ones are estimated apart and never
pooled; where both labeled a pair, how often they agree is reported. See
DESIGN.md#precision-sample.
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage.intervals import Z_95, wilson
from lineage.labels import CORRECT_FROM, MODEL, PERSON, Item, SectionKey
from lineage.pairs import load_cross_state_pairs
from lineage.semantic import MODEL as MODEL_NAME
from lineage.semantic_pairs import load_semantic_pairs, section_keys

# Each stratum's name and the stages that found its pairs, in report order.
STRATA: dict[str, tuple[str, ...]] = {
    "near_duplicate_only": ("near_duplicate",),
    "semantic_only": ("semantic",),
    "both": ("near_duplicate", "semantic"),
}
STAGES = ("near_duplicate", "semantic")
# Provisional: at a precision near 0.9, 100 labels give a Wilson interval about 0.12 wide per
# stratum. The final size is decided after release (DESIGN.md#precision-sample).
PER_STRATUM = 100
SEED = 20261112

SAMPLE = Path("evals/precision_sample.jsonl")
SAMPLE_REPORT = Path("reports/precision_sample.json")  # the draw: seed, and each stratum's size

PairKeys = NDArray[np.uint64]  # left << 32 | right, sorted and unique


class PrecisionError(Exception):
    pass


def pair_keys(left: NDArray[np.uint32], right: NDArray[np.uint32]) -> PairKeys:
    """One sortable key per section pair, from section numbers in the signatures' order."""
    return np.unique((left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64))


def strata(near_duplicate: PairKeys, semantic: PairKeys) -> dict[str, PairKeys]:
    """Every pair either stage found, in exactly one stratum."""
    return {
        "near_duplicate_only": np.setdiff1d(near_duplicate, semantic, assume_unique=True),
        "semantic_only": np.setdiff1d(semantic, near_duplicate, assume_unique=True),
        "both": np.intersect1d(near_duplicate, semantic, assume_unique=True),
    }


def stratum_of(stages: Sequence[str]) -> str:
    found = tuple(s for s in STAGES if s in stages)
    for name, covers in STRATA.items():
        if covers == found and len(found) == len(set(stages)):
            return name
    raise PrecisionError(f"no stratum is found by the stages {list(stages)}")


def draw(
    population: Mapping[str, PairKeys], *, per_stratum: int = PER_STRATUM, seed: int = SEED
) -> list[tuple[str, int]]:
    """(stratum, pair key) for up to `per_stratum` pairs of each stratum, in labeling order."""
    drawn: list[tuple[str, int]] = []
    for i, name in enumerate(STRATA):
        keys = population.get(name, np.empty(0, dtype=np.uint64))
        rng = np.random.default_rng([seed, i])
        chosen = rng.choice(len(keys), size=min(per_stratum, len(keys)), replace=False)
        drawn += [(name, int(k)) for k in keys[np.sort(chosen)].tolist()]
    order = np.random.default_rng([seed, len(STRATA)]).permutation(len(drawn))
    return [drawn[i] for i in order.tolist()]


def sample_rows(
    drawn: Sequence[tuple[str, int]], section: Callable[[int], SectionKey]
) -> list[dict[str, Any]]:
    """The sample file's rows (lineage.labels.load_sample), each section named by bill id and
    ordinal through `section`, which maps a section number to where it sits in the build."""
    rows: list[dict[str, Any]] = []
    for n, (name, key) in enumerate(drawn, start=1):
        query, match = section(key >> 32), section(key & 0xFFFFFFFF)
        rows.append(
            {
                "pair": f"p{n:04d}",  # numbered in labeling order, so it tells nothing of the stage
                "stages": list(STRATA[name]),
                "query": {"bill_id": query.bill_id, "ordinal": query.ordinal},
                "match": {"bill_id": match.bill_id, "ordinal": match.ordinal},
            }
        )
    return rows


def stratified(parts: Sequence[tuple[int, int, int]]) -> dict[str, Any] | None:
    """Precision over strata given as (size in the build, labeled, correct), weighted by size;
    None when a stratum with pairs has no labels yet."""
    weighted = [(size, n, x) for size, n, x in parts if size > 0]
    if not weighted or any(n == 0 for _, n, _ in weighted):
        return None
    if len(weighted) == 1:
        [(_, n, x)] = weighted
        return {"precision": round(x / n, 4), "interval_95": [round(v, 4) for v in wilson(x, n)]}
    total = sum(size for size, _, _ in weighted)
    z2 = Z_95 * Z_95
    point = center = variance = 0.0
    for size, n, x in weighted:
        w = size / total
        adjusted = (x + z2 / 2) / (n + z2)
        point += w * x / n
        center += w * adjusted
        variance += w * w * adjusted * (1 - adjusted) / (n + z2)
    margin = Z_95 * math.sqrt(variance)
    interval = [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]
    return {"precision": round(point, 4), "interval_95": interval}


def _latest(
    items: Sequence[Item], labels: Sequence[Mapping[str, Any]], prefix: str
) -> dict[str, int]:
    """Each pair's latest label among labelers named with `prefix`, by pair id."""
    by_sides = {(i.query.ref.key, i.match.ref.key): i.pair.pair for i in items}
    latest: dict[str, tuple[str, int]] = {}
    for row in labels:
        if not str(row["labeler"]).startswith(prefix):
            continue
        sides = (
            (row["query"]["bill_id"], row["query"]["text_sha256"]),
            (row["match"]["bill_id"], row["match"]["text_sha256"]),
        )
        pair = by_sides.get(sides)
        if pair is not None and (pair not in latest or str(row["at"]) >= latest[pair][0]):
            latest[pair] = (str(row["at"]), int(row["label"]))
    return {pair: label for pair, (_, label) in latest.items()}


def _estimate(
    items: Sequence[Item], labeled: Mapping[str, int], sizes: Mapping[str, int]
) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, Any]] = {}
    for name in STRATA:
        mine = [i.pair.pair for i in items if stratum_of(i.pair.stages) == name]
        got = [labeled[p] for p in mine if p in labeled]
        correct = sum(label >= CORRECT_FROM for label in got)
        by_stratum[name] = {
            "size": int(sizes.get(name, 0)),
            "sampled": len(mine),
            "labeled": len(got),
            "correct": correct,
            "precision": round(correct / len(got), 4) if got else None,
            "interval_95": [round(v, 4) for v in wilson(correct, len(got))],
        }

    def over(names: Sequence[str]) -> dict[str, Any] | None:
        s = [by_stratum[n] for n in names]
        return stratified([(x["size"], x["labeled"], x["correct"]) for x in s])

    stages = {
        stage: over([n for n, covers in STRATA.items() if stage in covers]) for stage in STAGES
    }
    return {"strata": by_stratum, "stages": stages, "overall": over(list(STRATA))}


def estimate(
    items: Sequence[Item], labels: Sequence[Mapping[str, Any]], sizes: Mapping[str, int]
) -> dict[str, Any]:
    """Precision from people's labels and, apart, from the model's provisional ones, with how
    often the two agree. `labels` are the current ones (lineage.labels.current_labels);
    `sizes` is each stratum's pair count in the build the sample was drawn from."""
    for row in labels:
        if not str(row["labeler"]).startswith((PERSON, MODEL)):
            raise PrecisionError(f"labeler {row['labeler']!r} is neither a person nor a model")
    for item in items:
        stratum_of(item.pair.stages)
    people, model = _latest(items, labels, PERSON), _latest(items, labels, MODEL)
    both = [p for p in people if p in model]
    return {
        "correct_from": CORRECT_FROM,
        "person": _estimate(items, people, sizes),
        "provisional_model": _estimate(items, model, sizes),
        "agreement": {
            "pairs": len(both),
            "same_label": sum(people[p] == model[p] for p in both),
            "same_correctness": sum(
                (people[p] >= CORRECT_FROM) == (model[p] >= CORRECT_FROM) for p in both
            ),
        },
    }


def load_sizes(path: Path = SAMPLE_REPORT) -> dict[str, int]:
    """Each stratum's pair count in the build, as the draw recorded it."""
    if not path.exists():
        raise PrecisionError(f"{path} is missing; it is written when the sample is drawn")
    doc = json.loads(path.read_text())
    try:
        return {name: int(doc["strata"][name]["size"]) for name in STRATA}
    except (KeyError, TypeError, ValueError) as e:
        raise PrecisionError(f"{path}: no size for every stratum ({e})") from e


def build_sample(
    tables_dir: Path,
    *,
    model: str = MODEL_NAME,
    per_stratum: int = PER_STRATUM,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sample's rows, drawn from both stages' stored cross-state pairs, and the draw's
    report: the seed and each stratum's size in the build, which the estimate weights by."""
    near = load_cross_state_pairs(tables_dir)
    meaning = load_semantic_pairs(tables_dir, model)
    population = strata(
        pair_keys(np.asarray(near.left), np.asarray(near.right)),
        pair_keys(np.asarray(meaning.left), np.asarray(meaning.right)),
    )
    drawn = draw(population, per_stratum=per_stratum, seed=seed)
    bills, ordinals = section_keys(tables_dir)
    rows = sample_rows(drawn, lambda s: SectionKey(bills[s], int(ordinals[s])))
    report = {
        "command": "uv run lineage precision-sample",
        "seed": seed,
        "per_stratum": per_stratum,
        "model": model,
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


def write_sample(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    sample_path: Path = SAMPLE,
    report_path: Path = SAMPLE_REPORT,
) -> None:
    sample_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report_path.write_text(json.dumps(report, indent=1) + "\n")
