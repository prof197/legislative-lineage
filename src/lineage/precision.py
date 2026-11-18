"""The precision sample (step 25): which of the build's matches are right, measured per stage.

Recall has an answer key; precision needs labels, so it is measured on a sample. The build's
cross-state section pairs fall into three strata by the stages that found them: the
near-duplicate stage alone, the semantic stage alone, or both. Each stratum is sampled apart,
PER_STRATUM pairs drawn without replacement under a fixed seed, so a stage that finds few pairs
is still measured, and the draw is the same on every run over the same build. The drawn pairs
are shuffled together into one labeling order, so a session stopped part way has labeled every
stratum about evenly.

A pair is correct when labeled CORRECT_FROM or higher (lineage.labels). Each stratum's
precision gets the Wilson interval (lineage.core.intervals). A stage's precision, and the overall
one, weight the strata it covers by their sizes in the build: a sample with as many pairs from a
small stratum as from a large one would otherwise over-count it. The weighted interval is
Agresti and Coull's: each stratum's share with two successes and two failures added, at 95%,
the variance summed by weight, so a stratum labeled all correct still leaves an interval of
some width. People's labels and a model's provisional ones are estimated apart and never
pooled; where both labeled a pair, how often they agree is reported. See
DESIGN.md#precision-sample.
"""

import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lineage.core.intervals import wilson
from lineage.labels import CORRECT_FROM, MODEL, PERSON, Item, SectionKey, is_correct
from lineage.semantic import MODEL as MODEL_NAME
from lineage.semantic_pairs import section_keys
from lineage.strata import (
    STAGES,
    STRATA,
    PairKeys,
    PrecisionError,
    stored_strata,
    stratum_of,
)

# Provisional: at a precision near 0.9, 100 labels give a Wilson interval about 0.12 wide per
# stratum. The final size is decided after release (DESIGN.md#precision-sample).
PER_STRATUM = 100
SEED = 20261112

SAMPLE = Path("evals/precision_sample.jsonl")
SAMPLE_REPORT = Path("reports/precision_sample.json")  # the draw: seed, and each stratum's size


def draw(
    population: Mapping[str, PairKeys],
    *,
    per_stratum: int = PER_STRATUM,
    seed: int = SEED,
    names: Sequence[str] = tuple(STRATA),
) -> list[tuple[str, int]]:
    """(stratum, pair key) for up to `per_stratum` pairs of each stratum, in labeling order.
    `names` are the strata in order, by stage unless a sample stratifies otherwise; each takes
    the seed with its place in that order."""
    drawn: list[tuple[str, int]] = []
    for i, name in enumerate(names):
        keys = population.get(name, np.empty(0, dtype=np.uint64))
        rng = np.random.default_rng([seed, i])
        chosen = rng.choice(len(keys), size=min(per_stratum, len(keys)), replace=False)
        drawn += [(name, int(k)) for k in keys[np.sort(chosen)].tolist()]
    order = np.random.default_rng([seed, len(names)]).permutation(len(drawn))
    return [drawn[i] for i in order.tolist()]


def sample_rows(
    drawn: Sequence[tuple[str, int]], section: Callable[[int], SectionKey], *, prefix: str = "p"
) -> list[dict[str, Any]]:
    """The sample file's rows (lineage.labels.load_sample), each section named by bill id and
    ordinal through `section`, which maps a section number to where it sits in the build."""
    rows: list[dict[str, Any]] = []
    for n, (name, key) in enumerate(drawn, start=1):
        query, match = section(key >> 32), section(key & 0xFFFFFFFF)
        rows.append(
            {
                # Numbered in labeling order, so the id tells nothing of the stage.
                "pair": f"{prefix}{n:04d}",
                "stages": list(STRATA[name]),
                "query": {"bill_id": query.bill_id, "ordinal": query.ordinal},
                "match": {"bill_id": match.bill_id, "ordinal": match.ordinal},
            }
        )
    return rows


def stratified(parts: Sequence[tuple[int, int, int]]) -> dict[str, Any] | None:
    """Precision over strata given as (size in the build, labeled, correct), weighted by size;
    None when a stratum with pairs has no labels yet.

    The interval is Wilson's at the effective sample size: the n whose binomial variance at
    the point equals the stratified estimate's, p(1 - p) / sum(w^2 p_h (1 - p_h) / n_h), capped
    at the labels taken, and the labels taken when that variance is zero. A Wilson interval
    holds its point, which the interval it replaced, centred on each stratum's Wilson-adjusted
    share, did not where small strata were all correct (fixed at the step 47 review)."""
    weighted = [(size, n, x) for size, n, x in parts if size > 0]
    if not weighted or any(n == 0 for _, n, _ in weighted):
        return None
    if len(weighted) == 1:
        [(_, n, x)] = weighted
        return {"precision": round(x / n, 4), "interval_95": [round(v, 4) for v in wilson(x, n)]}
    total = sum(size for size, _, _ in weighted)
    labeled = sum(n for _, n, _ in weighted)
    point = variance = 0.0
    for size, n, x in weighted:
        w, p = size / total, x / n
        point += w * p
        variance += w * w * p * (1 - p) / n
    effective = labeled if variance == 0 else min(labeled, point * (1 - point) / variance)
    interval = [round(v, 4) for v in wilson(point * effective, effective)]
    return {"precision": round(point, 4), "interval_95": interval}


def restratified(
    items: Sequence[Item],
    labeled: Mapping[str, bool],
    keys: Mapping[str, int],
    drawn: Mapping[str, PairKeys],
    now: Mapping[str, PairKeys],
) -> dict[str, Any]:
    """Precision over another population of pairs, `now` by stratum, from a sample drawn by
    stratum from `drawn`, each item's pair named by its key in `keys`. A sampled pair still in
    `now` counts in the cell of its stratum as drawn and its stratum now, and each cell is
    weighted by its size, so pairs that moved between strata, as a pair found by both stages
    becomes the near-duplicate stage's alone above its cosine, are weighted as they were drawn.
    A sampled pair no longer in `now` is counted and left out."""
    cells: dict[tuple[str, str], list[bool]] = {(o, n): [] for o in STRATA for n in STRATA}
    sampled: dict[tuple[str, str], int] = dict.fromkeys(cells, 0)
    left = 0
    for item in items:
        key = np.uint64(keys[item.pair.pair])
        where = [n for n in STRATA if _holds(now[n], key)]
        if not where:
            left += 1
            continue
        cell = (stratum_of(item.pair.stages), where[0])
        sampled[cell] += 1
        if item.pair.pair in labeled:
            cells[cell].append(labeled[item.pair.pair])
    size = {
        (o, n): int(np.intersect1d(drawn[o], now[n], assume_unique=True).size) for o, n in cells
    }

    def over(names: Sequence[str]) -> dict[str, Any] | None:
        return stratified([(size[c], len(cells[c]), sum(cells[c])) for c in cells if c[1] in names])

    by_stratum = {
        n: {
            "size": int(now[n].size),
            "sampled": sum(sampled[(o, n)] for o in STRATA),
            "labeled": sum(len(cells[(o, n)]) for o in STRATA),
            "correct": sum(sum(cells[(o, n)]) for o in STRATA),
            "estimate": over([n]),
        }
        for n in STRATA
    }
    return {
        "strata": by_stratum,
        "stages": {
            stage: over([n for n, covers in STRATA.items() if stage in covers]) for stage in STAGES
        },
        "overall": over(list(STRATA)),
        "sampled_pairs_no_longer_found": left,
    }


def _holds(keys: PairKeys, key: np.uint64) -> bool:
    at = int(np.searchsorted(keys, key))
    return at < keys.size and keys[at] == key


def latest_rows(
    items: Sequence[Item], labels: Sequence[Mapping[str, Any]], labeler: str, *, exact: bool = False
) -> dict[str, Mapping[str, Any]]:
    """Each pair's latest label row among labelers named with the prefix `labeler`, or with
    `exact`, by that labeler alone; by pair id. Of rows given the same day, the later wins."""
    by_sides = {(i.query.ref.key, i.match.ref.key): i.pair.pair for i in items}
    latest: dict[str, Mapping[str, Any]] = {}
    for row in labels:
        name = str(row["labeler"])
        if (name != labeler) if exact else not name.startswith(labeler):
            continue
        sides = (
            (row["query"]["bill_id"], row["query"]["text_sha256"]),
            (row["match"]["bill_id"], row["match"]["text_sha256"]),
        )
        pair = by_sides.get(sides)
        if pair is not None and (pair not in latest or str(row["at"]) >= str(latest[pair]["at"])):
            latest[pair] = row
    return latest


def latest_labels(
    items: Sequence[Item], labels: Sequence[Mapping[str, Any]], labeler: str, *, exact: bool = False
) -> dict[str, int]:
    """Each pair's latest label on the 0-4 scale, by pair id (latest_rows); refused for a row
    on another scale, whose labels say something else."""
    rows = latest_rows(items, labels, labeler, exact=exact)
    for pair, row in rows.items():
        if row.get("scale") is not None:
            raise PrecisionError(f"{pair}'s label by {row['labeler']} is not on the 0-4 scale")
    return {pair: int(row["label"]) for pair, row in rows.items()}


def latest_correct(
    items: Sequence[Item], labels: Sequence[Mapping[str, Any]], labeler: str, *, exact: bool = False
) -> dict[str, bool]:
    """Whether each pair's latest label (latest_rows) counts it correct, on whichever scale it
    was asked (lineage.labels.is_correct)."""
    return {p: is_correct(r) for p, r in latest_rows(items, labels, labeler, exact=exact).items()}


def _estimate(
    items: Sequence[Item], labeled: Mapping[str, bool], sizes: Mapping[str, int]
) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, Any]] = {}
    for name in STRATA:
        mine = [i.pair.pair for i in items if stratum_of(i.pair.stages) == name]
        got = [labeled[p] for p in mine if p in labeled]
        correct = sum(got)
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
    items: Sequence[Item],
    labels: Sequence[Mapping[str, Any]],
    sizes: Mapping[str, int],
    *,
    model_labeler: str | None = None,
) -> dict[str, Any]:
    """Precision from people's labels and, apart, from the model's provisional ones, with how
    often the two agree. `labels` are the current ones (lineage.labels.current_labels);
    `sizes` is each stratum's pair count in the build the sample was drawn from. With
    `model_labeler`, only that model labeler's labels count as the model's, so labels asked
    under another prompt or effort are not mixed in."""
    for row in labels:
        if not str(row["labeler"]).startswith((PERSON, MODEL)):
            raise PrecisionError(f"labeler {row['labeler']!r} is neither a person nor a model")
    for item in items:
        stratum_of(item.pair.stages)
    people = latest_rows(items, labels, PERSON)
    model = (
        latest_rows(items, labels, MODEL)
        if model_labeler is None
        else latest_rows(items, labels, model_labeler, exact=True)
    )
    both = [p for p in people if p in model]
    return {
        "correct_from": CORRECT_FROM,
        "model_labeler": model_labeler or MODEL,
        "person": _estimate(items, {p: is_correct(r) for p, r in people.items()}, sizes),
        "provisional_model": _estimate(items, {p: is_correct(r) for p, r in model.items()}, sizes),
        "agreement": {
            "pairs": len(both),
            # A label on another scale is never the same label.
            "same_label": sum(
                people[p].get("scale") == model[p].get("scale")
                and people[p]["label"] == model[p]["label"]
                for p in both
            ),
            "same_correctness": sum(is_correct(people[p]) == is_correct(model[p]) for p in both),
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


def without_sections(items: Sequence[Item], dropped: Collection[SectionKey]) -> list[Item]:
    """The sample's pairs neither side of which is among `dropped`."""
    return [i for i in items if i.pair.query not in dropped and i.pair.match not in dropped]


def build_sample(
    tables_dir: Path,
    *,
    model: str = MODEL_NAME,
    per_stratum: int = PER_STRATUM,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sample's rows, drawn from both stages' stored cross-state pairs, and the draw's
    report: the seed and each stratum's size in the build, which the estimate weights by."""
    population = stored_strata(tables_dir, model=model)
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
