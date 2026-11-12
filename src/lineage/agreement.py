"""How far the model's provisional labels can be trusted: a person labels a subsample of the
precision sample, and the two are compared.

PAIRS of the precision sample's pairs are drawn, stratified by the stages that found them (as
even a split as the count allows, the remainder to the strata first in report order), under a
fixed seed, and shuffled into one labeling order. The subsample file has the precision sample's
own rows and pair ids, so a person labels it with `lineage label` and the labels land in the
same file, keyed as every label is. The session is blind to the stages, as it always is, and
shows no model label.

Agreement is measured where the person and the named model labeler both labeled a pair: Cohen's
weighted kappa on the 0 to 4 scale, with linear weights, and the share of pairs on the same side
of the correct/incorrect cutoff (lineage.labels.CORRECT_FROM). Linear rather than quadratic
weights, because quadratic ones discount a one-step disagreement most, and a one-step
disagreement across the cutoff is the one precision turns on. The bars were set before any
person's label was given: kappa at least KAPPA_FROM and cutoff agreement at least
CUTOFF_AGREEMENT_FROM, or the model's labels do not stand as provisional.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lineage.intervals import wilson
from lineage.labels import CORRECT_FROM, PERSON, Item, Pair
from lineage.pair_data import LABELS
from lineage.precision import SEED as PRECISION_SEED
from lineage.precision import STRATA, latest_labels, stratified, stratum_of

PAIRS = 50
SEED = PRECISION_SEED + 2  # the flagged sample took + 1
KAPPA_FROM = 0.6
CUTOFF_AGREEMENT_FROM = 0.85
WEIGHTS = "linear"
SAMPLE = Path("evals/agreement_sample.jsonl")
SAMPLE_REPORT = Path("reports/agreement_sample.json")
COMMAND = "uv run lineage agreement-sample"


class AgreementError(Exception):
    pass


def allocation(size: int = PAIRS) -> dict[str, int]:
    """How many pairs each stratum gets: an even split, the remainder to the first strata."""
    base, extra = divmod(size, len(STRATA))
    return {name: base + (i < extra) for i, name in enumerate(STRATA)}


def draw_subsample(pairs: Sequence[Pair], *, size: int = PAIRS, seed: int = SEED) -> list[Pair]:
    """`size` of the sample's pairs, stratified by stage, in a shuffled labeling order."""
    wanted = allocation(size)
    drawn: list[Pair] = []
    for i, name in enumerate(STRATA):
        mine = [p for p in pairs if stratum_of(p.stages) == name]
        if len(mine) < wanted[name]:
            raise AgreementError(
                f"the sample has {len(mine)} {name} pairs, fewer than the {wanted[name]} wanted"
            )
        chosen = np.random.default_rng([seed, i]).choice(len(mine), wanted[name], replace=False)
        drawn += [mine[j] for j in np.sort(chosen).tolist()]
    order = np.random.default_rng([seed, len(STRATA)]).permutation(len(drawn))
    return [drawn[j] for j in order.tolist()]


def subsample_rows(pairs: Sequence[Pair]) -> list[dict[str, Any]]:
    """The subsample's rows, in the precision sample's own format and with its pair ids."""
    return [
        {
            "pair": p.pair,
            "stages": list(p.stages),
            "query": {"bill_id": p.query.bill_id, "ordinal": p.query.ordinal},
            "match": {"bill_id": p.match.bill_id, "ordinal": p.match.ordinal},
        }
        for p in pairs
    ]


def subsample_report(
    pairs: Sequence[Pair], *, source: Path, size: int, seed: int
) -> dict[str, Any]:
    return {
        "command": COMMAND,
        "source": str(source),
        "seed": seed,
        "pairs": size,
        "strata": {name: sum(1 for p in pairs if stratum_of(p.stages) == name) for name in STRATA},
        "kappa_from": KAPPA_FROM,
        "cutoff_agreement_from": CUTOFF_AGREEMENT_FROM,
        "weights": WEIGHTS,
    }


def write_subsample(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    sample_path: Path = SAMPLE,
    report_path: Path = SAMPLE_REPORT,
) -> None:
    sample_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report_path.write_text(json.dumps(report, indent=1) + "\n")


def weighted_kappa(first: Sequence[int], second: Sequence[int], categories: int) -> float | None:
    """Cohen's kappa with linear weights over labels 0 .. categories - 1; None when chance
    alone would leave no disagreement to measure against (both raters used one label)."""
    if len(first) != len(second):
        raise AgreementError("the two raters labeled different numbers of pairs")
    if not first:
        return None
    observed = np.zeros((categories, categories))
    for a, b in zip(first, second, strict=True):
        observed[a, b] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    index = np.arange(categories)
    weights = np.abs(index[:, None] - index[None, :]) / (categories - 1)
    chance = float((weights * expected).sum())
    if chance == 0:
        return None
    return 1.0 - float((weights * observed).sum()) / chance


def agreement(
    items: Sequence[Item],
    labels: Sequence[Mapping[str, Any]],
    *,
    model_labeler: str,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The person's labels against `model_labeler`'s on the pairs both labeled, and whether the
    model's labels pass the bars. With `sizes`, each stratum's size in the build, also the
    precision each labeler's labels give on those same pairs, weighted as the precision sample
    is, so the two are compared on the same pairs."""
    people = latest_labels(items, labels, PERSON)
    model = latest_labels(items, labels, model_labeler, exact=True)
    both = [i.pair.pair for i in items if i.pair.pair in people and i.pair.pair in model]
    person_labels = [people[p] for p in both]
    model_labels = [model[p] for p in both]
    kappa = weighted_kappa(person_labels, model_labels, len(LABELS))
    same_side = sum(
        (a >= CORRECT_FROM) == (b >= CORRECT_FROM)
        for a, b in zip(person_labels, model_labels, strict=True)
    )
    confusion = [[0] * len(LABELS) for _ in LABELS]
    for a, b in zip(person_labels, model_labels, strict=True):
        confusion[a][b] += 1
    stratum = {i.pair.pair: stratum_of(i.pair.stages) for i in items}
    by_stratum: dict[str, dict[str, Any]] = {}
    for name in STRATA:
        mine = [p for p in both if stratum[p] == name]
        by_stratum[name] = {
            "pairs": len(mine),
            "same_label": sum(people[p] == model[p] for p in mine),
            "same_side_of_cutoff": sum(
                (people[p] >= CORRECT_FROM) == (model[p] >= CORRECT_FROM) for p in mine
            ),
            "person_correct": sum(people[p] >= CORRECT_FROM for p in mine),
            "model_correct": sum(model[p] >= CORRECT_FROM for p in mine),
        }
    share = same_side / len(both) if both else None
    same_pairs: dict[str, Any] | None = None
    if sizes is not None:
        same_pairs = {
            who: stratified(
                [
                    (
                        int(sizes.get(name, 0)),
                        by_stratum[name]["pairs"],
                        by_stratum[name][f"{who}_correct"],
                    )
                    for name in STRATA
                ]
            )
            for who in ("person", "model")
        }
    return {
        "model_labeler": model_labeler,
        "pairs_in_subsample": len(items),
        "pairs_labeled_by_both": len(both),
        "weights": WEIGHTS,
        "weighted_kappa": None if kappa is None else round(kappa, 4),
        "same_label": sum(a == b for a, b in zip(person_labels, model_labels, strict=True)),
        "same_side_of_cutoff": same_side,
        "cutoff_agreement": None if share is None else round(share, 4),
        "cutoff_agreement_interval_95": [round(v, 4) for v in wilson(same_side, len(both))],
        "mean_model_minus_person": (
            round(float(np.mean(np.subtract(model_labels, person_labels))), 4) if both else None
        ),
        "confusion_person_by_model": confusion,
        "strata": by_stratum,
        "precision_on_these_pairs": same_pairs,
        "kappa_from": KAPPA_FROM,
        "cutoff_agreement_from": CUTOFF_AGREEMENT_FROM,
        "complete": len(both) == len(items),
        "passes": (
            kappa is not None
            and share is not None
            and len(both) == len(items)
            and kappa >= KAPPA_FROM
            and share >= CUTOFF_AGREEMENT_FROM
        ),
    }
