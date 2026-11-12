"""`lineage eval threshold`: the semantic threshold's options, what each shows and finds.

The threshold of 0.8905 was set on the labeled pair dataset (lineage.semantic) and changes
only on precision-sample evidence (DESIGN.md#precision-sample). The options are the build's
threshold and stricter ones in steps of 0.01; the near-duplicate stage alone; and keeping a
threshold but listing the relatives found by meaning alone apart, under their own heading, so
each list has its own precision. A lower threshold would list relatives the samples never drew,
whose precision nothing measures, so it is not offered until both samples are topped up from
the new band.

Precision is a page's, averaged over pages (lineage.relatives_sample). Each sampled page drew
one relative uniformly from the L it lists now. Under an option that keeps S of them, the
page's precision is the share correct of those S, and the drawn relative, counted when it is
kept and correct and weighted L / S, estimates that share without bias. The estimate is the
mean over the pages that would still list something, so at the build's threshold, where every
weight is 1, it is the share of sampled relatives labeled correct. Its 95% interval is a seeded
bootstrap over pages. A page longer than RELATIVES_SHOWN shows only its best, and a stricter
option could bring a relative from beyond that cut onto the page; the sample never drew those,
so the estimate counts only what the page shows now.

Development recall for each option comes from the commission eval (`recall_by_semantic_
threshold`, lineage.commission), counted on each query's own hits, so no option needs a search.
The proposal reads the tuning sample only; the reporting sample is read once the threshold is
final.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lineage.labels import Item
from lineage.precision import latest_correct
from lineage.semantic import THRESHOLD

# Up to 0.97, the first step at which development recall falls: the two enactments found by
# meaning alone reach 0.9677 and 0.9632 (reports/results.json, each bill's best_cosine).
CANDIDATES = (THRESHOLD, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97)
RESAMPLES = 2000
BOOTSTRAP_SEED = 20261116
REPORT = Path("reports/threshold_proposal.json")
COMMAND = "uv run lineage eval threshold"


class ThresholdError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Listed:
    """What a sampled page lists, as an option needs it."""

    listed: int  # relatives the page shows now
    near: bool  # whether the near-duplicate stage found the drawn relative
    cosine: float | None  # the drawn relative's best cosine by meaning, if found so
    by_meaning: tuple[float, ...]  # best cosines of the shown relatives found by meaning alone

    @property
    def near_shown(self) -> int:
        return self.listed - len(self.by_meaning)


# An option: whether it keeps the drawn relative, and how many of the page's relatives it keeps.
Keep = Callable[[Listed], tuple[bool, int]]


def listed_of(page: Mapping[str, Any]) -> Listed:
    if "shown_semantic_only" not in page:
        raise ThresholdError("the sample predates the fields a threshold needs; draw it again")
    cosine = page.get("best_cosine")
    return Listed(
        int(page["listed"]),
        "near_duplicate" in page["relative_stages"],
        None if cosine is None else float(cosine),
        tuple(float(c) for c in page["shown_semantic_only"]),
    )


def at_threshold(t: float) -> Keep:
    """Both stages, with the semantic stage's threshold at `t`."""

    def keep(p: Listed) -> tuple[bool, int]:
        kept = p.near or (p.cosine is not None and p.cosine >= t)
        return kept, p.near_shown + sum(c >= t for c in p.by_meaning)

    return keep


def near_duplicate_only(p: Listed) -> tuple[bool, int]:
    """The relatives the near-duplicate stage found: the page without the semantic stage, or
    the list found by wording when the two are listed apart."""
    return p.near, p.near_shown


def by_meaning_only(t: float) -> Keep:
    """The relatives found by meaning alone, at `t`: the list shown apart."""

    def keep(p: Listed) -> tuple[bool, int]:
        kept = not p.near and p.cosine is not None and p.cosine >= t
        return kept, sum(c >= t for c in p.by_meaning)

    return keep


def page_precision(
    pages: Sequence[Listed],
    correct: Sequence[bool | None],
    keep: Keep,
    *,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """A page's precision under `keep`, averaged over the pages that would list something. A
    page whose drawn relative has no label is left out and counted."""
    values: list[float] = []
    kept_labeled = unlabeled = emptied = 0
    for page, right in zip(pages, correct, strict=True):
        kept, shown = keep(page)
        if shown == 0:
            emptied += 1
            continue
        if right is None:
            unlabeled += 1
            continue
        kept_labeled += kept
        values.append(page.listed / shown if kept and right else 0.0)
    point: float | None = None
    interval: list[float] | None = None
    if values:
        point = round(float(np.mean(values)), 4)
        draws = np.random.default_rng(seed).integers(len(values), size=(RESAMPLES, len(values)))
        means = np.asarray(values)[draws].mean(axis=1)
        interval = [round(float(v), 4) for v in np.quantile(means, [0.025, 0.975])]
    return {
        "precision": point,
        "interval_95": interval,
        "pages_listing": len(pages) - emptied,
        "pages_emptied": emptied,
        "pages_unlabeled": unlabeled,
        "sampled_relatives_kept": kept_labeled,
    }


def _recall(found: int, enactments: int) -> dict[str, Any]:
    return {"found": found, "enactments": enactments, "recall": round(found / enactments, 4)}


def sample_inputs(
    items: Sequence[Item],
    labels: Sequence[Mapping[str, Any]],
    pages: Mapping[str, Mapping[str, Any]],
    labeler: str,
) -> tuple[list[Listed], list[bool | None]]:
    got = latest_correct(items, labels, labeler, exact=True)
    ids = [i.pair.pair for i in items]
    return [listed_of(pages[p]) for p in ids], [got.get(p) for p in ids]


def proposal(
    items: Sequence[Item],
    labels: Sequence[Mapping[str, Any]],
    pages: Mapping[str, Mapping[str, Any]],
    commission: Mapping[str, Any],
    *,
    labeler: str,
) -> dict[str, Any]:
    """Every option's page precision on the tuning sample, beside its development recall."""
    listed, correct = sample_inputs(items, labels, pages, labeler)
    by_threshold = {
        float(r["threshold"]): int(r["found"])
        for r in commission.get("recall_by_semantic_threshold", [])
    }
    missing = [t for t in CANDIDATES if t not in by_threshold]
    if missing:
        raise ThresholdError(
            f"the commission results count no recall at {missing}; run `uv run lineage eval "
            "commission`"
        )
    ablation = commission["stage_ablation"]
    near = int(ablation["near_duplicate"]["pooled"]["all"]["found"])
    both = ablation["both"]["pooled"]["all"]
    enactments = int(both["enactments"])
    options: list[dict[str, Any]] = [
        {
            "option": "threshold",
            "threshold": t,
            "page": page_precision(listed, correct, at_threshold(t)),
            "by_meaning_alone": page_precision(listed, correct, by_meaning_only(t)),
            "development_recall": _recall(by_threshold[t], enactments),
        }
        for t in CANDIDATES
    ]
    options.append(
        {
            "option": "near_duplicate_only",
            "page": page_precision(listed, correct, near_duplicate_only),
            "development_recall": _recall(near, enactments),
        }
    )
    options.append(
        {
            "option": "listed_apart",
            "threshold": THRESHOLD,
            "by_wording": page_precision(listed, correct, near_duplicate_only),
            "by_meaning_alone": page_precision(listed, correct, by_meaning_only(THRESHOLD)),
            "development_recall": {
                "on_the_page": _recall(int(both["found"]), enactments),
                "by_wording": _recall(near, enactments),
            },
        }
    )
    return {
        "command": COMMAND,
        "sample": "tuning",
        "labeler": labeler,
        "sampled": len(items),
        "labeled": sum(c is not None for c in correct),
        "resamples": RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "options": options,
    }
