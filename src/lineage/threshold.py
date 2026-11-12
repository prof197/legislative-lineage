"""`lineage eval threshold`: the semantic threshold's options, what each shows and finds.

The threshold of 0.8905 was set on the labeled pair dataset (lineage.semantic) and changes
only on precision-sample evidence (DESIGN.md#precision-sample). The options are the build's
threshold and stricter ones in steps of 0.01; the near-duplicate stage alone; and keeping a
threshold but listing the relatives found by meaning alone apart, under their own heading, so
each list has its own precision. A lower threshold would list relatives the samples never drew,
whose precision nothing measures, so it is not offered until both samples are topped up from
the new band.

Each sampled page drew one relative uniformly from what it lists now (lineage.relatives_
sample). An option's precision is the share labeled correct of the sampled relatives it would
still list, with its Wilson interval: at the build's threshold, the share over every page, and
under a stricter option, over the relatives that clear it. A page then counts in proportion to
how much of its list the option keeps. Weighting each kept relative by the page's list before
over after (L / S) would undo that and estimate the mean page's precision without bias, but on
the tuning sample, where a strict option keeps few drawn relatives, its variance put estimates
above 1 (DESIGN.md#precision-sample), so it is not used. Each option also counts the pages that
would list nothing. A page longer than RELATIVES_SHOWN shows only its best, and a stricter
option could bring a relative from beyond that cut onto the page; the sample never drew those.

Development recall for each option comes from the commission eval (`recall_by_semantic_
threshold`, lineage.commission), counted on each query's own hits, so no option needs a search.
The proposal reads the tuning sample only; the reporting sample is read once the threshold is
final.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage.intervals import weighted_share, wilson
from lineage.labels import Item
from lineage.listing import ListedPairs, listed_pairs, shipped_flag
from lineage.precision import STRATA, latest_correct, pair_keys, restratified, stored_strata, strata
from lineage.relatives import bill_relatives
from lineage.semantic import BUILD_THRESHOLD, CORROBORATING, MODEL, THRESHOLD
from lineage.semantic_pairs import section_keys
from lineage.state_boilerplate import dropped_strata

# Up to 0.97, the first step at which development recall falls: the two enactments found by
# meaning alone reach 0.9677 and 0.9632 (reports/results.json, each bill's best_cosine).
CANDIDATES = (BUILD_THRESHOLD, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97)
# The first review's choice, which the second moved to THRESHOLD; the tuning pages are listed
# again at both, so the record of each decision reproduces.
FIRST_CHOICE = 0.93
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
    pages: Sequence[Listed], correct: Sequence[bool | None], keep: Keep
) -> dict[str, Any]:
    """The share correct of the sampled relatives `keep` would still list, and how many pages
    would list nothing. A drawn relative with no label is left out and counted."""
    kept_right: list[bool] = []
    unlabeled = emptied = 0
    for page, right in zip(pages, correct, strict=True):
        kept, shown = keep(page)
        emptied += shown == 0
        if not kept:
            continue
        if right is None:
            unlabeled += 1
            continue
        kept_right.append(right)
    hits = sum(kept_right)
    return {
        "labeled": len(kept_right),
        "correct": hits,
        "precision": round(hits / len(kept_right), 4) if kept_right else None,
        "interval_95": [round(v, 4) for v in wilson(hits, len(kept_right))],
        "pages_listing": len(pages) - emptied,
        "pages_emptied": emptied,
        "unlabeled": unlabeled,
    }


def by_band(pages: Sequence[Listed], correct: Sequence[bool | None]) -> list[dict[str, Any]]:
    """The sampled relatives found by meaning alone, by their best cosine between options, and
    those the near-duplicate stage found: where precision changes."""
    edges = [*CANDIDATES, 1.0 + 1e-9]
    rows: list[dict[str, Any]] = []
    for lo, hi in pairwise(edges):
        mine = [
            r
            for p, r in zip(pages, correct, strict=True)
            if not p.near and p.cosine is not None and lo <= p.cosine < hi and r is not None
        ]
        rows.append(_band(f"by meaning, {lo} to {min(hi, 1.0)}", mine))
    near = [r for p, r in zip(pages, correct, strict=True) if p.near and r is not None]
    rows.append(_band("by wording", near))
    return rows


def _band(name: str, right: Sequence[bool]) -> dict[str, Any]:
    hits = sum(right)
    return {
        "band": name,
        "labeled": len(right),
        "correct": hits,
        "precision": round(hits / len(right), 4) if right else None,
        "interval_95": [round(v, 4) for v in wilson(hits, len(right))],
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
            "threshold": BUILD_THRESHOLD,
            "by_wording": page_precision(listed, correct, near_duplicate_only),
            "by_meaning_alone": page_precision(listed, correct, by_meaning_only(BUILD_THRESHOLD)),
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
        "by_band": by_band(listed, correct),
        "options": options,
    }


# The flag's variants measured at the stage's threshold: (name, flag on, the cosine from which
# a pair both stages found is exempt: the one that corroborates a match, the stage's own as the
# flag first shipped, or none).
FLAG_VARIANTS: tuple[tuple[str, bool, str | None], ...] = (
    ("flag_off", False, None),
    ("flag", True, "corroborating"),
    ("flag_exempt_at_threshold", True, "threshold"),
    ("flag_without_exemption", True, None),
)


def _exempt_at(rule: str | None, threshold: float) -> float | None:
    return {"corroborating": CORROBORATING, "threshold": threshold}.get(rule or "")


def _pair_key(a: int, b: int) -> int:
    lo, hi = min(a, b), max(a, b)
    return (lo << 32) | hi


def flag_at_threshold(
    tables_dir: Path,
    items: Sequence[Item],
    labels: Sequence[Mapping[str, Any]],
    pages: Mapping[str, Mapping[str, Any]],
    *,
    labeler: str,
    threshold: float = THRESHOLD,
    model: str = MODEL,
) -> dict[str, Any]:
    """The tuning sample's pages listed again at `threshold` with the flag off, on, on with the
    exemption for pairs both stages found only at `threshold`, and on without it. A drawn
    relative keeps its label when its page still lists its family and the pair it was judged on
    still stands; a family listed on other pairs only is counted apart, unlabeled. Each kept
    relative is weighted by its page's list as drawn over its list now, so the self-normalized
    share estimates the precision of what a page lists."""
    flag = shipped_flag(tables_dir)
    if flag is None:
        raise ThresholdError("no state boilerplate counts; run `uv run lineage state-boilerplate`")
    bills, ordinals = section_keys(tables_dir)
    number = {(b, o): i for i, (b, o) in enumerate(zip(bills, ordinals.tolist(), strict=True))}
    got = latest_correct(items, labels, labeler, exact=True)
    ids = [i.pair.pair for i in items]
    judged = {
        i.pair.pair: _pair_key(
            number[(i.pair.query.bill_id, i.pair.query.ordinal)],
            number[(i.pair.match.bill_id, i.pair.match.ordinal)],
        )
        for i in items
    }
    page_bills = sorted({str(pages[p]["bill_id"]) for p in ids})
    variants: dict[str, Any] = {}
    for name, on, rule in FLAG_VARIANTS:
        listed = listed_pairs(
            tables_dir, model=model, threshold=threshold, flag=flag if on else None,
            exempt_at=_exempt_at(rule, threshold),
        )  # fmt: skip
        standing = np.unique(
            np.concatenate(
                [
                    (np.asarray(p.left).astype(np.uint64) << np.uint64(32))
                    | np.asarray(p.right).astype(np.uint64)
                    for p in (listed.near, listed.meaning)
                ]
            )
        )
        stands = dict(
            zip(
                ids,
                np.isin(np.array([judged[p] for p in ids], dtype=np.uint64), standing).tolist(),
                strict=True,
            )
        )
        relatives = bill_relatives(tables_dir, page_bills, model=model, listed=listed)
        kept: list[tuple[float, bool]] = []
        relisted = dropped = unlabeled = 0
        for p in ids:
            page = pages[p]
            now = relatives[str(page["bill_id"])].matches
            if page["family"] not in {m.family for m in now}:
                dropped += 1
            elif not stands[p]:
                relisted += 1
            elif got.get(p) is None:
                unlabeled += 1
            else:
                kept.append((int(page["listed"]) / len(now), bool(got[p])))
        hits = sum(c for _, c in kept)
        share, interval = weighted_share([w for w, _ in kept], [c for _, c in kept])
        emptied = sum(not relatives[b].matches for b in page_bills)
        variants[name] = {
            "labeled": len(kept),
            "correct": hits,
            "precision": round(hits / len(kept), 4) if kept else None,
            "interval_95": [round(v, 4) for v in wilson(hits, len(kept))],
            "page_weighted": {
                "precision": None if share is None else round(share, 4),
                "interval_95": [round(v, 4) for v in interval],
            },
            "relatives_dropped": dropped,
            "relisted_on_another_pair": relisted,
            "unlabeled": unlabeled,
            "pages_listing": len(page_bills) - emptied,
            "pages_emptied": emptied,
            "relatives_listed": sum(len(relatives[b].matches) for b in page_bills),
            "build_pairs_exempted": listed.exempted,
        }
    return {"threshold": threshold, "labeler": labeler, "sampled": len(items), **variants}


def _strata_of(listed: ListedPairs) -> dict[str, NDArray[np.uint64]]:
    return strata(
        pair_keys(listed.near.left, listed.near.right),
        pair_keys(listed.meaning.left, listed.meaning.right),
    )


def _keys_of(items: Sequence[Item], tables_dir: Path) -> dict[str, int]:
    bills, ordinals = section_keys(tables_dir)
    number = {(b, o): i for i, (b, o) in enumerate(zip(bills, ordinals.tolist(), strict=True))}
    return {
        i.pair.pair: _pair_key(
            number[(i.pair.query.bill_id, i.pair.query.ordinal)],
            number[(i.pair.match.bill_id, i.pair.match.ordinal)],
        )
        for i in items
    }


def pair_level_at(
    tables_dir: Path,
    items: Sequence[Item],
    labels: Sequence[Mapping[str, Any]],
    sizes: Mapping[str, int],
    *,
    labeler: str,
    dropped_items: Sequence[Item] | None = None,
    dropped_labeler: str | None = None,
    threshold: float = THRESHOLD,
    model: str = MODEL,
) -> dict[str, Any]:
    """The pair-level diagnostic at the stage's threshold: the precision sample, drawn from
    every pair the build stores, re-weighted to the pairs found at `threshold` with the flag off
    and as shipped (lineage.precision.restratified); and the sample of the pairs the flag drops,
    re-weighted to the pairs it drops at `threshold` and to the pairs its exemption keeps: all of
    them, and apart those it keeps only because the semantic stage found them at the cosine that
    corroborates a match, not at `threshold`."""
    flag = shipped_flag(tables_dir)
    if flag is None:
        raise ThresholdError("no state boilerplate counts; run `uv run lineage state-boilerplate`")
    drawn = stored_strata(tables_dir, model=model)
    if {n: int(k.size) for n, k in drawn.items()} != dict(sizes):
        raise ThresholdError(
            "the build's strata are not the ones the precision sample was drawn from"
        )
    every = _strata_of(listed_pairs(tables_dir, model=model, threshold=threshold, flag=None))
    kept = _strata_of(listed_pairs(tables_dir, model=model, threshold=threshold, flag=flag))
    keys = _keys_of(items, tables_dir)
    correct = latest_correct(items, labels, labeler, exact=True)
    out: dict[str, Any] = {
        "threshold": threshold,
        "model_labeler": labeler,
        "flag_off": restratified(items, correct, keys, drawn, every),
        "flag_on": restratified(items, correct, keys, drawn, kept),
        "dropped_pairs": None,
    }
    if dropped_items is not None:
        strict = _strata_of(
            listed_pairs(tables_dir, model=model, threshold=threshold, flag=flag, exempt_at=None)
        )
        at_stage = _strata_of(
            listed_pairs(
                tables_dir, model=model, threshold=threshold, flag=flag, exempt_at=threshold
            )
        )
        who = dropped_labeler or labeler
        dropped_correct = latest_correct(dropped_items, labels, who, exact=True)
        dropped_keys = _keys_of(dropped_items, tables_dir)
        dropped_drawn = dropped_strata(tables_dir, model=model)

        def among(population: Mapping[str, NDArray[np.uint64]]) -> dict[str, Any]:
            return {
                "model_labeler": who,
                **restratified(
                    dropped_items, dropped_correct, dropped_keys, dropped_drawn, population
                ),
            }

        out["dropped_pairs"] = among(
            {n: np.setdiff1d(every[n], kept[n], assume_unique=True) for n in STRATA}
        )
        # The pairs the exemption keeps: a flagged section's pairs both stages found.
        out["exempted_pairs"] = among(
            {n: np.setdiff1d(kept[n], strict[n], assume_unique=True) for n in STRATA}
        )
        # Of those, the ones the semantic stage found under `threshold`.
        out["exempted_by_corroboration"] = among(
            {n: np.setdiff1d(kept[n], at_stage[n], assume_unique=True) for n in STRATA}
        )
    return out
