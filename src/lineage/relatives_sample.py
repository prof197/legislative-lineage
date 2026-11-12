"""The precision of what a bill page lists: a sample of listed relatives, judged on the passages
that put them there.

Pair-level precision (lineage.precision) counts every cross-state section pair once, so it is
dominated by long sections and compact templates that pair with many others, and it judges
whole sections where matching compares 94-word windows. It stays as each stage's diagnostic.
This measure asks what a reader meets instead: of the relatives a bill's page lists, how many
are right.

BILLS bills of STATE are drawn at random, under a fixed seed, from its bills with at least one
listed relative, and one relative is drawn at random from each page's list (the RELATIVES_SHOWN
best, lineage.relatives). Each page counts once and each listed relative of a page equally, so
the share correct estimates a page's precision averaged over pages, with no further weights.

A relative is judged on one section pair: the one its listing rests on most. A pair the
near-duplicate stage found comes first, the most band agreements first, since its copied text is
what a page highlights; failing that, the semantic stage's pair at the best cosine. The two
sections are aligned (lineage.align), and the copied and paraphrased passages are given
together, unmarked, alongside both sections. See DESIGN.md#precision-sample.

There are two samples (PARTS). The tuning sample, which the threshold proposal reads
(lineage.threshold), was drawn from pages as listed at the build's threshold without the state
boilerplate flag; each of its rows records what a threshold above the build's needs: the
relative's best cosine, whether the near-duplicate stage found it, and the best cosines of
every relative its page shows by meaning alone. The reporting sample was drawn afresh once the
threshold and the flag were final, under its own seed, from pages as they list then, and from
bills the tuning sample did not draw; it is read once. So its pages, too, were drawn at random
with one relative each from the lists measured, and the share correct is their precision.
"""

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.align import Embedder, align
from lineage.intervals import wilson
from lineage.labels import Item
from lineage.listing import listed_pairs
from lineage.precision import SEED as PRECISION_SEED
from lineage.precision import STRATA, latest_correct, stratum_of
from lineage.relatives import RELATIVES_SHOWN, bill_relatives
from lineage.semantic import BUILD_THRESHOLD, MODEL, THRESHOLD
from lineage.semantic_pairs import section_keys

STATE = "GA"
BILLS = 100
SEED = PRECISION_SEED + 3  # the flagged sample took + 1, the agreement subsample + 2
REPORTING_SEED = PRECISION_SEED + 4


@dataclass(frozen=True, slots=True)
class Part:
    """One sample's draw: its seed, the threshold and flag its pages were listed under, its
    first pair id's number, its files, and the part whose bills it never draws."""

    seed: int
    threshold: float
    flag: bool
    first: int
    sample: Path
    report: Path
    excludes: str | None = None


PARTS: dict[str, Part] = {
    # Drawn at the build's threshold without the flag: what the threshold proposal chose from.
    "tuning": Part(
        SEED,
        BUILD_THRESHOLD,
        False,
        1,
        Path("evals/relatives_tuning.jsonl"),
        Path("reports/relatives_tuning.json"),
    ),
    # Drawn afresh once the threshold and the flag were final, from what pages list then.
    "reporting": Part(
        REPORTING_SEED,
        THRESHOLD,
        True,
        BILLS + 1,
        Path("evals/relatives_reporting.jsonl"),
        Path("reports/relatives_reporting.json"),
        excludes="tuning",
    ),
}
SAMPLE = PARTS["tuning"].sample
SAMPLE_REPORT = PARTS["tuning"].report
LABELS = Path("evals/relatives_labels.jsonl")
COMMAND = "uv run lineage relatives-sample"
TOP = 5  # a page's first relatives, reported apart from the rest


class RelativesSampleError(Exception):
    pass


def _touching(
    left: NDArray[np.uint32], right: NDArray[np.uint32], mine: NDArray[np.bool_]
) -> NDArray[np.int64]:
    return np.flatnonzero(mine[left] | mine[right])


def _best_pair(
    rows: NDArray[np.int64],
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
    scores: NDArray[np.generic],
    query_sections: NDArray[np.bool_],
    match_sections: NDArray[np.bool_],
) -> tuple[int, int] | None:
    """The best-scoring pair among `rows` with one side in each bill, as (query, match)."""
    a, b = left[rows].astype(np.int64), right[rows].astype(np.int64)
    forward = query_sections[a] & match_sections[b]
    backward = query_sections[b] & match_sections[a]
    keep = forward | backward
    if not keep.any():
        return None
    q = np.where(forward, a, b)[keep]
    m = np.where(forward, b, a)[keep]
    s = np.asarray(scores)[rows][keep].astype(np.float64)
    best = np.lexsort((m, q, -s))[0]  # highest score, then the lowest sections
    return int(q[best]), int(m[best])


def build_relatives_sample(
    tables_dir: Path,
    *,
    model: str = MODEL,
    state: str = STATE,
    bills: int = BILLS,
    seed: int = SEED,
    threshold: float = BUILD_THRESHOLD,
    flag: NDArray[np.bool_] | None = None,
    excluded: Collection[str] = frozenset(),
    first: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The sample's rows, in the precision sample's format with the page each relative is
    listed on, and the draw's report: `bills` bills of `state` not in `excluded`, their pages
    listed at `threshold` with `flag` (lineage.listing), pair ids numbered from `first`."""
    section_bills, ordinals = section_keys(tables_dir)
    with queries.connect_quiet() as con:
        states = queries.bill_states(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    in_state = np.fromiter(
        (states.get(b) == state for b in section_bills), dtype=bool, count=len(section_bills)
    )
    pairs = listed_pairs(tables_dir, model=model, threshold=threshold, flag=flag)
    near, meaning = pairs.near, pairs.meaning
    paired = np.zeros(len(section_bills), dtype=bool)
    for left, right in ((near.left, near.right), (meaning.left, meaning.right)):
        paired[np.asarray(left)] = True
        paired[np.asarray(right)] = True
    signed = sorted({section_bills[s] for s in np.flatnonzero(in_state).tolist()})
    listing = sorted({section_bills[s] for s in np.flatnonzero(in_state & paired).tolist()})
    candidates = [b for b in listing if b not in excluded]
    if len(candidates) < bills:
        raise RelativesSampleError(
            f"{state} has {len(candidates)} bills with a relative to draw, fewer than {bills}"
        )
    order: list[int] = np.random.default_rng(seed).permutation(len(candidates)).tolist()
    chosen = [candidates[i] for i in order[:bills]]
    relatives = bill_relatives(tables_dir, chosen, model=model, listed=pairs)
    wanted = set(chosen) | {m.bill_id for page in relatives.values() for m in page.matches}
    numbers: dict[str, list[int]] = {}
    for number, bill_id in enumerate(section_bills):
        if bill_id in wanted:
            numbers.setdefault(bill_id, []).append(number)
    rows: list[dict[str, Any]] = []
    listed: list[int] = []
    for n, bill_id in enumerate(chosen, start=first):
        page = relatives[bill_id]
        if not page.matches:
            raise RelativesSampleError(f"{bill_id} has stored pairs but lists no relative")
        rank = int(np.random.default_rng([seed, n]).integers(len(page.matches)))
        relative = page.matches[rank]
        query_sections = _mask(len(section_bills), numbers[bill_id])
        match_sections = _mask(len(section_bills), numbers[relative.bill_id])
        found = _best_pair(
            _touching(near.left, near.right, query_sections),
            near.left,
            near.right,
            near.agreements,
            query_sections,
            match_sections,
        )
        if found is None:
            found = _best_pair(
                _touching(meaning.left, meaning.right, query_sections),
                meaning.left,
                meaning.right,
                meaning.cosine,
                query_sections,
                match_sections,
            )
        if found is None:
            raise RelativesSampleError(f"no stored pair links {bill_id} to {relative.bill_id}")
        q, m = found
        rows.append(
            {
                "pair": f"r{n:04d}",
                "stages": _stages_of(q, m, near.left, near.right, meaning.left, meaning.right),
                "query": {"bill_id": bill_id, "ordinal": int(ordinals[q])},
                "match": {"bill_id": relative.bill_id, "ordinal": int(ordinals[m])},
                "page": {
                    "bill_id": bill_id,
                    "listed": len(page.matches),
                    "families": page.families,
                    "rank": rank + 1,
                    "family": relative.family,
                    "state": relative.state,
                    "share": round(relative.share, 4),
                    "relative_stages": sorted(relative.sections_by_stage),
                    "best_cosine": relative.best_cosine,
                    # Every relative the page shows by meaning alone, by its best cosine: what
                    # a higher threshold would take off the page.
                    "shown_semantic_only": sorted(
                        float(m.best_cosine)
                        for m in page.matches
                        if "near_duplicate" not in m.sections_by_stage and m.best_cosine is not None
                    ),
                },
            }
        )
        listed.append(len(page.matches))
    report = {
        "command": COMMAND,
        "state": state,
        "seed": seed,
        "threshold": threshold,
        "state_boilerplate_flag": flag is not None,
        "bills_excluded": len(set(listing) & set(excluded)),
        "bills": bills,
        "model": model,
        "relatives_shown": RELATIVES_SHOWN,
        "state_bills_signed": len(signed),
        "state_bills_with_a_relative": len(listing),
        "listed_quantiles": {
            str(q): float(np.quantile(listed, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "relative_strata": {
            name: sum(1 for r in rows if _stratum(r["page"]["relative_stages"]) == name)
            for name in STRATA
        },
    }
    return rows, report


def _mask(size: int, numbers: Sequence[int]) -> NDArray[np.bool_]:
    mask = np.zeros(size, dtype=bool)
    mask[np.asarray(numbers, dtype=np.int64)] = True
    return mask


def _has(left: NDArray[np.uint32], right: NDArray[np.uint32], a: int, b: int) -> bool:
    lo, hi = min(a, b), max(a, b)
    at = np.flatnonzero(np.asarray(left) == lo)
    return bool((np.asarray(right)[at] == hi).any())


def _stages_of(
    q: int,
    m: int,
    near_left: NDArray[np.uint32],
    near_right: NDArray[np.uint32],
    meaning_left: NDArray[np.uint32],
    meaning_right: NDArray[np.uint32],
) -> list[str]:
    stages: list[str] = []
    if _has(near_left, near_right, q, m):
        stages.append("near_duplicate")
    if _has(meaning_left, meaning_right, q, m):
        stages.append("semantic")
    return stages


def _stratum(stages: Sequence[str]) -> str:
    return stratum_of(tuple(stages))


def with_passages(items: Sequence[Item], embed: Embedder | None) -> list[Item]:
    """Each item with the passages its alignment matched, copied and paraphrased alike."""
    out: list[Item] = []
    for item in items:
        a, b = item.query.text, item.match.text
        passages = tuple(
            (a[s.a[0] : s.a[1]], b[s.b[0] : s.b[1]])
            for s in align(a, b, embed=embed).spans
            if s.kind != "unmatched" and s.a is not None and s.b is not None
        )
        out.append(replace(item, passages=passages))
    return out


def write_sample(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    sample_path: Path = SAMPLE,
    report_path: Path = SAMPLE_REPORT,
) -> None:
    sample_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report_path.write_text(json.dumps(report, indent=1) + "\n")


def load_pages(path: Path = SAMPLE) -> dict[str, dict[str, Any]]:
    """Each sampled relative's page, by pair id."""
    pages: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            pages[str(row["pair"])] = dict(row["page"])
    return pages


def _share(labels: Sequence[bool]) -> dict[str, Any]:
    correct = sum(labels)
    return {
        "labeled": len(labels),
        "correct": correct,
        "precision": round(correct / len(labels), 4) if labels else None,
        "interval_95": [round(v, 4) for v in wilson(correct, len(labels))],
    }


def estimate(
    items: Sequence[Item],
    labels: Sequence[Mapping[str, Any]],
    pages: Mapping[str, Mapping[str, Any]],
    *,
    labeler: str,
    exact: bool = False,
) -> dict[str, Any]:
    """The share of sampled relatives labeled correct, overall with its Wilson interval, by the
    stages that found the relative, and for a page's first TOP relatives against the rest."""
    got = latest_correct(items, labels, labeler, exact=exact)
    ids = [i.pair.pair for i in items if i.pair.pair in got]
    by_stage = {
        name: _share([got[p] for p in ids if _stratum(pages[p]["relative_stages"]) == name])
        for name in STRATA
    }
    return {
        "labeler": labeler,
        "sampled": len(items),
        "overall": _share([got[p] for p in ids]),
        "relative_strata": by_stage,
        "rank": {
            f"first_{TOP}": _share([got[p] for p in ids if int(pages[p]["rank"]) <= TOP]),
            f"after_{TOP}": _share([got[p] for p in ids if int(pages[p]["rank"]) > TOP]),
        },
        "with_passages": _share([got[p] for p in ids if _passages_of(items, p)]),
        "without_passages": _share([got[p] for p in ids if not _passages_of(items, p)]),
    }


def _passages_of(items: Sequence[Item], pair: str) -> bool:
    return any(i.passages for i in items if i.pair.pair == pair)
