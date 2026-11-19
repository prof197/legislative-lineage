"""`lineage eval origin-reach`: each candidate reach scale's precision on the labeled origin
matches, beside its development recall, to set how fast a widespread passage counts for less.

A match's weighted shares count a passage in full up to the scale and scale / reach past it
(lineage.origins.matches.weight); a page lists a match whose larger weighted share reaches the
floor. Weighting only lowers a share, so a scale can only take matches off pages. Every labeled
match a label still answers (on the sections it rests on now, lineage.origins.labels
.answers_by_match) is read under each scale, unweighted: the labels come from samples of
different designs, drawn before matching changed, so this is a reading of where the errors sit,
not an estimate. The estimate is a fresh reporting sample, drawn at the scale chosen here.

The scale is the smallest that still pays: walking from no weighing down through the candidates,
each step must take more wrong labeled matches off pages than correct ones, and keep every
development enactment. A first rule, the highest precision that keeps development recall, was
fixed before the reading and amended after it (M13 step 6, recorded in DESIGN.md): it picked the
smallest scale, which takes 71 correct labeled matches off for 29 wrong ones, a loss development
recall cannot see, since uniform acts are not widespread passages. See DESIGN.md#origin-matches.
"""

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from lineage.core.intervals import wilson
from lineage.origins.matches import FLOOR, OriginMatch, is_listed
from lineage.origins.recall import LoadedKey, evaluate

COMMAND = "uv run lineage eval origin-reach"
REPORT = Path("reports/origin_reach_proposal.json")
# Candidate scales, in families. 274 is 0.1% of the corpus's 274,311 families: v1's shingle
# stoplist's share (lineage.match.near.stoplist), read at the passage's scale. None is no weighing.
SCALES: tuple[int | None, ...] = (20, 50, 100, 200, 274, None)
UNWEIGHED = 10**12  # a scale no passage reaches: every weight 1


def _share(answers: Sequence[bool]) -> dict[str, Any]:
    n, x = len(answers), sum(answers)
    return {
        "labeled": n,
        "correct": x,
        "precision": round(x / n, 4) if n else None,
        "interval_95": [round(v, 4) for v in wilson(x, n)],
    }


def _reading(
    scale: int | None,
    matches: Sequence[OriginMatch],
    answers: Mapping[tuple[str, str], bool],
    key: LoadedKey,
    docs: Mapping[str, Collection[str]],
) -> dict[str, Any]:
    s = UNWEIGHED if scale is None else scale
    listed = [m for m in matches if is_listed(m, FLOOR, s)]
    read = [(m, answers[(m.bill_id, m.doc_id)]) for m in listed if (m.bill_id, m.doc_id) in answers]
    sources = sorted({m.doc_id.split("/", 1)[0] for m, _ in read})
    recall = evaluate(matches, key, docs, floor=FLOOR, scale=s)["recall"]["pooled"]
    return {
        "scale": scale,
        "matches_listed": len(listed),
        "bills_listing_an_origin": len({m.bill_id for m in listed}),
        "documents_listed": len({m.doc_id for m in listed}),
        "labeled": _share([a for _, a in read]),
        "by_source": {
            src: _share([a for m, a in read if m.doc_id.split("/", 1)[0] == src]) for src in sources
        },
        "development_recall": recall,
    }


def _size(scale: int | None) -> int:
    return UNWEIGHED if scale is None else scale


def choose(readings: Sequence[Mapping[str, Any]]) -> int | None:
    """The rule: from no weighing down through the smaller scales, a step that takes labeled
    matches off pages is taken only while it takes more wrong ones off than correct ones and
    keeps every development enactment no weighing finds; the walk stops at the first step that
    fails. The scale is the last step taken that changed anything, or no weighing."""
    ordered = sorted(readings, key=lambda r: _size(r["scale"]), reverse=True)
    plain = ordered[0]
    found = plain["development_recall"]["found"]
    chosen, previous = plain, plain
    for r in ordered[1:]:
        before, now = previous["labeled"], r["labeled"]
        correct_off = before["correct"] - now["correct"]
        wrong_off = (before["labeled"] - before["correct"]) - (now["labeled"] - now["correct"])
        if correct_off == wrong_off == 0:
            continue  # takes nothing labeled off: neither a reason to go on nor to stop
        if wrong_off <= correct_off or r["development_recall"]["found"] < found:
            break
        chosen, previous = r, r
    return chosen["scale"]


def proposal(
    matches: Sequence[OriginMatch],
    answers: Mapping[tuple[str, str], bool],
    key: LoadedKey,
    docs: Mapping[str, Collection[str]],
) -> dict[str, Any]:
    """The proposal's report: each candidate scale's listing, labeled precision by source and
    development recall, and the scale the rule picks."""
    readings = [_reading(s, matches, answers, key, docs) for s in SCALES]
    reach = Counter(min(w.reach for w in m.weighed_bill) for m in matches if m.weighed_bill)
    return {
        "command": COMMAND,
        "floor": FLOOR,
        "weight": "a passage counts in full up to the scale and scale / reach past it",
        "answered_matches": len(answers),
        "matches": len(matches),
        "matches_with_reach": sum(1 for m in matches if m.model_words is not None),
        "least_reach_of_a_match": {
            "at_most_scale": {
                str(s): sum(n for r, n in reach.items() if r <= s) for s in SCALES if s is not None
            },
            "over_1000": sum(n for r, n in reach.items() if r > 1000),
        },
        "scales": readings,
        "chosen": choose(readings),
        "rule": "from no weighing down, each step taken only while it takes more wrong labeled "
        "matches off pages than correct ones and keeps every development enactment; the last "
        "step that changed anything",
    }
