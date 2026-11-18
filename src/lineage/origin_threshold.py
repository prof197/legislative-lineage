"""`lineage eval origin-threshold`: each candidate floor's precision on the tuning sample, beside
its development recall, for the ruling that sets origin matching's threshold.

The threshold is a floor on a match's larger share (lineage.origins.share_of): a page
lists an origin only at or above it. The tuning sample is drawn apart from each band between the
candidate floors, so a floor's precision is the bands it keeps, each weighted by its size in the
matches table (lineage.precision.stratified), and the band just above it is reported on its own.
Each floor's development recall is the origin recall eval (lineage.origin_recall) on the
matches the floor keeps, so the two are read from the same table. By stage, a floor's labels are
counted unweighted, as a reading of where its errors sit, not an estimate. See
DESIGN.md#origin-matches.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lineage.core.intervals import wilson
from lineage.labels import is_correct
from lineage.origin_labels import PROMPT
from lineage.origin_precision import BANDS, FLOORS, band_name, estimate
from lineage.origin_recall import LoadedKey, evaluate
from lineage.origins import OriginMatch, share_of
from lineage.precision import stratified
from lineage.strata import STRATA

COMMAND = "uv run lineage eval origin-threshold"
REPORT = Path("reports/origin_threshold_proposal.json")


def correct_by_pair(labels: Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    """Each labeled pair id's answer under the origin prompt; a pair labeled twice is refused,
    since which answer stood would depend on order."""
    out: dict[str, bool] = {}
    for row in labels:
        if not str(row["labeler"]).endswith(f"@{PROMPT.version}"):
            continue
        pair = str(row["pair"])
        if pair in out:
            raise ValueError(f"{pair} is labeled twice under {PROMPT.version}")
        out[pair] = is_correct(row)
    return out


def correct_by_match(labels: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], bool]:
    """Each labeled match's answer under the origin prompt, by (bill id, document id), whichever
    part drew it; two different answers for one match are refused."""
    out: dict[tuple[str, str], bool] = {}
    for row in labels:
        if not str(row["labeler"]).endswith(f"@{PROMPT.version}"):
            continue
        key = (str(row["bill_id"]), str(row["doc_id"]))
        answer = is_correct(row)
        if out.get(key, answer) != answer:
            raise ValueError(f"{key[0]} x {key[1]} has two different labels")
        out[key] = answer
    return out


def _share(xs: Sequence[bool]) -> dict[str, Any]:
    n, x = len(xs), sum(xs)
    return {
        "labeled": n,
        "correct": x,
        "precision": round(x / n, 4) if n else None,
        "interval_95": [round(v, 4) for v in wilson(x, n)],
    }


def proposal(
    rows: Sequence[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    sizes: Mapping[str, int],
    matches: Sequence[OriginMatch],
    key: LoadedKey,
    docs: Mapping[str, Collection[str]],
) -> dict[str, Any]:
    """The proposal's report: per band, and per candidate floor its weighted precision, how
    much it lists, its development recall and its labels by stage."""
    if list(sizes) != list(BANDS):
        raise ValueError(f"the tuning sample's strata are {list(sizes)}, not the bands {BANDS}")
    correct = correct_by_pair(labels)
    bands = estimate(rows, correct, sizes)["strata"]
    floors: list[dict[str, Any]] = []
    for floor in FLOORS:
        kept = [b for f, b in zip(FLOORS, BANDS, strict=True) if f >= floor]
        listed = [m for m in matches if share_of(m) >= floor]
        recall = evaluate(matches, key, docs, floor=floor)["recall"]["pooled"]
        sampled = [r for r in rows if r["stratum"] in kept and r["pair"] in correct]
        floors.append(
            {
                "floor": floor,
                "precision": stratified(
                    [(sizes[b], bands[b]["labeled"], bands[b]["correct"]) for b in kept]
                ),
                "band_above": bands[band_name(floor)],
                "matches_listed": len(listed),
                "bills_listing_an_origin": len({m.bill_id for m in listed}),
                "development_recall": recall,
                "by_stage": {
                    name: _share([correct[r["pair"]] for r in sampled if tuple(r["stages"]) == st])
                    for name, st in STRATA.items()
                },
            }
        )
    return {
        "command": COMMAND,
        "prompt": PROMPT.version,
        "score": "a match's larger share: of the model's words, or of the bill's",
        "labeled": len([r for r in rows if r["pair"] in correct]),
        "sampled": len(rows),
        "bands": bands,
        "floors": floors,
    }
