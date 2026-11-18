"""Origin recall: of the bills that enacted a model, the share whose origin matches name it.

v1's commission eval asks the same question with an act's text as a query
(lineage.evals.commission); here the act is a collected model document and the answer is the origin
matches table (lineage.origins.matches), so recall is measured on what the site would list rather
than on a query run for the eval. The ground truth is a resolved answer key, the shape of
`reports/answer_key.csv` (lineage.evals.answer_key): one row per bill a commission list row resolved
to, and the rows not counted with the reason. Resolution is v1's, so the two evals count the
same enactments.

An enactment is found when a document of its act has a match with one of its bills whose model
share, the share of the model's words in the sections matched, is above MIN_SHARE: the score v1
calls a family's share (DESIGN.md#bill-scoring), at v1's threshold of one matched section.
Recall is reported pooled, per act and per stratum, for each stage alone and both together,
each with its Wilson 95% interval (lineage.core.intervals), and at higher minimum shares.

A sealed answer key, one whose matches must not be inspected while matching is tuned, is
evaluated only with `final`, as v1's held-out acts are only with `--final`, and never twice at
one configuration (lineage.evals.commission.check_held_out_configuration). See
DESIGN.md#commission-recall and DESIGN.md#origin-matches.
"""

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from lineage.core.intervals import wilson
from lineage.evals.answer_key import ANSWER_KEY, HELD_OUT_KEY, ORIGINS_HELD_OUT_KEY, read_answer_key
from lineage.evals.commission import MIN_SHARE, SHARE_SWEEP
from lineage.evals.ulc import ULC_STATES, Stratum
from lineage.origins.matches import FLOOR, REACH_SCALE, OriginMatch, Stage, is_listed

COMMAND = "uv run lineage eval origins"
STAGES: tuple[Stage, ...] = ("near_duplicate", "semantic")
STRATA: tuple[Stratum, ...] = ("enacted", "substantially_similar")
# Ordered by how far an enactment got; an enactment's reason is its furthest bill's.
MissReason = Literal["not_in_corpus", "not_matched", "below_threshold"]
MISS_REASONS: tuple[MissReason, ...] = get_args(MissReason)


class OriginRecallError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class AnswerKey:
    """A resolved answer key, whether it is sealed (held out from tuning), and what a reader
    must be told about it beside its figures."""

    name: str
    path: Path
    sealed: bool
    status: str | None = None


# The sealed key is the held-out lists' two reserve acts, which v1 never ran (ruled at the M11
# review, 2026-11-16). A wider pull of the commission's lists is dropped for v2, since each new
# act needs a text downloaded by hand. v1's four held-out acts are reported too, as extra
# evidence: v1's final run opened them, so they are not a sealed test, and say so.
V1_OPENED = "opened in v1; not a sealed test"
KEYS: dict[str, AnswerKey] = {
    "development": AnswerKey("development", ANSWER_KEY, sealed=False),
    "v1_held_out": AnswerKey("v1_held_out", HELD_OUT_KEY, sealed=False, status=V1_OPENED),
    "held_out": AnswerKey("held_out", ORIGINS_HELD_OUT_KEY, sealed=True),
}


@dataclass(frozen=True, slots=True)
class KeyEnactment:
    """One counted list row of an answer key, and the corpus bills it resolved to."""

    act: str
    jurisdiction: str
    year: int
    listed: str
    stratum: Stratum
    rule: str | None  # the step that resolved it (lineage.evals.ulc.find_bills), None for no bill
    bill_ids: tuple[str, ...]

    @property
    def state(self) -> str | None:
        return ULC_STATES[self.jurisdiction]


@dataclass(frozen=True, slots=True)
class LoadedKey:
    note: str
    enactments: list[KeyEnactment]
    excluded: list[dict[str, Any]]  # rows not counted, with the reason


def _stratum(status: str) -> Stratum:
    # As lineage.evals.ulc reads the lists: "Enacted - Substantially Similiar", the lists' own
    # spelling, also starts with "Enacted", so the longer prefix is tried first.
    if status.startswith("Enacted - Substantially Simil"):
        return "substantially_similar"
    if status.startswith("Enacted"):
        return "enacted"
    raise OriginRecallError(f"a counted row has status {status!r}, which is not an enactment")


def load_key(path: Path) -> LoadedKey:
    """The counted enactments of a resolved answer key, each with its bills, in file order."""
    note, rows = read_answer_key(path)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    excluded: list[dict[str, Any]] = []
    for row in rows:
        if row["jurisdiction"] not in ULC_STATES:
            raise OriginRecallError(f"{path}: unknown jurisdiction {row['jurisdiction']!r}")
        if row["counted"] != "yes":
            excluded.append(
                {
                    "act": row["act"],
                    "jurisdiction": row["jurisdiction"],
                    "year": int(row["year"]),
                    "why": row["counted"].removeprefix("no: "),
                }
            )
            continue
        key = (row["act"], row["jurisdiction"], row["year"], row["listed"])
        grouped.setdefault(key, []).append(row)
    enactments = [
        KeyEnactment(
            act=act,
            jurisdiction=jurisdiction,
            year=int(year),
            listed=listed,
            stratum=_stratum(group[0]["listed_status"]),
            rule=group[0]["rule"] or None,
            bill_ids=tuple(dict.fromkeys(r["bill_id"] for r in group if r["bill_id"])),
        )
        for (act, jurisdiction, year, listed), group in grouped.items()
    ]
    return LoadedKey(note, enactments, excluded)


def same_slug(acts: Iterable[str], source: str) -> dict[str, frozenset[str]]:
    """Each act's documents when a source names its documents by the act's slug."""
    return {act: frozenset({f"{source}/{act}"}) for act in acts}


def check_final(key: AnswerKey, final: bool) -> None:
    """Refuse a sealed key without `final`, and `final` on a key that is not sealed."""
    if key.sealed and not final:
        raise OriginRecallError(
            f"the {key.name} answer key is sealed: it is evaluated once matching is settled, "
            "once per configuration (DESIGN.md#commission-recall). Pass final for that run."
        )
    if final and not key.sealed:
        raise OriginRecallError(f"final applies only to a sealed key; {key.name} is not sealed")


def check_configuration(
    previous: Mapping[str, Any] | None, configuration: Mapping[str, Any]
) -> None:
    """Refuse a sealed run at a configuration it has already been run at: rerunning one
    settles nothing, and a second look at the same matches is tuning on them."""
    runs = [] if previous is None else [previous, *previous.get("earlier_runs", [])]
    for run in runs:
        if run.get("configuration") == dict(configuration):
            raise OriginRecallError(
                f"the sealed key was already run at this configuration ({dict(configuration)})"
            )


def _rank(m: OriginMatch, floor: float, scale: int) -> tuple[bool, float]:
    return (is_listed(m, floor, scale), m.model_share)


def _cell(records: Sequence[Mapping[str, Any]], key: str = "found") -> dict[str, Any]:
    found = sum(bool(r[key]) for r in records)
    n = len(records)
    return {
        "enactments": n,
        "found": found,
        "recall": round(found / n, 4) if n else None,
        "interval_95": [round(v, 4) for v in wilson(found, n)],
    }


def _breakdown(records: Sequence[Mapping[str, Any]], acts: Sequence[str]) -> dict[str, Any]:
    return {
        "pooled": _cell(records),
        "by_act": {a: _cell([r for r in records if r["act"] == a]) for a in acts},
        "by_stratum": {s: _cell([r for r in records if r["stratum"] == s]) for s in STRATA},
    }


def _best_matches(
    matches: Iterable[OriginMatch], wanted: Mapping[str, str], floor: float, scale: int
) -> dict[tuple[str, str], OriginMatch]:
    """Each bill's best match with each act `wanted` names a document of, by (bill, act)."""
    by_pair: dict[tuple[str, str], OriginMatch] = {}
    for m in matches:
        act = wanted.get(m.doc_id)
        if act is None:
            continue
        # Two documents of one act can both match a bill; the act's best match stands, a listed
        # one before any the floor leaves off.
        best = by_pair.get((m.bill_id, act))
        if best is None or _rank(m, floor, scale) > _rank(best, floor, scale):
            by_pair[m.bill_id, act] = m
    return by_pair


def _bill_record(
    bill_id: str, m: OriginMatch | None, *, min_share: float, floor: float, scale: int
) -> dict[str, Any]:
    return {
        "id": bill_id,
        "doc_id": m.doc_id if m else None,
        "stages": list(m.stages) if m else [],
        "model_share": m.model_share if m else None,
        "bill_share": m.bill_share if m else None,
        "listed": m is not None and is_listed(m, floor, scale),
        "found": m is not None and m.model_share > min_share and is_listed(m, floor, scale),
    }


def _miss_reason(bills: Sequence[Mapping[str, Any]]) -> MissReason:
    """Why an enactment none of whose bills was found was missed: how far its bills got."""
    if not bills:
        return "not_in_corpus"
    if any(b["doc_id"] is not None for b in bills):
        return "below_threshold"
    return "not_matched"


def _enactment_record(
    e: KeyEnactment,
    by_pair: Mapping[tuple[str, str], OriginMatch],
    *,
    min_share: float,
    floor: float,
    scale: int,
) -> dict[str, Any]:
    bills = [
        _bill_record(
            bill_id, by_pair.get((bill_id, e.act)), min_share=min_share, floor=floor, scale=scale
        )
        for bill_id in e.bill_ids
    ]
    hits = [b for b in bills if b["found"]]
    matched = [b for b in bills if b["doc_id"] is not None]
    best = max(hits, key=lambda b: b["model_share"]) if hits else None
    return {
        "act": e.act,
        "jurisdiction": e.jurisdiction,
        "state": e.state,
        "year": e.year,
        "listed": e.listed,
        "stratum": e.stratum,
        "match_rule": e.rule,
        "found": best is not None,
        "found_by_stage": {s: any(b["found"] and s in b["stages"] for b in bills) for s in STAGES},
        "found_by": best["id"] if best else None,
        "model_share": best["model_share"] if best else None,
        "bill_share": best["bill_share"] if best else None,
        # The larger share of its best match, found or not: what a miss fell short by.
        "best_share": max((max(b["model_share"], b["bill_share"]) for b in matched), default=None),
        "miss_reason": _miss_reason(bills) if best is None else None,
        "bills": bills,
    }


def _miss_counts(records: Sequence[Mapping[str, Any]]) -> dict[MissReason, int]:
    return dict(
        sorted(
            Counter(r["miss_reason"] for r in records if r["miss_reason"]).items(),
            key=lambda kv: MISS_REASONS.index(kv[0]),
        )
    )


def evaluate(
    matches: Iterable[OriginMatch],
    key: LoadedKey,
    docs: Mapping[str, Collection[str]],
    *,
    min_share: float = MIN_SHARE,
    floor: float = FLOOR,
    scale: int = REACH_SCALE,
) -> dict[str, Any]:
    """Recall of each act's enactments from the origin matches of the act's documents.

    `docs` names each act's documents (lineage.origins.modeldocs.ModelDoc.id); a source may hold
    more than one per act, an edition per revision, so any of them finds an enactment. A match finds
    one only when a page lists it, its larger share at `floor` or above
    (lineage.origins.matches.FLOOR); one below is a miss below the threshold, its best share kept,
    so the floor's cost is read among the misses."""
    acts = sorted({e.act for e in key.enactments})
    missing = [a for a in acts if not docs.get(a)]
    if missing:
        raise OriginRecallError(f"no model document named for {', '.join(missing)}")
    by_pair = _best_matches(matches, {doc: act for act in acts for doc in docs[act]}, floor, scale)
    records = [
        _enactment_record(e, by_pair, min_share=min_share, floor=floor, scale=scale)
        for e in key.enactments
    ]
    ablation = {
        s: _breakdown([r | {"found": r["found_by_stage"][s]} for r in records], acts)
        for s in STAGES
    } | {"both": _breakdown(records, acts)}
    return {
        "threshold": {
            "score": "share of the model's words in the sections the bill matches",
            "min_share": min_share,
            "floor": floor,
            "reach_scale": scale,
            "found_when": "score above min_share, and the larger weighted share at floor or above",
        },
        "documents": {a: sorted(docs[a]) for a in acts},
        "recall": _breakdown(records, acts),
        "stage_ablation": ablation,
        "recall_by_min_share": [
            {"min_share": s, "found": sum(r["found"] and r["model_share"] >= s for r in records)}
            for s in SHARE_SWEEP
        ],
        "miss_reasons": _miss_counts(records),
        "excluded": key.excluded,
        "enactments": records,
    }


def origin_recall(
    matches: Iterable[OriginMatch],
    key: AnswerKey,
    docs: Mapping[str, Collection[str]],
    *,
    final: bool = False,
    configuration: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
    min_share: float = MIN_SHARE,
    floor: float = FLOOR,
) -> dict[str, Any]:
    """The eval's results section for one answer key.

    A sealed key needs `final` and a `configuration`, what matching was set to, which its
    section records; `previous` is the key's section from an earlier run, whose configurations
    are refused and kept under `earlier_runs`."""
    check_final(key, final)
    if key.sealed:
        if configuration is None:
            raise OriginRecallError("a sealed run records its configuration; none was given")
        check_configuration(previous, configuration)
    loaded = load_key(key.path)
    result = evaluate(matches, loaded, docs, min_share=min_share, floor=floor)
    section: dict[str, Any] = {
        "command": COMMAND,
        "answer_key": {"name": key.name, "path": str(key.path), "read": loaded.note},
        "sealed": key.sealed,
        "status": key.status,
        "configuration": None if configuration is None else dict(configuration),
    } | result
    if key.sealed:
        section["earlier_runs"] = _earlier_runs(previous)
    return section


def _earlier_runs(previous: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if previous is None:
        return []
    summary = {
        "configuration": previous.get("configuration"),
        "recall": previous["recall"]["pooled"],
        "stage_ablation": {s: r["pooled"] for s, r in previous["stage_ablation"].items()},
        "found": sorted(f"{e['act']} {e['state']}" for e in previous["enactments"] if e["found"]),
    }
    return [*previous.get("earlier_runs", []), summary]


def matches_from_commission(
    section: Mapping[str, Any], doc_of: Callable[[str], str]
) -> list[OriginMatch]:
    """v1's commission results as origin matches: for each bill an enactment names whose
    family the act's query listed, a match of the act's document with the family's share as
    the model share and the family's stages.

    This is v1's own answer, put in the shape lane C's matcher writes, so the recall code can
    be shown to count what v1 counted before any origin match exists. v1 records neither the
    share of the bill's words matched nor the spans, so the bill share is 0 and there are no
    spans; recall reads neither."""
    out: dict[tuple[str, str], OriginMatch] = {}
    for e in section["enactments"]:
        for bill in e["bills"]:
            if bill["rank"] is None:
                continue
            match = OriginMatch(
                bill_id=bill["id"],
                doc_id=doc_of(e["act"]),
                stages=bill["stages"],
                model_share=bill["share"],
                bill_share=0.0,
                spans=[],
            )
            out[match.bill_id, match.doc_id] = match
    return [out[k] for k in sorted(out)]
