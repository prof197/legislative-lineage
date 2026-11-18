"""The precision sample for origins: which origin matches are right, and what labeling them costs.

Recall has an answer key (lineage.origins.recall); precision needs labels, so it is measured on a
sample of the origin matches table (lineage.origins.matches), drawn the way v1 draws its pair sample
(lineage.evals.precision.draw): the matches fall into three strata by the stages that found them,
each stratum is sampled apart without replacement under a fixed seed, and the drawn matches are
shuffled into one labeling order. A sample row names a match by bill and document and holds no
text, of the bill or of the model.

The tuning sample is stratified otherwise, by share, since it is read to set the threshold: a
floor on a match's larger share, of the model's words or of the bill's, below which a page would
not list it. Its strata are the bands between the candidate floors (FLOORS), each drawn apart,
so every candidate's precision rests on labels drawn from what it would list, and the band just
above each floor, where a floor's choice is decided, is never left to a few labels: v1's first
tuning sample, drawn from long lists below the candidates, held 3 relatives where the threshold
was chosen (DESIGN.md#precision-sample).

There are two samples (PARTS), each under its own seed, as v1 learned to keep them
(DESIGN.md#precision-sample): v1's first sample was read to choose a threshold and then
overstated precision there. The tuning sample may be drawn and read while matching is set. The
reporting sample is drawn only once matching is final, from matches the tuning sample did not
draw, and only once; what it says is the published number.

Labels would come from a model, as v1's provisional labels do (lineage.evals.model_labels). Nothing
here asks one. A run is projected before anything could be sent, as v1 projects its label runs:
each prompt's tokens from its characters at CHARS_PER_TOKEN, each reply at its token limit,
thinking included, and half that when sent through the Batches API. The origin prompt is not
written yet, so v1's passage prompt (passages-v1-high) stands in for it: its system text, its
limit, and as the message the whole model document beside every bill section the match's spans
touch, or every matchable section of the bill when it has none, which errs high. A run projected
over MAX_COST, or over v2's BUDGET together with what v2's ledger records as spent, is refused.
The exact projection, from the prompts as they would be sent, is lineage.origins.labels'.

The reporting part draws only what a page lists, the matches at lineage.origins.matches.FLOOR, and
is read once (`reading`): overall, by stage, each source's, and one source's against the rest,
each weighted by that source's matches per stage, since errors concentrated on one
organization would be unfair to it whatever the overall figure. See DESIGN.md#precision-sample.

See DESIGN.md#dr-12.
"""

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from lineage.core import db
from lineage.core.intervals import wilson
from lineage.evals.model_labels import MAX_COST, MODEL_ID, PASSAGES_HIGH
from lineage.evals.model_labels import spent as label_spent
from lineage.evals.precision import PER_STRATUM, draw, stratified
from lineage.evals.precision import SEED as PRECISION_SEED
from lineage.explain.spend import PRICES, upper_bound
from lineage.match.strata import STRATA, PrecisionError, stratum_of
from lineage.origins import queries
from lineage.origins.matches import FLOOR, OriginMatch, is_listed, share_of
from lineage.origins.modeldocs import MODELDOCS_DIR, ModelDoc, load_text

COMMAND = "uv run lineage origin-sample"
# v1's samples took the precision seed + 0 to + 5 (lineage.evals.relatives_sample).
SEED = PRECISION_SEED + 6
REPORTING_SEED = PRECISION_SEED + 7
REPORT = Path("reports/origin_precision_sample.json")
# v2's model calls are capped at $25 together, spent and projected, in v2's own ledger, apart
# from v1's precision labels' (lineage.evals.model_labels.SPEND).
SPEND = Path("reports/origin_label_spend.json")
BUDGET = 25.0
# The question an origin's label would ask is v1's passage question: do the passages state the
# same provision, beyond boilerplate? Its prompt stands in for the projection until the origin
# prompt is written.
STAND_IN = PASSAGES_HIGH

MatchKey = tuple[str, str]  # (bill id, document id)
# The candidate floors on a match's larger share, and the tuning sample's draw from each band
# between them: four bands of 50 is 200 labels, about $6.50 at v1's measured passage-label cost.
FLOORS = (0.0, 0.05, 0.10, 0.25)
PER_BAND = 50


def band_name(floor: float) -> str:
    return f"share_from_{floor:.2f}"


BANDS = tuple(band_name(f) for f in FLOORS)


def band_of(share: float) -> str:
    """The band from the highest floor at or below `share`."""
    return band_name(max(f for f in FLOORS if f <= share))


class OriginPrecisionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Part:
    """One sample: its seed, its first pair id's number, its file, whether it may be drawn only
    once matching is final, and the parts whose matches it never draws."""

    name: str
    seed: int
    first: int
    sample: Path
    final_only: bool
    excludes: tuple[str, ...] = ()
    by_share: bool = False  # stratified by share band (BANDS), not by stage
    floor: float | None = None  # drawn only from the matches a page lists at this floor
    source: str | None = None  # drawn only from this source's matches
    census: bool = False  # every match it may draw, not a sample
    # Its labels are found by the sections its rows rest on, whichever part's row took them
    # (lineage.origins.labels.correct_by_sections): drawn after matching changed, it may share
    # matches with a census, and only a label on the sections a row now rests on answers it.
    by_sections: bool = False

    @property
    def strata(self) -> tuple[str, ...]:
        return BANDS if self.by_share else tuple(STRATA)

    @property
    def per_stratum(self) -> int:
        return PER_BAND if self.by_share else PER_STRATUM


_TUNING = Part(
    "tuning", SEED, 1, Path("evals/origin_precision_tuning.jsonl"), final_only=False, by_share=True
)
PARTS: dict[str, Part] = {
    "tuning": _TUNING,
    "reporting": Part(
        "reporting",
        REPORTING_SEED,
        # Numbered after every id the tuning sample can take, so no id names two matches.
        len(STRATA) * PER_STRATUM + 1,
        Path("evals/origin_precision_reporting.jsonl"),
        final_only=True,
        excludes=(_TUNING.name,),
        # What the published precision describes is what a page lists.
        floor=FLOOR,
    ),
    # Ruled at the step 47 review: before the parked fix is built, every EPPC match a page
    # lists is labeled, since its reporting labels (one correct of four, the errors one
    # document's) were too few to read. Numbered after every id the reporting sample can take.
    # Ruled at the step 47 review: generic sections are left out of matches, so the published
    # precision is read again, on a fresh sample drawn once from the table matched after the fix,
    # apart from both earlier samples' matches; the census's are left in, since leaving out every
    # EPPC match listed would leave EPPC none to read. Numbered after every id the census can take.
    "reporting_after_fix": Part(
        "reporting_after_fix",
        REPORTING_SEED + 2,
        1001,
        Path("evals/origin_precision_reporting_after_fix.jsonl"),
        final_only=True,
        excludes=("tuning", "reporting"),
        floor=FLOOR,
        by_sections=True,
    ),
    # Ruled at the step 48 review: generic text inside a section is left out of matches too, the
    # last change to matching before launch, so precision is read once more, on a fresh sample
    # drawn once from the table matched after it, apart from every earlier reporting and tuning
    # sample's matches, and published whatever it says. Numbered after every id the first
    # sample after the fix took.
    "reporting_after_compact_fix": Part(
        "reporting_after_compact_fix",
        REPORTING_SEED + 3,
        1301,
        Path("evals/origin_precision_reporting_after_compact_fix.jsonl"),
        final_only=True,
        excludes=("tuning", "reporting", "reporting_after_fix"),
        floor=FLOOR,
        by_sections=True,
    ),
    "census_eppc": Part(
        "census_eppc",
        REPORTING_SEED + 1,
        2 * len(STRATA) * PER_STRATUM + 1,
        Path("evals/origin_precision_census_eppc.jsonl"),
        final_only=False,
        floor=FLOOR,
        source="eppc",
        census=True,
    ),
}


def match_key(m: OriginMatch) -> MatchKey:
    return (m.bill_id, m.doc_id)


def by_stratum(
    matches: Iterable[OriginMatch], *, by_share: bool = False
) -> dict[str, list[OriginMatch]]:
    """Every match in exactly one stratum, by stage or by share band, each stratum in key
    order, so a draw repeats over the same table whatever order it was written in."""
    out: dict[str, list[OriginMatch]] = {name: [] for name in (BANDS if by_share else STRATA)}
    seen: set[MatchKey] = set()
    for m in sorted(matches, key=match_key):
        if match_key(m) in seen:
            raise OriginPrecisionError(f"{m.bill_id} matches {m.doc_id} twice in the table")
        seen.add(match_key(m))
        try:
            stage = stratum_of(m.stages)
        except PrecisionError as e:
            raise OriginPrecisionError(f"{m.bill_id} x {m.doc_id}: {e}") from e
        out[band_of(share_of(m)) if by_share else stage].append(m)
    return out


def draw_part(
    matches: Iterable[OriginMatch],
    part: Part,
    *,
    excluded: Collection[MatchKey] = frozenset(),
    final: bool = False,
    per_stratum: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The part's sample rows in labeling order, and its report: the seed, and each stratum's
    size in the table, which an estimate weights by."""
    if part.final_only and not final:
        raise OriginPrecisionError(
            f"the {part.name} sample is drawn only once matching is final; pass final for it"
        )
    per = part.per_stratum if per_stratum is None else per_stratum
    if part.floor is not None:
        matches = [m for m in matches if is_listed(m, part.floor)]
    if part.source is not None:
        matches = [m for m in matches if source_of(m.doc_id) == part.source]
    strata = by_stratum(matches, by_share=part.by_share)
    if part.census:
        per = max([len(ms) for ms in strata.values()], default=0)
    eligible = {
        name: [m for m in ms if match_key(m) not in excluded] for name, ms in strata.items()
    }
    population = {name: np.arange(len(ms), dtype=np.uint64) for name, ms in eligible.items()}
    drawn = draw(population, per_stratum=per, seed=part.seed, names=part.strata)
    rows: list[dict[str, Any]] = []
    for n, (name, index) in enumerate(drawn):
        m = eligible[name][index]
        rows.append(
            {
                # Numbered in labeling order, so the id tells nothing of the stratum.
                "pair": f"o{part.first + n:04d}",
                "stratum": name,
                "stages": list(STRATA[stratum_of(m.stages)]),
                "bill_id": m.bill_id,
                "doc_id": m.doc_id,
                "ordinals": sorted({s.ordinal for s in m.spans}),
                "model_share": m.model_share,
                "bill_share": m.bill_share,
            }
        )
    report: dict[str, Any] = {
        "command": f"{COMMAND} --part {part.name}",
        "seed": part.seed,
        "per_stratum": per,
        "strata": {name: len(ms) for name, ms in strata.items()},
        "excluded": {name: len(strata[name]) - len(eligible[name]) for name in part.strata},
        "drawn": {name: sum(r["stratum"] == name for r in rows) for name in part.strata},
    }
    if part.by_share:
        report["floors"] = list(FLOORS)
    if part.source is not None:
        report |= {"source": part.source, "census": part.census}
    if part.floor is not None:
        report["floor"] = part.floor
        # Each source's matches by stratum, which a reading by source weights by.
        report["by_source"] = {
            source: {
                name: sum(source_of(m.doc_id) == source for m in ms) for name, ms in strata.items()
            }
            for source in sorted({source_of(m.doc_id) for ms in strata.values() for m in ms})
        }
    return rows, report


def source_of(doc_id: str) -> str:
    """The source a document id names: "alec/some-act" -> "alec"."""
    return doc_id.split("/", 1)[0]


def load_sample(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def excluded_keys(part: Part, root: Path = Path()) -> set[MatchKey]:
    """The matches the parts `part` excludes drew; each must have been drawn first."""
    keys: set[MatchKey] = set()
    for name in part.excludes:
        path = root / PARTS[name].sample
        if not path.exists():
            raise OriginPrecisionError(f"draw the {name} sample before the {part.name} sample")
        keys |= {(r["bill_id"], r["doc_id"]) for r in load_sample(path)}
    return keys


def write_part(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    part: Part,
    *,
    root: Path = Path(),
    report_path: Path = REPORT,
) -> Path:
    """Write the part's sample, and its section of the report. A final-only part is drawn
    once: a second draw would let its reading be chosen."""
    path = root / part.sample
    if part.final_only and path.exists():
        raise OriginPrecisionError(f"the {part.name} sample is drawn once; {path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report_file = root / report_path
    doc: dict[str, Any] = json.loads(report_file.read_text()) if report_file.exists() else {}
    doc[part.name] = dict(report)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(dict(sorted(doc.items())), indent=1) + "\n")
    return path


def section_chars(tables_dir: Path, bill_ids: Sequence[str]) -> dict[str, dict[int, int]]:
    """Each bill's matchable sections' lengths in characters, by ordinal."""
    with db.connect_quiet() as con:
        rows = queries.matchable_section_texts(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
            ids=sorted(set(bill_ids)),
        )
    out: dict[str, dict[int, int]] = {}
    for bill_id, ordinal, text in rows:
        out.setdefault(bill_id, {})[ordinal] = len(text)
    return out


def doc_chars(docs: Iterable[ModelDoc], root: Path = MODELDOCS_DIR) -> dict[str, int]:
    """Each document's cleaned text's length in characters."""
    return {d.id: len(load_text(d, root)) for d in docs}


def prompt_chars(
    rows: Sequence[Mapping[str, Any]],
    *,
    sections: Mapping[str, Mapping[int, int]],
    docs: Mapping[str, int],
    system: str = STAND_IN.system,
) -> list[int]:
    """Each row's prompt in characters: the system text, the whole model document, and the
    bill sections its spans touch, or all of the bill's when it has none."""
    out: list[int] = []
    for r in rows:
        if r["doc_id"] not in docs:
            raise OriginPrecisionError(f"no text for {r['doc_id']}, so its cost is unknown")
        bill = sections.get(r["bill_id"])
        if not bill:
            raise OriginPrecisionError(f"no matchable section of {r['bill_id']}")
        ordinals = r["ordinals"] or sorted(bill)
        missing = [o for o in ordinals if o not in bill]
        if missing:
            raise OriginPrecisionError(f"{r['bill_id']} has no matchable section {missing[0]}")
        out.append(len(system) + docs[r["doc_id"]] + sum(bill[o] for o in ordinals))
    return out


def projected_cost(
    chars: Sequence[int],
    *,
    model: str = MODEL_ID,
    max_tokens: int = STAND_IN.max_tokens,
    batch: bool = False,
) -> float:
    """At most what asking each prompt once costs, as lineage.evals.model_labels.projected_cost
    counts it: its tokens from its characters, its reply at the token limit, at half the price
    through the Batches API."""
    if model not in PRICES:
        raise OriginPrecisionError(f"no price for {model}; add it to spend.PRICES first")
    cost = upper_bound(model, chars, max_tokens, attempts=1)
    return cost / 2 if batch else cost


@dataclass(frozen=True, slots=True)
class LabelPlan:
    """What labeling a sample would ask and cost; nothing is asked."""

    pairs: int
    model: str
    prompt: str  # the prompt the projection stands in
    batch: bool
    projected: float
    spent: float
    max_cost: float
    budget: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "model": self.model,
            "prompt": self.prompt,
            "batch": self.batch,
            "projected": round(self.projected, 4),
            "spent": round(self.spent, 4),
            "max_cost": self.max_cost,
            "budget": self.budget,
        }


def plan_labels(
    chars: Sequence[int],
    *,
    model: str = MODEL_ID,
    batch: bool = False,
    spend_path: Path = SPEND,
    max_cost: float = MAX_COST,
    budget: float = BUDGET,
) -> LabelPlan:
    """The projection, beside what the label ledger records as spent."""
    return LabelPlan(
        pairs=len(chars),
        model=model,
        prompt=STAND_IN.version,
        batch=batch,
        projected=projected_cost(chars, model=model, batch=batch),
        spent=label_spent(spend_path),
        max_cost=max_cost,
        budget=budget,
    )


def refuse_over_cap(plan: LabelPlan) -> None:
    """Refuse a run projected over the run's cap, or over the budget with what is spent, as
    `lineage label-model --live` refuses one before anything is sent."""
    if plan.projected > plan.max_cost:
        raise OriginPrecisionError(
            f"{plan.pairs} matches projected at most ${plan.projected:,.2f}, over the "
            f"${plan.max_cost:,.2f} cap; nothing asked"
        )
    if plan.spent + plan.projected > plan.budget:
        raise OriginPrecisionError(
            f"{plan.pairs} matches projected at most ${plan.projected:,.2f}, with "
            f"${plan.spent:,.2f} spent, over the ${plan.budget:,.2f} budget; nothing asked"
        )


def estimate(
    rows: Sequence[Mapping[str, Any]], correct: Mapping[str, bool], sizes: Mapping[str, int]
) -> dict[str, Any]:
    """Each stratum's precision with its Wilson interval, and the overall precision with the
    strata weighted by their sizes in the table (lineage.evals.precision.stratified), from the rows
    labeled in `correct`, by pair id; the strata are those `sizes` names, by stage or by band.
    The overall is None until every stratum with matches has a label."""
    per: dict[str, dict[str, Any]] = {}
    parts: list[tuple[int, int, int]] = []
    for name in sizes:
        labeled = [
            correct[r["pair"]] for r in rows if r["stratum"] == name and r["pair"] in correct
        ]
        n, x = len(labeled), sum(labeled)
        per[name] = {
            "size": sizes.get(name, 0),
            "labeled": n,
            "correct": x,
            "precision": round(x / n, 4) if n else None,
            "interval_95": [round(v, 4) for v in wilson(x, n)],
        }
        parts.append((sizes.get(name, 0), n, x))
    return {"strata": per, "overall": stratified(parts)}


def census_reading(
    rows: Sequence[Mapping[str, Any]],
    correct: Mapping[MatchKey, bool],
    sections: Mapping[MatchKey, Sequence[Mapping[str, Any]]],
    *,
    source: str,
) -> dict[str, Any]:
    """A census read: every listed match of a source labeled, so its precision is a count and
    needs no interval. A match's label may come from any part, since a label answers the
    match's strongest pair whichever sample drew it. Each wrong match is listed with the
    section pairs it rests on (`sections`, by match). Refused until every match is labeled."""
    keys = [(str(r["bill_id"]), str(r["doc_id"])) for r in rows]
    missing = [k for k in keys if k not in correct]
    if missing:
        raise OriginPrecisionError(
            f"{len(missing)} of the {len(rows)} matches have no label yet, {missing[0]} first"
        )
    by_stage: dict[str, dict[str, int]] = {name: {"matches": 0, "correct": 0} for name in STRATA}
    by_document: dict[str, dict[str, int]] = {}
    wrong: list[dict[str, Any]] = []
    for r, k in zip(rows, keys, strict=True):
        ok = correct[k]
        for cell in (
            by_stage[str(r["stratum"])],
            by_document.setdefault(k[1], {"matches": 0, "correct": 0}),
        ):
            cell["matches"] += 1
            cell["correct"] += ok
        if not ok:
            wrong.append(
                {
                    "pair": r["pair"],
                    "bill_id": k[0],
                    "doc_id": k[1],
                    "stratum": r["stratum"],
                    "model_share": r["model_share"],
                    "bill_share": r["bill_share"],
                    "sections": list(sections.get(k, [])),
                }
            )
    n, x = len(rows), sum(correct[k] for k in keys)
    return {
        "source": source,
        "floor": FLOOR,
        "census": True,
        "matches": n,
        "correct": x,
        "precision": round(x / n, 4) if n else None,
        "by_stage": by_stage,
        "by_document": dict(sorted(by_document.items())),
        "wrong": sorted(wrong, key=lambda w: (w["doc_id"], w["bill_id"])),
    }


def without_intervals(node: Any) -> Any:
    """A reading with every interval left out: what must not move when only its intervals are
    regenerated."""
    if isinstance(node, Mapping):
        return {
            str(k): without_intervals(v)
            for k, v in cast(Mapping[str, Any], node).items()
            if k != "interval_95"
        }
    if isinstance(node, list):
        return [without_intervals(v) for v in cast(list[Any], node)]
    return node


# The reporting reading's comparison: errors concentrated on one organization would be unfair
# to it, whatever the overall figure, so its precision is read against every other source's.
COMPARED = "alec"
RESULTS_SECTION = "origins_precision"
# Each reporting part's reading in the results. The published one is the last part's; each
# earlier reading, taken before a later change to matching, is kept under its own name.
RESULTS_SECTIONS = {
    "reporting": "origins_precision_before_fix",
    "reporting_after_fix": "origins_precision_before_compact_fix",
    "reporting_after_compact_fix": RESULTS_SECTION,
}


def part_of(section: Mapping[str, Any]) -> str:
    """The reporting part a stored reading was read from, by the command it records."""
    command = str(section.get("command", ""))
    if "--part " not in command:
        raise OriginPrecisionError(f"a stored reading names no part: {command}")
    return command.split("--part ", 1)[1].split()[0]


def retired(results: Mapping[str, Any]) -> dict[str, Any]:
    """The results with each stored reading under its own part's section: a reading published
    before a later part existed moves to the name its part now keeps, unchanged, and is never
    read again. A reading whose new place is taken is refused."""
    out = dict(results)
    for name in set(RESULTS_SECTIONS.values()):
        if name not in out:
            continue
        home = RESULTS_SECTIONS.get(part_of(out[name]))
        if home is None or home == name:
            continue
        if home in out:
            raise OriginPrecisionError(f"{name} holds {home}'s reading, and {home} is taken")
        out[home] = out.pop(name)
    return out


def reading(
    rows: Sequence[Mapping[str, Any]],
    correct: Mapping[str, bool],
    report: Mapping[str, Any],
    *,
    compared: str = COMPARED,
) -> dict[str, Any]:
    """The reporting sample read: precision overall and by stage, each source's, and one
    source's against the rest, every figure weighted by stage as the table holds it (the
    report's `strata` and `by_source`), with its Wilson interval. Refused until every row has a
    label, since the sample is read once."""
    unlabeled = [r["pair"] for r in rows if r["pair"] not in correct]
    if unlabeled:
        raise OriginPrecisionError(
            f"{len(unlabeled)} of the {len(rows)} sampled matches have no label yet, "
            f"{unlabeled[0]} first; the reporting sample is read once, when all are labeled"
        )
    by_source: Mapping[str, Mapping[str, int]] = report["by_source"]
    if compared not in by_source:
        raise OriginPrecisionError(f"no match of {compared} is listed, so none can be compared")

    def group(sources: Collection[str]) -> dict[str, Any]:
        sizes = {name: sum(by_source[s][name] for s in sources) for name in STRATA}
        kept = [r for r in rows if source_of(str(r["doc_id"])) in sources]
        est = estimate(kept, correct, sizes)
        return {
            "sources": sorted(sources),
            "matches": sum(sizes.values()),
            "labeled": len(kept),
            "correct": sum(correct[r["pair"]] for r in kept),
            "precision": est["overall"],
            "by_stage": est["strata"],
        }

    rest = [s for s in by_source if s != compared]
    overall = group(list(by_source))
    return {
        "floor": report["floor"],
        "seed": report["seed"],
        "overall": overall,
        "by_source": {s: group([s]) for s in by_source},
        "compared": {compared: group([compared]), "rest": group(rest)},
    }
