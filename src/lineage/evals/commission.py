"""`lineage eval commission`: what share of a uniform act's enactments its text finds.

Each act's text is the query. An enactment is found when a bill its list row names (see
`lineage.evals.ulc`) belongs to a family among the query's relatives with a score above the
threshold. The score is the share of the query's words in the sections the family matches
(DESIGN.md#bill-scoring), and the commission's standard closing sections are boilerplate, so
they never count. Rank is reported, not used: an
act's relatives can outnumber any fixed rank once every state's versions count. Every enactment
that is not found gets one miss reason, taken from the bill that got furthest through the
pipeline: no stored text, an amendment stored instead of the bill, an excluded document type,
no matchable section, never proposed by LSH, proposed only through a bucket over the cap,
dropped by verification, or scored below the threshold. The stage reasons are measured from
the bill's own signatures, with the band keys, cap and verification threshold the query uses.
Recall is reported pooled, per act and per stratum, over all enactments and over those with
text. See DESIGN.md#commission-recall.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import NDArray

from lineage.core import db
from lineage.data.sessions import Session, load_sessions
from lineage.data.stats import AMENDMENT_CHARS, AMENDMENT_OPENING, AMENDMENT_SUBSTITUTE
from lineage.evals import queries
from lineage.evals.acts import ACTS, ACTS_DIR, enacted_text
from lineage.evals.ulc import (
    ENACTMENTS,
    NO_BILLS,
    BillRecord,
    Enactment,
    Found,
    Snapshot,
    canonical_number,
    development_bills,
    find_bills,
    load_enactments,
)
from lineage.match.near.bands import load_band_index
from lineage.match.near.lsh import BUCKET_CAP, BandIndex, band_keys
from lineage.match.near.minhash import NUM_PERM
from lineage.match.near.signatures import SignatureFiles, open_signatures
from lineage.match.near.verify import THRESHOLD, min_agreements
from lineage.match.query import (
    QueryResult,
    bucket_sizes,
    matchable_sections,
    query_text,
    regrouped,
    require_index,
    sign_query,
)
from lineage.match.semantic.stage import (
    BUILD_THRESHOLD,
    CORROBORATING,
    SemanticSearch,
    best_cosines_to_bills,
)
from lineage.match.semantic.stage import THRESHOLD as THRESHOLD_SEMANTIC

COMMAND = "uv run lineage eval commission"
RESULTS = Path("reports/results.json")
# The corpus is the sessions of 2023-2026 (README.md#data); earlier enactments are listed but
# cannot be found, so they are left out of recall rather than counted as misses.
FIRST_YEAR = 2023
# Set from the measured scores: with the commission's standard sections as boilerplate,
# every relative of the three act texts is a version of that act (each act's relatives are in
# the results), so no score separates unrelated bills and one verified section is enough. Every
# found enactment's rank and share are in the results, and recall at higher thresholds is
# reported, so another threshold can be read from them without rerunning.
MIN_SHARE = 0.0
SHARE_SWEEP = (0.0, 0.1, 0.25, 0.5)

# Ordered by how far through the pipeline a bill got; an enactment's reason is its furthest bill's.
MissReason = Literal[
    "not_in_corpus",
    "no_text_in_source",
    "amendment_only",
    "excluded_type",
    "no_usable_text",
    "not_proposed",
    "bucket_cap",
    "verification",
    "below_threshold",
]
MISS_REASONS: tuple[MissReason, ...] = get_args(MissReason)
Stage = Literal["verified", "bucket_cap", "verification", "not_proposed"]


class CommissionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Probe:
    """How far a bill's sections got against the query's, stage by stage."""

    stage: Stage
    best_estimate: float  # the highest similarity of any query section to any of its sections


def probe(
    query: NDArray[np.uint32],
    query_sizes: NDArray[np.int64],
    target: NDArray[np.uint32],
    *,
    cap: int = BUCKET_CAP,
    threshold: float = THRESHOLD,
) -> Probe:
    """Replay LSH and verification for one bill's signatures against the query's.

    Both sides are signed by window, and a section pair is as far along as its furthest pair of
    windows. A pair is proposed when the two share a band key in a bucket no larger than the
    cap, and verified when its full signatures agree in enough positions. A bill blocked only by
    the cap is one whose pair would have verified had its bucket been under the cap.
    """
    agree = (query[:, None, :] == target[None, :, :]).sum(axis=2)
    shared = band_keys(query)[:, None, :] == band_keys(target)[None, :, :]
    proposed = (shared & (query_sizes <= cap)[:, None, :]).any(axis=2)
    capped = (shared & (query_sizes > cap)[:, None, :]).any(axis=2) & ~proposed
    passes = agree >= min_agreements(threshold)
    best = float(agree.max()) / NUM_PERM if agree.size else 0.0
    if (proposed & passes).any():
        return Probe("verified", best)
    if (capped & passes).any():
        return Probe("bucket_cap", best)
    if (proposed | capped).any():
        return Probe("verification", best)
    return Probe("not_proposed", best)


@dataclass(frozen=True, slots=True)
class Relative:
    rank: int
    share: float
    shown: str  # the identifier the family is shown through
    stages: frozenset[str] = frozenset()  # the stages that matched the family


def target_reason(
    bill: queries.EnactmentBill,
    *,
    signed_sections: int,
    probed: Probe | None,
    relative: Relative | None,
    min_share: float = MIN_SHARE,
) -> MissReason | None:
    """Why one bill was not found, or None when it was."""
    if relative is not None:
        return None if relative.share > min_share else "below_threshold"
    if bill.text_chars == 0:
        return "no_text_in_source"
    if bill.amendment_only:
        return "amendment_only"
    if not bill.eligible:
        return "excluded_type"
    if signed_sections == 0 or probed is None:
        return "no_usable_text"
    if probed.stage == "verified":
        raise CommissionError(f"{bill.id} verified against the query but its family is missing")
    return probed.stage


def _recall(records: list[dict[str, Any]]) -> dict[str, Any]:
    found = sum(r["found"] for r in records)
    return {
        "enactments": len(records),
        "found": found,
        "recall": round(found / len(records), 4) if records else None,
    }


def _both(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"all": _recall(records), "with_text": _recall([r for r in records if r["with_text"]])}


def excluded_reason(
    row: Enactment,
    first_year: int,
    development: frozenset[tuple[str, int, str]] = frozenset(),
    acts: frozenset[str] | None = None,
) -> str | None:
    """Why a list row is not counted, or None.

    `development` holds the development set's bills, which a held-out row may not share: a bill
    that enacted two acts was already studied. `acts`, when given, are the acts selected; the
    held-out lists also carry the reserve acts, kept for the record.
    """
    if acts is not None and row.act not in acts:
        return "a reserve act, not selected"
    if row.state is None:
        return "District of Columbia: the corpus is the 50 states"
    if row.year < first_year:
        return f"listed year {row.year} is before the corpus ({first_year}-2026)"
    if any((row.state, row.year, canonical_number(n)) in development for n in row.bill_numbers):
        return "names a bill of the development set"
    return None


def session_years(
    sessions: dict[tuple[str, str], Session],
) -> dict[tuple[str, str], tuple[int, int]]:
    return {
        key: (s.start_date.year, s.end_date.year)
        for key, s in sessions.items()
        if s.start_date is not None and s.end_date is not None
    }


@dataclass(frozen=True, slots=True)
class _Settings:
    """What every act's query is run and judged with."""

    tables_dir: Path
    stages: tuple[str, ...]
    semantic: SemanticSearch | None
    semantic_thresholds: Sequence[float]
    flagged: Collection[tuple[str, int]]
    min_share: float


@dataclass(frozen=True, slots=True)
class _Corpus:
    """The corpus bills the counted rows may name, which rows name which, and where each
    bill's signatures are."""

    by_id: dict[str, queries.EnactmentBill]
    found: list[Found]  # one per counted row, in order
    files: SignatureFiles
    rows_of: dict[str, list[tuple[int, int]]]  # bill id -> (signatures file, row)
    sections_of: dict[str, set[int]]  # bill id -> ordinals of its signed sections
    index: BandIndex


@dataclass(frozen=True, slots=True)
class _ActRun:
    """One act's query: what it matched, and what probing its enactments' bills needs."""

    result: QueryResult
    relatives: dict[str, Relative]  # by family
    families_at: dict[float, set[str]]  # the families found at each semantic threshold option
    cosines: dict[str, float]  # each enactment bill's best cosine to the query, by bill id
    signed: NDArray[np.uint32]
    sizes: NDArray[np.int64] | None  # the bucket size of each signed window's band keys


_Rows = list[tuple[Enactment, Found]]


def _counted(
    snapshot: Snapshot,
    first_year: int,
    development: frozenset[tuple[str, int, str]],
    acts: frozenset[str] | None,
) -> tuple[list[Enactment], list[tuple[Enactment, str]]]:
    """The list rows recall counts, and those it leaves out with the reason."""
    judged = [
        (r, excluded_reason(r, first_year, development, acts))
        for r in snapshot.rows
        if r.stratum is not None
    ]
    counted = [r for r, why in judged if why is None]
    excluded = [(r, why) for r, why in judged if why is not None]
    return counted, excluded


def _corpus(
    tables_dir: Path, counted: list[Enactment], sessions: dict[tuple[str, str], Session]
) -> _Corpus:
    with db.connect_quiet() as con:
        bills = queries.enactment_bills(
            con,
            bills_glob=str(tables_dir / "bills" / "*.parquet"),
            texts_glob=str(tables_dir / "texts" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
            families_path=tables_dir / "families" / "families.parquet",
            states=sorted({r.state for r in counted if r.state is not None}),
            opening=AMENDMENT_OPENING,
            chars=AMENDMENT_CHARS,
            substitute=AMENDMENT_SUBSTITUTE,
        )
    years = session_years(sessions)
    records_for_matching = [
        BillRecord(b.id, b.state, b.session, b.identifier, b.status, b.status_date,
                   b.passage_chambers)
        for b in bills
    ]  # fmt: skip
    found = [find_bills(r, records_for_matching, years) for r in counted]
    target_ids = sorted({b.id for f in found for b in f.bills})

    files = open_signatures(tables_dir)
    with db.connect_quiet() as con:
        located = queries.signature_rows_of(con, keys_paths=files.keys, bill_ids=target_ids)
    rows_of: dict[str, list[tuple[int, int]]] = {}
    sections_of: dict[str, set[int]] = {}
    for f, row, bill_id, ordinal in located:
        rows_of.setdefault(bill_id, []).append((f, row))
        sections_of.setdefault(bill_id, set()).add(ordinal)
    by_id = {b.id: b for b in bills}
    return _Corpus(by_id, found, files, rows_of, sections_of, load_band_index(tables_dir))


def _run_act(text: str, mine: _Rows, corpus: _Corpus, settings: _Settings) -> _ActRun:
    """Search once at the lowest threshold asked for, and regroup at each and at the stage's."""
    tables_dir, semantic = settings.tables_dir, settings.semantic
    lowest = min([THRESHOLD_SEMANTIC, *settings.semantic_thresholds])
    result = query_text(
        text, tables_dir, semantic=semantic, semantic_threshold=lowest, flagged=settings.flagged
    )
    if lowest < THRESHOLD_SEMANTIC:
        result = replace(result, matches=regrouped(result, tables_dir, THRESHOLD_SEMANTIC))
    families_at = {
        t: {m.family for m in regrouped(result, tables_dir, t) if m.share > settings.min_share}
        for t in settings.semantic_thresholds
    }
    relatives = {
        m.family: Relative(rank, round(m.share, 4), m.identifier, frozenset(m.sections_by_stage))
        for rank, m in enumerate(result.matches, start=1)
    }
    cosines: dict[str, float] = {}
    if semantic is not None:
        act_bills = [b.id for _, match in mine for b in match.bills]
        _, matchable = matchable_sections(text)
        cosines = best_cosines_to_bills([t for t, _ in matchable], act_bills, semantic)
    signed = sign_query(text, tables_dir).values
    sizes = bucket_sizes(band_keys(signed), corpus.index) if signed.shape[0] else None
    return _ActRun(result, relatives, families_at, cosines, signed, sizes)


def _query_stats(
    run: _ActRun, mine: _Rows, corpus: _Corpus, stages: Sequence[str]
) -> dict[str, Any]:
    result = run.result
    targets_by_family = {
        corpus.by_id[b.id].family
        for _, match in mine
        for b in match.bills
        if corpus.by_id[b.id].family
    }
    return {
        "sections": result.sections,
        "signed": result.signed,
        "searched": result.searched,
        "families": len(result.matches),
        "families_by_stage": {
            stage: sum(stage in m.sections_by_stage for m in result.matches) for stage in stages
        },
        # Every relative, so the threshold's premise can be checked by reading them.
        "relatives": [
            {
                "rank": rank,
                "share": round(m.share, 4),
                "sections_matched": m.sections_matched,
                "sections_by_stage": m.sections_by_stage,
                "state": m.state,
                "session": m.session,
                "identifier": m.identifier,
                "enactment": m.family in targets_by_family,
            }
            for rank, m in enumerate(result.matches, start=1)
        ],
    }


def _target(bill_id: str, run: _ActRun, corpus: _Corpus, min_share: float) -> dict[str, Any]:
    """One bill an enactment names: how far it got against the act's query, and why it was
    not found, if it was not."""
    record = corpus.by_id[bill_id]
    located_rows = corpus.rows_of.get(bill_id, [])
    probed = None
    if located_rows and run.sizes is not None:
        values = np.stack([corpus.files.matrices[f][r] for f, r in located_rows])
        probed = probe(run.signed, run.sizes, values)
    relative = run.relatives.get(record.family) if record.family else None
    signed_sections = len(corpus.sections_of.get(bill_id, ()))
    reason = target_reason(
        record,
        signed_sections=signed_sections,
        probed=probed,
        relative=relative,
        min_share=min_share,
    )
    return {
        "id": record.id,
        "session": record.session,
        "identifier": record.identifier,
        "status": record.status,
        "text_chars": record.text_chars,
        "signed_sections": signed_sections,
        "stage": probed.stage if probed else None,
        "best_estimate": round(probed.best_estimate, 4) if probed else None,
        "best_cosine": (round(run.cosines[bill_id], 4) if bill_id in run.cosines else None),
        "stages": sorted(relative.stages) if relative else [],
        "rank": relative.rank if relative else None,
        "share": relative.share if relative else None,
        "found": relative is not None and relative.share > min_share,
        "reason": reason,
    }


def _enactment(
    act: str, row: Enactment, match: Found, run: _ActRun, corpus: _Corpus, settings: _Settings
) -> dict[str, Any]:
    """One list row's record: found by its best-ranked bill, else its furthest bill's reason."""
    targets = [_target(b.id, run, corpus, settings.min_share) for b in match.bills]
    reasons: list[MissReason] = [t["reason"] for t in targets if t["reason"] is not None]
    hits = [t for t in targets if t["found"]]
    best = min(hits, key=lambda t: t["rank"]) if hits else None
    miss: MissReason | None = None
    if best is None:
        miss = max(reasons, key=MISS_REASONS.index) if reasons else "not_in_corpus"
    families = [corpus.by_id[b.id].family for b in match.bills]
    return {
        "act": act,
        "state": row.state,
        "year": row.year,
        "bill_cell": row.bill_cell,
        "stratum": row.stratum,
        "match_rule": match.rule if match.bills else None,
        "with_text": any(t["text_chars"] > 0 for t in targets),
        "found": best is not None,
        "found_by_stage": {
            stage: any(t["found"] and stage in t["stages"] for t in targets)
            for stage in settings.stages
        },
        "found_by": best["identifier"] if best else None,
        "found_at_semantic_threshold": {
            str(t): any(f in run.families_at[t] for f in families if f)
            for t in settings.semantic_thresholds
        },
        "rank": best["rank"] if best else None,
        "share": best["share"] if best else None,
        "miss_reason": miss,
        "bills": targets,
    }


_STRATA = ("enacted", "substantially_similar")


def _breakdown(records: list[dict[str, Any]], acts: Sequence[str]) -> dict[str, Any]:
    return {
        "pooled": _both(records),
        "by_act": {a: _both([r for r in records if r["act"] == a]) for a in acts},
        "by_stratum": {s: _both([r for r in records if r["stratum"] == s]) for s in _STRATA},
    }


def _recall_sweeps(
    records: list[dict[str, Any]], semantic_thresholds: Sequence[float]
) -> dict[str, Any]:
    """Recall as if the semantic stage's threshold, or the share found above, were another."""
    return {
        "recall_by_semantic_threshold": [
            {
                "threshold": t,
                "enactments": len(records),
                "found": sum(r["found_at_semantic_threshold"][str(t)] for r in records),
            }
            for t in semantic_thresholds
        ],
        "recall_by_min_share": [
            {
                "min_share": s,
                "found": sum(r["found"] and r["share"] >= s for r in records),
                "found_with_text": sum(
                    r["found"] and r["share"] >= s and r["with_text"] for r in records
                ),
            }
            for s in SHARE_SWEEP
        ],
    }


def _miss_reasons(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(r["miss_reason"] for r in records if r["miss_reason"]).items(),
            key=lambda kv: MISS_REASONS.index(kv[0]),
        )
    )


def _report(
    records: list[dict[str, Any]],
    query_stats: dict[str, dict[str, Any]],
    excluded: list[tuple[Enactment, str]],
    settings: _Settings,
    first_year: int,
) -> dict[str, Any]:
    """The results: recall broken down, by stage alone, and swept; the run's settings."""
    acts_run = sorted(query_stats)
    ablation = {
        stage: _breakdown([r | {"found": r["found_by_stage"][stage]} for r in records], acts_run)
        for stage in settings.stages
    } | {"both": _breakdown(records, acts_run)}
    semantic = settings.semantic is not None
    return {
        "stages": list(settings.stages),
        "stage_ablation": ablation if semantic else None,
        "threshold": {
            "score": "share of the query's words in the sections the family matches",
            "min_share": settings.min_share,
            "found_when": "score above min_share",
            "status": "approved",
        },
        "verification_threshold": THRESHOLD,
        **run_configuration(semantic, settings.flagged),
        "bucket_cap": BUCKET_CAP,
        "first_year": first_year,
        "queries": query_stats,
        "recall": _breakdown(records, acts_run),
        **_recall_sweeps(records, settings.semantic_thresholds),
        "miss_reasons": _miss_reasons(records),
        "excluded": [
            {"act": r.act, "jurisdiction": r.jurisdiction, "year": r.year, "why": why}
            for r, why in excluded
        ],
        "enactments": records,
    }


def evaluate(
    query_texts: dict[str, str],
    snapshot: Snapshot,
    tables_dir: Path,
    sessions: dict[tuple[str, str], Session],
    *,
    first_year: int = FIRST_YEAR,
    min_share: float = MIN_SHARE,
    development: frozenset[tuple[str, int, str]] = frozenset(),
    acts: frozenset[str] | None = None,
    semantic: SemanticSearch | None = None,
    semantic_thresholds: Sequence[float] = (),
    flagged: Collection[tuple[str, int]] = frozenset(),
) -> dict[str, Any]:
    """Recall of each act's enactments from its query text, with a reason for every miss.

    With `semantic`, both stages search, each enactment records which stages found it, and
    the stage ablation reports recall for each stage alone and for both together. With
    `semantic_thresholds`, each at or above the build's, recall is also counted as if the
    stage's threshold were each of them, for the threshold proposal: each query is searched
    once at the lowest, and its matches regrouped at each and at the stage's own
    (lineage.match.query.regrouped), so nothing else changes. Matches are listed as a page lists
    them, without one stage's hits on the `flagged` sections (lineage.match.listing)."""
    if any(t < BUILD_THRESHOLD for t in semantic_thresholds):
        raise CommissionError(
            f"a semantic threshold under the build's {BUILD_THRESHOLD} lists what the build "
            "never stored, so no page under it can be measured"
        )
    require_index(tables_dir)
    stages = ("near_duplicate", "semantic") if semantic is not None else ("near_duplicate",)
    settings = _Settings(tables_dir, stages, semantic, semantic_thresholds, flagged, min_share)
    counted, excluded = _counted(snapshot, first_year, development, acts)
    unknown = sorted({r.act for r in counted} - set(query_texts))
    if unknown:
        raise CommissionError(f"no query text for {', '.join(unknown)}")

    corpus = _corpus(tables_dir, counted, sessions)
    records: list[dict[str, Any]] = []
    query_stats: dict[str, dict[str, Any]] = {}
    for act in sorted(query_texts):
        mine = [(r, f) for r, f in zip(counted, corpus.found, strict=True) if r.act == act]
        run = _run_act(query_texts[act], mine, corpus, settings)
        query_stats[act] = _query_stats(run, mine, corpus, stages)
        records += [_enactment(act, row, match, run, corpus, settings) for row, match in mine]

    return _report(records, query_stats, excluded, settings, first_year)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_configuration(semantic: bool, flagged: Collection[tuple[str, int]]) -> dict[str, Any]:
    """What a run's listing was set to, as its results record it: the semantic stage's
    threshold, and the state boilerplate flag with the cosine its exemption counts from."""
    return {
        "semantic_threshold": THRESHOLD_SEMANTIC if semantic else None,
        "state_boilerplate_flag": (
            {"sections": len(flagged), "both_stages_exempt_from": CORROBORATING}
            if flagged
            else None
        ),
    }


def _configuration_of(run: dict[str, Any]) -> dict[str, Any]:
    return {k: run.get(k) for k in ("semantic_threshold", "state_boilerplate_flag")}


def check_held_out_configuration(
    previous: dict[str, Any] | None, configuration: dict[str, Any]
) -> None:
    """Refuse a held-out run at a configuration it has already been run at. The held-out acts
    were run once at the end of M5 and once more at the configuration the M5 review settled,
    each recorded (DESIGN.md#commission-recall); rerunning one settles nothing."""
    runs = [] if previous is None else [previous, *previous.get("earlier_runs", [])]
    for run in runs:
        if _configuration_of(run) == configuration:
            raise CommissionError(
                "the held-out acts were already run at this configuration "
                f"({configuration}); see DESIGN.md#commission-recall"
            )


def with_earlier_runs(section: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    """A held-out run's section, keeping each earlier run's configuration and what it found."""
    if previous is None:
        return section | {"earlier_runs": []}
    ablation: dict[str, dict[str, Any]] = previous.get("stage_ablation") or {}
    summary: dict[str, Any] = _configuration_of(previous) | {
        "recall": previous["recall"]["pooled"],
        "stage_ablation": {stage: r["pooled"] for stage, r in ablation.items()},
        "miss_reasons": previous["miss_reasons"],
        "found": sorted(f"{e['act']} {e['state']}" for e in previous["enactments"] if e["found"]),
    }
    return section | {"earlier_runs": [*previous.get("earlier_runs", []), summary]}


def commission_recall(
    tables_dir: Path,
    *,
    acts_dir: Path = ACTS_DIR,
    enactments_path: Path = ENACTMENTS,
    sessions_path: Path,
    act_ids: tuple[str, ...] = ACTS,
    development_path: Path | None = None,
    semantic: SemanticSearch | None = None,
    semantic_thresholds: Sequence[float] = (),
    flagged: Collection[tuple[str, int]] = frozenset(),
) -> dict[str, Any]:
    """The headline eval: each act's own text as the query.

    The held-out run passes its own acts and the development lists, whose bills it skips.
    """
    texts: dict[str, str] = {}
    acts: dict[str, dict[str, Any]] = {}
    for act in act_ids:
        path = acts_dir / f"{act}.txt"
        if not path.exists():
            raise CommissionError(
                f"{path} not found: see README.md#uniform-act-texts for how to get it"
            )
        raw = path.read_bytes()
        texts[act] = enacted_text(raw.decode("utf-8-sig"))
        acts[act] = {
            "file_sha256": _sha256(raw),
            "query_sha256": _sha256(texts[act].encode()),
            "words_in_file": len(raw.decode("utf-8-sig").split()),
            "words_kept": len(texts[act].split()),
        }
    snapshot = load_enactments(enactments_path)
    development = (
        development_bills(load_enactments(development_path)) if development_path else NO_BILLS
    )
    result = evaluate(
        texts,
        snapshot,
        tables_dir,
        load_sessions(sessions_path),
        development=development,
        # The held-out lists carry reserve acts; the development lists hold only the acts run.
        acts=frozenset(act_ids) if development_path else None,
        semantic=semantic,
        semantic_thresholds=semantic_thresholds,
        flagged=flagged,
    )
    for act, stats in acts.items():
        result["queries"][act] = stats | result["queries"][act]
    return {
        "command": COMMAND,
        "enactments_file": {
            "path": str(enactments_path),
            "read": snapshot.read_note,
            "sha256": _sha256(enactments_path.read_bytes()),
        },
    } | result


def write_results(path: Path, name: str, section: dict[str, Any]) -> None:
    """Replace one eval's section of the results file, keeping the others."""
    doc: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    doc[name] = section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(doc.items())), indent=1, ensure_ascii=False) + "\n")
