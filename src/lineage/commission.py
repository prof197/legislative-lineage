"""`lineage eval commission`: what share of a uniform act's enactments its text finds.

Each act's text is the query. An enactment is found when a bill its list row names (see
`lineage.ulc`) belongs to a family among the query's relatives with a score above the threshold.
The score is the share of the query's signed sections the family matches, and the commission's
standard closing sections are boilerplate, so they never count. Rank is reported, not used: an
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.acts import ACTS, ACTS_DIR, enacted_text
from lineage.bands import load_band_index
from lineage.lsh import BUCKET_CAP, band_keys
from lineage.minhash import NUM_PERM
from lineage.query import (
    bucket_sizes,
    matchable_sections,
    query_text,
    require_index,
    sign_query,
)
from lineage.semantic import SemanticSearch, best_cosines_to_bills
from lineage.sessions import Session, load_sessions
from lineage.signatures import open_signatures
from lineage.stats import AMENDMENT_CHARS, AMENDMENT_OPENING, AMENDMENT_SUBSTITUTE
from lineage.ulc import (
    ENACTMENTS,
    NO_BILLS,
    BillRecord,
    Enactment,
    Snapshot,
    canonical_number,
    development_bills,
    find_bills,
    load_enactments,
)
from lineage.verify import THRESHOLD, min_agreements

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
MISS_REASONS = (
    "not_in_corpus",
    "no_text_in_source",
    "amendment_only",
    "excluded_type",
    "no_usable_text",
    "not_proposed",
    "bucket_cap",
    "verification",
    "below_threshold",
)
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
) -> dict[str, Any]:
    """Recall of each act's enactments from its query text, with a reason for every miss.

    With `semantic`, both stages search, each enactment records which stages found it, and
    the stage ablation reports recall for each stage alone and for both together."""
    require_index(tables_dir)
    stages = ("near_duplicate", "semantic") if semantic is not None else ("near_duplicate",)
    rows = [r for r in snapshot.rows if r.stratum is not None]
    excluded = [
        (r, why)
        for r in rows
        if (why := excluded_reason(r, first_year, development, acts)) is not None
    ]
    counted = [r for r in rows if excluded_reason(r, first_year, development, acts) is None]
    unknown = sorted({r.act for r in counted} - set(query_texts))
    if unknown:
        raise CommissionError(f"no query text for {', '.join(unknown)}")

    with queries.connect_quiet() as con:
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
    by_id = {b.id: b for b in bills}
    years = session_years(sessions)
    records_for_matching = [
        BillRecord(b.id, b.state, b.session, b.identifier, b.status, b.status_date,
                   b.passage_chambers)
        for b in bills
    ]  # fmt: skip
    found = [find_bills(r, records_for_matching, years) for r in counted]
    target_ids = sorted({b.id for f in found for b in f.bills})

    files = open_signatures(tables_dir)
    with queries.connect_quiet() as con:
        located = queries.signature_rows_of(con, keys_paths=files.keys, bill_ids=target_ids)
    rows_of: dict[str, list[tuple[int, int]]] = {}
    sections_of: dict[str, set[int]] = {}
    for f, row, bill_id, ordinal in located:
        rows_of.setdefault(bill_id, []).append((f, row))
        sections_of.setdefault(bill_id, set()).add(ordinal)

    index = load_band_index(tables_dir)
    records: list[dict[str, Any]] = []
    query_stats: dict[str, dict[str, Any]] = {}
    for act in sorted(query_texts):
        result = query_text(query_texts[act], tables_dir, semantic=semantic)
        relatives = {
            m.family: Relative(
                rank, round(m.share, 4), m.identifier, frozenset(m.sections_by_stage)
            )
            for rank, m in enumerate(result.matches, start=1)
        }
        cosines: dict[str, float] = {}
        if semantic is not None:
            act_bills = [
                b.id
                for row, match in zip(counted, found, strict=True)
                if row.act == act
                for b in match.bills
            ]
            _, matchable = matchable_sections(query_texts[act])
            cosines = best_cosines_to_bills([t for t, _ in matchable], act_bills, semantic)
        signed = sign_query(query_texts[act], tables_dir).values
        sizes = bucket_sizes(band_keys(signed), index) if signed.shape[0] else None
        targets_by_family = {
            by_id[b.id].family
            for row, match in zip(counted, found, strict=True)
            if row.act == act
            for b in match.bills
            if by_id[b.id].family
        }
        query_stats[act] = {
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
        for row, match in zip(counted, found, strict=True):
            if row.act != act:
                continue
            targets: list[dict[str, Any]] = []
            reasons: list[MissReason] = []
            for bill in match.bills:
                record = by_id[bill.id]
                located_rows = rows_of.get(bill.id, [])
                probed = None
                if located_rows and sizes is not None:
                    values = np.stack([files.matrices[f][r] for f, r in located_rows])
                    probed = probe(signed, sizes, values)
                relative = relatives.get(record.family) if record.family else None
                reason = target_reason(
                    record,
                    signed_sections=len(sections_of.get(bill.id, ())),
                    probed=probed,
                    relative=relative,
                    min_share=min_share,
                )
                if reason is not None:
                    reasons.append(reason)
                targets.append(
                    {
                        "id": record.id,
                        "session": record.session,
                        "identifier": record.identifier,
                        "status": record.status,
                        "text_chars": record.text_chars,
                        "signed_sections": len(sections_of.get(bill.id, ())),
                        "stage": probed.stage if probed else None,
                        "best_estimate": round(probed.best_estimate, 4) if probed else None,
                        "best_cosine": (round(cosines[bill.id], 4) if bill.id in cosines else None),
                        "stages": sorted(relative.stages) if relative else [],
                        "rank": relative.rank if relative else None,
                        "share": relative.share if relative else None,
                        "found": relative is not None and relative.share > min_share,
                        "reason": reason,
                    }
                )
            hits = [t for t in targets if t["found"]]
            best = min(hits, key=lambda t: t["rank"]) if hits else None
            miss: MissReason | None = None
            if best is None:
                miss = max(reasons, key=MISS_REASONS.index) if reasons else "not_in_corpus"
            records.append(
                {
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
                        for stage in stages
                    },
                    "found_by": best["identifier"] if best else None,
                    "rank": best["rank"] if best else None,
                    "share": best["share"] if best else None,
                    "miss_reason": miss,
                    "bills": targets,
                }
            )

    strata = ("enacted", "substantially_similar")
    ablation = {
        stage: _both([r | {"found": r["found_by_stage"][stage]} for r in records])
        for stage in stages
    } | {"both": _both(records)}
    return {
        "stages": list(stages),
        "stage_ablation": ablation if semantic is not None else None,
        "threshold": {
            "score": "share of the query's signed sections that the family matches",
            "min_share": min_share,
            "found_when": "score above min_share",
            "status": "approved",
        },
        "verification_threshold": THRESHOLD,
        "bucket_cap": BUCKET_CAP,
        "first_year": first_year,
        "queries": query_stats,
        "recall": {
            "pooled": _both(records),
            "by_act": {
                a: _both([r for r in records if r["act"] == a]) for a in sorted(query_texts)
            },
            "by_stratum": {s: _both([r for r in records if r["stratum"] == s]) for s in strata},
        },
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
        "miss_reasons": dict(
            sorted(
                Counter(r["miss_reason"] for r in records if r["miss_reason"]).items(),
                key=lambda kv: MISS_REASONS.index(kv[0]),
            )
        ),
        "excluded": [
            {"act": r.act, "jurisdiction": r.jurisdiction, "year": r.year, "why": why}
            for r, why in excluded
        ],
        "enactments": records,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def commission_recall(
    tables_dir: Path,
    *,
    acts_dir: Path = ACTS_DIR,
    enactments_path: Path = ENACTMENTS,
    sessions_path: Path,
    act_ids: tuple[str, ...] = ACTS,
    development_path: Path | None = None,
    semantic: SemanticSearch | None = None,
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
