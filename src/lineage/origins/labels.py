"""Origin precision labels: each sampled origin match judged by a model on its strongest pair.

A match rests on section pairs, a model section and a bill section a stage matched
(lineage.origins.matches.SectionPair). It is judged, as v1 judges a listed relative
(DESIGN.md#precision-sample), on the pair its listing rests on most: the near-duplicate pair with
the most band agreements, since copied text is what a page highlights, and failing that the pair
found by meaning at the best cosine. The two sections are aligned (lineage.align) and the model
is asked v1's passage-level question, unchanged (lineage.evals.model_labels.PASSAGES_HIGH): do the
matched passages state the same provision, beyond boilerplate? Unchanged so that its labels mean
what v1's mean, the question a person's labels were checked against at the cutoff. The model's
section is the first, the bill's the second, as a page lists the bill under the model. Neither
is named or dated, so the judge sees no stage, source or share.

A label row names its match by bill and document and each section by the SHA-256 of its
normalized text, and holds no text. Spend goes in v2's own ledger, under v2's cap for model
calls (lineage.origins.precision.BUDGET), apart from v1's precision-label ledger. See
DESIGN.md#origin-matches.

See DESIGN.md#dr-12.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lineage.core import db
from lineage.evals.labels import (
    Item,
    Pair,
    Section,
    SectionKey,
    SectionRef,
    is_correct,
    model_labeler,
    text_sha256,
)
from lineage.evals.model_labels import (
    MAX_COST,
    MODEL_ID,
    PASSAGES_HIGH,
    Batches,
    batch_usage,
    projected_cost,
    record_spend,
    request_params,
    spent,
)
from lineage.match.query import matchable_sections
from lineage.origins import queries
from lineage.origins.matches import OriginMatch, SectionPair
from lineage.origins.modeldocs import MODELDOCS_DIR, ModelDoc, load_text
from lineage.origins.precision import BUDGET, SPEND, MatchKey, OriginPrecisionError, match_key
from lineage.text.generic_sections import generic_kind

PROMPT = PASSAGES_HIGH
COMMAND = "uv run lineage origin-label --live"
LABELS = Path("evals/origin_precision_labels.jsonl")


def strongest_pair(m: OriginMatch) -> SectionPair:
    """The near-duplicate pair with the most agreements, else the best cosine's; ties go to
    the lowest sections."""
    near = [p for p in m.pairs if p.agreements]
    if near:
        return min(near, key=lambda p: (-(p.agreements or 0), p.model_section, p.ordinal))
    by_meaning = [p for p in m.pairs if p.cosine is not None]
    if not by_meaning:
        raise OriginPrecisionError(
            f"{m.bill_id} x {m.doc_id} keeps no section pairs; run `lineage origins` again"
        )
    return min(by_meaning, key=lambda p: (-(p.cosine or 0.0), p.model_section, p.ordinal))


@dataclass(frozen=True, slots=True)
class OriginItem:
    """A sampled match's strongest pair as the judge reads it: the model's section first."""

    row: Mapping[str, Any]  # the sample row
    item: Item

    @property
    def pair(self) -> str:
        return str(self.row["pair"])


def build_items(
    rows: Sequence[Mapping[str, Any]],
    matches: Iterable[OriginMatch],
    docs: Iterable[ModelDoc],
    tables_dir: Path,
    *,
    root: Path = MODELDOCS_DIR,
) -> list[OriginItem]:
    """Each sample row's strongest pair with both sections' texts, from the matches table, the
    document store and the sections table; no passages yet (lineage.evals.relatives_sample
    .with_passages aligns them)."""
    by_key: dict[MatchKey, OriginMatch] = {match_key(m): m for m in matches}
    held = {d.id: d for d in docs}
    chosen: list[tuple[Mapping[str, Any], SectionPair]] = []
    for r in rows:
        m = by_key.get((r["bill_id"], r["doc_id"]))
        if m is None:
            raise OriginPrecisionError(
                f"{r['pair']}: {r['bill_id']} x {r['doc_id']} is not in the matches table"
            )
        chosen.append((r, strongest_pair(m)))
    model_sections: dict[str, list[tuple[str, str]]] = {}
    for doc_id in sorted({str(r["doc_id"]) for r, _ in chosen}):
        if doc_id not in held:
            raise OriginPrecisionError(f"no source holds {doc_id}")
        _, matchable = matchable_sections(load_text(held[doc_id], root))
        model_sections[doc_id] = matchable
    with db.connect_quiet() as con:
        found = queries.matchable_sections_of(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
            ids=sorted({str(r["bill_id"]) for r, _ in chosen}),
        )
    bill_sections = {(b, o): (display, normalized) for b, o, display, normalized in found}
    out: list[OriginItem] = []
    for r, p in chosen:
        doc_id, bill_id = str(r["doc_id"]), str(r["bill_id"])
        sections = model_sections[doc_id]
        if p.model_section >= len(sections):
            raise OriginPrecisionError(f"{doc_id} has no section {p.model_section} now")
        if (bill_id, p.ordinal) not in bill_sections:
            raise OriginPrecisionError(f"{bill_id} has no matchable section {p.ordinal}")
        m_display, m_normalized = sections[p.model_section]
        b_display, b_normalized = bill_sections[(bill_id, p.ordinal)]
        model_ref = SectionRef(doc_id, None, p.model_section, text_sha256(m_normalized))
        bill_ref = SectionRef(bill_id, None, p.ordinal, text_sha256(b_normalized))
        item = Item(
            Pair(
                str(r["pair"]),
                tuple(r["stages"]),
                SectionKey(doc_id, p.model_section),
                SectionKey(bill_id, p.ordinal),
            ),
            Section(model_ref, m_display),
            Section(bill_ref, b_display),
        )
        out.append(OriginItem(r, item))
    return out


def pair_sections(
    keys: Iterable[MatchKey],
    matches: Iterable[OriginMatch],
    docs: Iterable[ModelDoc],
    tables_dir: Path,
    *,
    root: Path = MODELDOCS_DIR,
) -> dict[MatchKey, list[dict[str, Any]]]:
    """Every section pair each match rests on, the strongest (the one labeled) marked, each
    side's length in words and its generic kind (lineage.text.generic_sections), with no text."""
    wanted = set(keys)
    by_key = {match_key(m): m for m in matches if match_key(m) in wanted}
    held = {d.id: d for d in docs}
    model: dict[str, list[tuple[str, str]]] = {}
    for doc_id in sorted({d for _, d in by_key}):
        model[doc_id] = matchable_sections(load_text(held[doc_id], root))[1]
    with db.connect_quiet() as con:
        found = queries.matchable_sections_of(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
            ids=sorted({b for b, _ in by_key}),
        )
    bill = {(b, o): normalized for b, o, _, normalized in found}
    out: dict[MatchKey, list[dict[str, Any]]] = {}
    for key, m in sorted(by_key.items()):
        strongest = strongest_pair(m)
        rows: list[dict[str, Any]] = []
        for p in sorted(m.pairs, key=lambda p: (p.model_section, p.ordinal)):
            m_norm = model[m.doc_id][p.model_section][1]
            b_norm = bill.get((m.bill_id, p.ordinal), "")
            rows.append(
                {
                    "model_section": p.model_section,
                    "bill_section": p.ordinal,
                    "strongest": p == strongest,
                    "agreements": p.agreements,
                    "cosine": p.cosine,
                    "model_words": len(m_norm.split()),
                    "bill_words": len(b_norm.split()),
                    "model_kind": generic_kind(m_norm),
                    "bill_kind": generic_kind(b_norm),
                }
            )
        out[key] = rows
    return out


def load_labels(path: Path = LABELS) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def pending(items: Sequence[OriginItem], labels: Iterable[Mapping[str, Any]]) -> list[OriginItem]:
    """The items no label under this prompt answers yet, for the same sections."""
    done = {
        (row["bill_id"], row["doc_id"], row["model_section"], row["bill_section"])
        for row in labels
        if str(row["labeler"]).endswith(f"@{PROMPT.version}")
    }
    return [i for i in items if _key(i) not in done]


def correct_by_sections(
    items: Sequence[OriginItem], labels: Iterable[Mapping[str, Any]]
) -> dict[str, bool]:
    """Each item's answer under this prompt, by its pair id, from the label on the same bill,
    document and both sections, whichever part's row took it: a sample drawn after matching
    changed may share matches with an earlier part, and only a label on the sections its row
    now rests on answers it. An item no such label answers is absent; two different answers for
    the same sections are refused."""
    by_sections: dict[tuple[str, str, str, str], bool] = {}
    for row in labels:
        if not str(row["labeler"]).endswith(f"@{PROMPT.version}"):
            continue
        key = (str(row["bill_id"]), str(row["doc_id"]), row["model_section"], row["bill_section"])
        answer = is_correct(row)
        if by_sections.get(key, answer) != answer:
            raise OriginPrecisionError(f"{key[0]} x {key[1]} has two different labels")
        by_sections[key] = answer
    return {i.pair: by_sections[_key(i)] for i in items if _key(i) in by_sections}


def labelers(items: Sequence[OriginItem], labels: Iterable[Mapping[str, Any]]) -> list[str]:
    """Who labeled the items, under this prompt, on the sections each rests on: what a reading
    records as its labeler."""
    keys = {_key(i) for i in items}
    return sorted(
        {
            str(row["labeler"])
            for row in labels
            if str(row["labeler"]).endswith(f"@{PROMPT.version}")
            and (str(row["bill_id"]), str(row["doc_id"]), row["model_section"], row["bill_section"])
            in keys
        }
    )


def _key(i: OriginItem) -> tuple[str, str, str, str]:
    return (
        i.item.match.ref.bill_id,
        i.item.query.ref.bill_id,
        i.item.query.ref.text_sha256,
        i.item.match.ref.text_sha256,
    )


def label_row(i: OriginItem, label: int, served_by: str, today: str) -> dict[str, Any]:
    bill_id, doc_id, model_sha, bill_sha = _key(i)
    return {
        "pair": i.pair,
        "bill_id": bill_id,
        "doc_id": doc_id,
        "model_section": model_sha,
        "bill_section": bill_sha,
        "label": label,
        "labeler": model_labeler(served_by, PROMPT.version),
        "scale": PROMPT.scale,
        "at": today,
    }


@dataclass(frozen=True, slots=True)
class Plan:
    items: int
    projected: float
    spent: float
    max_cost: float
    budget: float

    def refusal(self) -> str | None:
        if self.projected > self.max_cost:
            return (
                f"{self.items} matches projected at most ${self.projected:,.2f}, over the "
                f"${self.max_cost:,.2f} cap; nothing asked"
            )
        if self.spent + self.projected > self.budget:
            return (
                f"{self.items} matches projected at most ${self.projected:,.2f}, with "
                f"${self.spent:,.2f} spent, over v2's ${self.budget:,.2f} budget; nothing asked"
            )
        return None


def plan(
    items: Sequence[OriginItem],
    *,
    model: str = MODEL_ID,
    spend_path: Path = SPEND,
    max_cost: float = MAX_COST,
    budget: float = BUDGET,
) -> Plan:
    """Each prompt exactly as it would be sent, through the Batches API, every reply at its
    token limit (lineage.evals.model_labels.projected_cost)."""
    cost = projected_cost([i.item for i in items], model=model, batch=True, prompt=PROMPT)
    return Plan(len(items), cost, spent(spend_path), max_cost, budget)


def label_in_batch(
    items: Sequence[OriginItem],
    *,
    batches: Batches,
    model: str,
    labels_path: Path,
    today: str,
    wait: Callable[[], None],
    batch_id: str | None = None,
    submitted: Callable[[str], None] = lambda _: None,
) -> tuple[int, int, str]:
    """Send every item in one batch, or collect `batch_id`; wait until it ends, then append
    each usable label. Returns how many were labeled, how many replies were unusable, and the
    batch id."""
    by_id = {i.pair: i for i in items}
    if len(by_id) != len(items):
        raise OriginPrecisionError("two matches asked in one batch share an id")
    if batch_id is None:
        batch_id = batches.submit([(i.pair, request_params(i.item, model, PROMPT)) for i in items])
        submitted(batch_id)
    while not batches.ended(batch_id):
        wait()
    rows: list[dict[str, Any]] = []
    unusable = 0
    for reply in batches.results(batch_id, PROMPT.schema):
        item = by_id.get(reply.custom_id)
        if item is None:
            raise OriginPrecisionError(f"batch {batch_id} answered {reply.custom_id}, not asked")
        if reply.label is None or reply.served_by is None:
            unusable += 1
            continue
        rows.append(label_row(item, reply.label, reply.served_by, today))
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["pair"]))
    return len(rows), unusable, batch_id


def record(
    batches: Batches,
    batch_id: str,
    *,
    model: str,
    today: str,
    spend_path: Path = SPEND,
    planned: Plan | None = None,
) -> dict[str, Any]:
    """An ended batch's tokens and cost, in v2's ledger under v2's budget. With its plan, the
    batch's entry also records what it was projected at and the cap it ran under beside the
    per-run cap, so a run allowed past that cap (ruled at M12's start, when its projection fits
    v2's budget) says so in the ledger."""
    ledger = record_spend(
        spend_path,
        batch_id=batch_id,
        model=model,
        prompt=PROMPT.version,
        usage=batch_usage(batches, batch_id),
        today=today,
        budget=BUDGET,
        command=COMMAND,
    )
    if planned is None:
        return ledger
    for entry in ledger["batches"]:
        if entry["batch_id"] == batch_id:
            entry |= {
                "projected": round(planned.projected, 4),
                "max_cost": planned.max_cost,
                "per_run_cap": MAX_COST,
            }
    spend_path.write_text(json.dumps(ledger, indent=1) + "\n")
    return ledger


def judged_correct(
    keys: Iterable[MatchKey],
    matches: Iterable[OriginMatch],
    docs: Iterable[ModelDoc],
    tables_dir: Path,
    labels: Iterable[Mapping[str, Any]],
    *,
    root: Path = MODELDOCS_DIR,
) -> set[MatchKey]:
    """Of the matches `keys` names, those a label under this prompt judged correct on the
    sections their strongest pair rests on now: the matches a featured story may feature
    (lineage.export.origin_story, ruled at the step 48 review). A match never labeled, or labeled on
    sections it no longer rests on, is not judged."""
    rows: list[dict[str, Any]] = [
        {"pair": f"{bill_id} {doc_id}", "bill_id": bill_id, "doc_id": doc_id, "stages": ()}
        for bill_id, doc_id in sorted(set(keys))
    ]
    if not rows:
        return set()
    items = build_items(rows, matches, docs, tables_dir, root=root)
    answers = correct_by_sections(items, labels)
    return {(str(i.row["bill_id"]), str(i.row["doc_id"])) for i in items if answers.get(i.pair)}
