"""Model documents withheld from what pages show, because their labeled matches are mostly wrong.

Ruled at the deploy review (2026-11-17), for every source alike, and amended 2026-11-19: a
document fewer than half of whose labeled listed matches are judged correct, judged on at least
MIN_LABELED labeled matches or on all of its listed matches, is withheld from bill pages, the feed
and the origin story until a fix lands. The minimum guards against judging a document from a small
sample; when every listed match is labeled there is no sample. A labeled match is one a page
lists (at the floor, lineage.origins.matches.is_listed) whose strongest section pair a label under
the current prompt answers, from whichever sample drew it (lineage.origins.labels
.correct_by_sections), so a label on sections the match no longer rests on does not count. The
sources page names each withheld document and why: its matches rest on generic text.

The withheld set is a committed report, written by `lineage withhold` after matching changes,
so the daily live track reads the same set the reviewed export did and never recounts it.
Precision readings are not redrawn: they were measured with these documents' matches in, so
they are conservative for what pages show (DESIGN.md#origin-matches).

A stopgap: the fix on the roadmap is frequency-weighted shares, a passage that recurs across many
unrelated bills and models counting for less, for every source (DESIGN.md#future-work). Once it
lands, `lineage withhold` runs again on a fresh reading and a document whose matches then pass the
rule returns by itself.

See DESIGN.md#dr-11.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lineage.match.strata import STRATA, stratum_of
from lineage.origins.labels import answers_by_match
from lineage.origins.matches import FLOOR, OriginMatch, is_listed
from lineage.origins.modeldocs import MODELDOCS_DIR, ModelDoc
from lineage.origins.precision import MatchKey, Part, match_key

COMMAND = "uv run lineage withhold"
REPORT = Path("reports/withheld_documents.json")
MIN_LABELED = 3
Reason = Literal["generic_text", "review"]
REASON: Reason = "generic_text"
REVIEW: Reason = "review"


@dataclass(frozen=True)
class Tally:
    doc_id: str
    title: str
    listed: int
    labeled: int
    correct: int

    @property
    def withheld(self) -> bool:
        judged = self.labeled >= MIN_LABELED or self.labeled == self.listed
        return self.labeled > 0 and judged and 2 * self.correct < self.labeled


def tallies(
    matches: Iterable[OriginMatch],
    docs: Iterable[ModelDoc],
    tables_dir: Path,
    labels: Iterable[Mapping[str, Any]],
    *,
    root: Path = MODELDOCS_DIR,
) -> list[Tally]:
    """Each document with a labeled listed match: how many matches pages list for it, how many
    of those a label answers, and how many of those it judged correct, by document id."""
    listed = [m for m in matches if is_listed(m)]
    shown = Counter(m.doc_id for m in listed)
    docs = list(docs)
    titles = {d.id: d.title for d in docs}
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    answers = answers_by_match(listed, docs, tables_dir, labels, root=root)
    for (_, doc_id), correct in answers.items():
        c = counts[doc_id]
        c[0] += 1
        c[1] += int(correct)
    return [
        Tally(doc_id=d, title=titles.get(d, d), listed=shown[d], labeled=n, correct=k)
        for d, (n, k) in sorted(counts.items())
    ]


@dataclass(frozen=True)
class Reviewed:
    """A document the maintainer withheld after reviewing a held day's claims: its matches were
    found wrong, whatever the labels say (ruled at the deploy review)."""

    doc_id: str
    title: str
    listed: int
    note: str
    on: str  # ISO 8601


def report(found: Iterable[Tally], reviewed: Iterable[Reviewed] = ()) -> dict[str, Any]:
    found = list(found)
    by_rule = [t for t in found if t.withheld]
    ruled = {t.doc_id for t in by_rule}
    by_review = [r for r in reviewed if r.doc_id not in ruled]
    return {
        "command": COMMAND,
        "rule": {
            "min_labeled": MIN_LABELED,
            "withheld_when": "fewer than half of the labeled listed matches judged correct, "
            "judged on at least min_labeled labeled matches or on all of the listed matches",
            "floor": FLOOR,
        },
        "documents_with_labels": len(found),
        "listed_matches_withheld": sum(t.listed for t in by_rule)
        + sum(r.listed for r in by_review),
        "withheld": [{**asdict(t), "reason": REASON} for t in by_rule],
        # Withheld by hand after a held day's review (`lineage withhold --doc`); kept whenever
        # the rule's part is written again.
        "reviewed": [{**asdict(r), "reason": REVIEW} for r in by_review],
    }


def read_reviewed(path: Path = REPORT) -> list[Reviewed]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text()).get("reviewed", [])
    return [
        Reviewed(
            doc_id=r["doc_id"], title=r["title"], listed=r["listed"], note=r["note"], on=r["on"]
        )
        for r in rows
    ]


def write_report(
    found: Iterable[Tally], path: Path = REPORT, reviewed: Iterable[Reviewed] | None = None
) -> Path:
    """The report, its rule's part from `found`, the reviewed part kept as the file has it
    unless `reviewed` is given."""
    kept = read_reviewed(path) if reviewed is None else list(reviewed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report(found, kept), indent=1) + "\n")
    return path


def listed_count(matches: Iterable[OriginMatch], doc_id: str) -> int:
    return sum(1 for m in matches if m.doc_id == doc_id and is_listed(m))


@dataclass(frozen=True)
class Withheld:
    """The withheld documents as the committed report gives them: each one's reason, by id, and
    how many listed matches they held nationally when it was written."""

    reasons: Mapping[str, Reason] = field(default_factory=dict[str, Reason])
    matches: int = 0

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self.reasons)


NONE = Withheld()


def load_withheld(path: Path = REPORT) -> Withheld:
    """The committed report's withheld documents, by the rule and by review; none before it
    exists."""
    if not path.exists():
        return NONE
    doc = json.loads(path.read_text())
    reasons: dict[str, Reason] = {str(d["doc_id"]): REASON for d in doc["withheld"]}
    reasons |= {str(d["doc_id"]): REVIEW for d in doc.get("reviewed", [])}
    return Withheld(reasons=reasons, matches=int(doc["listed_matches_withheld"]))


TOPUP_PER_DOCUMENT = 10


def unsettled(matches: Iterable[OriginMatch], answers: Mapping[MatchKey, bool]) -> list[str]:
    """The documents the rule cannot settle yet: a labeled listed match judged wrong, but fewer
    labeled listed matches than MIN_LABELED and some listed match still unlabeled."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for m in matches:
        if is_listed(m):
            c = counts[m.doc_id]
            c[2] += 1
            if (key := match_key(m)) in answers:
                c[0] += 1
                c[1] += int(answers[key])
    return sorted(
        d for d, (n, k, listed) in counts.items() if 0 < n < MIN_LABELED and k < n and n < listed
    )


def topup(
    matches: Sequence[OriginMatch],
    answers: Mapping[MatchKey, bool],
    part: Part,
    *,
    per_document: int = TOPUP_PER_DOCUMENT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each unsettled document's listed matches no label answers yet, up to `per_document` in
    a seeded order, as rows to label, and the draw's report."""
    docs = unsettled(matches, answers)
    rng = np.random.default_rng(part.seed)
    rows: list[dict[str, Any]] = []
    drawn: dict[str, dict[str, int]] = {}
    for doc in docs:
        listed = sorted((m for m in matches if m.doc_id == doc and is_listed(m)), key=match_key)
        open_ = [m for m in listed if match_key(m) not in answers]
        order: list[int] = rng.permutation(len(open_)).tolist()
        chosen = [open_[i] for i in order[:per_document]]
        labeled = [answers[match_key(m)] for m in listed if match_key(m) in answers]
        drawn[doc] = {
            "listed": len(listed),
            "labeled": len(labeled),
            "correct": sum(labeled),
            "drawn": len(chosen),
        }
        for m in chosen:
            stratum = stratum_of(m.stages)
            rows.append(
                {
                    "pair": f"o{part.first + len(rows):04d}",
                    "stratum": stratum,
                    "stages": list(STRATA[stratum]),
                    "bill_id": m.bill_id,
                    "doc_id": m.doc_id,
                    "ordinals": sorted({s.ordinal for s in m.spans}),
                    "model_share": m.model_share,
                    "bill_share": m.bill_share,
                }
            )
    report = {
        "command": f"uv run lineage origin-sample --part {part.name}",
        "seed": part.seed,
        "per_document": per_document,
        "floor": FLOOR,
        "documents": drawn,
    }
    return rows, report
