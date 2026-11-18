"""Model documents withheld from what pages show, because their labeled matches are mostly wrong.

Ruled at the deploy review (2026-11-17), for every source alike: a document with at least
MIN_LABELED labeled matches, fewer than half of them judged correct, is withheld from bill pages,
the feed and the origin story until a fix lands. A labeled match is one a page lists (at the
floor, lineage.origins.matches.is_listed) whose strongest section pair a label under the current
prompt answers, from whichever sample drew it (lineage.origins.labels.correct_by_sections), so a
label on sections the match no longer rests on does not count. The sources page names each withheld
document and why: its matches rest on generic text.

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
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from lineage.origins.labels import build_items, correct_by_sections
from lineage.origins.matches import FLOOR, OriginMatch, is_listed
from lineage.origins.modeldocs import MODELDOCS_DIR, ModelDoc

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
        return self.labeled >= MIN_LABELED and 2 * self.correct < self.labeled


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
    labels = list(labels)
    labeled_keys = {(str(r["bill_id"]), str(r["doc_id"])) for r in labels}
    listed = [m for m in matches if is_listed(m)]
    shown = Counter(m.doc_id for m in listed)
    rows: list[dict[str, Any]] = [
        {"pair": f"{m.bill_id} {m.doc_id}", "bill_id": m.bill_id, "doc_id": m.doc_id, "stages": ()}
        for m in sorted(listed, key=lambda m: (m.bill_id, m.doc_id))
        if (m.bill_id, m.doc_id) in labeled_keys
    ]
    docs = list(docs)
    titles = {d.id: d.title for d in docs}
    if not rows:
        return []
    items = build_items(rows, listed, docs, tables_dir, root=root)
    answers = correct_by_sections(items, labels)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i in items:
        if i.pair in answers:
            c = counts[str(i.row["doc_id"])]
            c[0] += 1
            c[1] += int(answers[i.pair])
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
            "withheld_when": "fewer than half of the labeled listed matches judged correct",
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
