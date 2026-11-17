"""Model documents withheld from what pages show, because their labeled matches are mostly wrong.

Ruled at the deploy review (2026-11-17), for every source alike: a document with at least
MIN_LABELED labeled matches, fewer than half of them judged correct, is withheld from bill pages,
the feed and the origin story until a fix lands. A labeled match is one a page lists (at the
floor, lineage.origins.is_listed) whose strongest section pair a label under the current prompt
answers, from whichever sample drew it (lineage.origin_labels.correct_by_sections), so a label on
sections the match no longer rests on does not count. The sources page names each withheld
document and why: its matches rest on generic text.

The withheld set is a committed report, written by `lineage withhold` after matching changes,
so the daily live track reads the same set the reviewed export did and never recounts it.
Precision readings are not redrawn: they were measured with these documents' matches in, so
they are conservative for what pages show (DESIGN.md#origin-matches).

See DESIGN.md#dr-11.
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from lineage.modeldocs import MODELDOCS_DIR, ModelDoc
from lineage.origin_labels import build_items, correct_by_sections
from lineage.origins import FLOOR, OriginMatch, is_listed

COMMAND = "uv run lineage withhold"
REPORT = Path("reports/withheld_documents.json")
MIN_LABELED = 3
REASON: Literal["generic_text"] = "generic_text"


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


def report(found: Iterable[Tally]) -> dict[str, Any]:
    found = list(found)
    return {
        "command": COMMAND,
        "rule": {
            "min_labeled": MIN_LABELED,
            "withheld_when": "fewer than half of the labeled listed matches judged correct",
            "floor": FLOOR,
        },
        "documents_with_labels": len(found),
        "listed_matches_withheld": sum(t.listed for t in found if t.withheld),
        "withheld": [{**asdict(t), "reason": REASON} for t in found if t.withheld],
    }


def write_report(found: Iterable[Tally], path: Path = REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report(found), indent=1) + "\n")
    return path


@dataclass(frozen=True)
class Withheld:
    """The withheld documents as the committed report gives them: their ids, and how many
    listed matches they held nationally when it was written."""

    ids: frozenset[str] = frozenset()
    matches: int = 0


NONE = Withheld()


def load_withheld(path: Path = REPORT) -> Withheld:
    """The committed report's withheld documents; none before it exists."""
    if not path.exists():
        return NONE
    doc = json.loads(path.read_text())
    return Withheld(
        ids=frozenset(str(d["doc_id"]) for d in doc["withheld"]),
        matches=int(doc["listed_matches_withheld"]),
    )
