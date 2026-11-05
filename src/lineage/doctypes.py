"""Which records are matched: the per-state document-type rules of DESIGN.md#dr-5.

The rules live in one committed table, `rules/document_types.csv`, read top to bottom; the
first rule whose state, type and title patterns fit a record decides it, and a record no rule
fits is excluded. `lineage eligible` applies them to the bills table and reports how many
records each rule decided, so every exclusion is counted rather than asserted.
"""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lineage import queries
from lineage.stoplist import file_sha256

COMMAND = "uv run lineage eligible"
RULES = Path(__file__).parents[2] / "rules" / "document_types.csv"
_COLUMNS = ["state", "type", "title", "unless", "decision", "reason"]
_DECISIONS = {"match": True, "exclude": False}
# Titles shown per title-pattern rule, so a reviewer can see what a pattern caught.
_SAMPLE = 60


class DocTypesError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Rule:
    number: int  # position in the table, from 1
    state: str  # a state code, or "*" for every state
    type: str  # the record's classifications joined by "+", e.g. "bill+appropriation"
    title: re.Pattern[str] | None
    unless: re.Pattern[str] | None
    match: bool
    reason: str

    def fits(self, state: str, kind: str, title: str) -> bool:
        return (
            self.state in ("*", state)
            and self.type == kind
            and (self.title is None or self.title.search(title) is not None)
            and (self.unless is None or self.unless.search(title) is None)
        )


def _pattern(text: str, line: int) -> re.Pattern[str] | None:
    if not text:
        return None
    try:
        return re.compile(text, re.IGNORECASE)
    except re.error as e:
        raise DocTypesError(f"line {line}: bad pattern {text!r}: {e}") from e


def load_rules(path: Path) -> list[Rule]:
    lines = path.read_text().splitlines()
    numbered = [(i, line) for i, line in enumerate(lines, start=1) if not line.startswith("#")]
    reader = csv.DictReader((line for _, line in numbered))
    if reader.fieldnames != _COLUMNS:
        raise DocTypesError(f"{path}: expected columns {_COLUMNS}, got {reader.fieldnames}")
    rules: list[Rule] = []
    # Row k of the reader is file line numbered[k]; the header is numbered[0].
    for number, row in enumerate(reader, start=1):
        line = numbered[number][0]
        if row["decision"] not in _DECISIONS:
            raise DocTypesError(f"line {line}: decision must be match or exclude")
        if not row["reason"]:
            raise DocTypesError(f"line {line}: every rule needs a reason")
        rules.append(
            Rule(
                number=number,
                state=row["state"],
                type=row["type"],
                title=_pattern(row["title"], line),
                unless=_pattern(row["unless"], line),
                match=_DECISIONS[row["decision"]],
                reason=row["reason"],
            )
        )
    return rules


def decide(rules: list[Rule], state: str, kind: str, title: str) -> Rule | None:
    """The rule that decides a record, or None when no rule fits and it is excluded."""
    return next((r for r in rules if r.fits(state, kind, title)), None)


def _stable(record_id: str) -> str:
    # Samples are chosen by a hash of the record id, so reruns show the same titles.
    return hashlib.sha256(record_id.encode()).hexdigest()


def build_eligibility(tables_dir: Path, *, rules_path: Path, report_path: Path) -> dict[str, Any]:
    """Write `<tables>/eligible.parquet` (id, eligible, rule) and the report."""
    rules = load_rules(rules_path)
    bills = str(tables_dir / "bills" / "*.parquet")
    with queries.connect_quiet() as con:
        records = queries.record_types(con, bills_glob=bills)
        decided = [(r, decide(rules, r.state, r.type, r.title)) for r in records]
        partial = tables_dir / "eligible.parquet.partial"
        queries.write_eligibility(
            con,
            parquet_path=partial,
            ids=[r.id for r, _ in decided],
            eligible=[rule is not None and rule.match for _, rule in decided],
            rules=[rule.number if rule else None for _, rule in decided],
        )
        partial.replace(tables_dir / "eligible.parquet")
        with_text = queries.eligible_text_by_state(
            con,
            bills_glob=bills,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            eligible_path=str(tables_dir / "eligible.parquet"),
        )

    by_rule: dict[int, Counter[str]] = defaultdict(Counter)
    samples: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    unlisted: Counter[str] = Counter()
    for record, rule in decided:
        if rule is None:
            unlisted[record.type] += 1
            continue
        by_rule[rule.number][record.state] += 1
        if rule.title is not None or rule.unless is not None:
            samples[rule.number].append((_stable(record.id), record.state, record.title))
    # Titles a rule's `unless` pattern spared: the ones a looser pattern would have caught.
    spared: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for record, _ in decided:
        for rule in rules:
            if (
                rule.unless is not None
                and rule.state in ("*", record.state)
                and rule.type == record.type
                and (rule.title is None or rule.title.search(record.title))
                and rule.unless.search(record.title)
            ):
                spared[rule.number].append((record.state, record.title))

    doc: dict[str, Any] = {
        "command": COMMAND,
        "rules_sha256": file_sha256(rules_path),
        "totals": {
            "records": len(decided),
            "eligible": sum(1 for _, rule in decided if rule is not None and rule.match),
            "eligible_with_sections": sum(n for n, _ in with_text.values()),
        },
        "rules": [
            {
                "rule": rule.number,
                "state": rule.state,
                "type": rule.type,
                "title": rule.title.pattern if rule.title else None,
                "unless": rule.unless.pattern if rule.unless else None,
                "decision": "match" if rule.match else "exclude",
                "reason": rule.reason,
                "records": sum(by_rule[rule.number].values()),
                "by_state": dict(sorted(by_rule[rule.number].items())),
            }
            | (
                {
                    "sample_titles": [
                        f"{state}: {title}"
                        for _, state, title in sorted(samples[rule.number])[:_SAMPLE]
                    ]
                }
                if rule.number in samples
                else {}
            )
            | (
                {"spared_titles": [f"{s}: {t}" for s, t in sorted(set(spared[rule.number]))]}
                if rule.number in spared
                else {}
            )
            for rule in rules
        ],
        "unlisted_types": dict(sorted(unlisted.items())),
        # States with eligible records but no section from any of them: no text in the source.
        "states_without_text": sorted(
            state
            for state, (sectioned, eligible) in with_text.items()
            if eligible and not sectioned
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc
