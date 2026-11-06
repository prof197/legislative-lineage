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
SAMPLES_COMMAND = "uv run lineage stats --samples"
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


def _spares(rule: Rule, record: queries.RecordType) -> bool:
    """Whether `rule` would have caught `record` but for its `unless` pattern."""
    return (
        rule.unless is not None
        and rule.state in ("*", record.state)
        and rule.type == record.type
        and (rule.title is None or rule.title.search(record.title) is not None)
        and rule.unless.search(record.title) is not None
    )


def build_eligibility(tables_dir: Path, *, rules_path: Path, report_path: Path) -> dict[str, Any]:
    """Write `<tables>/eligible.parquet` (id, eligible, rule) and the report of counts."""
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
    spared: dict[int, Counter[str]] = defaultdict(Counter)
    unlisted: Counter[str] = Counter()
    for record, rule in decided:
        for candidate in rules:
            if _spares(candidate, record):
                spared[candidate.number][record.state] += 1
        if rule is None:
            unlisted[record.type] += 1
            continue
        by_rule[rule.number][record.state] += 1

    # Counts only. Titles of commemorations name private people, so the committed report
    # carries none; `lineage stats --samples` writes them to a local file instead.
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
                {"spared_by_state": dict(sorted(spared[rule.number].items()))}
                if rule.unless is not None
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


def rule_samples(tables_dir: Path, *, rules_path: Path) -> dict[str, Any]:
    """Sample titles for each rule that reads titles, and every title an `unless` spared.

    Written only to a local, gitignored file. The titles are public records, and anyone can
    reproduce the file from the pinned snapshot, but commemorations name private people.
    """
    rules = load_rules(rules_path)
    with queries.connect_quiet() as con:
        records = queries.record_types(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
    samples: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    spared: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        rule = decide(rules, record.state, record.type, record.title)
        if rule is not None and (rule.title is not None or rule.unless is not None):
            samples[rule.number].append((_stable(record.id), record.state, record.title))
        for candidate in rules:
            if _spares(candidate, record):
                spared[candidate.number].add((record.state, record.title))
    return {
        "command": SAMPLES_COMMAND,
        "rules_sha256": file_sha256(rules_path),
        "rules": [
            {
                "rule": rule.number,
                "reason": rule.reason,
                "sample_titles": [
                    f"{state}: {title}"
                    for _, state, title in sorted(samples[rule.number])[:_SAMPLE]
                ],
                "spared_titles": [f"{s}: {t}" for s, t in sorted(spared[rule.number])],
            }
            for rule in rules
            if rule.number in samples or rule.number in spared
        ],
    }
