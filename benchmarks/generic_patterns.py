"""What the hand-written generic patterns still add over frequency weighting, for
DESIGN.md#origin-matches (M13 step 6).

    uv run lineage origins --tables WITH                    # the patterns on (as published)
    uv run lineage origins --tables WITHOUT --no-patterns   # every pattern off
    uv run python -m benchmarks.generic_patterns WITH WITHOUT

Both tables directories are copies of the corpus's tables with passage reach built. A match a
page lists (its weighted share at the floor) without the patterns, but not with them, is one the
patterns still keep off pages; each is put down to the kinds of generic text its section pairs
touch (lineage.text.generic_sections, whole sections by their kind, or generic text inside a
section), and read against every label that answers it on the sections it rests on now. The
definitions pattern is also read alone, by source, since it is the one whose extra matches are
mostly correct. Development recall is read both ways. Writes reports/generic_patterns.json.
"""

import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lineage.core import db
from lineage.origins import queries
from lineage.origins.collect import collectors
from lineage.origins.labels import LABELS, answers_by_match, load_labels
from lineage.origins.matches import OriginMatch, is_listed, read_origin_matches
from lineage.origins.matching import BillSection, ModelText, bill_sections, model_text
from lineage.origins.modeldocs import MODELDOCS_DIR, ModelDoc, load_text, read_docs
from lineage.origins.recall import KEYS, evaluate, load_key, same_slug
from lineage.text.generic_sections import generic_kind
from lineage.text.sections import normalize

REPORT = Path("reports/generic_patterns.json")
DEFINITIONS = frozenset({"model:definitions", "bill:definitions"})


def _source(m: OriginMatch) -> str:
    return m.doc_id.split("/", 1)[0]


def _bills(tables: Path, ids: list[str]) -> dict[str, dict[int, BillSection]]:
    with db.connect_quiet() as con:
        rows = queries.matchable_sections_of(
            con,
            sections_glob=str(tables / "sections" / "*.parquet"),
            eligible_path=tables / "eligible.parquet",
            ids=ids,
        )
        by_heading = queries.split_by_heading_of(
            con, sections_glob=str(tables / "sections" / "*.parquet"), ids=ids
        )
    of_bill: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for b, o, d, n in rows:
        of_bill[b].append((o, d, n))
    return {b: bill_sections(r, by_heading=by_heading[b]) for b, r in of_bill.items()}


def _kinds(m: OriginMatch, text: ModelText, bill: Mapping[int, BillSection]) -> frozenset[str]:
    """The kinds of generic text the match's pairs touch, on each side."""
    out: set[str] = set()
    for p in m.pairs:
        q, o = p.model_section, p.ordinal
        if q in text.generic:
            out.add(f"model:{generic_kind(normalize(text.display[q])) or 'mostly_inside'}")
        elif q in text.inside:
            out.add("model:inside")
        if o in bill and bill[o].generic:
            out.add(f"bill:{generic_kind(normalize(bill[o].display)) or 'mostly_inside'}")
        elif o in bill and bill[o].inside is not None:
            out.add("bill:inside")
    return frozenset(out) or frozenset({"none"})


def _read(ms: Sequence[OriginMatch], answers: Mapping[tuple[str, str], bool]) -> dict[str, Any]:
    labeled = [answers[(m.bill_id, m.doc_id)] for m in ms if (m.bill_id, m.doc_id) in answers]
    return {"matches": len(ms), "labeled": len(labeled), "correct": sum(labeled)}


def report(with_patterns: Path, without: Path) -> dict[str, Any]:
    docs: dict[str, ModelDoc] = {
        d.id: d for c in collectors().values() for d in read_docs(c.source, MODELDOCS_DIR)
    }
    labels = load_labels(LABELS)
    on = read_origin_matches(with_patterns)
    off = read_origin_matches(without)
    listed_on = [m for m in on if is_listed(m)]
    shown = {(m.bill_id, m.doc_id) for m in listed_on}
    extra = [m for m in off if is_listed(m) and (m.bill_id, m.doc_id) not in shown]
    answers = answers_by_match(extra, docs.values(), without, labels)
    base = answers_by_match(listed_on, docs.values(), with_patterns, labels)
    bills = _bills(without, sorted({m.bill_id for m in extra}))
    texts = {d: model_text(d, load_text(docs[d], MODELDOCS_DIR)) for d in {m.doc_id for m in extra}}
    kinds = {(m.bill_id, m.doc_id): _kinds(m, texts[m.doc_id], bills[m.bill_id]) for m in extra}
    by_kind: dict[str, list[OriginMatch]] = defaultdict(list)
    for m in extra:
        for k in kinds[(m.bill_id, m.doc_id)]:
            by_kind[k].append(m)
    only_definitions = [m for m in extra if kinds[(m.bill_id, m.doc_id)] <= DEFINITIONS]
    key = KEYS["development"]
    loaded = load_key(key.path)
    acts = {e.act for e in loaded.enactments}
    recall = {
        name: evaluate(ms, loaded, same_slug(acts, "commission"))["recall"]["pooled"]
        for name, ms in (("with_patterns", on), ("without_patterns", off))
    }
    return {
        "command": "uv run python -m benchmarks.generic_patterns",
        "listed_with_patterns": _read(listed_on, base),
        "extra_listed_without": _read(extra, answers),
        "extra_by_source": dict(Counter(_source(m) for m in extra).most_common()),
        "by_kind": {
            k: _read(ms, answers) for k, ms in sorted(by_kind.items(), key=lambda kv: -len(kv[1]))
        },
        "definitions_only": _read(only_definitions, answers)
        | {
            "by_source": {
                s: _read([m for m in only_definitions if _source(m) == s], answers)
                for s in sorted({_source(m) for m in only_definitions})
            }
        },
        "development_recall": recall,
    }


if __name__ == "__main__":
    doc = report(Path(sys.argv[1]), Path(sys.argv[2]))
    REPORT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("listed_with_patterns", "extra_listed_without")}))
    print(f"-> {REPORT}")
