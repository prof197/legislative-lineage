"""The fixture export every v2 lane builds against: Utah's three bills from the commission
fixture, with synthetic model documents and hand-made origin matches (DESIGN.md#origin-matches).

The model documents are synthetic: each is the longest sections of a fixture bill, standing in
for a published model so the export, the site and the tests have an origin to show before any
real source is collected. The matches are written by hand, as matching (lane C) would write them
for a model copied whole. One source is marked leaked, so its label shows.

Regenerate the committed copy with
`LINEAGE_REGENERATE=1 uv run pytest tests/test_fixture_export.py`.
"""

import json
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

from lineage import queries
from lineage.collect import collectors
from lineage.core import db
from lineage.evals.labels import SAME_PROVISION, text_sha256
from lineage.export import RESULTS, build_export
from lineage.match.query import matchable_sections
from lineage.match.semantic.pairs import build_semantic_pairs
from lineage.modeldocs import ModelDoc, Source, sha256, store
from lineage.origin_labels import PROMPT
from lineage.origins import BillSpan, OriginMatch, SectionPair, write_origin_matches
from lineage.text.generic_sections import generic_kind
from lineage.text.sections import normalize
from lineage.withheld import Tally
from lineage.withheld import write_report as write_withheld
from tests.conftest import (
    COMMISSION,
    FAKE_MODEL,
    build_raw_dir,
    embed_with_stand_in,
    tables_from_raw,
)

REPO = Path(__file__).parent.parent
COMMITTED = Path(__file__).parent / "fixtures" / "export"
PACKAGE = "tests.collector_fixtures"
STATE = "UT"
COLLECTED_ON = date(2026, 11, 15)
# Which fixture bill each synthetic document is drawn from.
DRAWN_FROM = {
    "example-library/public-expression-protection": (
        "ocd-bill/7968b9d9-25ee-4684-a92e-c76167d6f917",  # Utah SB 18 (2023)
        "Model Public Expression Protection Act (synthetic fixture)",
    ),
    "example-archive/estate-planning-documents": (
        "ocd-bill/370e175f-246b-4076-a6b8-62312800299c",  # Utah HB 181 (2026)
        "Model Electronic Estate Planning Documents Act (synthetic fixture)",
    ),
    # Matched, then withheld (lineage.withheld), so no page lists it and the sources page does.
    "example-library/speech-procedures": (
        "ocd-bill/7968b9d9-25ee-4684-a92e-c76167d6f917",  # Utah SB 18 (2023)
        "Model Speech Procedures Act (synthetic fixture)",
    ),
}
WITHHELD = "example-library/speech-procedures"
SECTIONS_DRAWN = 2


def hand_made_origins(
    tables: Path, modeldocs: Path, sources: dict[str, Source], labels: Path
) -> list[OriginMatch]:
    """The hand-made matches, and in `labels` a label judging the first correct, as a labeler
    would on its strongest pair, so the fixture's origin story can feature it (a featured
    story features only a match judged correct, lineage.origin_story)."""
    ids = [bill for bill, _ in DRAWN_FROM.values()]
    with db.connect_quiet() as con:
        rows = queries.matchable_section_texts(
            con,
            sections_glob=str(tables / "sections" / "*.parquet"),
            eligible_path=tables / "eligible.parquet",
            ids=ids,
        )
    by_bill: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for bill_id, ordinal, text in rows:
        by_bill[bill_id].append((ordinal, text))
    out: list[OriginMatch] = []
    # The first is found by both stages, so the fixture carries an origin story
    # (lineage.origin_story), and the rest by wording alone.
    first = next(iter(DRAWN_FROM))
    for doc_id, (bill_id, title) in DRAWN_FROM.items():
        sections = by_bill[bill_id]
        drawn = sorted(sorted(sections, key=lambda s: -len(s[1]))[:SECTIONS_DRAWN])
        text = "\n\n".join(t for _, t in drawn)
        source = sources[doc_id.partition("/")[0]]
        raw = text.encode()
        doc = ModelDoc(
            id=doc_id,
            source=source.key,
            title=title,
            url=f"{source.library_url}/{doc_id.partition('/')[2]}",
            collected_on=COLLECTED_ON,
            robots="allowed",
            leaked=source.leaked,
            drafted_by="Example Council" if source.leaked else None,
            raw_sha256=sha256(raw),
            media_type="text/plain",
            text_sha256=sha256(raw),
        )
        store(source, doc, raw=raw, text=text, root=modeldocs)
        # Generic sections are left out of a match's shares (lineage.text.generic_sections).
        words = sum(len(t.split()) for _, t in sections if generic_kind(normalize(t)) is None)
        model_sections = [norm for _, norm in matchable_sections(text)[1]]
        pairs = [
            SectionPair(model_section=model_sections.index(normalize(t)), ordinal=o, agreements=9)
            for o, t in drawn
        ]
        out.append(
            OriginMatch(
                bill_id=bill_id,
                doc_id=doc_id,
                stages=["near_duplicate", "semantic"] if doc_id == first else ["near_duplicate"],
                model_share=1.0,
                bill_share=round(sum(len(t.split()) for _, t in drawn) / words, 4),
                spans=[BillSpan(ordinal=o, start=0, end=len(t), kind="exact") for o, t in drawn],
                pairs=pairs,
            )
        )
        if doc_id == first:
            strongest = min(pairs, key=lambda p: (p.model_section, p.ordinal))
            bill_text = dict(drawn)[strongest.ordinal]
            label = {
                "pair": "fixture-1",
                "bill_id": bill_id,
                "doc_id": doc_id,
                "model_section": text_sha256(model_sections[strongest.model_section]),
                "bill_section": text_sha256(normalize(bill_text)),
                "label": 1,
                "labeler": f"fixture@{PROMPT.version}",
                "scale": SAME_PROVISION,
                "at": COLLECTED_ON.isoformat(),
            }
            labels.write_text(json.dumps(label) + "\n")
    return out


def build_fixture_export(root: Path) -> Path:
    """The fixture export under `root/export`, and the model documents it names under
    `root/modeldocs`; returns the export's directory."""
    built = tables_from_raw(root, build_raw_dir(root, COMMISSION), COMMISSION / "sessions.csv")
    tables = embed_with_stand_in(built, root / "embedded")
    build_semantic_pairs(tables, model=FAKE_MODEL, floor_bytes=0)
    sources = {c.source.key: c.source for c in collectors(PACKAGE).values()}
    modeldocs = root / "modeldocs"
    labels = root / "origin_labels.jsonl"
    write_origin_matches(tables, hand_made_origins(tables, modeldocs, sources, labels))
    no_pairs, no_stories, rules = root / "none.jsonl", root / "none.json", root / "stories.toml"
    no_pairs.touch()
    no_stories.write_text('{"stories": {}}')
    rules.write_text('[stories.origin]\nstates = "all"\n')
    withheld = root / "withheld.json"
    write_withheld(
        [Tally(doc_id=WITHHELD, title=DRAWN_FROM[WITHHELD][1], listed=1, labeled=3, correct=0)],
        withheld,
    )
    out = root / "export"
    build_export(
        tables,
        state=STATE,
        out=out,
        models_dir=None,
        cache=root / "no-cache",
        featured_pairs=no_pairs,
        featured_report=no_stories,
        stories_rules=rules,
        results=REPO / RESULTS,
        modeldocs_root=modeldocs,
        collector_package=PACKAGE,
        model=FAKE_MODEL,
        require_flag=False,
        floor=0,
        origin_labels=labels,
        withheld_report=withheld,
    )
    return out


def files_of(directory: Path) -> dict[str, object]:
    """Every JSON file under `directory`, parsed, by its path."""
    return {
        str(p.relative_to(directory)): json.loads(p.read_text())
        for p in sorted(directory.rglob("*.json"))
    }


def replace_committed(built: Path) -> None:
    if COMMITTED.exists():
        shutil.rmtree(COMMITTED)
    shutil.copytree(built, COMMITTED)
