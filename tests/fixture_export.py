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
from lineage.export import RESULTS, build_export
from lineage.modeldocs import ModelDoc, Source, sha256, store
from lineage.origins import BillSpan, OriginMatch, write_origin_matches
from lineage.semantic_pairs import build_semantic_pairs
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
}
SECTIONS_DRAWN = 2


def hand_made_origins(
    tables: Path, modeldocs: Path, sources: dict[str, Source]
) -> list[OriginMatch]:
    ids = [bill for bill, _ in DRAWN_FROM.values()]
    with queries.connect_quiet() as con:
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
            raw_sha256=sha256(raw),
            media_type="text/plain",
            text_sha256=sha256(raw),
        )
        store(source, doc, raw=raw, text=text, root=modeldocs)
        words = sum(len(t.split()) for _, t in sections)
        out.append(
            OriginMatch(
                bill_id=bill_id,
                doc_id=doc_id,
                stages=["near_duplicate"],
                model_share=1.0,
                bill_share=round(sum(len(t.split()) for _, t in drawn) / words, 4),
                spans=[BillSpan(ordinal=o, start=0, end=len(t), kind="exact") for o, t in drawn],
            )
        )
    return out


def build_fixture_export(root: Path) -> Path:
    """The fixture export under `root/export`, and the model documents it names under
    `root/modeldocs`; returns the export's directory."""
    built = tables_from_raw(root, build_raw_dir(root, COMMISSION), COMMISSION / "sessions.csv")
    tables = embed_with_stand_in(built, root / "embedded")
    build_semantic_pairs(tables, model=FAKE_MODEL, floor_bytes=0)
    sources = {c.source.key: c.source for c in collectors(PACKAGE).values()}
    modeldocs = root / "modeldocs"
    write_origin_matches(tables, hand_made_origins(tables, modeldocs, sources))
    no_pairs, no_stories, rules = root / "none.jsonl", root / "none.json", root / "stories.toml"
    no_pairs.touch()
    no_stories.write_text('{"stories": {}}')
    rules.write_text("")
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
