"""The committed fixture export (tests/fixtures/export/) is what the export writes today, valid
against the schema the site's types come from, with an origin from a leaked source among them."""

import json
import os
from pathlib import Path

import pytest

from lineage.export.build import RESULTS, carries_origin_story
from lineage.export.origin_story import FILE as ORIGIN_STORY_FILE
from lineage.export.origin_story import story_file
from lineage.export.schema import BillPage, Meta, Sources
from lineage.live.publish import bill_list, bill_page, plan_export, texts_by_slug
from lineage.origins.collect import collectors
from lineage.origins.labels import judged_correct, load_labels
from lineage.origins.matches import read_origin_matches
from lineage.origins.modeldocs import read_docs
from tests.conftest import FAKE_MODEL
from tests.fixture_export import (
    COMMITTED,
    DRAWN_FROM,
    PACKAGE,
    REPO,
    STATE,
    WITHHELD,
    build_fixture_export,
    files_of,
    replace_committed,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = build_fixture_export(tmp_path_factory.mktemp("fixture-export"))
    if os.environ.get("LINEAGE_REGENERATE") == "1":
        replace_committed(out)
    return out


def test_the_committed_fixture_export_is_what_the_export_writes(built: Path) -> None:
    # Regenerate with `LINEAGE_REGENERATE=1 uv run pytest tests/test_fixture_export.py`.
    assert files_of(COMMITTED) == files_of(built)


def test_the_fixture_shows_origins_and_their_sources() -> None:
    meta = Meta.model_validate_json((COMMITTED / "meta.json").read_text())
    sources = Sources.model_validate_json((COMMITTED / "sources.json").read_text())
    pages = [BillPage.model_validate_json(p.read_text()) for p in (COMMITTED / "bills").glob("*")]

    assert meta.state == "UT"
    assert {s.key for s in sources.sources} == {"example-archive", "example-library"}
    origins = [o for p in pages for o in p.origins]
    assert {o.doc.source for o in origins} == {"example-archive", "example-library"}
    assert any(o.doc.leaked for o in origins)
    assert all(0 < o.bill_share <= 1 and o.sections for o in origins)


def test_a_match_is_judged_correct_only_on_the_sections_its_label_names(built: Path) -> None:
    """The featured story's gate (ruled at the step 48 review): a match is judged by a label on
    the sections its strongest pair rests on now, and by no other."""
    root = built.parent
    tables, modeldocs = root / "embedded", root / "modeldocs"
    matches = read_origin_matches(tables)
    docs = [d for c in collectors(PACKAGE).values() for d in read_docs(c.source, modeldocs)]
    labels = load_labels(root / "origin_labels.jsonl")
    keys = {(m.bill_id, m.doc_id) for m in matches}
    first = next(iter(DRAWN_FROM))
    judged = judged_correct(keys, matches, docs, tables, labels, root=modeldocs)
    assert judged == {(DRAWN_FROM[first][0], first)}

    elsewhere = [r | {"bill_section": "0" * 64} for r in labels]
    assert judged_correct(keys, matches, docs, tables, elsewhere, root=modeldocs) == set()
    wrong = [r | {"label": 0} for r in labels]
    assert judged_correct(keys, matches, docs, tables, wrong, root=modeldocs) == set()
    assert json.loads((COMMITTED / "origin_story.json").read_text())["passed_over"] == 0


def test_the_live_tracks_rendering_gives_the_exports_pages_byte_for_byte(built: Path) -> None:
    """Ruled at the step 48 review: until lineage.live.publish shares the export's rendering, its
    own copy must give the same files for the same bills, read from the same tables: every bill
    page, the bill index, every bill's section texts and the origin story."""
    root = built.parent
    plan = plan_export(
        root / "embedded",
        state=STATE,
        cache=root / "no-cache",
        featured_pairs=root / "none.jsonl",
        featured_report=root / "none.json",
        stories_rules=root / "stories.toml",
        results=REPO / RESULTS,
        modeldocs_root=root / "modeldocs",
        collector_package=PACKAGE,
        model=FAKE_MODEL,
        require_flag=False,
        origin_labels=root / "origin_labels.jsonl",
        withheld_report=root / "withheld.json",
    )

    def same(name: str, rendered: bytes) -> None:
        assert rendered == (built / name).read_bytes(), name

    assert plan.hosted
    for bill_id in plan.hosted:
        same(
            f"bills/{plan.refs[bill_id].slug}.json",
            bill_page(plan, bill_id).model_dump_json().encode(),
        )
    same("bills.json", bill_list(plan).model_dump_json().encode())
    texts = texts_by_slug(plan)
    assert {f"texts/{s}.json" for s in texts} == {
        p.relative_to(built).as_posix() for p in built.glob("texts/*.json")
    }
    for slug, doc in texts.items():
        same(f"texts/{slug}.json", doc.model_dump_json().encode())
    assert carries_origin_story(STATE, root / "stories.toml")
    story = story_file((bill_page(plan, b) for b in plan.hosted), plan.judged)
    assert story is not None
    same(ORIGIN_STORY_FILE, story.model_dump_json().encode())


def test_a_withheld_document_is_on_the_sources_page_and_on_no_bill_page(built: Path) -> None:
    """Ruled at the deploy review: a document whose labeled matches are mostly wrong is withheld
    from bill pages, the feed and stories, and the sources page lists it, with the reason."""
    sources = json.loads((built / "sources.json").read_text())["sources"]
    withheld = [w for s in sources for w in s["withheld"]]
    assert [(w["id"], w["reason"]) for w in withheld] == [(WITHHELD, "generic_text")]
    named = {
        o["doc"]["id"]
        for p in (built / "bills").glob("*.json")
        for o in json.loads(p.read_text())["origins"]
    }
    assert named, "the fixture lists its other origins"
    assert WITHHELD not in named
    story = json.loads((built / "origin_story.json").read_text())
    assert story["origin"]["doc"]["id"] != WITHHELD
    meta = json.loads((built / "meta.json").read_text())["validation"]["origin_precision"]
    assert (meta["withheld_documents"], meta["withheld_matches"]) == (1, 1)
