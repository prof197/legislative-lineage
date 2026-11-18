"""`lineage export`: the site's JSON for one state, each file valid against the schema the
site's types come from, every comparison's spans indexing the section texts written beside it,
and featured pairs carrying their cached explanation (DESIGN.md#export)."""

import json
from pathlib import Path
from typing import Any

import pytest

from lineage import queries
from lineage.align import align
from lineage.core import db
from lineage.explain.explain import NotCachedError, explain, prompt_spans
from lineage.explain.explanations import ChangeExplanation, Claim
from lineage.export import (
    RESULTS,
    SCHEMA,
    ExportError,
    align_all,
    build_export,
    comparison_slug,
    schema_text,
    slug,
    stories_for,
    validation_from,
)
from lineage.export_schema import BillList, BillPage, BillTexts, Comparison, Meta
from lineage.match.semantic.pairs import build_semantic_pairs
from tests.conftest import FAKE_MODEL
from tests.test_explain import scripted
from tests.test_relatives import bill_of, most_related

REPO = Path(__file__).parent.parent


def test_a_bills_file_name_is_its_state_session_and_identifier() -> None:
    assert slug("GA", "2023_24", "SB 351") == "ga-2023_24-sb-351"
    assert slug("NJ", "2024-2025", "A 1234/S 99") == "nj-2024-2025-a-1234-s-99"
    assert comparison_slug("a", "b") == "a--b"


def test_the_committed_schema_is_the_one_the_models_emit() -> None:
    # Regenerate with `uv run lineage export --schema`.
    assert (REPO / SCHEMA).read_text() == schema_text()


def test_the_validation_figures_are_the_committed_results() -> None:
    results = json.loads((REPO / RESULTS).read_text())

    got = validation_from(results)

    assert got.development_recall.found == results["commission"]["recall"]["pooled"]["all"]["found"]
    assert got.faithfulness.share == results["faithfulness"]["overall"]["share"]
    assert got.faithfulness_judge == results["faithfulness"]["judge"]
    assert len(got.development_enactments) == len(results["commission"]["enactments"])
    # Ruled at the step 48 review: origin precision, ALEC's matches against every other
    # source's, with intervals, as the published reading holds them.
    origins = results["origins_precision"]
    assert got.origin_precision.overall.share == origins["overall"]["precision"]["precision"]
    assert got.origin_precision.compared.sources == ["alec"]
    alec = origins["compared"]["alec"]
    assert got.origin_precision.compared.interval_95 == tuple(alec["precision"]["interval_95"])
    assert got.origin_precision.rest.matches == origins["compared"]["rest"]["matches"]
    assert got.origin_precision.labeler == origins["labeler"]


@pytest.fixture
def tables(embedded_commission: Path) -> Path:
    build_semantic_pairs(embedded_commission, model=FAKE_MODEL, floor_bytes=0)
    return embedded_commission


def _export(tables: Path, out: Path, **kw: Any) -> dict[str, Any]:
    state = bill_of(tables, most_related(tables))[0]
    rules = out.parent / "stories.toml"
    rules.write_text('[stories.s]\nstates = "all"\n')
    no_pairs, no_stories = out.parent / "none.jsonl", out.parent / "none.json"
    no_pairs.touch()
    no_stories.write_text('{"stories": {}}')
    return build_export(
        tables,
        state=state,
        out=out,
        models_dir=None,
        cache=kw.pop("cache", out.parent / "no-cache"),
        featured_pairs=kw.pop("featured_pairs", no_pairs),
        featured_report=kw.pop("featured_report", no_stories),
        stories_rules=rules,
        results=kw.pop("results", REPO / RESULTS),
        model=FAKE_MODEL,
        require_flag=False,
        floor=0,
        **kw,
    )


def test_every_hosted_bill_gets_a_page_and_every_relative_a_comparison(
    tables: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    out.mkdir()
    (out / "stale.json").write_text("{}")

    report = _export(tables, out)

    assert not (out / "stale.json").exists()  # the output is replaced whole
    meta = Meta.model_validate_json((out / "meta.json").read_text())
    listing = BillList.model_validate_json((out / "bills.json").read_text())
    assert meta.hosted_bills == len(listing.bills) == report["hosted_bills"]
    assert report["command"].endswith(f"--state {meta.state}")
    assert report["files"]["bills"] == len(listing.bills)
    assert all(b.bill.state == meta.state for b in listing.bills)
    relatives = 0
    for entry in listing.bills:
        page = BillPage.model_validate_json((out / "bills" / f"{entry.bill.slug}.json").read_text())
        assert page.families == entry.families
        assert (entry.closest is None) == (not page.relatives)
        for rel in page.relatives:
            relatives += 1
            assert rel.bill.state != meta.state
            comparison = Comparison.model_validate_json(
                (out / "compare" / f"{rel.comparison}.json").read_text()
            )
            assert (comparison.a.slug, comparison.b.slug) == (page.bill.slug, rel.bill.slug)
            assert [(p.ordinal_a, p.ordinal_b) for p in comparison.pairs] == [
                (p.ordinal, p.other_ordinal) for p in rel.pairs
            ]
            assert all(p.explanation is None for p in comparison.pairs)  # nothing featured
        matched = {p.ordinal for rel in page.relatives for p in rel.pairs}
        assert {s.ordinal for s in page.sections if s.with_relatives} == matched
    assert relatives == report["relatives_listed"] > 0


def test_each_comparisons_spans_index_the_section_texts_written_beside_it(
    tables: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    report = _export(tables, out)

    texts: dict[str, dict[int, str]] = {}
    for path in (out / "texts").glob("*.json"):
        doc = BillTexts.model_validate_json(path.read_text())
        texts[doc.bill] = {s.ordinal: s.text for s in doc.sections}
    pairs = 0
    for path in (out / "compare").glob("*.json"):
        c = Comparison.model_validate_json(path.read_text())
        for p in c.pairs:
            pairs += 1
            ta, tb = texts[c.a.slug][p.ordinal_a], texts[c.b.slug][p.ordinal_b]
            assert [s.id for s in p.spans] == [s.id for s in align(ta, tb).spans]
            for s in p.spans:
                assert s.a is None or 0 <= s.a[0] < s.a[1] <= len(ta)
                assert s.b is None or 0 <= s.b[0] < s.b[1] <= len(tb)
    assert pairs == report["section_pairs_aligned"]


def test_a_sections_text_is_written_once_however_many_comparisons_show_it(
    tables: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    _export(tables, out)

    for path in (out / "texts").glob("*.json"):
        ordinals = [s.ordinal for s in BillTexts.model_validate_json(path.read_text()).sections]
        assert len(ordinals) == len(set(ordinals))
    for path in (out / "compare").glob("*.json"):
        assert '"text"' not in path.read_text()


def _one_pair(out: Path) -> tuple[Comparison, Any]:
    path = sorted((out / "compare").glob("*.json"))[0]
    c = Comparison.model_validate_json(path.read_text())
    return c, c.pairs[0]


def _bill_id(tables: Path, state: str, session: str, identifier: str) -> str:
    with db.connect_quiet() as con:
        found = queries.bill_id_of(
            con,
            bills_glob=str(tables / "bills" / "*.parquet"),
            state=state,
            session=session,
            identifier=identifier,
        )
    assert found is not None
    return found


def test_a_featured_pair_carries_its_cached_explanation_and_an_uncached_one_is_refused(
    tables: Path, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    _export(tables, first)
    c, pair = _one_pair(first)
    a = _bill_id(tables, c.a.state, c.a.session, c.a.identifier)
    b = _bill_id(tables, c.b.state, c.b.session, c.b.identifier)
    texts = {
        doc.bill: {s.ordinal: s.text for s in doc.sections}
        for doc in (
            BillTexts.model_validate_json((first / "texts" / f"{x}.json").read_text())
            for x in (c.a.slug, c.b.slug)
        )
    }
    ta, tb = texts[c.a.slug][pair.ordinal_a], texts[c.b.slug][pair.ordinal_b]
    featured = tmp_path / "featured.jsonl"
    featured.write_text(
        json.dumps(
            {
                "story": "s",
                "origin": {"bill_id": a, "ordinal": pair.ordinal_a},
                "descendant": {"bill_id": b, "ordinal": pair.ordinal_b},
            }
        )
        + "\n"
    )

    with pytest.raises(NotCachedError):
        _export(tables, tmp_path / "uncached", featured_pairs=featured)

    spans = align(ta, tb).spans
    cited = next(s for s in spans if s.kind != "copied" and s.b is not None)
    claim = Claim(
        kind="added", summary="Adds a clause.", origin_span_ids=[], descendant_span_ids=[cited.id]
    )
    cache = tmp_path / "cache"
    explain(
        prompt_spans(spans, ta, tb, origin_side="a"),
        ask=scripted(ChangeExplanation(claims=[claim]))[1],
        cache=cache,
    )
    report = _export(tables, tmp_path / "second", featured_pairs=featured, cache=cache)

    got = Comparison.model_validate_json(
        (
            tmp_path / "second" / "compare" / f"{comparison_slug(c.a.slug, c.b.slug)}.json"
        ).read_text()
    )
    explained = [p for p in got.pairs if p.explanation is not None]
    assert [(p.ordinal_a, p.ordinal_b) for p in explained] == [(pair.ordinal_a, pair.ordinal_b)]
    assert explained[0].explanation is not None
    assert [x.summary for x in explained[0].explanation.claims] == ["Adds a clause."]
    assert report["featured_pairs_explained"] == 1


def test_a_featured_pair_the_build_does_not_list_is_refused(tables: Path, tmp_path: Path) -> None:
    featured = tmp_path / "featured.jsonl"
    row = {"story": "s", "origin": {"bill_id": "x", "ordinal": 0}, "descendant": {}}
    row["descendant"] = {"bill_id": "y", "ordinal": 0}
    featured.write_text(json.dumps(row) + "\n")

    with pytest.raises(ExportError, match="not listed"):
        _export(tables, tmp_path / "site", featured_pairs=featured)


def test_a_missing_or_stale_committed_input_fails_before_any_alignment(
    tables: Path, tmp_path: Path
) -> None:
    # Run from a checkout whose committed files are older than the code, the export once
    # aligned for hours and then failed on the results; with the stories' files missing it
    # would have built a site without stories and said nothing.
    stale = tmp_path / "results.json"
    stale.write_text(json.dumps({"commission": {}}))
    aligned: list[int] = []

    def watch(line: str) -> None:
        if "align" in line:
            aligned.append(1)

    with pytest.raises(ExportError, match="results"):
        _export(tables, tmp_path / "stale", results=stale, progress=watch)
    with pytest.raises(ExportError, match=r"missing\.json"):
        _export(tables, tmp_path / "gone", featured_report=tmp_path / "missing.json")
    assert not aligned
    assert not (tmp_path / "stale").exists()
    assert not (tmp_path / "gone").exists()


def test_aligning_in_worker_processes_gives_what_one_process_does() -> None:
    texts = [
        ("The fee is set each fiscal year by the board.", "The fee is set every biennium."),
        ("Records are kept for five years.", "Records are kept for five years by the clerk."),
    ] * 70  # more than one task's worth

    serial = align_all(texts, models_dir=None, workers=1)
    parallel = align_all(texts, models_dir=None, workers=2)

    assert parallel == serial == [align(a, b) for a, b in texts]


def test_a_state_carries_the_stories_the_rules_give_it(tmp_path: Path) -> None:
    rules = tmp_path / "stories.toml"
    rules.write_text('[stories.shared]\nstates = "all"\n\n[stories.local]\nstates = ["GA"]\n')

    assert stories_for("GA", ["shared", "local"], rules) == {"shared", "local"}
    assert stories_for("AK", ["shared", "local"], rules) == {"shared"}
    with pytest.raises(ExportError, match="new-story"):
        stories_for("AK", ["shared", "new-story"], rules)


def test_the_committed_rules_place_every_committed_story() -> None:
    stories = json.loads((REPO / "reports" / "featured_pairs.json").read_text())["stories"]

    assert stories_for("GA", stories, REPO / "rules" / "stories.toml") == set(stories)
    assert "georgia-social-media" not in stories_for("AK", stories, REPO / "rules" / "stories.toml")
