"""Origin matching: model documents queried against the corpus through both stages.

The synthetic documents of the fixture export (tests/fixture_export.py) are copied whole from
fixture bills, so each must come back as its hand-made match says. The commission's act texts,
which cannot be committed, are matched as model documents against the commission fixture when
they are present locally, and must find every enactment the v1 eval finds there.
"""

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from lineage import origin_matching
from lineage.acts import ACTS, ACTS_DIR, enacted_text
from lineage.collect import collectors
from lineage.commission import evaluate
from lineage.listing import listed_hits
from lineage.modeldocs import (
    ModelDoc,
    ModelDocError,
    Source,
    load_text,
    read_docs,
    sha256,
    store,
    text_path,
)
from lineage.origin_labels import build_items, strongest_pair
from lineage.origin_matching import bill_spans, listed, match_models
from lineage.origin_precision import OriginPrecisionError
from lineage.origins import BillSpan, OriginMatch, read_origin_matches, write_origin_matches
from lineage.semantic import CORROBORATING, THRESHOLD, open_search
from lineage.sessions import load_sessions
from lineage.ulc import ENACTMENTS, load_enactments
from tests.conftest import COMMISSION, FAKE_MODEL
from tests.fakes import MARK, bag_of_words
from tests.fixture_export import PACKAGE
from tests.fixture_export import hand_made_origins as hand_made


def _sources() -> dict[str, Source]:
    return {c.source.key: c.source for c in collectors(PACKAGE).values()}


def _docs(root: Path) -> list[ModelDoc]:
    return [d for s in _sources().values() for d in read_docs(s, root)]


def _match(tables: Path, docs: list[ModelDoc], root: Path) -> list[OriginMatch]:
    return match_models(
        tables, docs, modeldocs_root=root, model=FAKE_MODEL, floor_bytes=0, embed=bag_of_words
    )


def _lines(matches: list[OriginMatch]) -> str:
    return "".join(m.model_dump_json() + "\n" for m in matches)


def _digests(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def synthetic(embedded_commission: Path, tmp_path: Path) -> tuple[Path, Path, list[OriginMatch]]:
    """The embedded commission fixture, the fixture export's model documents stored under a
    root of their own, and the matches written by hand for them."""
    root = tmp_path / "modeldocs"
    return embedded_commission, root, hand_made(embedded_commission, root, _sources())


def test_a_model_copied_whole_yields_its_hand_made_match(
    synthetic: tuple[Path, Path, list[OriginMatch]],
) -> None:
    tables, root, expected = synthetic

    matches = {(m.bill_id, m.doc_id): m for m in _match(tables, _docs(root), root)}

    for want in expected:
        got = matches[(want.bill_id, want.doc_id)]
        assert "near_duplicate" in got.stages
        assert got.model_share == want.model_share == 1.0
        # The hand-made share counts display words, matching counts normalized ones, as a
        # bill's relatives do (lineage.relatives); the two differ by a word or so.
        assert got.bill_share == pytest.approx(want.bill_share, abs=0.01)
        assert got.spans == want.spans


def test_every_match_is_kept_by_bill_with_shares_and_spans_on_the_bill_side(
    synthetic: tuple[Path, Path, list[OriginMatch]],
) -> None:
    tables, root, _ = synthetic

    matches = _match(tables, _docs(root), root)

    assert matches == sorted(matches, key=lambda m: (m.bill_id, m.doc_id))
    assert len({(m.bill_id, m.doc_id) for m in matches}) == len(matches)
    for m in matches:
        assert m.stages == sorted(set(m.stages))
        assert 0 < m.model_share <= 1
        assert 0 < m.bill_share <= 1
        assert all(s.end > s.start for s in m.spans)
        assert m.spans == sorted(m.spans, key=lambda s: (s.ordinal, s.start, s.end))
    # Other states' enactments of the same act match the documents drawn from Utah's.
    assert len({m.bill_id for m in matches}) > len(_docs(root))


def test_each_match_keeps_its_section_pairs_with_each_stages_evidence(
    synthetic: tuple[Path, Path, list[OriginMatch]],
) -> None:
    tables, root, _ = synthetic

    for m in _match(tables, _docs(root), root):
        assert m.pairs == sorted(m.pairs, key=lambda p: (p.model_section, p.ordinal))
        # Every section a span is in was paired, and a stage's evidence appears only when the
        # stage found the match.
        assert {s.ordinal for s in m.spans} <= {p.ordinal for p in m.pairs}
        assert any(p.agreements for p in m.pairs) == ("near_duplicate" in m.stages)
        assert any(p.cosine is not None for p in m.pairs) == ("semantic" in m.stages)
        assert all(p.cosine is None or p.cosine >= CORROBORATING for p in m.pairs)


def test_matching_is_deterministic_and_leaves_the_bill_index_alone(
    synthetic: tuple[Path, Path, list[OriginMatch]], tmp_path: Path
) -> None:
    tables, root, _ = synthetic
    before = _digests(tables)
    docs = _docs(root)

    first = _match(tables, docs, root)
    again = _match(tables, list(reversed(docs)), root)
    alone = [m for d in docs for m in _match(tables, [d], root)]

    assert _lines(first) == _lines(again)
    assert _lines(first) == _lines(sorted(alone, key=lambda m: (m.bill_id, m.doc_id)))
    assert _digests(tables) == before
    out = tmp_path / "out"
    write_origin_matches(out, first)
    assert read_origin_matches(out) == first


def test_a_document_listed_twice_is_refused(
    synthetic: tuple[Path, Path, list[OriginMatch]],
) -> None:
    tables, root, _ = synthetic
    doc = _docs(root)[0]

    with pytest.raises(ModelDocError, match="twice"):
        _match(tables, [doc, doc], root)


def test_a_model_two_sources_hold_is_matched_once_and_credited_to_each(
    synthetic: tuple[Path, Path, list[OriginMatch]], monkeypatch: pytest.MonkeyPatch
) -> None:
    tables, root, _ = synthetic
    original = _docs(root)[0]
    copy = original.model_copy(
        update={"id": f"second-library/{original.id.partition('/')[2]}", "source": "second-library"}
    )
    held = text_path(copy, root)
    held.parent.mkdir(parents=True)
    held.write_text(load_text(original, root))
    seen: list[int] = []
    real = origin_matching.match_texts

    def counted(*args: Any, **kwargs: Any) -> list[OriginMatch]:
        seen.append(len(args[1]))
        return real(*args, **kwargs)

    monkeypatch.setattr(origin_matching, "match_texts", counted)
    matches = _match(tables, [original, copy], root)

    assert seen == [1]  # one text, matched once
    by_doc = {d: [m for m in matches if m.doc_id == d] for d in (original.id, copy.id)}
    assert by_doc[original.id]
    strip = [m.model_copy(update={"doc_id": ""}) for m in by_doc[copy.id]]
    assert strip == [m.model_copy(update={"doc_id": ""}) for m in by_doc[original.id]]
    assert matches == sorted(matches, key=lambda m: (m.bill_id, m.doc_id))


def test_no_documents_no_matches(embedded_commission: Path, tmp_path: Path) -> None:
    assert _match(embedded_commission, [], tmp_path) == []


def test_hits_are_listed_as_a_page_lists_them() -> None:
    near = [(0, "a", 1, 90), (0, "b", 2, 70), (1, "b", 2, 66), (2, "c", 0, 80)]
    by_meaning = [
        (0, "a", 1, 0.99),
        (1, "b", 2, CORROBORATING),
        (0, "d", 4, THRESHOLD),
        (2, "d", 4, THRESHOLD - 0.001),
        (3, "c", 0, 0.97),
    ]
    flagged = {("b", 2), ("c", 0)}

    kept_near, kept_meaning = listed(near, by_meaning, flagged=flagged)

    page_near, page_meaning = listed_hits(near, by_meaning, threshold=THRESHOLD, flagged=flagged)
    assert [(q, b, a) for q, b, _, a in kept_near] == page_near
    assert [(q, b, c) for q, b, _, c in kept_meaning] == page_meaning
    # b's section 2 stands from section 1, where both stages hit it, and c's from neither.
    assert kept_near == [(0, "a", 1, 90), (1, "b", 2, 66)]
    assert kept_meaning == [(0, "a", 1, 0.99), (0, "d", 4, THRESHOLD)]


def test_a_bill_section_keeps_what_each_model_section_copied_or_paraphrased() -> None:
    first = "The court shall hear the motion within sixty days of its filing."
    second = "A party may appeal an order granting or denying the motion as of right."
    reworded = " ".join(w + MARK for w in second.split())
    bill = f"{first} Something the model never said appears here. {reworded}"

    spans = bill_spans(3, bill, [first, second], bag_of_words)

    assert spans == [
        BillSpan(ordinal=3, start=0, end=len(first), kind="exact"),
        BillSpan(ordinal=3, start=bill.index(reworded), end=len(bill), kind="paraphrase"),
    ]
    # Without a model only copies are marked, and a copy found twice is one span.
    assert bill_spans(3, bill, [first, first], None) == spans[:1]


def _act_docs(root: Path) -> list[ModelDoc]:
    """Each development act's text, as the v1 eval queries it, stored as a model document."""
    source = Source(
        key="acts",
        organization="Uniform Law Commission (local eval copy)",
        descriptor="the development acts, read from the local data directory",
        library_url="https://example.org/acts",
        terms_note="local eval only; never committed",
        terms_read_on=date(2026, 11, 15),
    )
    docs: list[ModelDoc] = []
    for act in ACTS:
        raw = (ACTS_DIR / f"{act}.txt").read_bytes()
        text = enacted_text(raw.decode("utf-8-sig"))
        doc = ModelDoc(
            id=f"acts/{act}",
            source="acts",
            title=act,
            url=f"https://example.org/acts/{act}",
            collected_on=date(2026, 11, 15),
            robots="no_robots_txt",
            leaked=False,
            raw_sha256=sha256(raw),
            media_type="text/plain",
            text_sha256=sha256(text.encode()),
        )
        store(source, doc, raw=raw, text=text, root=root)
        docs.append(doc)
    return docs


@pytest.mark.skipif(
    not all((ACTS_DIR / f"{act}.txt").exists() for act in ACTS),
    reason="the act texts are local data (README.md#uniform-act-texts)",
)
def test_the_act_texts_as_models_find_every_enactment_v1_finds(
    embedded_commission: Path, tmp_path: Path
) -> None:
    root = tmp_path / "modeldocs"
    docs = _act_docs(root)
    v1 = evaluate(
        {act: enacted_text((ACTS_DIR / f"{act}.txt").read_text("utf-8-sig")) for act in ACTS},
        load_enactments(ENACTMENTS),
        embedded_commission,
        load_sessions(COMMISSION / "sessions.csv"),
        semantic=open_search(embedded_commission, bag_of_words, FAKE_MODEL),
    )

    matched = {(m.doc_id, m.bill_id) for m in _match(embedded_commission, docs, root)}

    enactments: list[dict[str, Any]] = v1["enactments"]
    found_v1 = {_enactment(e) for e in enactments if e["found"]}
    found = {
        _enactment(e)
        for e in enactments
        if any((f"acts/{e['act']}", b["id"]) in matched for b in e["bills"])
    }
    assert found >= found_v1, sorted(found_v1 - found)


def _enactment(e: dict[str, Any]) -> str:
    return f"{e['act']} {e['state']} {e['year']}"


def test_a_sampled_match_is_read_back_as_its_strongest_pair_both_sections_whole(
    synthetic: tuple[Path, Path, list[OriginMatch]],
) -> None:
    tables, root, _ = synthetic
    docs = _docs(root)
    matches = _match(tables, docs, root)
    rows = [
        {"pair": f"o{n:04d}", "stages": m.stages, "bill_id": m.bill_id, "doc_id": m.doc_id}
        for n, m in enumerate(matches[:5], start=1)
    ]

    items = build_items(rows, matches, docs, tables, root=root)

    assert [i.pair for i in items] == [r["pair"] for r in rows]
    for i, m in zip(items, matches, strict=False):
        best = strongest_pair(m)
        # The model's section is asked first, the bill's second, each whole and each keyed by
        # its normalized text's hash, as a label row names it.
        assert (i.item.query.ref.bill_id, i.item.query.ref.ordinal) == (
            m.doc_id,
            best.model_section,
        )
        assert (i.item.match.ref.bill_id, i.item.match.ref.ordinal) == (m.bill_id, best.ordinal)
        assert i.item.query.text.strip()
        assert i.item.match.text.strip()
    stale = [r | {"doc_id": "nowhere/else"} for r in rows[:1]]
    with pytest.raises(OriginPrecisionError, match="not in the matches table"):
        build_items(stale, matches, docs, tables, root=root)
