"""`lineage origins`: model documents as a second collection, each queried against the corpus.

A document's text is treated exactly as `lineage query` treats a bill's (lineage.query): cleaned,
split and normalized, boilerplate-only sections set aside, the rest cut into windows, signed,
looked up in the band index under its cap and verified at the build's threshold, and with the
semantic stage embedded and searched exactly, each window keeping its K nearest corpus texts.
Hits are listed as a page lists them (lineage.listing): by meaning at the stage's threshold, and
without the state boilerplate flag's sections unless both stages found them. The bill index is
only read, never written, so a document added or removed changes no bill's relatives.

A match is kept per bill, not per family, since a bill page shows its own origins. It gives two
shares, both scored by words as a query's family is (DESIGN.md#bill-scoring): of the model's
words searched, the share in its sections the bill matches, and of the bill's matchable words,
the share in its sections the model matches. The spans are the bill's side of each matched
section pair's alignment (lineage.align): copied blocks as exact, sentences paired by meaning as
paraphrase, merged over the model's sections, an exact span winning where the two overlap. The
model's own text appears nowhere in a match.

Every document's windows are searched together, in id order, a pass over the corpus's vectors
per lineage.semantic.WINDOWS_PER_PASS windows rather than one per document, and everything is
ordered before it is returned, so the same documents and tables give the same matches, byte for
byte, whichever order they come in. Matched alone, a document gives the same matches unless a
cosine sits within float rounding of a threshold, since a matrix product's last bits can depend
on how many rows it multiplies.

A model window keeps its K nearest corpus texts, as a query's does, so an act copied with
rewording into more than K distinct versions is found by meaning in only the K closest; the
near-duplicate stage has no such limit. See DESIGN.md#origin-matches.
"""

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.align import align
from lineage.disk import require_free_space
from lineage.embed import embed as embed_texts
from lineage.embed import load_model
from lineage.listing import MeaningHit, NearHit, SectionKey, flagged_sections
from lineage.modeldocs import ModelDoc, ModelDocError, load_text
from lineage.models import MODELS_DIR
from lineage.origins import BillSpan, OriginMatch, SectionPair, Stage
from lineage.query import matchable_sections, near_duplicate_hits, require_index, sign_query
from lineage.semantic import CORROBORATING, THRESHOLD, Embedder, open_search, search_sections

SHARE_PLACES = 4  # as the export and the results round a share
COSINE_PLACES = 4


@dataclass(frozen=True, slots=True)
class ModelText:
    """A document's matchable sections, as a query's are."""

    doc_id: str
    raw: str
    display: list[str]
    words: list[int]  # each section's normalized words, the shares' unit

    @property
    def searched(self) -> set[int]:
        """The sections either stage searches: with the semantic stage, every matchable section
        with a display word (lineage.query.query_text)."""
        return {q for q, text in enumerate(self.display) if text.split()}


def model_text(doc_id: str, raw: str) -> ModelText:
    _, matchable = matchable_sections(raw)
    return ModelText(
        doc_id,
        raw,
        [text for text, _ in matchable],
        [len(normalized.split()) for _, normalized in matchable],
    )


def listed(
    near: Sequence[NearHit],
    by_meaning: Sequence[MeaningHit],
    *,
    threshold: float = THRESHOLD,
    flagged: Collection[SectionKey] = frozenset(),
    exempt_at: float = CORROBORATING,
) -> tuple[list[NearHit], list[MeaningHit]]:
    """The hits a page lists, by the rule of lineage.listing.listed_hits, keeping each hit's
    corpus section: by meaning at or above `threshold`, and without a hit on a flagged section
    unless both stages hit it from the same model section, by meaning at `exempt_at` or above."""
    near_at = {(q, b, o) for q, b, o, _ in near}
    meaning_at = {(q, b, o) for q, b, o, c in by_meaning if c >= exempt_at}
    return (
        [h for h in near if (h[1], h[2]) not in flagged or h[:3] in meaning_at],
        [
            h
            for h in by_meaning
            if h[3] >= threshold and ((h[1], h[2]) not in flagged or h[:3] in near_at)
        ],
    )


@dataclass(slots=True)
class _BillHits:
    stages: set[Stage] = field(default_factory=set[Stage])
    pairs: set[tuple[int, int]] = field(default_factory=set[tuple[int, int]])  # (model, bill)
    agreements: dict[tuple[int, int], int] = field(default_factory=dict[tuple[int, int], int])
    cosine: dict[tuple[int, int], float] = field(default_factory=dict[tuple[int, int], float])


def _merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _less(intervals: list[tuple[int, int]], cut: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """`intervals` with every character in `cut` removed; both merged and sorted."""
    out: list[tuple[int, int]] = []
    for start, end in intervals:
        at = start
        for c_start, c_end in cut:
            if c_end <= at or c_start >= end:
                continue
            if c_start > at:
                out.append((at, c_start))
            at = max(at, c_end)
        if at < end:
            out.append((at, end))
    return out


def bill_spans(
    ordinal: int,
    bill_text: str,
    model_sections: Sequence[str],
    embed: Embedder | None,
) -> list[BillSpan]:
    """The characters of one bill section that match any of `model_sections`, from each pair's
    alignment, merged by kind; exact wins over paraphrase."""
    exact: list[tuple[int, int]] = []
    paraphrase: list[tuple[int, int]] = []
    for model_section in model_sections:
        for span in align(model_section, bill_text, embed=embed).spans:
            if span.b is None:
                continue
            if span.kind == "copied":
                exact.append(span.b)
            elif span.kind == "paraphrased":
                paraphrase.append(span.b)
    exact = _merged(exact)
    paraphrase = _less(_merged(paraphrase), exact)
    spans = [BillSpan(ordinal=ordinal, start=s, end=e, kind="exact") for s, e in exact] + [
        BillSpan(ordinal=ordinal, start=s, end=e, kind="paraphrase") for s, e in paraphrase
    ]
    return sorted(spans, key=lambda s: (s.start, s.end))


def match_texts(
    tables_dir: Path,
    texts: Sequence[ModelText],
    embed: Embedder,
    *,
    model: str,
    flagged: Collection[SectionKey] = frozenset(),
) -> list[OriginMatch]:
    """Every bill each text matches, through both stages, its shares and its spans."""
    require_index(tables_dir)
    search = open_search(tables_dir, embed, model)
    # Every document's sections in one search, which reads the corpus's vectors once per
    # WINDOWS_PER_PASS windows, where a search per document would read them once each.
    offsets: list[tuple[int, int]] = []  # (document, section) of each searched section
    for d, text in enumerate(texts):
        offsets += [(d, q) for q in range(len(text.display))]
    found = search_sections(
        [s for text in texts for s in text.display],
        search,
        threshold=min(THRESHOLD, CORROBORATING),
    )
    meaning_of: dict[int, list[MeaningHit]] = defaultdict(list)
    for hit in found:
        d, q = offsets[hit.query_section]
        meaning_of[d].append((q, hit.bill_id, hit.ordinal, hit.cosine))

    hits: dict[tuple[int, str], _BillHits] = defaultdict(_BillHits)  # (document, bill)
    for d, text in enumerate(texts):
        near, by_meaning = listed(
            near_duplicate_hits(sign_query(text.raw, tables_dir), tables_dir),
            meaning_of[d],
            flagged=flagged,
        )
        for q, bill_id, ordinal, count in near:
            entry = hits[(d, bill_id)]
            entry.stages.add("near_duplicate")
            entry.pairs.add((q, ordinal))
            entry.agreements[(q, ordinal)] = max(count, entry.agreements.get((q, ordinal), 0))
        for q, bill_id, ordinal, cosine in by_meaning:
            entry = hits[(d, bill_id)]
            entry.stages.add("semantic")
            entry.pairs.add((q, ordinal))
            best = max(cosine, entry.cosine.get((q, ordinal), -1.0))
            entry.cosine[(q, ordinal)] = best
    if not hits:
        return []

    with queries.connect_quiet() as con:
        rows = queries.matchable_sections_of(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
            ids=sorted({bill_id for _, bill_id in hits}),
        )
    sections: dict[str, dict[int, tuple[str, int]]] = defaultdict(dict)
    for bill_id, ordinal, display, normalized in rows:
        sections[bill_id][ordinal] = (display, len(normalized.split()))

    out: list[OriginMatch] = []
    for (d, bill_id), entry in sorted(hits.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        text = texts[d]
        bill = sections[bill_id]
        model_words = sum(text.words[q] for q in text.searched)
        bill_words = sum(words for _, words in bill.values())
        matched_model = {q for q, _ in entry.pairs}
        matched_bill = {o for _, o in entry.pairs}
        by_ordinal: dict[int, list[int]] = defaultdict(list)
        for q, o in sorted(entry.pairs):
            by_ordinal[o].append(q)
        spans = [
            span
            for o in sorted(by_ordinal)
            for span in bill_spans(o, bill[o][0], [text.display[q] for q in by_ordinal[o]], embed)
        ]
        out.append(
            OriginMatch(
                bill_id=bill_id,
                doc_id=text.doc_id,
                stages=sorted(entry.stages),
                model_share=_share(sum(text.words[q] for q in matched_model), model_words),
                bill_share=_share(sum(bill[o][1] for o in matched_bill), bill_words),
                spans=spans,
                pairs=[
                    SectionPair(
                        model_section=q,
                        ordinal=o,
                        agreements=entry.agreements.get((q, o)),
                        cosine=None
                        if (q, o) not in entry.cosine
                        else round(float(entry.cosine[(q, o)]), COSINE_PLACES),
                    )
                    for q, o in sorted(entry.pairs)
                ],
            )
        )
    return out


def _share(part: int, whole: int) -> float:
    return round(min(part / whole, 1.0), SHARE_PLACES) if whole else 0.0


def _embedder(model: str, models_dir: Path) -> Embedder:
    loaded = load_model(model, models_dir)

    def embed(texts: Sequence[str]) -> NDArray[np.float32]:
        return embed_texts(loaded, texts)

    return embed


def match_models(
    tables_dir: Path,
    docs: Sequence[ModelDoc],
    *,
    modeldocs_root: Path,
    model: str,
    floor_bytes: int,
    embed: Embedder | None = None,
    models_dir: Path = MODELS_DIR,
) -> list[OriginMatch]:
    """Every bill each document's text matches, through the existing sections, windows and both
    stages, without changing the bill index; deterministic. The semantic stage embeds with
    `model` from `models_dir`, unless `embed` stands in for it."""
    ids = [d.id for d in docs]
    if len(set(ids)) != len(ids):
        raise ModelDocError("a document is listed twice; which text stands would depend on order")
    require_index(tables_dir)
    # A model several sources hold is one text: it is matched once, under the first of its
    # documents by id, and each document holding it gets the same matches, so every source's
    # provenance is kept and none is matched twice.
    holders: dict[str, list[str]] = defaultdict(list)
    for d in sorted(docs, key=lambda d: d.id):
        holders[d.text_sha256].append(d.id)
    first = {ids[0]: ids for ids in holders.values()}
    texts = [
        model_text(d.id, load_text(d, modeldocs_root))
        for d in sorted(docs, key=lambda d: d.id)
        if d.id in first
    ]
    if not texts:
        return []
    matched = match_texts(
        tables_dir,
        texts,
        embed if embed is not None else _embedder(model, models_dir),
        model=model,
        flagged=flagged_sections(tables_dir),
    )
    matches = sorted(
        (m.model_copy(update={"doc_id": held}) for m in matched for held in first[m.doc_id]),
        key=lambda m: (m.bill_id, m.doc_id),
    )
    # The caller writes them (lineage.origins.write_origin_matches); this bounds that write.
    needed = sum(len(m.model_dump_json()) + 1 for m in matches)
    require_free_space(tables_dir, needed=needed, floor=floor_bytes)
    return matches
