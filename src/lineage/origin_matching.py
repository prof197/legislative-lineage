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

See DESIGN.md#dr-8.
"""

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.align import Alignment, align
from lineage.core import db
from lineage.core.disk import require_free_space
from lineage.embed import embed as embed_texts
from lineage.embed import load_model
from lineage.listing import MeaningHit, NearHit, SectionKey, flagged_sections
from lineage.modeldocs import ModelDoc, ModelDocError, load_text
from lineage.models import MODELS_DIR
from lineage.origins import BillSpan, OriginMatch, SectionPair, Stage
from lineage.query import near_duplicate_hits, require_index, sign_query
from lineage.semantic import CORROBORATING, THRESHOLD, Embedder, open_search, search_sections
from lineage.text.boilerplate import boilerplate_kind
from lineage.text.generic_sections import (
    MIN_WORDS,
    Inside,
    generic_inside,
    generic_kind,
    words_outside,
)
from lineage.text.sections import normalize, split_sections

SHARE_PLACES = 4  # as the export and the results round a share
COSINE_PLACES = 4


@dataclass(frozen=True, slots=True)
class ModelText:
    """A document's matchable sections, as a query's are."""

    doc_id: str
    raw: str
    display: list[str]
    words: list[int]  # each section's normalized words, the shares' unit
    # Sections a match leaves out on the model's side (lineage.text.generic_sections)
    generic: frozenset[int] = frozenset()
    # Generic text inside a section that is not generic (lineage.text.generic_sections), by section
    inside: dict[int, Inside] = field(default_factory=dict[int, Inside])

    def counted(self, q: int) -> int:
        """Section q's words a share counts: those outside its generic text."""
        return self.words[q] - (self.inside[q].words if q in self.inside else 0)

    @property
    def searched(self) -> set[int]:
        """The sections either stage searches: with the semantic stage, every matchable section
        with a display word (lineage.query.query_text)."""
        return {q for q, text in enumerate(self.display) if text.split()}


def model_text(doc_id: str, raw: str) -> ModelText:
    """The document's matchable sections as lineage.query.matchable_sections gives them, and
    their generic text, read over every section in order so that an article running through
    the window fallback's windows is read whole."""
    split = split_sections(raw)
    texts = [section.text for section in split.sections]
    inside_of = generic_inside(texts, continuous=not split.by_heading)
    matchable = [
        (text, norm, inside)
        for text, inside in zip(texts, inside_of, strict=True)
        if boilerplate_kind(norm := normalize(text)) is None
    ]
    generic, inside = _generic(
        [(len(norm.split()), generic_kind(norm), ins) for _, norm, ins in matchable]
    )
    return ModelText(
        doc_id,
        raw,
        [text for text, _, _ in matchable],
        [len(norm.split()) for _, norm, _ in matchable],
        generic,
        inside,
    )


def _generic(
    sections: Sequence[tuple[int, str | None, Inside | None]],
) -> tuple[frozenset[int], dict[int, Inside]]:
    """Which sections are generic, whole, and the generic text inside the rest: a section is
    generic by its kind, or when less than MIN_WORDS of it lies outside its generic text."""
    whole: set[int] = set()
    inside: dict[int, Inside] = {}
    for n, (words, kind, ins) in enumerate(sections):
        if kind is not None or (ins is not None and words - ins.words < MIN_WORDS):
            whole.add(n)
        elif ins is not None:
            inside[n] = ins
    return frozenset(whole), inside


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
class BillHits:
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
    return _spans_of(ordinal, [align(m, bill_text, embed=embed) for m in model_sections])


def _spans_of(
    ordinal: int, alignments: Sequence[Alignment], cut: Sequence[tuple[int, int]] = ()
) -> list[BillSpan]:
    """A bill section's spans from its pairs' alignments, merged by kind, exact winning over
    paraphrase, with the characters in `cut`, its generic text, left out."""
    exact: list[tuple[int, int]] = []
    paraphrase: list[tuple[int, int]] = []
    for alignment in alignments:
        for span in alignment.spans:
            if span.b is None:
                continue
            if span.kind == "copied":
                exact.append(span.b)
            elif span.kind == "paraphrased":
                paraphrase.append(span.b)
    exact = _less(_merged(exact), list(cut))
    paraphrase = _less(_less(_merged(paraphrase), exact), list(cut))
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

    hits: dict[tuple[int, str], BillHits] = defaultdict(BillHits)  # (document, bill)
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

    with db.connect_quiet() as con:
        rows = queries.matchable_sections_of(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
            ids=sorted({bill_id for _, bill_id in hits}),
        )
        split_by_heading = queries.split_by_heading_of(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            ids=sorted({bill_id for _, bill_id in hits}),
        )
    by_bill: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for bill_id, ordinal, display, normalized in rows:
        by_bill[bill_id].append((ordinal, display, normalized))
    sections = {
        bill_id: bill_sections(rows_of, by_heading=split_by_heading[bill_id])
        for bill_id, rows_of in by_bill.items()
    }

    out: list[OriginMatch] = []
    for (d, bill_id), entry in sorted(hits.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if (m := match_of(texts[d], bill_id, sections[bill_id], entry, embed)) is not None:
            out.append(m)
    return out


@dataclass(frozen=True, slots=True)
class BillSection:
    display: str
    words: int
    generic: bool  # left out of a match on the bill's side (lineage.text.generic_sections)
    inside: Inside | None = None  # generic text inside a section that is not generic

    @property
    def cut(self) -> tuple[tuple[int, int], ...]:
        """Its generic text's characters, left out of its spans."""
        return self.inside.ranges if self.inside is not None else ()

    @property
    def counted(self) -> int:
        """The words a share counts: those outside its generic text."""
        return self.words - (self.inside.words if self.inside is not None else 0)


def bill_sections(
    rows: Sequence[tuple[int, str, str]], *, by_heading: bool
) -> dict[int, BillSection]:
    """A bill's matchable sections, (ordinal, display, normalized) in order, with their generic
    text; sections from the window fallback are read as one text, as a model's are."""
    whole, inside = _generic(
        [
            (len(norm.split()), generic_kind(norm), ins)
            for (_, _, norm), ins in zip(
                rows,
                generic_inside([d for _, d, _ in rows], continuous=not by_heading),
                strict=True,
            )
        ]
    )
    return {
        ordinal: BillSection(display, len(norm.split()), n in whole, inside.get(n))
        for n, (ordinal, display, norm) in enumerate(rows)
    }


def match_of(
    text: ModelText,
    bill_id: str,
    bill: dict[int, BillSection],
    entry: BillHits,
    embed: Embedder | None,
) -> OriginMatch | None:
    """A bill's match with a model from the section pairs the stages found, generic sections
    left out on both sides (ruled at the step 47 review): a pair counts only when neither of its
    sections is generic, both shares are of the sections that are not, and the stages, spans
    and pairs are the counted pairs'. A bill whose every pair touches a generic section has no
    match."""
    aligned: dict[tuple[int, int], Alignment] = {}

    def alignment(q: int, o: int) -> Alignment:
        if (q, o) not in aligned:
            aligned[(q, o)] = align(text.display[q], bill[o].display, embed=embed)
        return aligned[(q, o)]

    def outside(q: int, o: int) -> bool:
        """Ruled at the step 48 review: where either section holds generic text, a pair counts
        only when its alignment shares MIN_WORDS outside it on both sides."""
        model_in, bill_in = text.inside.get(q), bill[o].inside
        if model_in is None and bill_in is None:
            return True
        return any(
            span.a is not None
            and span.b is not None
            and words_outside(text.display[q], span.a, model_in.ranges if model_in else ())
            >= MIN_WORDS
            and words_outside(bill[o].display, span.b, bill_in.ranges if bill_in else ())
            >= MIN_WORDS
            for span in alignment(q, o).spans
        )

    pairs = sorted(
        (q, o)
        for q, o in entry.pairs
        if q not in text.generic and not bill[o].generic and outside(q, o)
    )
    if not pairs:
        return None
    model_words = sum(text.counted(q) for q in text.searched if q not in text.generic)
    bill_words = sum(s.counted for s in bill.values() if not s.generic)
    matched_model = {q for q, _ in pairs}
    matched_bill = {o for _, o in pairs}
    by_ordinal: dict[int, list[int]] = defaultdict(list)
    for q, o in pairs:
        by_ordinal[o].append(q)
    spans = [
        span
        for o in sorted(by_ordinal)
        for span in _spans_of(
            o,
            [alignment(q, o) for q in by_ordinal[o]],
            bill[o].cut,
        )
    ]
    stages: list[Stage] = []
    if any(p in entry.agreements for p in pairs):
        stages.append("near_duplicate")
    if any(p in entry.cosine for p in pairs):
        stages.append("semantic")
    return OriginMatch(
        bill_id=bill_id,
        doc_id=text.doc_id,
        stages=stages,
        model_share=_share(sum(text.counted(q) for q in matched_model), model_words),
        bill_share=_share(sum(bill[o].counted for o in matched_bill), bill_words),
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
            for q, o in pairs
        ],
    )


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
