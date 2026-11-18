"""The export's files as both exports render them: the plan (everything the files hold but the
alignments, read from the tables once) and each file rendered from it. `lineage export` renders
every file (lineage.export.build); the live track rewrites only the files a day's changes touch
(lineage.live.publish), from these same functions, so the two cannot drift apart.
See DESIGN.md#export.
"""

import json
import re
import tomllib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from lineage.align import Alignment
from lineage.core import db
from lineage.data.stats import (
    AMENDMENT_CHARS,
    AMENDMENT_OPENING,
    AMENDMENT_SUBSTITUTE,
    GEORGIA_ENACTING,
    GEORGIA_NOTICE,
)
from lineage.explain import queries as explain_queries
from lineage.explain.explain import CACHE_DIR, explain, prompt_spans
from lineage.explain.explain import MODEL as EXPLAIN_MODEL
from lineage.export import queries
from lineage.export.origin_story import BOTH as BOTH_STAGES
from lineage.export.origin_story import STORY as ORIGIN_STORY
from lineage.export.schema import (
    AlignedPair,
    ArchiveStamp,
    BillList,
    BillListEntry,
    BillPage,
    BillRef,
    BillTexts,
    Claim,
    Closest,
    Comparison,
    Explanation,
    Meta,
    ModelDocRef,
    Origin,
    Relative,
    SectionInfo,
    SectionPair,
    SectionText,
    SourceEntry,
    Sources,
    Span,
    Stage,
    Story,
    StoryRelative,
    TextInfo,
    Validation,
    WithheldDocument,
)
from lineage.export.validation import validation_from
from lineage.match.listing import ListedPairs, listed_pairs, shipped_flag
from lineage.match.near.minhash import NUM_PERM
from lineage.match.near.pairs import load_families
from lineage.match.relatives import BillRelatives, bill_relatives
from lineage.match.semantic.pairs import section_keys
from lineage.match.semantic.stage import MODEL
from lineage.match.state_boilerplate import COMPACT_PHRASES
from lineage.origins.collect import PACKAGE as COLLECT_PACKAGE
from lineage.origins.collect import collectors
from lineage.origins.labels import LABELS as ORIGIN_LABELS
from lineage.origins.labels import judged_correct, load_labels
from lineage.origins.matches import OriginMatch, is_listed, read_origin_matches
from lineage.origins.modeldocs import MODELDOCS_DIR, ModelDoc, Source, read_docs
from lineage.origins.withheld import REPORT as WITHHELD_REPORT
from lineage.origins.withheld import Reason as WithheldReason
from lineage.origins.withheld import Withheld, load_withheld

FEATURED_PAIRS = Path("evals/featured_pairs.jsonl")
STORIES_RULES = Path("rules/stories.toml")
FEATURED_REPORT = Path("reports/featured_pairs.json")
RESULTS = Path("reports/results.json")
_COMPACT = re.compile("|".join(re.escape(p) for p in COMPACT_PHRASES))


class ExportError(Exception):
    pass


def slug(state: str, session: str, identifier: str) -> str:
    """A bill's file name: its state, session and identifier, lower case, each run of other
    characters one hyphen."""
    return re.sub(r"[^a-z0-9_]+", "-", f"{state}-{session}-{identifier}".lower()).strip("-")


def comparison_slug(a: str, b: str) -> str:
    return f"{a}--{b}"


BillPair = tuple[str, str]  # (first bill id, second bill id)
OrdinalPair = tuple[int, int]
Featured = tuple[str, int, str, int]  # (first bill, its ordinal, second bill, its ordinal)
SectionKey = tuple[str, int]


class Found(Protocol):
    """What the stages found of one section pair (pairs_between)."""

    stages: set[str]
    estimate: float | None
    cosine: float | None


@dataclass
class _PairFound:
    stages: set[str]
    estimate: float | None = None
    cosine: float | None = None


def pairs_between(
    listed: ListedPairs,
    bills: Sequence[str],
    ordinals: NDArray[np.int64],
    wanted: Iterable[BillPair],
) -> dict[BillPair, dict[OrdinalPair, _PairFound]]:
    """Every listed section pair between the bills of each wanted pair, from the first bill's
    side, with what each stage found."""
    wanted_set = set(wanted)
    firsts = {a for a, _ in wanted_set}
    first = np.fromiter((b in firsts for b in bills), dtype=bool, count=len(bills))
    out: dict[BillPair, dict[OrdinalPair, _PairFound]] = defaultdict(dict)
    stages: tuple[tuple[str, Any, Any, Any], ...] = (
        ("near_duplicate", listed.near.left, listed.near.right, listed.near.agreements),
        ("semantic", listed.meaning.left, listed.meaning.right, listed.meaning.cosine),
    )
    for stage, left, right, scores in stages:
        lo, hi = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
        values = np.asarray(scores)
        for x, y in ((lo, hi), (hi, lo)):
            at = np.flatnonzero(first[x])
            for s, t, v in zip(x[at].tolist(), y[at].tolist(), values[at].tolist(), strict=True):
                key = (str(bills[s]), str(bills[t]))
                if key not in wanted_set:
                    continue
                found = out[key].setdefault((int(ordinals[s]), int(ordinals[t])), _PairFound(set()))
                found.stages.add(stage)
                if stage == "near_duplicate":
                    found.estimate = round(float(v) / NUM_PERM, 4)
                else:
                    found.cosine = round(float(v), 4)
    return dict(out)


def stages_of(found: Found) -> list[Stage]:
    return [s for s in ("near_duplicate", "semantic") if s in found.stages]


def stories_for(state: str, stories: Iterable[str], rules: Path = STORIES_RULES) -> set[str]:
    """The featured stories `state`'s site carries, by `rules`: a story listed for "all" states,
    or for this one. A story the rules do not list is an error, so a new story is placed
    deliberately rather than shown on every state's site by default."""
    listed: dict[str, Any] = tomllib.loads(rules.read_text()).get("stories", {})
    unlisted = sorted(set(stories) - set(listed))
    if unlisted:
        raise ExportError(
            f"stories not in {rules}: {', '.join(unlisted)}; say which states carry them"
        )
    out: set[str] = set()
    for key in stories:
        states = listed[key].get("states")
        if states == "all" or (isinstance(states, list) and state in states):
            out.add(key)
    return out


def carries_origin_story(state: str, rules: Path = STORIES_RULES) -> bool:
    """Whether `state`'s site carries the origin story: only where `rules` place it, so a rules
    file without it carries none, the safe reading for a story that shows nothing unplaced."""
    listed: dict[str, Any] = tomllib.loads(rules.read_text()).get("stories", {})
    return ORIGIN_STORY in listed and bool(stories_for(state, [ORIGIN_STORY], rules))


def load_featured(path: Path) -> dict[str, set[Featured]]:
    """The featured pairs, by story."""
    out: dict[str, set[Featured]] = defaultdict(set)
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            o, d = row["origin"], row["descendant"]
            pair = (str(o["bill_id"]), int(o["ordinal"]), str(d["bill_id"]), int(d["ordinal"]))
            out[str(row["story"])].add(pair)
    return out


def origins_by_bill(
    matches: Sequence[OriginMatch],
    docs: Mapping[str, tuple[ModelDoc, Source]],
    hosted: set[str],
    withheld: frozenset[str] = frozenset(),
) -> dict[str, list[Origin]]:
    """Each hosted bill's origins at the listing floor (lineage.origins.matches.FLOOR), the most
    of its words first, less those to a withheld document (lineage.origins.withheld). A match
    whose document no source holds is an error: the page would name a model the sources page
    does not list."""
    out: dict[str, list[Origin]] = defaultdict(list)
    for m in matches:
        if m.bill_id not in hosted or not is_listed(m) or m.doc_id in withheld:
            continue
        if m.doc_id not in docs:
            raise ExportError(f"origin match names {m.doc_id}, which no collected source holds")
        doc, source = docs[m.doc_id]
        ref = ModelDocRef(
            id=doc.id,
            source=source.key,
            organization=source.organization,
            title=doc.title,
            url=doc.url,
            published=None if doc.published is None else doc.published.isoformat(),
            leaked=doc.leaked,
            co_publishers=list(doc.co_publishers),
            drafted_by=doc.drafted_by,
        )
        out[m.bill_id].append(
            Origin(
                doc=ref,
                stages=list(m.stages),
                bill_share=round(m.bill_share, 4),
                model_share=round(m.model_share, 4),
                sections=sorted({s.ordinal for s in m.spans}),
            )
        )
    return {b: sorted(os, key=lambda o: (-o.bill_share, o.doc.id)) for b, os in out.items()}


def judged_origins(
    tables: Path,
    origins: Mapping[str, Sequence[Origin]],
    refs: Mapping[str, BillRef],
    matches: Sequence[OriginMatch],
    docs: Mapping[str, tuple[ModelDoc, Source]],
    *,
    modeldocs_root: Path,
    labels: Path,
) -> frozenset[tuple[str, str]]:
    """The origin story's candidates a labeler judged correct, by (bill slug, document id):
    each passed bill's listed origins both stages found, judged on the sections its match's
    strongest pair rests on now (lineage.origins.labels.judged_correct)."""
    keys = {
        (bill_id, o.doc.id)
        for bill_id, listed in origins.items()
        if refs[bill_id].status == "passed"
        for o in listed
        if frozenset(o.stages) >= BOTH_STAGES
    }
    judged = judged_correct(
        keys,
        matches,
        [d for d, _ in docs.values()],
        tables,
        load_labels(labels),
        root=modeldocs_root,
    )
    return frozenset((refs[b].slug, d) for b, d in judged)


def sources_doc(
    sources: Sequence[Source],
    docs: Mapping[str, tuple[ModelDoc, Source]],
    withheld: Mapping[str, WithheldReason] | None = None,
) -> Sources:
    """The sources page: every library, with how many documents were collected and when, and
    the documents withheld from pages because their matches rest on generic text."""
    entries: list[SourceEntry] = []
    for src in sorted(sources, key=lambda s: s.key):
        held = [d for d, s in docs.values() if s.key == src.key]
        reasons = withheld or {}
        kept_back = sorted((d for d in held if d.id in reasons), key=lambda d: d.id)
        latest = max((d.collected_on for d in held), default=None)
        entries.append(
            SourceEntry(
                key=src.key,
                organization=src.organization,
                descriptor=src.descriptor,
                members=src.members,
                members_source=src.members_source,
                library_url=src.library_url,
                terms_url=src.terms_url,
                terms_note=src.terms_note,
                terms_read_on=src.terms_read_on.isoformat(),
                leaked=src.leaked,
                leak_note=src.leak_note,
                documents=len(held),
                collected_on=None if latest is None else latest.isoformat(),
                withheld=[
                    WithheldDocument(id=d.id, title=d.title, url=d.url, reason=reasons[d.id])
                    for d in kept_back
                ],
            )
        )
    return Sources(sources=entries)


def bill_refs(
    con: Any, tables: Path, ids: list[str]
) -> tuple[dict[str, BillRef], dict[str, TextInfo]]:
    """Each wanted bill as a page names it, and the text matching read, as the export gives
    them; a bill missing from the tables or two bills sharing a file name are errors."""
    rows = queries.export_bills(
        con,
        bills_glob=str(tables / "bills" / "*.parquet"),
        texts_glob=str(tables / "texts" / "*.parquet"),
        ids=ids,
        notice=GEORGIA_NOTICE,
        enacting=GEORGIA_ENACTING,
        amendment_opening=AMENDMENT_OPENING,
        amendment_chars=AMENDMENT_CHARS,
        substitute=AMENDMENT_SUBSTITUTE,
    )
    refs: dict[str, BillRef] = {}
    texts: dict[str, TextInfo] = {}
    for row in rows:
        (bill_id, state, session, identifier, title, status, reason, status_date, first) = row[:9]
        source, version, url, notice_only, amendment_only = row[9:]
        refs[bill_id] = BillRef.model_validate(
            {
                "slug": slug(state, session, identifier),
                "state": state,
                "session": session,
                "identifier": identifier,
                "title": title,
                "status": status,
                "status_reason": reason,
                "status_date": None if status_date is None else status_date.isoformat(),
                "first_action": first,
            }
        )
        if source is not None:
            texts[bill_id] = TextInfo.model_validate(
                {
                    "source": source,
                    "version": version,
                    "url": url,
                    "notice_only": notice_only,
                    "amendment_only": amendment_only,
                }
            )
    missing = sorted(set(ids) - set(refs))
    if missing:
        raise ExportError(f"{len(missing)} bills are not in the bills table, {missing[0]} first")
    clashes: dict[str, list[str]] = defaultdict(list)
    for bill_id, ref in refs.items():
        clashes[ref.slug].append(bill_id)
    for name, holders in clashes.items():
        if len(holders) > 1:
            raise ExportError(f"bills {', '.join(sorted(holders))} share the file name {name}")
    return refs, texts


@dataclass(frozen=True, slots=True)
class Plan:
    """Everything the export's files hold but the alignments, read from the tables."""

    state: str
    hosted: list[str]
    relatives: Mapping[str, BillRelatives]
    found: Mapping[BillPair, Mapping[OrdinalPair, Found]]
    featured: frozenset[Featured]
    refs: Mapping[str, BillRef]
    text_infos: Mapping[str, TextInfo]
    sections_of: Mapping[str, list[tuple[int, str | None, int, str | None]]]
    shown: Mapping[SectionKey, tuple[str | None, str, str]]
    origins: Mapping[str, list[Origin]]
    judged: frozenset[
        tuple[str, str]
    ]  # origins a labeler judged correct (lineage.export.origin_story)
    stories: Mapping[str, Story]
    sources: Sources
    meta: Meta
    cache: Path

    def matchable(self, bill_id: str) -> int:
        return sum(1 for *_, bp in self.sections_of.get(bill_id, []) if bp is None)


SectionRow = tuple[int, str | None, int, str | None]  # (ordinal, heading, words, boilerplate)


def _committed(
    inputs: Sequence[Path], results: Path, withheld_report: Path
) -> tuple[Withheld, Validation]:
    """The withheld documents and the validation, refused when a committed input is missing or
    the results are older than this code."""
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise ExportError(f"committed inputs not found: {', '.join(missing)}")
    kept_back = load_withheld(withheld_report)
    try:
        return kept_back, validation_from(json.loads(results.read_text()), kept_back)
    except KeyError as e:
        raise ExportError(f"{results} lacks {e}; is it older than this code?") from e


def _carried_stories(
    state: str, featured_report: Path, featured_pairs: Path, stories_rules: Path
) -> tuple[dict[str, Any], set[Featured]]:
    """The featured report's stories `state`'s site carries, and their featured pairs."""
    all_stories: dict[str, Any] = json.loads(featured_report.read_text())["stories"]
    by_story = load_featured(featured_pairs)
    keys = set(all_stories) | set(by_story)
    carried: set[str] = stories_for(state, keys, stories_rules) if keys else set()
    stories_doc = {k: v for k, v in all_stories.items() if k in carried}
    featured = {pair for key in carried for pair in by_story.get(key, set())}
    return stories_doc, featured


def _found_pairs(
    tables: Path,
    listed: ListedPairs,
    relatives: Mapping[str, BillRelatives],
    featured: set[Featured],
) -> tuple[set[BillPair], dict[BillPair, dict[OrdinalPair, _PairFound]], int]:
    """The bill pairs the pages compare, their listed section pairs, and how many sections are
    signed; a featured pair this build does not list is an error."""
    wanted: set[BillPair] = {(b, m.bill_id) for b, r in relatives.items() for m in r.matches}
    wanted |= {(a, b) for a, _, b, _ in featured}
    bills, ordinals = section_keys(tables)
    found = pairs_between(listed, bills, ordinals, wanted)
    for a, oa, b, ob in sorted(featured):
        if (oa, ob) not in found.get((a, b), {}):
            raise ExportError(f"featured pair {a} §{oa} / {b} §{ob} is not listed in this build")
    return wanted, found, len(bills)


def _sections(
    tables: Path, ids: list[str], found: Mapping[BillPair, Mapping[OrdinalPair, Found]]
) -> tuple[
    dict[str, BillRef],
    dict[str, TextInfo],
    dict[str, list[SectionRow]],
    dict[SectionKey, tuple[str | None, str, str]],
]:
    """The bills as pages name them, their texts, every section of each, and the text of each
    section a comparison shows."""
    shown_keys = sorted(
        {(a, oa) for (a, _), ps in found.items() for oa, _ in ps}
        | {(b, ob) for (_, b), ps in found.items() for _, ob in ps}
    )
    sections_glob = str(tables / "sections" / "*.parquet")
    with db.connect_quiet() as con:
        refs, text_infos = bill_refs(con, tables, ids)
        section_rows = queries.section_infos(con, sections_glob=sections_glob, ids=ids)
        shown = explain_queries.sections_at(con, sections_glob=sections_glob, keys=shown_keys)
    sections_of: dict[str, list[SectionRow]] = defaultdict(list)
    for bill_id, ordinal, heading, words, boilerplate in section_rows:
        sections_of[bill_id].append((ordinal, heading, words, boilerplate))
    return refs, text_infos, dict(sections_of), shown


def _stories(
    tables: Path,
    stories_doc: Mapping[str, Any],
    refs: Mapping[str, BillRef],
    *,
    model: str,
    listed: ListedPairs,
    signed: int,
) -> dict[str, Story]:
    """Each carried story's file, each relative with its share of the anchor's page."""
    anchors = sorted({str(st["anchor"]["bill_id"]) for st in stories_doc.values()})
    anchor_scores: dict[str, dict[str, float]] = {}
    family_of: dict[str, str] = {}
    if anchors:
        listed_for = bill_relatives(tables, anchors, model=model, listed=listed, shown=signed)
        family_of = load_families(tables)
        anchor_scores = {x: {m.family: m.share for m in r.matches} for x, r in listed_for.items()}

    def story_share(anchor: str, bill_id: str) -> float | None:
        score = anchor_scores.get(anchor, {}).get(family_of.get(bill_id, bill_id))
        return None if score is None else round(score, 4)

    stories: dict[str, Story] = {}
    for key, s in sorted(stories_doc.items()):
        anchor = str(s["anchor"]["bill_id"])
        stories[key] = Story(
            slug=key,
            anchor=refs[anchor],
            relatives_rule=s["relatives_rule"],
            scope_note=s["scope_note"],
            relatives=[
                StoryRelative(
                    bill=refs[str(rel["bill_id"])],
                    share=story_share(anchor, str(rel["bill_id"])),
                    section_pairs=int(rel["section_pairs"]),
                    comparison=comparison_slug(refs[anchor].slug, refs[str(rel["bill_id"])].slug)
                    if int(rel["section_pairs"])
                    else None,
                )
                for rel in s["relatives"]
            ],
        )
    return stories


@dataclass(frozen=True, slots=True)
class ExportInputs:
    """What an export reads besides the tables: the committed inputs (the featured pairs and
    stories, their rules, the results, the origin labels, the withheld report), the cached
    explanations, the collected model documents and their collectors, the embedding model, and
    whether the state boilerplate counts must exist. By default the repository's own."""

    cache: Path = CACHE_DIR
    featured_pairs: Path = FEATURED_PAIRS
    featured_report: Path = FEATURED_REPORT
    stories_rules: Path = STORIES_RULES
    results: Path = RESULTS
    modeldocs_root: Path = MODELDOCS_DIR
    collector_package: str = COLLECT_PACKAGE
    model: str = MODEL
    require_flag: bool = True
    origin_labels: Path = ORIGIN_LABELS
    withheld_report: Path = WITHHELD_REPORT


DEFAULT_INPUTS = ExportInputs()


def plan_export(
    tables: Path,
    *,
    state: str,
    inputs: ExportInputs = DEFAULT_INPUTS,
    progress: Callable[[str], None] = lambda _: None,
) -> Plan:
    """What the export reads before it aligns anything, with its inputs and checks: everything
    its files hold but the alignments (lineage.export.build.build_export)."""
    kept_back, validation = _committed(
        (inputs.featured_pairs, inputs.featured_report, inputs.stories_rules, inputs.results),
        inputs.results,
        inputs.withheld_report,
    )
    sources = [c.source for c in collectors(inputs.collector_package).values()]
    docs = {d.id: (d, src) for src in sources for d in read_docs(src, inputs.modeldocs_root)}
    bills_glob = str(tables / "bills" / "*.parquet")
    flag = shipped_flag(tables)
    if flag is None and inputs.require_flag:
        raise ExportError("no state boilerplate counts; run `uv run lineage state-boilerplate`")
    with db.connect_quiet() as con:
        hosted = queries.hosted_bill_ids(
            con, bills_glob=bills_glob, eligible_path=tables / "eligible.parquet", state=state
        )
        stamps = queries.archive_stamps(con, bills_glob=bills_glob, state=state)
        indexed = queries.states_indexed(con, bills_glob=bills_glob)
    if not hosted:
        raise ExportError(f"no {state} bill is matched by the document-type rules")
    progress(f"{len(hosted)} hosted bills")
    listed = listed_pairs(tables, model=inputs.model, flag=flag)
    relatives = bill_relatives(tables, hosted, model=inputs.model, listed=listed)
    stories_doc, featured = _carried_stories(
        state, inputs.featured_report, inputs.featured_pairs, inputs.stories_rules
    )
    wanted, found, signed = _found_pairs(tables, listed, relatives, featured)
    story_ids = {
        str(r["bill_id"]) for s in stories_doc.values() for r in [s["anchor"], *s["relatives"]]
    }
    ids = sorted(set(hosted) | {x for pair in wanted for x in pair} | story_ids)
    refs, text_infos, sections_of, shown = _sections(tables, ids, found)
    stories = _stories(tables, stories_doc, refs, model=inputs.model, listed=listed, signed=signed)
    meta = Meta(
        state=state,
        archives=[
            ArchiveStamp(session=session, generated_at=at.isoformat()) for session, at in stamps
        ],
        states_indexed=indexed,
        hosted_bills=len(hosted),
        stories=list(stories),
        validation=validation,
    )
    matches = read_origin_matches(tables)
    origins = origins_by_bill(matches, docs, set(hosted), kept_back.ids)
    judged = (
        judged_origins(
            tables,
            origins,
            refs,
            matches,
            docs,
            modeldocs_root=inputs.modeldocs_root,
            labels=inputs.origin_labels,
        )
        if carries_origin_story(state, inputs.stories_rules)
        else frozenset[tuple[str, str]]()
    )
    return Plan(
        state=state,
        hosted=list(hosted),
        relatives=relatives,
        found=found,
        featured=frozenset(featured),
        refs=refs,
        text_infos=text_infos,
        sections_of=sections_of,
        shown=shown,
        origins=origins,
        judged=judged,
        stories=stories,
        sources=sources_doc(sources, docs, kept_back.reasons),
        meta=meta,
        cache=inputs.cache,
    )


def bill_page(plan: Plan, bill_id: str, updated_on: date | None = None) -> BillPage:
    r = plan.relatives[bill_id]
    listed_here: list[Relative] = []
    per_section: dict[int, int] = defaultdict(int)
    for m in r.matches:
        pairs = plan.found.get((bill_id, m.bill_id), {})
        compact = any(
            _COMPACT.search(plan.shown[(bill_id, oa)][2])
            or _COMPACT.search(plan.shown[(m.bill_id, ob)][2])
            for oa, ob in pairs
        )
        for oa in {oa for oa, _ in pairs}:
            per_section[oa] += 1
        listed_here.append(
            Relative(
                bill=plan.refs[m.bill_id],
                family_bills=m.family_bills,
                share=round(m.share, 4),
                best_estimate=round(m.best_estimate, 4),
                best_cosine=None if m.best_cosine is None else round(m.best_cosine, 4),
                found_by="wording" if m.found_by == "wording" else "meaning",
                pairs=[
                    SectionPair(
                        ordinal=oa,
                        other_ordinal=ob,
                        stages=stages_of(pf),
                        estimate=pf.estimate,
                        cosine=pf.cosine,
                    )
                    for (oa, ob), pf in sorted(pairs.items())
                ],
                names_compact=compact,
                comparison=comparison_slug(plan.refs[bill_id].slug, plan.refs[m.bill_id].slug),
            )
        )
    return BillPage(
        bill=plan.refs[bill_id],
        text=plan.text_infos.get(bill_id),
        sections=[
            SectionInfo(
                ordinal=o, heading=h, words=w, boilerplate=bp, with_relatives=per_section.get(o, 0)
            )
            for o, h, w, bp in plan.sections_of.get(bill_id, [])
        ],
        words_searched=r.words_searched,
        families=r.families,
        relatives=listed_here,
        origins=plan.origins.get(bill_id, []),
        updated_on=None if updated_on is None else updated_on.isoformat(),
    )


def bill_list(plan: Plan) -> BillList:
    entries: list[BillListEntry] = []
    for bill_id in plan.hosted:
        r = plan.relatives[bill_id]
        best = r.matches[0] if r.matches else None
        entries.append(
            BillListEntry(
                bill=plan.refs[bill_id],
                families=r.families,
                closest=None
                if best is None
                else Closest(
                    slug=plan.refs[best.bill_id].slug,
                    state=plan.refs[best.bill_id].state,
                    identifier=plan.refs[best.bill_id].identifier,
                    share=round(best.share, 4),
                ),
            )
        )
    return BillList(bills=entries)


def texts_by_slug(plan: Plan) -> dict[str, BillTexts]:
    """Each shown bill's texts file, by its slug."""
    by_bill: dict[str, list[SectionText]] = defaultdict(list)
    for (bill_id, ordinal), (heading, text, _) in sorted(plan.shown.items()):
        by_bill[bill_id].append(SectionText(ordinal=ordinal, heading=heading, text=text))
    return {
        plan.refs[b].slug: BillTexts(bill=plan.refs[b].slug, sections=s) for b, s in by_bill.items()
    }


def comparison_name(plan: Plan, pair: BillPair) -> str:
    a, b = pair
    return f"compare/{comparison_slug(plan.refs[a].slug, plan.refs[b].slug)}.json"


def comparison(
    plan: Plan, pair: BillPair, alignments: Mapping[OrdinalPair, Alignment]
) -> Comparison:
    """A comparison file, each section pair aligned as given, a featured pair explained from
    the cache as the export explains it."""
    a, b = pair
    aligned: list[AlignedPair] = []
    for (oa, ob), pf in sorted(plan.found[pair].items()):
        alignment = alignments[(oa, ob)]
        explanation = None
        if (a, oa, b, ob) in plan.featured:
            ta, tb = plan.shown[(a, oa)][1], plan.shown[(b, ob)][1]
            outcome = explain(
                prompt_spans(alignment.spans, ta, tb, origin_side="a"), ask=None, cache=plan.cache
            )
            explanation = Explanation(
                model=EXPLAIN_MODEL,
                withheld=outcome.withheld,
                claims=[Claim.model_validate(c.model_dump()) for c in outcome.claims],
            )
        aligned.append(
            AlignedPair(
                ordinal_a=oa,
                ordinal_b=ob,
                stages=stages_of(pf),
                spans=[Span(id=x.id, kind=x.kind, a=x.a, b=x.b) for x in alignment.spans],
                copied_share_a=round(alignment.copied_share_a, 4),
                copied_share_b=round(alignment.copied_share_b, 4),
                explanation=explanation,
            )
        )
    return Comparison(
        a=plan.refs[a],
        b=plan.refs[b],
        sections_a=plan.matchable(a),
        sections_b=plan.matchable(b),
        text_a=plan.text_infos.get(a),
        text_b=plan.text_infos.get(b),
        pairs=aligned,
    )
