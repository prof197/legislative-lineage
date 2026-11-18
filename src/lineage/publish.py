"""The incremental export: the hosted state's files a day's changes touch, rebuilt from the
updated tables, and every other file left as it was, byte for byte and untouched.

The live track's change log (lineage.updates) names each bill a day added, updated or replaced,
and each relative and origin a bill gained or lost. From it:

- **Bill pages.** A hosted bill's page is rewritten when the log names the bill, or names a
  record the page shows as a relative, since a relative's status or title is on the page. A
  rewritten page carries `updated_on`, the last day applied that touched it.
- **Comparisons.** One is rebuilt when the log names either of its bills. A section pair whose
  two texts the export already shows unchanged keeps its alignment from the file in place,
  since an alignment is a function of the two texts alone; only a pair with a new text is
  aligned, which is the export's whole cost (DESIGN.md#export).
- **Section texts** of every bill in a rebuilt or dropped comparison, and the bill index, the
  stories, the sources list and the data stamp, whose `as_of` is the last day applied.

A file is written only when its bytes change, and a file the tables no longer give is removed.
The log carries which relatives a bill gained or lost, not every change to one it kept: a new
window can push one section pair of a kept relative out of a bucket under the cap. So every
other page and comparison is also checked against its file in place, which needs no alignment,
and any that differ are rebuilt and reported as `beyond_log`; the recorded days have none.

The pages are rendered as `lineage.export.build_export` renders them, from its functions, and a
test holds the result equal to a full export of the same tables. The live export is never
written: the day is applied to a staging copy whose unchanged files are hard links to the live
ones, and `promote` swaps it in only once every step has succeeded.

Owned by lane L3 (M12). See DESIGN.md#updates, DESIGN.md#export.

See DESIGN.md#dr-13.
"""

import json
import os
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, TypeAdapter

from lineage import queries
from lineage.align import Alignment
from lineage.align import Span as AlignSpan
from lineage.collect import PACKAGE as COLLECT_PACKAGE
from lineage.collect import collectors
from lineage.core import db
from lineage.core.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.data.stats import (
    AMENDMENT_CHARS,
    AMENDMENT_OPENING,
    AMENDMENT_SUBSTITUTE,
    GEORGIA_ENACTING,
    GEORGIA_NOTICE,
)
from lineage.explain import CACHE_DIR, explain, prompt_spans
from lineage.explain import MODEL as EXPLAIN_MODEL
from lineage.export import (
    FEATURED_PAIRS,
    FEATURED_REPORT,
    RESULTS,
    STORIES_RULES,
    BillPair,
    ExportError,
    OrdinalPair,
    align_all,
    carries_origin_story,
    comparison_slug,
    judged_origins,
    origins_by_bill,
    pairs_between,
    slug,
    sources_doc,
    stories_for,
    validation_from,
)
from lineage.export_schema import (
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
    Origin,
    Relative,
    SectionInfo,
    SectionPair,
    SectionText,
    Sources,
    Span,
    Stage,
    Story,
    StoryRelative,
    TextInfo,
)
from lineage.incremental import applied_days
from lineage.listing import listed_pairs, shipped_flag
from lineage.modeldocs import MODELDOCS_DIR, read_docs
from lineage.origin_labels import LABELS as ORIGIN_LABELS
from lineage.origin_story import FILE as ORIGIN_STORY_FILE
from lineage.origin_story import story_file as origin_story_file
from lineage.origins import read_origin_matches
from lineage.pairs import load_families
from lineage.relatives import BillRelatives, bill_relatives
from lineage.semantic import MODEL
from lineage.semantic_pairs import section_keys
from lineage.state_boilerplate import COMPACT_PHRASES
from lineage.updates import (
    CHANGES_DIR,
    Change,
    RelativeGained,
    RelativeLost,
)
from lineage.withheld import REPORT as WITHHELD_REPORT
from lineage.withheld import load_withheld

STAGING_SUFFIX = ".staging"
OLD_SUFFIX = ".old"
META = "meta.json"
BILLS = "bills.json"
SOURCES = "sources.json"
FEED_FILE = "feed.xml"  # written by lineage.feed; carried over untouched here
_COMPACT = re.compile("|".join(re.escape(p) for p in COMPACT_PHRASES))
_CHANGE: TypeAdapter[Change] = TypeAdapter(Change)

Featured = tuple[str, int, str, int]  # (first bill, its ordinal, second bill, its ordinal)
SectionKey = tuple[str, int]


class PublishError(Exception):
    pass


class Found(Protocol):
    """What the stages found of one section pair (lineage.export.pairs_between)."""

    stages: set[str]
    estimate: float | None
    cosine: float | None


def _stages(found: Found) -> list[Stage]:
    return [s for s in ("near_duplicate", "semantic") if s in found.stages]


# The change log, read by day.


def logged_days(changes_dir: Path) -> list[date]:
    """Every day the change log holds, oldest first."""
    return sorted(date.fromisoformat(p.stem) for p in changes_dir.glob("*.jsonl"))


def changes_on(changes_dir: Path, day: date) -> list[Change]:
    """The changes logged for `day`, in the order written; none when it logged nothing."""
    path = changes_dir / f"{day.isoformat()}.jsonl"
    if not path.exists():
        return []
    return [_CHANGE.validate_json(line) for line in path.read_text().splitlines() if line]


@dataclass(frozen=True, slots=True)
class Touched:
    """What a run of days' changes name, each with the last day that names it."""

    bills: Mapping[str, date]  # every bill the log names, and every relative it gains or loses
    records: Mapping[str, date]  # records added, updated or replaced: what a page shows of them


def touched(changes: Iterable[Change]) -> Touched:
    bills: dict[str, date] = {}
    records: dict[str, date] = {}

    def mark(into: dict[str, date], bill_id: str, day: date) -> None:
        into[bill_id] = max(day, into.get(bill_id, day))

    for c in changes:
        mark(bills, c.bill_id, c.as_of)
        if isinstance(c, RelativeGained | RelativeLost):
            mark(bills, c.relative_id, c.as_of)
        if c.kind in ("added", "updated", "replaced"):
            mark(records, c.bill_id, c.as_of)
    return Touched(bills=bills, records=records)


# The export's files, rendered as lineage.export.build_export renders them.


def _load_featured(path: Path) -> dict[str, set[Featured]]:
    out: dict[str, set[Featured]] = defaultdict(set)
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            o, d = row["origin"], row["descendant"]
            pair = (str(o["bill_id"]), int(o["ordinal"]), str(d["bill_id"]), int(d["ordinal"]))
            out[str(row["story"])].add(pair)
    return out


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
    judged: frozenset[tuple[str, str]]  # origins a labeler judged correct (lineage.origin_story)
    stories: Mapping[str, Story]
    sources: Sources
    meta: Meta
    cache: Path

    def matchable(self, bill_id: str) -> int:
        return sum(1 for *_, bp in self.sections_of.get(bill_id, []) if bp is None)


def plan_export(
    tables: Path,
    *,
    state: str,
    cache: Path = CACHE_DIR,
    featured_pairs: Path = FEATURED_PAIRS,
    featured_report: Path = FEATURED_REPORT,
    stories_rules: Path = STORIES_RULES,
    results: Path = RESULTS,
    modeldocs_root: Path = MODELDOCS_DIR,
    collector_package: str = COLLECT_PACKAGE,
    model: str = MODEL,
    require_flag: bool = True,
    origin_labels: Path = ORIGIN_LABELS,
    withheld_report: Path = WITHHELD_REPORT,
) -> Plan:
    """What `build_export` reads before it aligns anything, with its inputs and checks."""
    inputs = (featured_pairs, featured_report, stories_rules, results)
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise ExportError(f"committed inputs not found: {', '.join(missing)}")
    kept_back = load_withheld(withheld_report)
    try:
        validation = validation_from(json.loads(results.read_text()), kept_back)
    except KeyError as e:
        raise ExportError(f"{results} lacks {e}; is it older than this code?") from e
    sources = [c.source for c in collectors(collector_package).values()]
    docs = {d.id: (d, src) for src in sources for d in read_docs(src, modeldocs_root)}
    bills_glob = str(tables / "bills" / "*.parquet")
    flag = shipped_flag(tables)
    if flag is None and require_flag:
        raise ExportError("no state boilerplate counts; run `uv run lineage state-boilerplate`")
    with db.connect_quiet() as con:
        hosted = queries.hosted_bill_ids(
            con, bills_glob=bills_glob, eligible_path=tables / "eligible.parquet", state=state
        )
        stamps = queries.archive_stamps(con, bills_glob=bills_glob, state=state)
        indexed = queries.states_indexed(con, bills_glob=bills_glob)
    if not hosted:
        raise ExportError(f"no {state} bill is matched by the document-type rules")
    listed = listed_pairs(tables, model=model, flag=flag)
    relatives = bill_relatives(tables, hosted, model=model, listed=listed)

    all_stories: dict[str, Any] = json.loads(featured_report.read_text())["stories"]
    by_story = _load_featured(featured_pairs)
    keys = set(all_stories) | set(by_story)
    carried: set[str] = stories_for(state, keys, stories_rules) if keys else set()
    stories_doc = {k: v for k, v in all_stories.items() if k in carried}
    featured = {pair for key in carried for pair in by_story.get(key, set())}
    wanted: set[BillPair] = {(b, m.bill_id) for b, r in relatives.items() for m in r.matches}
    wanted |= {(a, b) for a, _, b, _ in featured}
    bills, ordinals = section_keys(tables)
    found = pairs_between(listed, bills, ordinals, wanted)
    for a, oa, b, ob in sorted(featured):
        if (oa, ob) not in found.get((a, b), {}):
            raise ExportError(f"featured pair {a} §{oa} / {b} §{ob} is not listed in this build")
    story_ids = {
        str(r["bill_id"]) for s in stories_doc.values() for r in [s["anchor"], *s["relatives"]]
    }
    ids = sorted(set(hosted) | {x for pair in wanted for x in pair} | story_ids)
    shown_keys = sorted(
        {(a, oa) for (a, _), ps in found.items() for oa, _ in ps}
        | {(b, ob) for (_, b), ps in found.items() for _, ob in ps}
    )
    sections_glob = str(tables / "sections" / "*.parquet")
    with db.connect_quiet() as con:
        refs, text_infos = bill_refs(con, tables, ids)
        section_rows = queries.section_infos(con, sections_glob=sections_glob, ids=ids)
        shown = queries.sections_at(con, sections_glob=sections_glob, keys=shown_keys)
    sections_of: dict[str, list[tuple[int, str | None, int, str | None]]] = defaultdict(list)
    for bill_id, ordinal, heading, words, boilerplate in section_rows:
        sections_of[bill_id].append((ordinal, heading, words, boilerplate))

    anchors = sorted({str(st["anchor"]["bill_id"]) for st in stories_doc.values()})
    anchor_scores: dict[str, dict[str, float]] = {}
    family_of: dict[str, str] = {}
    if anchors:
        listed_for = bill_relatives(tables, anchors, model=model, listed=listed, shown=len(bills))
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
    return Plan(
        state=state,
        hosted=list(hosted),
        relatives=relatives,
        found=found,
        featured=frozenset(featured),
        refs=refs,
        text_infos=text_infos,
        sections_of=dict(sections_of),
        shown=shown,
        origins=origins,
        judged=judged_origins(
            tables,
            origins,
            refs,
            matches,
            docs,
            modeldocs_root=modeldocs_root,
            labels=origin_labels,
        )
        if carries_origin_story(state, stories_rules)
        else frozenset(),
        stories=stories,
        sources=sources_doc(sources, docs, kept_back.reasons),
        meta=meta,
        cache=cache,
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
                        stages=_stages(pf),
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
                stages=_stages(pf),
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


def _signature(c: Comparison) -> tuple[Any, ...]:
    """What a comparison holds but its alignments: its bills, texts and pairs by stage."""
    return (
        c.a,
        c.b,
        c.sections_a,
        c.sections_b,
        c.text_a,
        c.text_b,
        [(p.ordinal_a, p.ordinal_b, p.stages) for p in c.pairs],
    )


def _planned_signature(plan: Plan, pair: BillPair) -> tuple[Any, ...]:
    a, b = pair
    return (
        plan.refs[a],
        plan.refs[b],
        plan.matchable(a),
        plan.matchable(b),
        plan.text_infos.get(a),
        plan.text_infos.get(b),
        [(oa, ob, _stages(pf)) for (oa, ob), pf in sorted(plan.found[pair].items())],
    )


# Staging: the live export's files as hard links, so a file written here replaces the link and
# the live file is never touched; a file left alone is the live file itself.


def staging_dir(export: Path) -> Path:
    return export.with_name(export.name + STAGING_SUFFIX)


def stage(export: Path, staging: Path | None = None) -> Path:
    """A fresh staging copy of the live export, every file a hard link to the live one. A
    promote stopped between its two renames is finished first."""
    old = export.with_name(export.name + OLD_SUFFIX)
    if not export.exists() and old.exists():
        old.rename(export)
    if not (export / META).exists():
        raise PublishError(f"{export} holds no export; run `uv run lineage export` first")
    target = staging if staging is not None else staging_dir(export)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(export, target, copy_function=os.link)
    return target


def promote(export: Path, staging: Path | None = None) -> Path:
    """Swap the staging copy in as the live export; the export before it is removed."""
    source = staging if staging is not None else staging_dir(export)
    if not (source / META).exists():
        raise PublishError(f"{source} holds no staged export; nothing to promote")
    old = export.with_name(export.name + OLD_SUFFIX)
    if old.exists():
        shutil.rmtree(old)
    if export.exists():
        export.rename(old)
    source.rename(export)
    if old.exists():
        shutil.rmtree(old)
    return export


def _put(root: Path, name: str, data: bytes) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    partial.replace(path)  # a new file: the live export's link is left as it was


def _bytes(model: BaseModel) -> bytes:
    return model.model_dump_json().encode()


def _read(root: Path, name: str) -> bytes | None:
    path = root / name
    return path.read_bytes() if path.exists() else None


def _export_files(root: Path) -> set[str]:
    """The export's JSON files by their published paths; the feed is lineage.feed's."""
    return {p.relative_to(root).as_posix() for p in root.rglob("*.json") if p.is_file()}


def export_as_of(export: Path) -> date | None:
    """The last day applied to an export, or None for a full export."""
    meta = Meta.model_validate_json((export / META).read_text())
    return None if meta.as_of is None else date.fromisoformat(meta.as_of)


def days_to_apply(
    export: Path, tables: Path, changes_dir: Path, *, since: date | None = None
) -> list[date]:
    """The logged days the export has not had applied: those after its `as_of`, or from
    `since`, or every logged day for a full export, which is always safe, since every file a
    day touches is rewritten from the tables as they stand. Each must be applied to the tables;
    a day the tables have not had is an error, since its pages would show what is not there."""
    as_of = export_as_of(export)
    logged = logged_days(changes_dir)
    days = [d for d in logged if (d >= since if since is not None else as_of is None or d > as_of)]
    applied = applied_days(tables)
    missing = [d for d in days if d not in applied]
    if missing:
        raise PublishError(f"the tables have not had {missing[0]} applied; run `lineage update`")
    unlogged = sorted(d for d in applied if d not in logged and (as_of is None or d > as_of))
    if unlogged:
        raise PublishError(f"{unlogged[0]} is applied to the tables but not in {changes_dir}")
    return days


def export_changes(
    tables: Path,
    *,
    state: str,
    export: Path,
    staging: Path | None = None,
    changes_dir: Path = CHANGES_DIR,
    since: date | None = None,
    models_dir: Path | None,
    workers: int = 1,
    cache: Path = CACHE_DIR,
    featured_pairs: Path = FEATURED_PAIRS,
    featured_report: Path = FEATURED_REPORT,
    stories_rules: Path = STORIES_RULES,
    results: Path = RESULTS,
    modeldocs_root: Path = MODELDOCS_DIR,
    collector_package: str = COLLECT_PACKAGE,
    model: str = MODEL,
    require_flag: bool = True,
    floor: int = DEFAULT_FLOOR_BYTES,
    origin_labels: Path = ORIGIN_LABELS,
    withheld_report: Path = WITHHELD_REPORT,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Apply the logged days the live `export` lacks to a staging copy of it, from the updated
    `tables`; returns the report. The live export is left as it was: `promote` swaps the copy
    in. Paraphrase is marked with the model under `models_dir` unless it is None, which must be
    as the export in place was built, since its alignments are kept."""
    started = time.monotonic()
    days = days_to_apply(export, tables, changes_dir, since=since)
    target = stage(export, staging)
    log = touched(c for d in days for c in changes_on(changes_dir, d))
    through = days[-1] if days else export_as_of(export)
    progress(f"{len(days)} days to apply; {len(log.bills)} bills named in the log")
    plan = plan_export(
        tables,
        state=state,
        cache=cache,
        featured_pairs=featured_pairs,
        featured_report=featured_report,
        stories_rules=stories_rules,
        results=results,
        modeldocs_root=modeldocs_root,
        collector_package=collector_package,
        model=model,
        require_flag=require_flag,
        origin_labels=origin_labels,
        withheld_report=withheld_report,
    )
    before = _export_files(target)
    writes: dict[str, bytes] = {}
    beyond: list[str] = []

    def put(name: str, model: BaseModel) -> None:
        data = _bytes(model)
        if data != _read(target, name):
            writes[name] = data

    # Bill pages.
    for bill_id in plan.hosted:
        name = f"bills/{plan.refs[bill_id].slug}.json"
        days_named = [log.bills.get(bill_id)] + [
            log.records.get(m.bill_id) for m in plan.relatives[bill_id].matches
        ]
        named = [d for d in days_named if d is not None]
        if named:
            put(name, bill_page(plan, bill_id, max(named)))
            continue
        old = _read(target, name)
        page = bill_page(plan, bill_id)
        if (
            old is None
            or BillPage.model_validate_json(old).model_copy(update={"updated_on": None}) != page
        ):
            beyond.append(name)
            stamped = None if through is None else through.isoformat()
            put(name, page.model_copy(update={"updated_on": stamped}))

    # Comparisons: those the log names, and any other whose bills, texts or pairs moved.
    rebuild: list[BillPair] = []
    for pair in sorted(plan.found):
        name = comparison_name(plan, pair)
        if pair[0] in log.bills or pair[1] in log.bills:
            rebuild.append(pair)
            continue
        old = _read(target, name)
        if old is None or _signature(Comparison.model_validate_json(old)) != _planned_signature(
            plan, pair
        ):
            beyond.append(name)
            rebuild.append(pair)
    names_now = {comparison_name(plan, p) for p in plan.found}
    dropped = sorted(n for n in before if n.startswith("compare/") and n not in names_now)

    # Alignments kept where both texts are the ones the export already shows.
    old_texts: dict[str, dict[int, str]] = {}

    def shown_before(bill_slug: str) -> dict[int, str]:
        if bill_slug not in old_texts:
            data = _read(target, f"texts/{bill_slug}.json")
            old_texts[bill_slug] = (
                {}
                if data is None
                else {s.ordinal: s.text for s in BillTexts.model_validate_json(data).sections}
            )
        return old_texts[bill_slug]

    kept: dict[BillPair, dict[OrdinalPair, Alignment]] = defaultdict(dict)
    todo: list[tuple[BillPair, OrdinalPair]] = []
    reused = 0
    for pair in rebuild:
        a, b = pair
        sa, sb = plan.refs[a].slug, plan.refs[b].slug
        data = _read(target, comparison_name(plan, pair))
        reusable = (
            {(p.ordinal_a, p.ordinal_b): p for p in Comparison.model_validate_json(data).pairs}
            if data is not None
            else {}
        )
        for oa, ob in sorted(plan.found[pair]):
            was = reusable.get((oa, ob))
            same = (
                was is not None
                and shown_before(sa).get(oa) == plan.shown[(a, oa)][1]
                and shown_before(sb).get(ob) == plan.shown[(b, ob)][1]
            )
            if was is not None and same:
                reused += 1
                kept[pair][(oa, ob)] = Alignment(
                    spans=[AlignSpan(id=s.id, kind=s.kind, a=s.a, b=s.b) for s in was.spans],
                    copied_share_a=was.copied_share_a,
                    copied_share_b=was.copied_share_b,
                )
            else:
                todo.append((pair, (oa, ob)))
    progress(f"{len(rebuild)} comparisons to rebuild, {len(todo)} section pairs to align")
    aligned = align_all(
        [(plan.shown[(p[0], oa)][1], plan.shown[(p[1], ob)][1]) for p, (oa, ob) in todo],
        models_dir=models_dir,
        workers=workers,
        progress=progress,
    )
    for (pair, ordinals), alignment in zip(todo, aligned, strict=True):
        kept[pair][ordinals] = alignment
    for pair in rebuild:
        put(comparison_name(plan, pair), comparison(plan, pair, kept[pair]))

    # Section texts of every bill in a rebuilt or dropped comparison.
    texts = texts_by_slug(plan)
    candidates = {plan.refs[x].slug for pair in rebuild for x in pair}
    for name in dropped:
        candidates |= set(name.removeprefix("compare/").removesuffix(".json").split("--"))
    for bill_slug in sorted(candidates & set(texts)):
        put(f"texts/{bill_slug}.json", texts[bill_slug])

    # The bill index, the stories, the sources list and the data stamp.
    put(BILLS, bill_list(plan))
    for key, story in plan.stories.items():
        put(f"stories/{key}.json", story)
    # The origin story, chosen afresh from the hosted pages as a full export chooses it.
    origin_story: set[str] = set()
    if carries_origin_story(state, stories_rules):
        chosen = origin_story_file((bill_page(plan, b) for b in plan.hosted), plan.judged)
        if chosen is not None:
            put(ORIGIN_STORY_FILE, chosen)
            origin_story.add(ORIGIN_STORY_FILE)
    put(SOURCES, plan.sources)
    put(
        META,
        plan.meta.model_copy(update={"as_of": None if through is None else through.isoformat()}),
    )

    expected = (
        {f"bills/{plan.refs[b].slug}.json" for b in plan.hosted}
        | names_now
        | {f"texts/{s}.json" for s in texts}
        | {f"stories/{k}.json" for k in plan.stories}
        | origin_story
        | {BILLS, SOURCES, META}
    )
    removed = sorted(before - expected)
    require_free_space(
        target.parent, needed=sum(map(len, writes.values())) + (16 << 20), floor=floor
    )
    for name, data in sorted(writes.items()):
        _put(target, name, data)
    for name in removed:
        (target / name).unlink()
    written: dict[str, int] = defaultdict(int)
    for name in writes:
        written[name.split("/")[0] if "/" in name else name] += 1
    return {
        "state": state,
        "days": [d.isoformat() for d in days],
        "as_of": None if through is None else through.isoformat(),
        "bills_named": len(log.bills),
        "files_written": dict(sorted(written.items())),
        "files_removed": len(removed),
        "comparisons_rebuilt": len(rebuild),
        "section_pairs_aligned": len(todo),
        "section_pairs_kept": reused,
        "beyond_log": sorted(beyond),
        "written": sorted(writes),
        "removed": removed,
        "staging": str(target),
        "seconds": round(time.monotonic() - started, 1),
    }
