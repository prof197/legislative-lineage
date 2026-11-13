"""`lineage export --state GA`: the static site's JSON for one state's bills (DESIGN.md#export).

Every file is an instance of a model in lineage.export_schema, validated as it is written, and
the JSON Schema those models emit is the site's contract (DR-1). One build writes:

- `meta.json`: the data stamp (each hosted archive's Generated At), the stories, and the
  validation page's figures from the committed results.
- `bills.json`: every hosted bill, for the browse page's filter.
- `bills/<slug>.json`: one page per hosted bill: its record, its text's source, its sections,
  and its relatives as a page lists them (lineage.relatives), each with the section pairs it
  shares with the hosted bill.
- `compare/<a>--<b>.json`: each listed relative's, and each featured story pair's, matched
  sections aligned (lineage.align), with a featured pair's grounded explanation from the cache
  (lineage.explain); a pair is never sent to a model here.
- `texts/<slug>.json`: the display text of each section some comparison shows, once per bill.
  A section is shown beside many relatives, so keeping its text out of the comparisons keeps
  the site's size to its distinct text (measured in reports/export.json).
- `stories/<slug>.json`: each featured story's anchor and relatives (reports/featured_pairs.json).

The hosted bills are the state's records the document-type rules match. The first bill of a
comparison is the hosted bill, or a story's anchor, whose text the comparison starts from. The
output is written to a fresh directory and swapped in whole, so a failed export never leaves a
mix of two builds.
"""

import json
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from lineage import queries
from lineage.align import Embedder, align
from lineage.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.explain import CACHE_DIR, explain, prompt_spans
from lineage.explain import MODEL as EXPLAIN_MODEL
from lineage.export_schema import (
    TOP_LEVEL,
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
    Enactment,
    Explanation,
    Meta,
    Recall,
    Relative,
    SectionInfo,
    SectionPair,
    SectionText,
    Share,
    Span,
    Stage,
    Story,
    StoryRelative,
    TextInfo,
    Validation,
)
from lineage.listing import ListedPairs, listed_pairs, shipped_flag
from lineage.minhash import NUM_PERM
from lineage.relatives import bill_relatives
from lineage.semantic import MODEL
from lineage.semantic_pairs import section_keys
from lineage.state_boilerplate import COMPACT_PHRASES
from lineage.stats import (
    AMENDMENT_CHARS,
    AMENDMENT_OPENING,
    AMENDMENT_SUBSTITUTE,
    GEORGIA_ENACTING,
    GEORGIA_NOTICE,
)

EXPORT_DIR = Path("data/export")
REPORT = Path("reports/export.json")
SCHEMA = Path("schema/export.schema.json")
FEATURED_PAIRS = Path("evals/featured_pairs.jsonl")
FEATURED_REPORT = Path("reports/featured_pairs.json")
RESULTS = Path("reports/results.json")
COMMAND = "uv run lineage export --state GA"
# GitHub Pages' limit on a published site; DR-2 reverses at half of it.
SITE_LIMIT = 1_000_000_000
_COMPACT = re.compile("|".join(re.escape(p) for p in COMPACT_PHRASES))


class ExportError(Exception):
    pass


def slug(state: str, session: str, identifier: str) -> str:
    """A bill's file name: its state, session and identifier, lower case, each run of other
    characters one hyphen."""
    return re.sub(r"[^a-z0-9_]+", "-", f"{state}-{session}-{identifier}".lower()).strip("-")


def comparison_slug(a: str, b: str) -> str:
    return f"{a}--{b}"


def export_schema() -> dict[str, Any]:
    """The JSON Schema of every file the export writes, as the site reads them."""
    _, schema = models_json_schema(
        [(m, "serialization") for m in TOP_LEVEL], title="Legislative Lineage export"
    )
    return schema


def schema_text() -> str:
    return json.dumps(export_schema(), indent=1, sort_keys=True) + "\n"


BillPair = tuple[str, str]  # (first bill id, second bill id)
OrdinalPair = tuple[int, int]


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


def _stages(found: _PairFound) -> list[Stage]:
    return [s for s in ("near_duplicate", "semantic") if s in found.stages]


def validation_from(results: Mapping[str, Any]) -> Validation:
    """The validation page's figures, read from the committed results."""

    def recall(r: Mapping[str, Any]) -> Recall:
        return Recall(enactments=r["enactments"], found=r["found"], recall=r["recall"])

    dev, held = results["commission"], results["commission_held_out"]
    precision = results["relatives_precision"]["provisional_model"]
    faith = results["faithfulness"]
    return Validation(
        development_recall=recall(dev["recall"]["pooled"]["all"]),
        development_recall_by_act={a: recall(r["all"]) for a, r in dev["recall"]["by_act"].items()},
        development_enactments=[
            Enactment(
                act=e["act"],
                state=e["state"],
                year=e["year"],
                bill_cell=e["bill_cell"],
                stratum=e["stratum"],
                found=e["found"],
                miss_reason=e["miss_reason"],
            )
            for e in dev["enactments"]
        ],
        held_out_recall=recall(held["recall"]["pooled"]["all"]),
        relatives_precision=Share(
            labeled=precision["overall"]["labeled"],
            correct=precision["overall"]["correct"],
            share=precision["overall"]["precision"],
            interval_95=tuple(precision["overall"]["interval_95"]),
        ),
        relatives_precision_labeler=precision["labeler"],
        faithfulness=Share(
            labeled=faith["overall"]["judged"],
            correct=faith["overall"]["correct"],
            share=faith["overall"]["share"],
            interval_95=tuple(faith["overall"]["interval_95"]),
        ),
        faithfulness_judge=faith["judge"],
    )


def _refs(con: Any, tables: Path, ids: list[str]) -> tuple[dict[str, BillRef], dict[str, TextInfo]]:
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
    for (
        bill_id,
        state,
        session,
        identifier,
        title,
        status,
        reason,
        status_date,
        first,
        source,
        version,
        url,
        notice_only,
        amendment_only,
    ) in rows:
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


def _load_featured(path: Path) -> set[tuple[str, int, str, int]]:
    out: set[tuple[str, int, str, int]] = set()
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            o, d = row["origin"], row["descendant"]
            out.add((str(o["bill_id"]), int(o["ordinal"]), str(d["bill_id"]), int(d["ordinal"])))
    return out


@dataclass(frozen=True, slots=True)
class _Written:
    files: dict[str, int]  # per kind, how many
    bytes: dict[str, int]  # per kind, how many bytes
    largest: tuple[str, int]


def _write(root: Path, files: Mapping[str, BaseModel]) -> _Written:
    count: dict[str, int] = defaultdict(int)
    size: dict[str, int] = defaultdict(int)
    largest = ("", 0)
    for name, model in sorted(files.items()):
        data = model.model_dump_json().encode()
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        kind = name.split("/")[0] if "/" in name else name
        count[kind] += 1
        size[kind] += len(data)
        if len(data) > largest[1]:
            largest = (name, len(data))
    return _Written(dict(sorted(count.items())), dict(sorted(size.items())), largest)


def build_export(
    tables: Path,
    *,
    state: str,
    out: Path,
    embed: Embedder | None,
    cache: Path = CACHE_DIR,
    featured_pairs: Path = FEATURED_PAIRS,
    featured_report: Path = FEATURED_REPORT,
    results: Path = RESULTS,
    model: str = MODEL,
    require_flag: bool = True,
    floor: int = DEFAULT_FLOOR_BYTES,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Write `state`'s site data to `out`, replacing what was there; returns the report."""
    started = time.monotonic()
    bills_glob = str(tables / "bills" / "*.parquet")
    flag = shipped_flag(tables)
    if flag is None and require_flag:
        raise ExportError("no state boilerplate counts; run `uv run lineage state-boilerplate`")
    with queries.connect_quiet() as con:
        hosted = queries.hosted_bill_ids(
            con, bills_glob=bills_glob, eligible_path=tables / "eligible.parquet", state=state
        )
        stamps = queries.archive_stamps(con, bills_glob=bills_glob, state=state)
        indexed = queries.states_indexed(con, bills_glob=bills_glob)
    if not hosted:
        raise ExportError(f"no {state} bill is matched by the document-type rules")
    progress(f"{len(hosted)} hosted bills")
    listed = listed_pairs(tables, model=model, flag=flag)
    relatives = bill_relatives(tables, hosted, model=model, listed=listed)

    featured = _load_featured(featured_pairs)
    stories_doc: dict[str, Any] = (
        json.loads(featured_report.read_text())["stories"] if featured_report.exists() else {}
    )
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
    keys = sorted(
        {(a, oa) for (a, _), ps in found.items() for oa, _ in ps}
        | {(b, ob) for (_, b), ps in found.items() for _, ob in ps}
    )
    with queries.connect_quiet() as con:
        refs, text_infos = _refs(con, tables, ids)
        section_rows = queries.section_infos(
            con, sections_glob=str(tables / "sections" / "*.parquet"), ids=hosted
        )
        shown = queries.sections_at(
            con, sections_glob=str(tables / "sections" / "*.parquet"), keys=keys
        )
    text_bytes = sum(len(t.encode()) for _, t, _ in shown.values())
    n_pairs = sum(len(p) for p in found.values())
    # Each text written once, spans at most a few hundred bytes a pair, and the pages.
    require_free_space(out.parent, needed=2 * text_bytes + 4096 * n_pairs + (64 << 20), floor=floor)
    progress(f"{len(found)} comparisons, {n_pairs} section pairs to align")

    files: dict[str, BaseModel] = {}
    featured_explained = 0
    withheld = 0
    for done, ((a, b), pairs) in enumerate(sorted(found.items()), start=1):
        aligned: list[AlignedPair] = []
        for (oa, ob), pf in sorted(pairs.items()):
            ta, tb = shown[(a, oa)][1], shown[(b, ob)][1]
            alignment = align(ta, tb, embed=embed)
            explanation = None
            if (a, oa, b, ob) in featured:
                outcome = explain(
                    prompt_spans(alignment.spans, ta, tb, origin_side="a"), ask=None, cache=cache
                )
                explanation = Explanation(
                    model=EXPLAIN_MODEL,
                    withheld=outcome.withheld,
                    claims=[Claim.model_validate(c.model_dump()) for c in outcome.claims],
                )
                featured_explained += 1
                withheld += outcome.withheld
            aligned.append(
                AlignedPair(
                    ordinal_a=oa,
                    ordinal_b=ob,
                    stages=_stages(pf),
                    spans=[Span(id=s.id, kind=s.kind, a=s.a, b=s.b) for s in alignment.spans],
                    copied_share_a=round(alignment.copied_share_a, 4),
                    copied_share_b=round(alignment.copied_share_b, 4),
                    explanation=explanation,
                )
            )
        name = comparison_slug(refs[a].slug, refs[b].slug)
        files[f"compare/{name}.json"] = Comparison(a=refs[a], b=refs[b], pairs=aligned)
        if done % 500 == 0:
            progress(f"aligned {done} of {len(found)} comparisons")

    by_bill: dict[str, list[SectionText]] = defaultdict(list)
    for (bill_id, ordinal), (heading, text, _) in sorted(shown.items()):
        by_bill[bill_id].append(SectionText(ordinal=ordinal, heading=heading, text=text))
    for bill_id, sections in by_bill.items():
        name = refs[bill_id].slug
        files[f"texts/{name}.json"] = BillTexts(bill=name, sections=sections)

    sections_of: dict[str, list[tuple[int, str | None, int, str | None]]] = defaultdict(list)
    for bill_id, ordinal, heading, words, boilerplate in section_rows:
        sections_of[bill_id].append((ordinal, heading, words, boilerplate))
    entries: list[BillListEntry] = []
    for bill_id in hosted:
        r = relatives[bill_id]
        listed_here: list[Relative] = []
        per_section: dict[int, int] = defaultdict(int)
        for m in r.matches:
            pairs = found.get((bill_id, m.bill_id), {})
            compact = any(
                _COMPACT.search(shown[(bill_id, oa)][2])
                or _COMPACT.search(shown[(m.bill_id, ob)][2])
                for oa, ob in pairs
            )
            for oa in {oa for oa, _ in pairs}:
                per_section[oa] += 1
            listed_here.append(
                Relative(
                    bill=refs[m.bill_id],
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
                    comparison=comparison_slug(refs[bill_id].slug, refs[m.bill_id].slug),
                )
            )
        page = BillPage(
            bill=refs[bill_id],
            text=text_infos.get(bill_id),
            sections=[
                SectionInfo(
                    ordinal=o,
                    heading=h,
                    words=w,
                    boilerplate=bp,
                    with_relatives=per_section.get(o, 0),
                )
                for o, h, w, bp in sections_of[bill_id]
            ],
            words_searched=r.words_searched,
            families=r.families,
            relatives=listed_here,
        )
        files[f"bills/{refs[bill_id].slug}.json"] = page
        best = listed_here[0] if listed_here else None
        entries.append(
            BillListEntry(
                bill=refs[bill_id],
                families=r.families,
                closest=None
                if best is None
                else Closest(
                    slug=best.bill.slug,
                    state=best.bill.state,
                    identifier=best.bill.identifier,
                    share=best.share,
                ),
            )
        )
    files["bills.json"] = BillList(bills=entries)

    story_slugs: list[str] = []
    for key, s in sorted(stories_doc.items()):
        anchor = str(s["anchor"]["bill_id"])
        story_relatives = [
            StoryRelative(
                bill=refs[str(rel["bill_id"])],
                section_pairs=int(rel["section_pairs"]),
                comparison=comparison_slug(refs[anchor].slug, refs[str(rel["bill_id"])].slug)
                if int(rel["section_pairs"])
                else None,
            )
            for rel in s["relatives"]
        ]
        files[f"stories/{key}.json"] = Story(
            slug=key,
            anchor=refs[anchor],
            relatives_rule=s["relatives_rule"],
            scope_note=s["scope_note"],
            relatives=story_relatives,
        )
        story_slugs.append(key)

    files["meta.json"] = Meta(
        state=state,
        archives=[
            ArchiveStamp(session=session, generated_at=at.isoformat()) for session, at in stamps
        ],
        states_indexed=indexed,
        hosted_bills=len(hosted),
        stories=story_slugs,
        validation=validation_from(json.loads(results.read_text())),
    )

    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    written = _write(partial, files)
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    total = sum(written.bytes.values())
    with_relatives = sum(1 for r in relatives.values() if r.matches)
    return {
        "command": COMMAND,
        "state": state,
        "schema": str(SCHEMA),
        "hosted_bills": len(hosted),
        "bills_with_relatives": with_relatives,
        "relatives_listed": sum(len(r.matches) for r in relatives.values()),
        "comparisons": len(found),
        "section_pairs_aligned": n_pairs,
        "featured_pairs_explained": featured_explained,
        "featured_pairs_withheld": withheld,
        "files": written.files,
        "bytes": written.bytes,
        "total_bytes": total,
        "share_of_site_limit": round(total / SITE_LIMIT, 4),
        "largest_file": {"path": written.largest[0], "bytes": written.largest[1]},
        "seconds": round(time.monotonic() - started),
    }
