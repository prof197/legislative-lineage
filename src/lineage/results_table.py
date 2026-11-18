"""The README's results table, rendered from the committed results files, never typed by hand.

`lineage eval readme` rewrites the text between the README's results markers, and a test fails
when the committed README and results files disagree, so the numbers shown are the numbers
measured. Recall with the text layer comes from `reports/results.json`: both stages, and the
near-duplicate stage alone from its stage ablation. The last column is the near-duplicate stage
on Open States' text alone (`reports/results_exported_only.json`), so the table shows what the
text layer added, stage for stage. The export's text has no semantic run, since the corpus's
vectors are embedded from the fetched text. See DESIGN.md#dr-6.
"""

import json
import tomllib
from pathlib import Path
from typing import Any, cast

from lineage.evals.acts import ACT_NAMES
from lineage.evals.ulc import ULC_STATES
from lineage.export import STORIES_RULES, stories_for
from lineage.origin_story import STORY as ORIGIN_STORY
from lineage.release import RELEASE_MANIFEST

README = Path("README.md")
EXPORTED_ONLY = Path("reports/results_exported_only.json")
EXPORTED_ONLY_STEPS = (
    "`uv run lineage text merge --exported-only`, the pipeline, then "
    f"`uv run lineage eval commission --results {EXPORTED_ONLY}`"
)
START = "<!-- results:commission -->"
END = "<!-- /results:commission -->"
HELD_OUT_START = "<!-- results:held-out -->"
HELD_OUT_END = "<!-- /results:held-out -->"
HELD_OUT_COMMAND = "uv run lineage eval commission --held-out --final"
SUMMARY_START = "<!-- results:summary -->"
SUMMARY_END = "<!-- /results:summary -->"
ORIGINS_START = "<!-- results:origins -->"
ORIGINS_END = "<!-- /results:origins -->"
# The stories the hosted state's site carries, counted from rules/stories.toml.
STORIES_START = "<!-- stories -->"
STORIES_END = "<!-- /stories -->"
_COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
# v2's origin evals, each rendered only when its section is in the results: they are
# regenerated as matching settles, and a row for a run not yet made would be a number typed in.
ORIGIN_DEVELOPMENT = "origins_development"
ORIGIN_SEALED = "origins_held_out"
ORIGIN_V1_HELD_OUT = "origins_v1_held_out"
ORIGIN_PRECISION = "origins_precision"
# One footnote on the sealed row, filled in when the README is regenerated: the sealed key runs
# once per configuration, so a later change to matching is said beside it, never rerun to fit.
SEALED_NOTE = (
    "Run once, before generic sections were left out of origin matches; a sealed key is not run "
    "again, so it stands at the configuration it records ([DESIGN.md](DESIGN.md#origin-matches))."
)
_ORIGIN_STAGES = {
    "near_duplicate_only": "found by the near-duplicate stage alone",
    "semantic_only": "found by meaning alone",
    "both": "found by both stages",
}
# The precision sample's report, which names the state whose pages were sampled.
SAMPLED_STATE_REPORT = Path("reports/relatives_reporting.json")
_COUNTS = {2: "two", 3: "three", 4: "four"}

_REASONS = {
    "not_in_corpus": "not in the corpus",
    "no_text_in_source": "no text in the source",
    "amendment_only": "an amendment stored instead of the bill",
    "excluded_type": "an excluded document type",
    "no_usable_text": "no usable text",
    "not_proposed": "never proposed",
    "bucket_cap": "the bucket cap",
    "verification": "verification",
    "below_threshold": "below the threshold",
}
_STRATA = {"enacted": "Enacted", "substantially_similar": "Substantially similar"}


class ResultsTableError(Exception):
    pass


def _cell(r: dict[str, Any]) -> str:
    if not r["enactments"]:
        return "none"
    return f"{r['found']} / {r['enactments']} ({r['found'] / r['enactments']:.1%})"


def render_commission(section: dict[str, Any], exported: dict[str, Any]) -> str:
    if section["enactments_file"] != exported["enactments_file"]:
        raise ResultsTableError("the two results files differ in enactments_file; rerun both evals")
    # At a threshold of 0, any matched section is found, so how the score weighs sections does
    # not change what either column counts.
    mine, theirs = section["threshold"], exported["threshold"]
    if mine["min_share"] == 0:
        compared = ("min_share", "found_when")
        mine, theirs = {k: mine.get(k) for k in compared}, {k: theirs.get(k) for k in compared}
    if mine != theirs:
        raise ResultsTableError("the two results files differ in threshold; rerun both evals")
    if exported.get("stage_ablation") is not None:
        raise ResultsTableError(
            "the export-only results searched by meaning too, over vectors of other text; "
            "rerun them with --no-semantic"
        )
    ablation = section.get("stage_ablation")
    alone = exported["recall"]
    if ablation is None:
        heads = ["With the text layer", "Open States' text alone"]
        columns = [section["recall"], alone]
        matched = "sharing at least one verified section with it"
        misses = "Misses with the text layer"
    else:
        heads = [
            "Both stages",
            "Near-duplicate stage",
            "Near-duplicate stage, Open States' text alone",
        ]
        columns = [section["recall"], ablation["near_duplicate"], alone]
        matched = "matching at least one of its sections by wording or by meaning"
        misses = "Misses by both stages, at the step where the near-duplicate stage lost them"
    min_share = section["threshold"]["min_share"]
    found_when = (
        matched if min_share == 0 else f"sharing more than {min_share:.0%} of its sections with it"
    )
    read = section["enactments_file"]["read"].removeprefix("ULC Bill List, ")
    rows = [("Pooled, three acts", [c["pooled"] for c in columns])]
    rows += [(ACT_NAMES[a], [c["by_act"][a] for c in columns]) for a in section["recall"]["by_act"]]
    rows += [
        (_STRATA[s], [c["by_stratum"][s] for c in columns]) for s in section["recall"]["by_stratum"]
    ]
    lines = [
        "Commission recall: of the bills that enacted three uniform acts, the share that the "
        f"act's own text finds, {found_when}, not counting the commission's "
        f"standard closing sections. The lists were {read}. "
        f"Reproduce with `{section['command']}` → `reports/results.json`. The last column "
        "counts the same enactments with Open States' exported text alone, before the project "
        f"fetched any: {EXPORTED_ONLY_STEPS}.",
        "",
        "| | " + " | ".join(heads) + " |",
        "|---" * (len(heads) + 1) + "|",
        *(
            f"| {label} | " + " | ".join(_cell(c["all"]) for c in cells) + " |"
            for label, cells in rows
        ),
        "",
        f"{misses}: "
        + ", ".join(f"{_REASONS[k]} {v}" for k, v in section["miss_reasons"].items())
        + ". The reasons are defined in DESIGN.md, "
        "[Commission recall](DESIGN.md#commission-recall).",
    ]
    return "\n".join(lines)


def render_held_out(section: dict[str, Any]) -> str:
    """The held-out run's recall, both stages beside each stage alone."""
    ablation = section.get("stage_ablation")
    if ablation is None:
        raise ResultsTableError("the held-out results have no stage ablation; run both stages")
    heads = ["Both stages", "Near-duplicate stage", "Semantic stage"]
    columns = [section["recall"], ablation["near_duplicate"], ablation["semantic"]]
    acts = list(section["recall"]["by_act"])
    rows = [(f"Pooled, {_COUNTS.get(len(acts), len(acts))} acts", [c["pooled"] for c in columns])]
    rows += [(ACT_NAMES[a], [c["by_act"][a] for c in columns]) for a in acts]
    rows += [
        (_STRATA[s], [c["by_stratum"][s] for c in columns]) for s in section["recall"]["by_stratum"]
    ]
    flag = "with" if section.get("state_boilerplate_flag") else "without"
    read = section["enactments_file"]["read"].removeprefix("ULC Bill List, ")
    earlier = "".join(
        f" It was first run at a threshold of {r['semantic_threshold']}, which found "
        f"{r['recall']['all']['found']} of {r['recall']['all']['enactments']}; the threshold "
        "then moved on tuning data alone ([DESIGN.md](DESIGN.md#commission-recall))."
        for r in section.get("earlier_runs", [])
    )
    lines = [
        "Held-out recall: the same measure on four uniform acts never examined while matching "
        "was tuned, run at the end of M5 with the semantic stage's threshold at "
        f"{section['semantic_threshold']} and {flag} the state boilerplate flag.{earlier} The "
        f"lists were {read}. Reproduce with `{HELD_OUT_COMMAND}` → `reports/results.json`, "
        "`commission_held_out`.",
        "",
        "| | " + " | ".join(heads) + " |",
        "|---" * (len(heads) + 1) + "|",
        *(
            f"| {label} | " + " | ".join(_cell(c["all"]) for c in cells) + " |"
            for label, cells in rows
        ),
        "",
        "Misses by both stages, at the step where the near-duplicate stage lost them: "
        + ", ".join(f"{_REASONS[k]} {v}" for k, v in section["miss_reasons"].items())
        + ".",
    ]
    return "\n".join(lines)


def _share(x: dict[str, Any], key: str) -> str:
    low, high = x["interval_95"]
    return f"{x[key]:.2f} [{low:.2f}, {high:.2f}]"


def _counted(n: int, noun: str) -> str:
    return f"{_COUNTS.get(n, n)} {noun}{'' if n == 1 else 's'}"


def _origin_floor(results: dict[str, Any]) -> float | None:
    """The listing floor every origin eval present was read at; refused when they differ, since
    rows read at different floors would sit side by side as one configuration."""
    floors = {
        results[name]["threshold"]["floor"]
        for name in (ORIGIN_DEVELOPMENT, ORIGIN_SEALED, ORIGIN_V1_HELD_OUT)
        if name in results
    }
    if ORIGIN_PRECISION in results:
        floors.add(results[ORIGIN_PRECISION]["floor"])
    if len(floors) > 1:
        raise ResultsTableError(
            f"the origin evals were read at different floors ({sorted(floors)}); rerun them"
        )
    return next(iter(floors), None)


def _origin_recall_command(section: dict[str, Any]) -> str:
    key = section["answer_key"]["name"]
    if key == "development":
        return section["command"]
    return f"{section['command']} --key {key}" + (" --final" if section["sealed"] else "")


def _origin_recall(section: dict[str, Any]) -> str:
    r = section["recall"]["pooled"]
    if not r["enactments"]:
        return "none"
    low, high = r["interval_95"]
    return f"{r['found']} of {r['enactments']}, {r['recall']:.2f} [{low:.2f}, {high:.2f}]"


def _origin_precision(group: dict[str, Any]) -> str:
    """A stage's or the whole sample's precision; a stratum's figure sits on the group itself,
    the whole sample's in a nested estimate."""
    p = group["precision"]
    if p is None:
        return "no estimate: a stage holding matches has no label"
    if isinstance(p, float | int):
        return _share(group, "precision")
    return _share(cast(dict[str, Any], p), "precision")


V1_HELD_OUT_STATUS = "opened in v1; not a sealed test"


def _compared_name(name: str, group: dict[str, Any]) -> str:
    """A compared group by whose matches it holds: one source's, by its organization's name as
    its collector gives it, or every other source's."""
    if name == "rest":
        return "every other source's matches"
    from lineage.collect import collectors  # the registry imports every collector

    (source,) = group["sources"]
    organization = next(
        (c.source.organization for c in collectors().values() if c.source.key == source), source
    )
    return f"{organization}'s matches"


def render_origins(
    results: dict[str, Any],
    sealed_note: str = SEALED_NOTE,
    withheld: dict[str, Any] | None = None,
) -> str:
    """v2's origin evals, one row each for those in the results and nothing for those not:
    recall on the development key and on the sealed key, precision on the reporting sample
    overall and by stage, and apart from the sealed row, v1's held-out acts, which v1's final
    run had already opened. With `withheld`, the withheld documents' report (lineage.withheld),
    precision is said to be read before they were withheld."""
    floor = _origin_floor(results)
    if floor is None:
        return ""
    recall = [results[n] for n in (ORIGIN_DEVELOPMENT, ORIGIN_SEALED) if n in results]
    precision = results.get(ORIGIN_PRECISION)
    v1 = results.get(ORIGIN_V1_HELD_OUT)
    intro: list[str] = []
    if recall or v1:
        intro.append(
            "Origin recall: each uniform act's published text is matched against the corpus "
            "as a model document, through both stages and the rule a page lists by, and an "
            "enactment is found when the act's document matches one of its bills with the "
            "larger of the two shares, of the model's words or of the bill's, at the listing "
            f"floor of {floor:.2f} or above."
        )
    if precision:
        intro.append(
            "Origin precision: a model's judgment of a seeded sample of the matches pages list "
            "at the floor, drawn by stage and weighted by each stage's matches, each labeled "
            "on the section pair it rests on most with v1's passage-level question. One "
            "source's matches are read against every other source's together, each weighted "
            "by stage as the whole is (`compared`), and each source's own figure is in the "
            "same section (`by_source`)."
        )
        held = [*(withheld or {}).get("withheld", []), *(withheld or {}).get("reviewed", [])]
        if held:
            n = len(held)
            matches = int((withheld or {})["listed_matches_withheld"])
            intro.append(
                f"The reading was taken before {_counted(n, 'document')} whose matches were "
                f"found mostly wrong {'was' if n == 1 else 'were'} withheld from pages, "
                f"{matches:,} listed {'match' if matches == 1 else 'matches'} in all "
                "(`uv run lineage withhold`, "
                "`reports/withheld_documents.json`), so it likely understates the precision of "
                "what pages show."
            )
    intro.append(
        "Each result carries its 95% Wilson interval, and each command writes "
        "`reports/results.json` ([DESIGN.md](DESIGN.md#origin-matches))."
    )
    head = ["| | Result | Reproduce |", "|---|---|---|"]
    rows: list[tuple[str, str, str]] = []
    for s in recall:
        acts = _counted(len(s["documents"]), "act")
        if s["sealed"]:
            label = f"Recall on the sealed key, {acts} v1 never ran, run once with `--final`"
            label += "[^sealed]" if sealed_note else ""
        else:
            label = f"Recall on the development key, {acts}"
        rows.append((label, _origin_recall(s), _origin_recall_command(s)))
    if precision:
        overall = precision["overall"]
        rows.append(
            (
                f"Precision, model-judged, all stages ({overall['labeled']} labels of "
                f"{overall['matches']:,} matches)",
                _origin_precision(overall),
                precision["command"],
            )
        )
        rows += [
            (
                f"Precision, {_ORIGIN_STAGES[stage]} ({g['labeled']} labels of "
                f"{g['size']:,} matches)",
                _origin_precision(g),
                precision["command"],
            )
            for stage, g in overall["by_stage"].items()
        ]
        compared: dict[str, Any] = precision["compared"]
        rows += [
            (
                f"Precision, {_compared_name(name, g)} ({g['labeled']} labels of "
                f"{g['matches']:,} matches)",
                _origin_precision(g),
                precision["command"],
            )
            for name, g in compared.items()
        ]
    lines = [" ".join(intro)]
    if rows:
        lines += ["", *head, *(f"| {label} | {r} | `{c}` |" for label, r, c in rows)]
    if sealed_note and ORIGIN_SEALED in results:
        lines += ["", f"[^sealed]: {sealed_note}"]
    if v1:
        if v1["status"] != V1_HELD_OUT_STATUS:
            raise ResultsTableError(
                f"v1's held-out acts are reported as {V1_HELD_OUT_STATUS!r}, not {v1['status']!r}"
            )
        acts = _counted(len(v1["documents"]), "held-out act")
        lines += [
            "",
            "Apart from the sealed key, v1's held-out acts: v1's final run looked at their "
            "matches, so they test nothing sealed.",
            "",
            *head,
            f"| Recall on v1's {acts}: {V1_HELD_OUT_STATUS} | {_origin_recall(v1)} | "
            f"`{_origin_recall_command(v1)}` |",
        ]
    return "\n".join(lines)


def render_summary(results: dict[str, Any], sampled_state: str) -> str:
    """The README's opening table: one line per question the evals answer, each with the
    command that reproduces it. The detail is in the sections below it."""
    dev = results["commission"]["recall"]["pooled"]["all"]
    held = results["commission_held_out"]["recall"]["pooled"]["all"]
    precision = results["relatives_precision"]["provisional_model"]["overall"]
    faith = results["faithfulness"]["overall"]
    acts = len(results["commission"]["recall"]["by_act"])
    held_acts = len(results["commission_held_out"]["recall"]["by_act"])
    state = next(
        (name for name, code in ULC_STATES.items() if code == sampled_state), sampled_state
    )
    rows = [
        (
            f"Does it find the bills that enacted a uniform act? ({_COUNTS.get(acts, acts)} acts)",
            f"{dev['found']} of {dev['enactments']} ({dev['found'] / dev['enactments']:.1%})",
            "uv run lineage eval commission",
        ),
        (
            f"The same, on {_COUNTS.get(held_acts, held_acts)} acts never used while tuning",
            f"{held['found']} of {held['enactments']} ({held['found'] / held['enactments']:.1%})",
            HELD_OUT_COMMAND,
        ),
        (
            f"Judged by a model: do the related bills a page lists share the provision? "
            f"({precision['labeled']} sampled from {state}'s pages)",
            _share(precision, "precision"),
            "uv run lineage eval relatives",
        ),
        (
            f"Judged by a model: do the explanations describe the text they cite? "
            f"({faith['judged']} claims)",
            _share(faith, "share"),
            "uv run lineage eval faithfulness",
        ),
    ]
    if ORIGIN_DEVELOPMENT in results:
        s = results[ORIGIN_DEVELOPMENT]
        r = s["recall"]["pooled"]
        acts = len(s["documents"])
        rows.append(
            (
                "Matched as published model legislation, does a uniform act find the bills "
                f"that enacted it? ({_COUNTS.get(acts, acts)} acts, at the listing floor)",
                f"{r['found']} of {r['enactments']} ({r['found'] / r['enactments']:.1%})",
                _origin_recall_command(s),
            )
        )
    if ORIGIN_PRECISION in results:
        overall = results[ORIGIN_PRECISION]["overall"]
        rows.append(
            (
                "Judged by a model: does the model legislation a page lists as an origin share "
                f"the provision? ({overall['labeled']} sampled matches)",
                _origin_precision(overall),
                results[ORIGIN_PRECISION]["command"],
            )
        )
    lines = [
        "| Question | Result | Reproduce |",
        "|---|---|---|",
        *(f"| {q} | {r} | `{c}` |" for q, r, c in rows),
    ]
    return "\n".join(lines)


def hosted_state(manifest: Path = RELEASE_MANIFEST) -> str:
    """The state the site hosts: the one whose data release the deploy builds from."""
    tag = str(json.loads(manifest.read_text())["tag"])
    return tag.split("-")[1].upper()


def render_stories(state: str, rules: Path = STORIES_RULES) -> str:
    """The stories `state`'s site carries, counted from the stories rules: its featured stories,
    and the origin story when the rules place it, which a site shows only once a labeler has
    judged its match correct (lineage.origin_story)."""
    listed: dict[str, Any] = tomllib.loads(rules.read_text()).get("stories", {})
    carried = stories_for(state, listed, rules)
    n = len(carried - {ORIGIN_STORY})
    count = _COUNT_WORDS[n] if n < len(_COUNT_WORDS) else f"{n:,}"
    out = f"{count} featured stor{'y' if n == 1 else 'ies'}"
    if ORIGIN_STORY in carried:
        out += (
            ", and one on a passed bill that shares wording with a published model, once a model "
            "has judged such a match correct"
        )
    return out + "."


def _replaced(readme: str, start_mark: str, end_mark: str, body: str) -> str:
    start, end = readme.find(start_mark), readme.find(end_mark)
    if start < 0 or end < start:
        raise ResultsTableError(f"README has no {start_mark} ... {end_mark} section")
    return readme[: start + len(start_mark)] + "\n" + body + "\n" + readme[end:]


def rendered_readme(
    readme: str,
    results: dict[str, Any],
    exported: dict[str, Any],
    sampled_state: str | None = None,
    sealed_note: str = SEALED_NOTE,
    stories: str | None = None,
    withheld: dict[str, Any] | None = None,
) -> str:
    """The README with its results section replaced by one rendered from `results`, and from
    `exported`, the results on the export's text alone; its opening summary, when it has one,
    from `results` and the state the precision sample was drawn from; its held-out section,
    when it has one, from the held-out run's results, which must then exist; and its origins
    section, when it has one, from whichever origin evals the results hold, with `sealed_note`
    as the sealed row's footnote when it is given; and its story count, when it has one, from
    `stories` (render_stories)."""
    out = _replaced(
        readme, START, END, render_commission(results["commission"], exported["commission"])
    )
    if SUMMARY_START in out:
        if sampled_state is None:
            raise ResultsTableError("README has a summary but no precision sample state was given")
        out = _replaced(out, SUMMARY_START, SUMMARY_END, render_summary(results, sampled_state))
    if HELD_OUT_START in out:
        if "commission_held_out" not in results:
            raise ResultsTableError(
                "README has a held-out section but the held-out run has no results"
            )
        out = _replaced(
            out, HELD_OUT_START, HELD_OUT_END, render_held_out(results["commission_held_out"])
        )
    if ORIGINS_START in out:
        out = _replaced(
            out, ORIGINS_START, ORIGINS_END, render_origins(results, sealed_note, withheld)
        )
    if STORIES_START in out:
        if stories is None:
            raise ResultsTableError("README counts its stories but no count was given")
        out = _replaced(out, STORIES_START, STORIES_END, stories)
    return out
