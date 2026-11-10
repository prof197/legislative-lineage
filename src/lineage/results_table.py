"""The README's results table, rendered from the committed results files, never typed by hand.

`lineage eval readme` rewrites the text between the README's results markers, and a test fails
when the committed README and results files disagree, so the numbers shown are the numbers
measured. Recall with the text layer comes from `reports/results.json`: both stages, and the
near-duplicate stage alone from its stage ablation. The last column is the near-duplicate stage
on Open States' text alone (`reports/results_exported_only.json`), so the table shows what the
text layer added, stage for stage. The export's text has no semantic run, since the corpus's
vectors are embedded from the fetched text. See DESIGN.md#dr-6.
"""

from pathlib import Path
from typing import Any

from lineage.acts import ACT_NAMES

README = Path("README.md")
EXPORTED_ONLY = Path("reports/results_exported_only.json")
EXPORTED_ONLY_STEPS = (
    "`uv run lineage text merge --exported-only`, the pipeline, then "
    f"`uv run lineage eval commission --results {EXPORTED_ONLY}`"
)
START = "<!-- results:commission -->"
END = "<!-- /results:commission -->"

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
    for key in ("threshold", "enactments_file"):
        if section[key] != exported[key]:
            raise ResultsTableError(f"the two results files differ in {key}; rerun both evals")
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


def rendered_readme(readme: str, results: dict[str, Any], exported: dict[str, Any]) -> str:
    """The README with its results section replaced by one rendered from `results`, and from
    `exported`, the results on the export's text alone."""
    start, end = readme.find(START), readme.find(END)
    if start < 0 or end < start:
        raise ResultsTableError(f"README has no {START} ... {END} section")
    body = render_commission(results["commission"], exported["commission"])
    return readme[: start + len(START)] + "\n" + body + "\n" + readme[end:]
