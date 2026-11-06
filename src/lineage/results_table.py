"""The README's results table, rendered from `reports/results.json`, never typed by hand.

`lineage eval readme` rewrites the text between the README's results markers, and a test fails
when the committed README and results file disagree, so the numbers shown are the numbers
measured.
"""

from pathlib import Path
from typing import Any

from lineage.acts import ACT_NAMES

README = Path("README.md")
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
    "below_cutoff": "below the cutoff",
}
_STRATA = {"enacted": "Enacted", "substantially_similar": "Substantially similar"}


class ResultsTableError(Exception):
    pass


def _cell(r: dict[str, Any]) -> str:
    if not r["enactments"]:
        return "none"
    return f"{r['found']} / {r['enactments']} ({r['found'] / r['enactments']:.1%})"


def render_commission(section: dict[str, Any]) -> str:
    recall = section["recall"]
    cutoff = section["cutoff"]["max_rank"]
    read = section["enactments_file"]["read"].removeprefix("ULC Bill List, ")
    rows = [("Pooled, three acts", recall["pooled"])]
    rows += [(ACT_NAMES[act], r) for act, r in recall["by_act"].items()]
    rows += [(_STRATA[s], r) for s, r in recall["by_stratum"].items()]
    lines = [
        "Commission recall: of the bills that enacted three uniform acts, the share that the "
        f"act's own text finds among its top {cutoff} relatives. The lists were {read}. "
        f"Reproduce with `{section['command']}` → `reports/results.json`.",
        "",
        "| | All enactments | With text in the source |",
        "|---|---|---|",
        *(f"| {label} | {_cell(r['all'])} | {_cell(r['with_text'])} |" for label, r in rows),
        "",
        "Misses: "
        + ", ".join(f"{_REASONS[k]} {v}" for k, v in section["miss_reasons"].items())
        + ". The reasons are defined in DESIGN.md, "
        "[Commission recall](DESIGN.md#commission-recall).",
    ]
    return "\n".join(lines)


def rendered_readme(readme: str, results: dict[str, Any]) -> str:
    """The README with its results section replaced by one rendered from `results`."""
    start, end = readme.find(START), readme.find(END)
    if start < 0 or end < start:
        raise ResultsTableError(f"README has no {START} ... {END} section")
    body = render_commission(results["commission"])
    return readme[: start + len(START)] + "\n" + body + "\n" + readme[end:]
