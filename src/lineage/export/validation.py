"""The validation page's figures, read from the committed results (reports/results.json): the
commission recall by act and stage, the relatives' precision, the explanations' faithfulness and
the origin precision, with how many documents were withheld after it was measured. Every
export's meta.json carries them (lineage.export.pages). See DESIGN.md#export.
"""

from collections.abc import Mapping
from typing import Any

from lineage.export.schema import (
    Enactment,
    Excluded,
    OriginPrecision,
    OriginShare,
    Recall,
    Share,
    Validation,
)
from lineage.origins.withheld import NONE as NO_WITHHELD
from lineage.origins.withheld import Withheld


def validation_from(results: Mapping[str, Any], withheld: Withheld = NO_WITHHELD) -> Validation:
    """The validation page's figures, read from the committed results; origin precision with
    how many documents were withheld after it was measured (lineage.origins.withheld)."""

    def recall(r: Mapping[str, Any]) -> Recall:
        return Recall(enactments=r["enactments"], found=r["found"], recall=r["recall"])

    def origin_share(g: Mapping[str, Any]) -> OriginShare:
        p = g["precision"]
        return OriginShare(
            sources=list(g["sources"]),
            matches=g["matches"],
            labeled=g["labeled"],
            correct=g["correct"],
            share=None if p is None else p["precision"],
            interval_95=None if p is None else tuple(p["interval_95"]),
        )

    origins = results["origins_precision"]
    (compared,) = (k for k in origins["compared"] if k != "rest")
    dev, held = results["commission"], results["commission_held_out"]
    stages = dev["stage_ablation"]
    precision = results["relatives_precision"]["provisional_model"]
    faith = results["faithfulness"]
    return Validation(
        development_recall=recall(dev["recall"]["pooled"]["all"]),
        development_recall_by_act={a: recall(r["all"]) for a, r in dev["recall"]["by_act"].items()},
        development_recall_by_stage={
            "near_duplicate": recall(stages["near_duplicate"]["pooled"]["all"]),
            "semantic": recall(stages["semantic"]["pooled"]["all"]),
            "both": recall(dev["recall"]["pooled"]["all"]),
        },
        semantic_threshold=dev["semantic_threshold"],
        development_enactments=[
            Enactment(
                act=e["act"],
                state=e["state"],
                year=e["year"],
                bill_cell=e["bill_cell"],
                stratum=e["stratum"],
                found=e["found"],
                share=e["share"],
                miss_reason=e["miss_reason"],
            )
            for e in dev["enactments"]
        ],
        development_excluded=[
            Excluded(act=x["act"], jurisdiction=x["jurisdiction"], year=x["year"], why=x["why"])
            for x in dev["excluded"]
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
        origin_precision=OriginPrecision(
            floor=origins["floor"],
            overall=origin_share(origins["overall"]),
            compared=origin_share(origins["compared"][compared]),
            rest=origin_share(origins["compared"]["rest"]),
            labeler=list(origins["labeler"]),
            withheld_documents=len(withheld.ids),
            withheld_matches=withheld.matches,
        ),
    )
