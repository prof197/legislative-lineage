import json
from pathlib import Path

from lineage.models import MODELS
from lineage.pair_eval import REPORT, TARGET_PRECISION
from lineage.semantic import BUILD_THRESHOLD, MODEL, THRESHOLD
from lineage.threshold import CANDIDATES
from lineage.threshold import REPORT as PROPOSAL


def test_the_model_and_build_threshold_are_the_ones_the_pair_evaluation_chose() -> None:
    semantic = json.loads(Path(REPORT).read_text())["semantic"]
    scored = semantic["models"]
    at = {name: m["at_precision"][str(TARGET_PRECISION)] for name, m in scored.items()}

    assert MODEL in MODELS
    assert semantic["target_precision"] == TARGET_PRECISION
    assert at[MODEL]["threshold"] == BUILD_THRESHOLD
    # The model kept is the one with the most test recall at the target precision.
    assert max(at, key=lambda name: at[name]["test_recall"]) == MODEL


def test_the_stage_threshold_is_an_option_the_proposal_measured_above_the_build() -> None:
    options = json.loads(PROPOSAL.read_text())["options"]

    assert BUILD_THRESHOLD < THRESHOLD
    assert THRESHOLD in CANDIDATES
    assert any(o.get("threshold") == THRESHOLD and o["option"] == "threshold" for o in options)
