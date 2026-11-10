import json
from pathlib import Path

from lineage.models import MODELS
from lineage.pair_eval import REPORT, TARGET_PRECISION
from lineage.semantic import MODEL, THRESHOLD


def test_the_model_and_threshold_are_the_ones_the_pair_evaluation_chose() -> None:
    semantic = json.loads(Path(REPORT).read_text())["semantic"]
    scored = semantic["models"]
    at = {name: m["at_precision"][str(TARGET_PRECISION)] for name, m in scored.items()}

    assert MODEL in MODELS
    assert semantic["target_precision"] == TARGET_PRECISION
    assert at[MODEL]["threshold"] == THRESHOLD
    # The model kept is the one with the most test recall at the target precision.
    assert max(at, key=lambda name: at[name]["test_recall"]) == MODEL
