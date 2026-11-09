"""The semantic stage's choices: its model and its cosine threshold.

Both were chosen on the labeled pair dataset before development recall was measured, so the
36 development enactments tuned neither (`uv run lineage eval pair-data` ->
reports/pair_data.json). bge-small-en-v1.5 scored better than all-MiniLM-L6-v2 at every
precision measured and in ROC AUC. Its threshold is the lowest cosine whose precision on the
training and validation pairs reaches TARGET_PRECISION. See DESIGN.md#pair-dataset.
"""

from lineage.pair_eval import TARGET_PRECISION

MODEL = "bge-small-en-v1.5"
# reports/pair_data.json: semantic.models[MODEL].at_precision[TARGET_PRECISION].threshold
THRESHOLD = 0.8905

__all__ = ["MODEL", "TARGET_PRECISION", "THRESHOLD"]
