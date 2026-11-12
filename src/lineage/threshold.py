"""The semantic threshold's options: what each would show and what it would find.

The threshold of 0.8905 was set on the labeled pair dataset (lineage.semantic) and changes
only on precision-sample evidence (DESIGN.md#precision-sample). The options are the build's
threshold and stricter ones in steps of 0.01. A lower one would list relatives the samples
never drew, whose precision nothing measures, so it is not offered until both samples are
topped up from the new band.
"""

from lineage.semantic import THRESHOLD

# Up to 0.97, the first step at which development recall falls: the two enactments found by
# meaning alone reach 0.9677 and 0.9632 (reports/results.json, each bill's best_cosine).
CANDIDATES = (THRESHOLD, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97)
