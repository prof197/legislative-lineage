"""The faithfulness eval's hard check, run in CI: every committed explanation's claims cite
aligned spans that exist and differ between the two texts. It needs no corpus, since each cache
file keeps the spans' grounding form (DESIGN.md#explanations)."""

from pathlib import Path

from lineage.explain import CACHE_DIR, check_cache

ROOT = Path(__file__).resolve().parents[1]


def test_every_committed_explanation_cites_spans_that_exist_and_differ() -> None:
    assert check_cache(ROOT / CACHE_DIR) == []
