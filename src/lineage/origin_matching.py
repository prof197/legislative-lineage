"""Lane C's matching: model documents as a second collection, queried against the corpus.

Owned by lane C (M10). The contract it meets is `match_models` below and the OriginMatch record
(lineage.origins). See DESIGN.md#origin-matches.
"""

from collections.abc import Sequence
from pathlib import Path

from lineage.modeldocs import ModelDoc
from lineage.origins import OriginMatch


def match_models(
    tables_dir: Path,
    docs: Sequence[ModelDoc],
    *,
    modeldocs_root: Path,
    model: str,
    floor_bytes: int,
) -> list[OriginMatch]:
    """Every bill each document's text matches, through the existing sections, windows and both
    stages, without changing the bill index; deterministic."""
    raise NotImplementedError("lineage origins is lane C's (M10)")
