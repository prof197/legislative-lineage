"""Origin matches: a bill's text found in a published model document, and the table they live in.

An `OriginMatch` gives both shares, of the model's wording found in the bill and of the bill's
wording that comes from the model, since a long model a bill borrows one section from and a short
model a bill copies whole read differently. Spans are kept on the bill's side only: the model's
text is never shown, so nothing on the site could point into it. `lineage origins` (lane C,
lineage.origin_matching) writes the table; the export and the snapshot read it. See
DESIGN.md#origin-matches.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Stage = Literal["near_duplicate", "semantic"]
ORIGINS_FILE = Path("origins") / "matches.jsonl"


class BillSpan(BaseModel):
    """Characters of one of the bill's sections that match the model: copied or paraphrased."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ordinal: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    kind: Literal["exact", "paraphrase"]


class OriginMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    bill_id: str
    doc_id: str  # lineage.modeldocs.ModelDoc.id
    stages: list[Stage] = Field(min_length=1)
    model_share: float = Field(ge=0, le=1)  # of the model's words, the share found in the bill
    bill_share: float = Field(ge=0, le=1)  # of the bill's words, the share from the model
    spans: list[BillSpan]


def origins_path(tables_dir: Path) -> Path:
    return tables_dir / ORIGINS_FILE


def write_origin_matches(tables_dir: Path, matches: list[OriginMatch]) -> Path:
    """Replace the table with `matches`, sorted by bill then document."""
    path = origins_path(tables_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(matches, key=lambda m: (m.bill_id, m.doc_id))
    partial = path.with_name(path.name + ".partial")
    partial.write_text("".join(m.model_dump_json() + "\n" for m in rows))
    partial.replace(path)
    return path


def read_origin_matches(tables_dir: Path) -> list[OriginMatch]:
    """The table, or nothing before `lineage origins` has run."""
    path = origins_path(tables_dir)
    if not path.exists():
        return []
    return [OriginMatch.model_validate_json(line) for line in path.read_text().splitlines() if line]
