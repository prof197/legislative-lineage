"""Origin matches: a bill's text found in a published model document, and the table they live in.

An `OriginMatch` gives both shares, of the model's wording found in the bill and of the bill's
wording that comes from the model, since a long model a bill borrows one section from and a short
model a bill copies whole read differently. Spans are kept on the bill's side only: the model's
text is never shown, so nothing on the site could point into it. `lineage origins` (lane C,
lineage.origin_matching) writes the table; the export and the snapshot read it. See
DESIGN.md#origin-matches.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lineage.modeldocs import ModelDoc

Stage = Literal["near_duplicate", "semantic"]
ORIGINS_FILE = Path("origins") / "matches.jsonl"
REPORT = Path("reports/origins.json")


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


class SourceOrigins(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    documents: int
    documents_matched: int
    matches: int


class OriginsReport(BaseModel):
    """What a matching run found, counted from the table, for the docs to cite: the table itself
    names bills and is local."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    documents: int
    distinct_texts: int  # a model several sources hold is matched once (DESIGN.md#origin-matches)
    matches: int
    bills: int
    by_stage: dict[str, int]  # near_duplicate only, semantic only, both
    by_source: dict[str, SourceOrigins]
    seconds: float | None  # the run's wall time; None when counted from a table already written


def origins_report(
    matches: Sequence[OriginMatch], docs: Sequence[ModelDoc], *, seconds: float | None
) -> OriginsReport:
    source_of = {d.id: d.source for d in docs}
    unknown = sorted({m.doc_id for m in matches} - source_of.keys())
    if unknown:
        raise ValueError(f"matches name documents no source holds: {', '.join(unknown[:5])}")
    stage = Counter("both" if len(set(m.stages)) == 2 else m.stages[0] for m in matches)
    held = Counter(d.source for d in docs)
    found = Counter(source_of[m.doc_id] for m in matches)
    matched = Counter(source_of[i] for i in {m.doc_id for m in matches})
    return OriginsReport(
        documents=len(docs),
        distinct_texts=len({d.text_sha256 for d in docs}),
        matches=len(matches),
        bills=len({m.bill_id for m in matches}),
        by_stage={k: stage[k] for k in ("near_duplicate", "semantic", "both")},
        by_source={
            s: SourceOrigins(documents=held[s], documents_matched=matched[s], matches=found[s])
            for s in sorted(held)
        },
        seconds=None if seconds is None else round(seconds, 1),
    )


def write_origins_report(report: OriginsReport, path: Path = REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=1) + "\n")
    return path
