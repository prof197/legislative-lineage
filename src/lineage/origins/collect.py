"""The collector registry: one module per source under `lineage.origins.collectors`, found by name.

A collector module declares `SOURCE`, its `lineage.origins.modeldocs.Source`, and
`collect(context)`, which stores what it finds through `lineage.origins.modeldocs.store` and returns
a `CollectReport`. The registry imports every module in the package, so a new source is a new file
and nothing shared is edited. Every request goes through `lineage.core.http.open_url` and its
pacing, and a URL that robots.txt disallows is reported, never worked around. See
DESIGN.md#model-documents.

See DESIGN.md#dr-10.
"""

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lineage.origins.modeldocs import MODELDOCS_DIR, Source

PACKAGE = "lineage.origins.collectors"
REPORTS_DIR = Path("reports/modeldocs")


class CollectError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class CollectContext:
    root: Path = MODELDOCS_DIR  # lineage.origins.modeldocs stores under root/<source>/
    on: date = field(default_factory=date.today)
    floor_bytes: int = 0


class CollectFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    url: str
    reason: str


class CollectReport(BaseModel):
    """What one run found in a library and what became of each document: every document found
    is collected, unchanged, blocked by robots.txt or failed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    run_on: date
    found: int = Field(ge=0)
    collected: int = Field(ge=0)  # stored new, or with a changed text
    unchanged: int = Field(ge=0)
    blocked: list[str]  # URLs robots.txt disallows for this project
    failed: list[CollectFailure]
    # Documents stored before that this run read and found out of scope: no longer current,
    # and not among those found.
    withdrawn: list[str] = Field(default_factory=list[str])

    @model_validator(mode="after")
    def _every_document_is_accounted_for(self) -> Self:
        accounted = self.collected + self.unchanged + len(self.blocked) + len(self.failed)
        if accounted != self.found:
            raise ValueError(f"{self.found} found but {accounted} accounted for")
        return self


@dataclass(frozen=True, slots=True)
class Collector:
    source: Source
    collect: Callable[[CollectContext], CollectReport]


def _collector(module: ModuleType) -> Collector:
    name = module.__name__.rpartition(".")[2]
    source = getattr(module, "SOURCE", None)
    collect = getattr(module, "collect", None)
    if not isinstance(source, Source) or not callable(collect):
        raise CollectError(f"{module.__name__} must declare SOURCE (a Source) and collect()")
    if source.key != name.replace("_", "-"):
        raise CollectError(f"{module.__name__} declares source {source.key}; name them alike")
    return Collector(source, cast(Callable[[CollectContext], CollectReport], collect))


def collectors(package: str = PACKAGE) -> dict[str, Collector]:
    """Every collector in `package`, by source key."""
    root = importlib.import_module(package)
    out: dict[str, Collector] = {}
    for info in sorted(pkgutil.iter_modules(root.__path__), key=lambda i: i.name):
        if not info.name.startswith("_"):
            c = _collector(importlib.import_module(f"{package}.{info.name}"))
            out[c.source.key] = c
    return out


def run_collector(key: str, context: CollectContext, package: str = PACKAGE) -> CollectReport:
    """Collect one source; its report must be about that source."""
    known = collectors(package)
    if key not in known:
        raise CollectError(f"no collector for {key}; known: {', '.join(known) or 'none yet'}")
    report = known[key].collect(context)
    if report.source != key:
        raise CollectError(f"the {key} collector reported on {report.source}")
    return report


def write_report(report: CollectReport, directory: Path = REPORTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.source}.json"
    path.write_text(report.model_dump_json(indent=1) + "\n")
    return path
