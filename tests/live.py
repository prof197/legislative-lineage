"""The live track at fixture scale for lane L3's tests: the recorded days (tests/days.py)
applied by `lineage update` to the build after day 1, and full exports of each day's tables,
built once per session and shared by the publish, feed and schedule tests.

The hosted state is Georgia, whose synthetic record arrives on day 1, gains Utah SB 18 on day
2 and is replaced on day 3 (tests/days.py).
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest

from lineage.export import RESULTS, build_export
from lineage.updates import load_days, update
from tests.conftest import FAKE_MODEL
from tests.days import DAYS, rebuild, write_days

REPO = Path(__file__).parent.parent
STATE = "GA"
PACKAGE = "tests.collector_fixtures"


@dataclass(frozen=True, slots=True)
class Recorded:
    root: Path
    days: Path
    tables: dict[int, Path]  # the tables after day n: day 1 built, days 2 and 3 updated
    log: Path  # every day's change log, days 2 and 3

    def log_through(self, n: int, where: Path) -> Path:
        """A change log holding days 2 to `n` only."""
        where.mkdir(parents=True, exist_ok=True)
        for day in DAYS[1:n]:
            shutil.copy(self.log / f"{day}.jsonl", where / f"{day}.jsonl")
        return where


_built: dict[str, Recorded] = {}


def recorded(factory: pytest.TempPathFactory) -> Recorded:
    """The recorded days, the tables after each, and their change log; built once a session."""
    if "recorded" in _built:
        return _built["recorded"]
    root = factory.mktemp("live")
    days = write_days(root / "days")
    tables = {1: rebuild(days, 1, root / "built1")}
    log = root / "changes"
    for n, day in ((2, load_days(days)[1]), (3, load_days(days)[2])):
        tables[n] = root / f"updated{n}"
        shutil.copytree(tables[n - 1], tables[n], symlinks=False)
        update(tables[n], day, model=FAKE_MODEL, changes_dir=log, floor_bytes=0)
    _built["recorded"] = Recorded(root=root, days=days, tables=tables, log=log)
    return _built["recorded"]


class ExportInputs(TypedDict):
    cache: Path
    featured_pairs: Path
    featured_report: Path
    stories_rules: Path
    results: Path
    modeldocs_root: Path
    collector_package: str
    model: str
    require_flag: bool


def export_inputs(root: Path) -> ExportInputs:
    """The committed inputs a fixture export reads: no stories, the repository's results."""
    root.mkdir(parents=True, exist_ok=True)
    no_pairs, no_stories, rules = root / "none.jsonl", root / "none.json", root / "stories.toml"
    no_pairs.touch()
    no_stories.write_text('{"stories": {}}')
    rules.write_text("")
    return {
        "cache": root / "no-cache",
        "featured_pairs": no_pairs,
        "featured_report": no_stories,
        "stories_rules": rules,
        "results": REPO / RESULTS,
        "modeldocs_root": root / "modeldocs",
        "collector_package": PACKAGE,
        "model": FAKE_MODEL,
        "require_flag": False,
    }


def full_export(tables: Path, out: Path, inputs: ExportInputs, state: str = STATE) -> Path:
    build_export(tables, state=state, out=out, models_dir=None, floor=0, **inputs)
    return out
