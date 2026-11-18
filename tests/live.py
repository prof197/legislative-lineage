"""The live track at fixture scale for lane L3's tests: the recorded days (tests/days.py)
applied by `lineage update` to the build after day 1, and full exports of each day's tables,
built once per session and shared by the publish, feed and schedule tests.

The hosted state is Georgia, whose synthetic record arrives on day 1, gains Utah SB 18 on day
2 and is replaced on day 3 (tests/days.py).
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import pytest
from typer.testing import CliRunner

from lineage.cli import app
from lineage.export import RESULTS, build_export
from lineage.incremental import update
from lineage.match.semantic.pairs import build_semantic_pairs
from lineage.match.state_boilerplate import build_state_boilerplate
from lineage.updates import fold, load_days, write_archives
from tests.conftest import FAKE_MODEL, embed_with_stand_in
from tests.days import DAYS, rebuild, sessions_csv, write_days

REPO = Path(__file__).parent.parent
STATE = "GA"
PACKAGE = "tests.collector_fixtures"


@dataclass(frozen=True, slots=True)
class Recorded:
    root: Path
    days: Path
    tables: dict[int, Path]  # the tables after day n: day 1 built, days 2 and 3 updated
    log: Path  # every day's change log, days 2 and 3
    rebuilt: Path  # a full rebuild over the three days, with the base build's statistics
    refreshed: Path  # the same, with the stoplist and the state boilerplate flag counted afresh

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
    _built["recorded"] = Recorded(
        root=root,
        days=days,
        tables=tables,
        log=log,
        rebuilt=rebuild(days, 3, root / "rebuilt"),
        refreshed=refreshed_rebuild(days, 3, root / "refreshed"),
    )
    return _built["recorded"]


# The base build's stoplist is empty: a few dozen bills are too few for the corpus's share
# (tests/conftest.py). Refreshed, the stoplist takes shingles found in more than this share of
# sections, which at fixture scale are the acts' shared wording, so relatives move.
REFRESHED_SHARE = "0.02"


def _invoke(*args: str) -> None:
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output


def refreshed_rebuild(days: Path, upto: int, root: Path) -> Path:
    """A full rebuild over the first `upto` days with its corpus statistics counted afresh: a
    stoplist at REFRESHED_SHARE and the state boilerplate counts; returns its tables dir."""
    root.mkdir(parents=True)
    raw = write_archives(fold(load_days(days)[:upto]), root / "raw")
    manifest, built, reports = root / "manifest.json", root / "tables", root / "reports"
    no_floor = ("--min-free-gib", "0")
    _invoke("snapshot", "--raw", str(raw), "--manifest", str(manifest))
    _invoke("ingest", "--raw", str(raw), "--manifest", str(manifest), "--out", str(built),
         "--sessions", str(sessions_csv(root)), *no_floor)  # fmt: skip
    _invoke("text", "merge", "--tables", str(built), "--plan", str(root / "no-plan.parquet"),
         *no_floor)  # fmt: skip
    _invoke("split", "--tables", str(built), "--workers", "1", *no_floor)
    _invoke("eligible", "--tables", str(built), "--report", str(reports / "types.json"))
    _invoke("boilerplate", "--tables", str(built), "--manifest", str(manifest), "--max-share",
         REFRESHED_SHARE, "--report", str(reports / "boilerplate.json"),
         "--workers", "1")  # fmt: skip
    _invoke("index", "--tables", str(built), "--workers", "1", "--report",
         str(reports / "index.json"), *no_floor)  # fmt: skip
    tables = embed_with_stand_in(built, root / "embedded")
    build_semantic_pairs(tables, model=FAKE_MODEL, floor_bytes=0)
    build_state_boilerplate(tables, model=FAKE_MODEL, floor_bytes=0)
    return tables


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


# Comparing exports.


@dataclass(frozen=True, slots=True)
class Seen:
    data: bytes
    inode: int
    mtime_ns: int


def files_seen(root: Path) -> dict[str, Seen]:
    out: dict[str, Seen] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[p.relative_to(root).as_posix()] = Seen(p.read_bytes(), st.st_ino, st.st_mtime_ns)
    return out


def unstamped(name: str, data: bytes) -> Any:
    """A file as a full export writes it: no live-track dates."""
    doc = json.loads(data)
    if name.startswith("bills/") or name == "meta.json":
        doc.pop("updated_on", None)
        doc.pop("as_of", None)
    return doc


def normalized(root: Path) -> dict[str, Any]:
    """Every JSON file of an export as a full export writes it; the feed is not one."""
    return {n: unstamped(n, s.data) for n, s in files_seen(root).items() if n.endswith(".json")}
