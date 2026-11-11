import shutil
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from lineage import queries
from lineage.cli import app
from lineage.neighbors import build_neighbors
from lineage.vectors import build_plan, semantic_dir, shard_name, vectors_dir
from tests.fakes import bag_of_words

FIXTURES = Path(__file__).parent / "fixtures" / "openstates"
# Real session dates for the fixture archives, copied from snapshot/sessions.csv.
SESSIONS = Path(__file__).parent / "fixtures" / "sessions.csv"


def _readme_field(readme: str, key: str) -> str:
    for line in readme.splitlines():
        if line.startswith(f"{key}: "):
            return line.removeprefix(f"{key}: ")
    raise KeyError(key)


def build_raw_dir(root: Path, fixtures: Path = FIXTURES) -> Path:
    """Zip each fixture the way Open States does: README plus STATE/session/*_bills.json, stored."""
    raw = root / "raw"
    for fixture in sorted(p for p in fixtures.iterdir() if p.is_dir()):
        readme = (fixture / "README").read_text()
        state = _readme_field(readme, "State")
        session = _readme_field(readme, "Session")
        bills = next(fixture.glob("*_bills.json"))
        archive = raw / state / f"{state}_{session}_json_fixture.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("README", readme)
            z.write(bills, f"{state}/{session}/{bills.name}")
    return raw


@pytest.fixture(autouse=True)
def _own_pace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each test paces its requests apart from other tests and from any real run.
    monkeypatch.setenv("LINEAGE_PACE_DIR", str(tmp_path / "pace"))


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return build_raw_dir(tmp_path)


@dataclass
class Upstream:
    """A local stand-in for the Open States file server, serving archives by file name."""

    root: Path
    url: str
    user_agents: list[str]

    def file(self, name: str) -> Path:
        return self.root / name


@pytest.fixture
def upstream(raw_dir: Path, tmp_path: Path) -> Iterator[Upstream]:
    root = tmp_path / "upstream"
    root.mkdir()
    for archive in raw_dir.glob("*/*.zip"):
        shutil.copy(archive, root / archive.name)
    user_agents: list[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            user_agents.append(self.headers.get("User-Agent", ""))
            super().do_GET()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    yield Upstream(root, f"http://127.0.0.1:{server.server_address[1]}/", user_agents)
    server.shutdown()
    server.server_close()


def _invoke(*args: str) -> None:
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output


def build_tables(root: Path, fixtures: Path, sessions: Path) -> Path:
    """Fixture archives taken through every pipeline command; returns the tables dir."""
    raw, manifest, tables = build_raw_dir(root, fixtures), root / "manifest.json", root / "tables"
    reports = root / "reports"
    no_floor = ("--min-free-gib", "0")
    _invoke("snapshot", "--raw", str(raw), "--manifest", str(manifest))
    _invoke("ingest", "--raw", str(raw), "--manifest", str(manifest), "--out", str(tables),
         "--sessions", str(sessions), *no_floor)  # fmt: skip
    _invoke("text", "merge", "--tables", str(tables), "--plan", str(root / "no-plan.parquet"),
         *no_floor)  # fmt: skip
    _invoke("split", "--tables", str(tables), "--workers", "1", *no_floor)
    _invoke("eligible", "--tables", str(tables), "--report", str(reports / "types.json"))
    # A few dozen bills are too few for a document-frequency stoplist: at the corpus's 0.1%
    # share, every shingle would be stoplisted. A share of 1 stoplists nothing.
    _invoke("boilerplate", "--tables", str(tables), "--manifest", str(manifest), "--max-share", "1",
         "--report", str(reports / "boilerplate.json"), "--workers", "1")  # fmt: skip
    _invoke("index", "--tables", str(tables), "--workers", "1", "--report",
         str(reports / "index.json"), *no_floor)  # fmt: skip
    return tables


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture archives taken through every pipeline command; returns the tables dir."""
    return build_tables(tmp_path_factory.mktemp("corpus"), FIXTURES, SESSIONS)


COMMISSION = Path(__file__).parent / "fixtures" / "commission"


@pytest.fixture(scope="session")
def commission_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The commission enactments among distractors (built by `evals/fixture.py`), indexed."""
    root = tmp_path_factory.mktemp("commission")
    return build_tables(root, COMMISSION, COMMISSION / "sessions.csv")


FAKE_MODEL = "fake"


@pytest.fixture
def embedded_commission(commission_corpus: Path, tmp_path: Path) -> Path:
    """The commission fixture's tables, its windows embedded by the stand-in, neighbors built."""
    out = tmp_path / "tables"
    out.mkdir()
    for table in commission_corpus.iterdir():
        if table.name != "semantic":  # the stand-in's windows and vectors are built here
            (out / table.name).symlink_to(table)
    plan = build_plan(out, floor_bytes=0)
    vectors = vectors_dir(out, FAKE_MODEL)
    vectors.mkdir(parents=True)
    with queries.connect_quiet() as con:
        for s in range(plan.shards):
            texts = queries.read_semantic_texts(
                con, parquet_path=semantic_dir(out) / "texts" / f"{shard_name(s)}.parquet"
            )
            np.save(vectors / f"{shard_name(s)}.npy", bag_of_words(texts).astype(np.float16))
    build_neighbors(out, model=FAKE_MODEL, spec="Flat", ef_search=None, floor_bytes=0)
    return out
