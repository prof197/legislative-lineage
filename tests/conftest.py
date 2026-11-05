import shutil
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "openstates"


def _readme_field(readme: str, key: str) -> str:
    for line in readme.splitlines():
        if line.startswith(f"{key}: "):
            return line.removeprefix(f"{key}: ")
    raise KeyError(key)


def build_raw_dir(root: Path) -> Path:
    """Zip each fixture the way Open States does: README plus STATE/session/*_bills.json, stored."""
    raw = root / "raw"
    for fixture in sorted(FIXTURES.iterdir()):
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
