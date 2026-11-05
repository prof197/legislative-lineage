import zipfile
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
