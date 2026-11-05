from typer.testing import CliRunner

from lineage import __version__
from lineage.cli import app

runner = CliRunner()


def test_version_prints_installed_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"lineage {__version__}\n"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "Usage" in result.output
