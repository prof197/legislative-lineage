"""The collector registry (DESIGN.md#model-documents): a source is a module, found by name, and
its report accounts for every document it found."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from lineage.cli import app
from lineage.collect import (
    CollectContext,
    CollectError,
    CollectReport,
    collectors,
    run_collector,
    write_report,
)

FIXTURES = "tests.collector_fixtures"


def test_a_source_is_a_module_found_by_its_name() -> None:
    found = collectors(FIXTURES)

    assert list(found) == ["example-archive", "example-library"]
    assert found["example-archive"].source.leaked
    assert found["example-library"].source.organization == "Example Council"


def test_the_shipped_package_holds_only_collectors_the_registry_accepts() -> None:
    # Each lane adds its module here; any that declares itself wrongly fails this test.
    assert all(key == c.source.key for key, c in collectors().items())


def test_a_run_reports_on_its_own_source(tmp_path: Path) -> None:
    report = run_collector("example-library", CollectContext(on=date(2026, 11, 15)), FIXTURES)

    assert report.blocked == ["https://example.org/models/private.pdf"]
    path = write_report(report, tmp_path)
    assert CollectReport.model_validate_json(path.read_text()) == report
    with pytest.raises(CollectError, match="no collector for nowhere"):
        run_collector("nowhere", CollectContext(), FIXTURES)


def test_every_document_found_is_accounted_for() -> None:
    with pytest.raises(ValidationError, match="3 found but 2 accounted for"):
        CollectReport(
            source="s",
            run_on=date(2026, 11, 15),
            found=3,
            collected=1,
            unchanged=1,
            blocked=[],
            failed=[],
        )


def test_the_commands_say_what_is_missing(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = runner.invoke(
        app, ["collect-models", "nowhere", "--min-free-gib", "0", "--reports", str(tmp_path)]
    )
    assert missing.exit_code == 1
    assert "no collector for nowhere" in missing.output
    origins = runner.invoke(app, ["origins", "--tables", str(tmp_path), "--min-free-gib", "0"])
    assert origins.exit_code == 1
    assert "not built yet" in origins.output
