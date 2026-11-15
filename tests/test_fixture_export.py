"""The committed fixture export (tests/fixtures/export/) is what the export writes today, valid
against the schema the site's types come from, with an origin from a leaked source among them."""

import os
from pathlib import Path

import pytest

from lineage.export_schema import BillPage, Meta, Sources
from tests.fixture_export import COMMITTED, build_fixture_export, files_of, replace_committed


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = build_fixture_export(tmp_path_factory.mktemp("fixture-export"))
    if os.environ.get("LINEAGE_REGENERATE") == "1":
        replace_committed(out)
    return out


def test_the_committed_fixture_export_is_what_the_export_writes(built: Path) -> None:
    # Regenerate with `LINEAGE_REGENERATE=1 uv run pytest tests/test_fixture_export.py`.
    assert files_of(COMMITTED) == files_of(built)


def test_the_fixture_shows_origins_and_their_sources() -> None:
    meta = Meta.model_validate_json((COMMITTED / "meta.json").read_text())
    sources = Sources.model_validate_json((COMMITTED / "sources.json").read_text())
    pages = [BillPage.model_validate_json(p.read_text()) for p in (COMMITTED / "bills").glob("*")]

    assert meta.state == "UT"
    assert {s.key for s in sources.sources} == {"example-archive", "example-library"}
    origins = [o for p in pages for o in p.origins]
    assert {o.doc.source for o in origins} == {"example-archive", "example-library"}
    assert any(o.doc.leaked for o in origins)
    assert all(0 < o.bill_share <= 1 and o.sections for o in origins)
