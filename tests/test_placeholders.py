"""No draft number ships: docs written ahead of a measurement mark each number still to come
with one token, and this fails while any remain, locally and in CI."""

from pathlib import Path

import pytest

PLACEHOLDER = "{{TBD}}"
ROOT = Path(__file__).parent.parent
# Every committed document a reader sees; reports are written by commands, never by hand.
DOCS = sorted(p.relative_to(ROOT) for p in [*ROOT.glob("*.md"), *(ROOT / "evals").glob("*.md")])


@pytest.mark.parametrize("doc", DOCS, ids=str)
def test_no_placeholder_is_left_in(doc: Path) -> None:
    lines = [
        n for n, line in enumerate((ROOT / doc).read_text().splitlines(), 1) if PLACEHOLDER in line
    ]

    assert not lines, f"{doc} still has {PLACEHOLDER} on lines {lines}"
