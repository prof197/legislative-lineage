import pytest

from lineage import memory
from lineage.memory import MemoryBoundError, require_within


def test_a_peak_that_has_passed_still_crosses_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    # A spike inside one library call is over by the time the caller checks: the footprint is
    # back under the bound, and only the lifetime maximum shows it.
    monkeypatch.setattr(memory, "current_bytes", lambda: 10)
    monkeypatch.setattr(memory, "peak_bytes", lambda: 30)

    require_within(30, "building")
    with pytest.raises(MemoryBoundError, match="peaked at"):
        require_within(20, "building")


def test_the_footprint_now_crosses_the_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "current_bytes", lambda: 30)
    monkeypatch.setattr(memory, "peak_bytes", lambda: 30)

    with pytest.raises(MemoryBoundError, match="holds"):
        require_within(20, "building")
