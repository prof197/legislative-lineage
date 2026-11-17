"""Every point a committed report states lies within the interval it is given with (ruled at
the step 47 review): a point outside its own interval misleads whatever the method."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).parents[1]
REPORTS = sorted((ROOT / "reports").rglob("*.json"))
POINTS = ("precision", "recall", "share", "rate", "agreement", "kappa")
# Rounded to four places in the reports.
TOLERANCE = 1e-4


def _points(node: Any, where: str) -> Iterator[tuple[str, float, list[float]]]:
    if isinstance(node, dict):
        node = cast(dict[str, Any], node)
        interval = node.get("interval_95")
        if isinstance(interval, list) and len(cast(list[Any], interval)) == 2:
            interval = cast(list[float], interval)
            for key in POINTS:
                value = node.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    yield f"{where}.{key}", float(value), [float(v) for v in interval]
            found, total = node.get("found"), node.get("enactments")
            if isinstance(found, int) and isinstance(total, int) and total:
                yield f"{where}.found", found / total, [float(v) for v in interval]
        for key, child in node.items():
            yield from _points(child, f"{where}.{key}")
    elif isinstance(node, list):
        for i, child in enumerate(cast(list[Any], node)):
            yield from _points(child, f"{where}[{i}]")


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_reported_point_lies_within_its_interval(path: Path) -> None:
    outside = [
        (where, point, interval)
        for where, point, interval in _points(json.loads(path.read_text()), path.stem)
        if not interval[0] - TOLERANCE <= point <= interval[1] + TOLERANCE
    ]
    assert outside == []


def test_the_walk_finds_points() -> None:
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    assert sum(1 for _ in _points(results, "results")) > 100
