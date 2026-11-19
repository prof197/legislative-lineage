"""lineage.origins.reach: the reach scale is the smallest step down from no weighing that still
takes more wrong labeled matches off pages than correct ones, keeping every development
enactment no weighing finds."""

from typing import Any

from lineage.origins.reach import choose


def _reading(scale: int | None, correct: int, labeled: int, found: int = 31) -> dict[str, Any]:
    return {
        "scale": scale,
        "labeled": {"labeled": labeled, "correct": correct, "precision": correct / labeled},
        "development_recall": {"found": found, "enactments": 36},
    }


def test_the_walk_stops_where_a_step_takes_more_correct_matches_off_than_wrong() -> None:
    readings = [
        _reading(None, 764, 823),
        _reading(274, 764, 823),  # nothing off: passed through
        _reading(100, 762, 803),  # 2 correct, 18 wrong off: taken
        _reading(50, 757, 795),  # 5 correct, 3 wrong more: stops
        _reading(20, 693, 723),
    ]
    assert choose(readings) == 100


def test_a_step_that_loses_a_development_enactment_stops_the_walk() -> None:
    readings = [_reading(None, 80, 100), _reading(100, 80, 90, found=30), _reading(50, 80, 85)]
    assert choose(readings) is None


def test_no_weighing_stands_when_no_step_takes_a_labeled_match_off() -> None:
    readings = [_reading(None, 80, 100), _reading(200, 80, 100), _reading(50, 80, 100)]
    assert choose(readings) is None
