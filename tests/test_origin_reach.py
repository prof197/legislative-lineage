"""lineage.origins.reach: the reach scale is chosen by its rule, fixed before the reading: keep
every development enactment no weighing finds, then the highest precision, then the larger
scale of two within a label of each other."""

from typing import Any

from lineage.origins.reach import choose


def _reading(scale: int | None, correct: int, labeled: int, found: int) -> dict[str, Any]:
    return {
        "scale": scale,
        "labeled": {"labeled": labeled, "correct": correct, "precision": correct / labeled},
        "development_recall": {"found": found, "enactments": 36},
    }


def test_a_scale_that_loses_a_development_enactment_is_not_chosen() -> None:
    readings = [_reading(20, 99, 100, 30), _reading(100, 90, 100, 31), _reading(None, 80, 100, 31)]
    assert choose(readings) == 100


def test_of_two_within_a_label_the_larger_scale_is_chosen() -> None:
    readings = [_reading(50, 95, 100, 31), _reading(200, 94, 100, 31), _reading(None, 80, 100, 31)]
    assert choose(readings) == 200


def test_no_weighing_stands_when_no_scale_beats_it_by_more_than_a_label() -> None:
    readings = [_reading(50, 81, 100, 31), _reading(None, 80, 100, 31)]
    assert choose(readings) is None
