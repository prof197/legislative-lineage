from datetime import date
from pathlib import Path

import pytest

from lineage.snapshot import load_manifest
from lineage.status import EXPORT_TIMEZONES, Action, Outcome, Status, bill_status, local_date

SESSION_END = date(2026, 4, 2)
AFTER_SESSION = date(2026, 7, 25)
DURING_SESSION = date(2026, 3, 1)


def _actions(*steps: tuple[str, str]) -> list[Action]:
    """Actions in the order given, each a (date, classification) pair; '' means unclassified."""
    return [
        Action(date=d, classification=(c,) if c else (), order=i) for i, (d, c) in enumerate(steps)
    ]


def _status(actions: list[Action], *, as_of: date, end: date | None = SESSION_END) -> Outcome:
    return bill_status(actions, state="GA", session_end=end, as_of=as_of)


INTRODUCED = ("2025-01-15", "introduction")
PASSED_HOUSE = ("2025-02-20", "passage")
PASSED_SENATE = ("2025-03-25", "passage")


def test_a_signed_bill_passed() -> None:
    actions = _actions(
        INTRODUCED, PASSED_HOUSE, PASSED_SENATE, ("2025-05-09", "executive-signature")
    )

    assert _status(actions, as_of=AFTER_SESSION) == Outcome(
        Status.PASSED, "executive-signature", date(2025, 5, 9)
    )


def test_a_bill_that_became_law_without_a_signature_passed() -> None:
    actions = _actions(INTRODUCED, PASSED_HOUSE, PASSED_SENATE, ("2025-05-20", "became-law"))

    assert _status(actions, as_of=AFTER_SESSION) == Outcome(
        Status.PASSED, "became-law", date(2025, 5, 20)
    )


def test_a_vetoed_bill_died_even_while_the_session_runs() -> None:
    actions = _actions(INTRODUCED, PASSED_HOUSE, PASSED_SENATE, ("2026-02-10", "executive-veto"))

    assert _status(actions, as_of=DURING_SESSION) == Outcome(
        Status.DIED, "executive-veto", date(2026, 2, 10)
    )


def test_a_veto_override_passes_the_bill() -> None:
    actions = _actions(
        INTRODUCED,
        PASSED_HOUSE,
        PASSED_SENATE,
        ("2025-04-01", "executive-veto"),
        ("2025-04-01", "veto-override-passage"),
    )

    assert _status(actions, as_of=AFTER_SESSION) == Outcome(
        Status.PASSED, "veto-override-passage", date(2025, 4, 1)
    )


def test_a_failed_override_leaves_the_bill_dead() -> None:
    actions = _actions(
        INTRODUCED, ("2025-04-01", "executive-veto"), ("2025-04-03", "veto-override-failure")
    )

    assert _status(actions, as_of=DURING_SESSION) == Outcome(
        Status.DIED, "veto-override-failure", date(2025, 4, 3)
    )


def test_a_signature_after_a_conditional_veto_passes_the_bill() -> None:
    # New Jersey's conditional veto: the legislature adopts the governor's changes, who then signs.
    actions = _actions(
        INTRODUCED,
        ("2025-06-30", "executive-veto"),
        ("2025-09-22", "passage"),
        ("2025-10-01", "executive-signature"),
    )

    assert _status(actions, as_of=AFTER_SESSION).status is Status.PASSED


def test_a_failed_vote_kills_the_bill() -> None:
    actions = _actions(INTRODUCED, ("2026-02-12", "failure"))

    assert _status(actions, as_of=DURING_SESSION) == Outcome(
        Status.DIED, "failure", date(2026, 2, 12)
    )


def test_a_bill_passed_after_a_failed_vote_is_alive_again() -> None:
    actions = _actions(INTRODUCED, ("2026-02-12", "failure"), ("2026-02-13", "passage"))

    assert _status(actions, as_of=DURING_SESSION) == Outcome(Status.INTRODUCED, None, None)


def test_a_bill_without_an_outcome_died_when_the_session_ended() -> None:
    actions = _actions(INTRODUCED, PASSED_HOUSE)

    assert _status(actions, as_of=AFTER_SESSION) == Outcome(
        Status.DIED, "session-ended", SESSION_END
    )


def test_a_bill_without_an_outcome_is_introduced_while_the_session_runs() -> None:
    actions = _actions(INTRODUCED, PASSED_HOUSE)

    assert _status(actions, as_of=DURING_SESSION) == Outcome(Status.INTRODUCED, None, None)


def test_the_last_day_of_the_session_has_not_ended_it() -> None:
    assert _status(_actions(INTRODUCED), as_of=SESSION_END).status is Status.INTRODUCED


def test_an_unknown_session_end_leaves_the_bill_introduced() -> None:
    assert _status(_actions(INTRODUCED), as_of=AFTER_SESSION, end=None) == Outcome(
        Status.INTRODUCED, None, None
    )


def test_a_bill_with_no_actions_follows_the_session() -> None:
    assert _status([], as_of=AFTER_SESSION).status is Status.DIED
    assert _status([], as_of=DURING_SESSION).status is Status.INTRODUCED


def test_sequence_comes_from_the_action_order_not_list_position() -> None:
    veto = Action(date="2025-04-01", classification=("executive-veto",), order=3)
    override = Action(date="2025-04-01", classification=("veto-override-passage",), order=4)

    assert _status([override, veto], as_of=AFTER_SESSION).status is Status.PASSED


def test_an_action_can_carry_several_classifications() -> None:
    actions = [
        Action(date="2025-05-01", classification=("passage", "executive-signature"), order=0)
    ]

    assert _status(actions, as_of=AFTER_SESSION).reason == "executive-signature"


@pytest.mark.parametrize(
    ("value", "state", "expected"),
    [
        ("2025-03-25", "GA", date(2025, 3, 25)),
        # Most zoned exports write a plain date as midnight in the state's time zone.
        ("2025-03-04T06:00:00+00:00", "MO", date(2025, 3, 4)),
        ("2025-03-04T08:00:00+00:00", "NV", date(2025, 3, 4)),
        # Alabama's export writes midnight Eastern, though the state keeps Central time.
        ("2023-05-16T04:00:00+00:00", "AL", date(2023, 5, 16)),
        ("2024-01-09T05:00:00+00:00", "AL", date(2024, 1, 9)),
        # An evening action in Montana falls on the next day in UTC.
        ("2025-04-18T01:33:56+00:00", "MT", date(2025, 4, 17)),
        ("2025-04-18T01:33:56+00:00", "NY", date(2025, 4, 17)),
        ("2025-04-18T16:00:00+00:00", "HI", date(2025, 4, 18)),
    ],
)
def test_zoned_timestamps_become_the_states_local_date(
    value: str, state: str, expected: date
) -> None:
    assert local_date(value, state) == expected


def test_every_snapshot_state_has_an_export_time_zone() -> None:
    manifest = Path(__file__).parents[1] / "snapshot" / "manifest.json"

    assert {a.state for a in load_manifest(manifest)} <= set(EXPORT_TIMEZONES)
