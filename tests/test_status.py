from datetime import date
from pathlib import Path

import pytest

from lineage.snapshot import load_manifest
from lineage.status import EXPORT_TIMEZONES, Action, Outcome, Status, bill_status, local_date

SESSION_END = date(2026, 4, 2)
AFTER_SESSION = date(2026, 7, 25)
DURING_SESSION = date(2026, 3, 1)


def _actions(*steps: tuple[str, str] | tuple[str, str, str]) -> list[Action]:
    """Actions in the order given: (date, classification) or (date, classification, body).

    '' means unclassified. The body defaults to the House.
    """
    return [
        Action(
            date=step[0],
            classification=(step[1],) if step[1] else (),
            order=i,
            organization=step[2] if len(step) == 3 else "House",
        )
        for i, step in enumerate(steps)
    ]


def _status(
    actions: list[Action],
    *,
    as_of: date,
    end: date | None = SESSION_END,
    kinds: tuple[str, ...] = ("bill",),
) -> Outcome:
    return bill_status(actions, kinds=kinds, state="GA", session_end=end, as_of=as_of)


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
    veto = Action(date="2025-04-01", classification=("executive-veto",), order=3, organization="")
    override = Action(
        date="2025-04-01", classification=("veto-override-passage",), order=4, organization=""
    )

    assert _status([override, veto], as_of=AFTER_SESSION).status is Status.PASSED


def test_an_action_can_carry_several_classifications() -> None:
    actions = [
        Action(
            date="2025-05-01",
            classification=("passage", "executive-signature"),
            order=0,
            organization="Governor",
        )
    ]

    assert _status(actions, as_of=AFTER_SESSION).reason == "executive-signature"


# Resolutions are adopted, not signed. A simple resolution is adopted by its own chamber; a
# joint or concurrent one by both, or by a body that acts for the whole legislature.

ADOPTED_HOUSE = ("2025-02-20", "passage", "House")
ADOPTED_SENATE = ("2025-03-25", "passage", "Senate")


def test_a_simple_resolution_passed_when_its_chamber_adopted_it() -> None:
    actions = _actions(INTRODUCED, ADOPTED_HOUSE)

    assert _status(actions, as_of=DURING_SESSION, kinds=("resolution",)) == Outcome(
        Status.PASSED, "passage", date(2025, 2, 20)
    )


def test_a_memorial_is_adopted_like_a_simple_resolution() -> None:
    actions = _actions(INTRODUCED, ADOPTED_SENATE)

    assert _status(actions, as_of=AFTER_SESSION, kinds=("memorial",)).status is Status.PASSED


@pytest.mark.parametrize(
    "kind",
    [
        "joint resolution",
        "concurrent resolution",
        "joint memorial",
        "concurrent memorial",
        "constitutional amendment",
    ],
)
def test_a_joint_measure_passed_when_the_second_chamber_adopted_it(kind: str) -> None:
    actions = _actions(INTRODUCED, ADOPTED_HOUSE, ADOPTED_SENATE)

    # The deciding action is the second chamber's adoption, which a bill page names.
    assert _status(actions, as_of=AFTER_SESSION, kinds=(kind,)) == Outcome(
        Status.PASSED, "passage", date(2025, 3, 25)
    )


def test_a_joint_resolution_adopted_by_one_chamber_died_with_the_session() -> None:
    actions = _actions(INTRODUCED, ADOPTED_HOUSE, ("2025-03-01", "passage", "House"))

    assert _status(actions, as_of=AFTER_SESSION, kinds=("joint resolution",)) == Outcome(
        Status.DIED, "session-ended", SESSION_END
    )


def test_the_assembly_is_a_lower_chamber() -> None:
    actions = _actions(ADOPTED_SENATE, ("2025-04-02", "passage", "Assembly"))

    assert _status(actions, as_of=AFTER_SESSION, kinds=("concurrent resolution",)) == Outcome(
        Status.PASSED, "passage", date(2025, 4, 2)
    )


@pytest.mark.parametrize(
    "body",
    [
        "Nebraska Legislature",  # unicameral
        "Utah State Legislature",  # "signed by President/ sent for enrolling"
        "Ohio General Assembly",  # "Sent To The Governor"
    ],
)
def test_adoption_by_the_whole_legislature_counts_for_both_chambers(body: str) -> None:
    actions = _actions(INTRODUCED, ("2025-03-30", "passage", body))

    assert _status(actions, as_of=AFTER_SESSION, kinds=("joint resolution",)) == Outcome(
        Status.PASSED, "passage", date(2025, 3, 30)
    )


def test_a_vetoed_joint_resolution_died_though_both_chambers_adopted_it() -> None:
    actions = _actions(ADOPTED_HOUSE, ADOPTED_SENATE, ("2025-04-10", "executive-veto", "Governor"))

    assert _status(actions, as_of=AFTER_SESSION, kinds=("joint resolution",)) == Outcome(
        Status.DIED, "executive-veto", date(2025, 4, 10)
    )


def test_a_signed_joint_resolution_names_the_signature() -> None:
    actions = _actions(
        ADOPTED_HOUSE, ADOPTED_SENATE, ("2025-04-10", "executive-signature", "Governor")
    )

    assert _status(actions, as_of=AFTER_SESSION, kinds=("joint resolution",)).reason == (
        "executive-signature"
    )


def test_a_resolution_failed_then_adopted_was_reconsidered() -> None:
    actions = _actions(("2025-02-01", "failure", "House"), ADOPTED_HOUSE)

    assert _status(actions, as_of=AFTER_SESSION, kinds=("resolution",)).status is Status.PASSED


def test_votes_in_both_chambers_alone_do_not_show_the_legislature_finished() -> None:
    # Hawaii's conference deaths, Maryland's refused concurrences and New Jersey's bills
    # replaced by their companions all passed both chambers in some form.
    actions = _actions(INTRODUCED, ADOPTED_HOUSE, ADOPTED_SENATE)

    assert _status(actions, as_of=AFTER_SESSION, kinds=("bill",)).status is Status.DIED


@pytest.mark.parametrize(
    "description",
    [
        "Concurred in House amendments",  # Ohio
        "PASSED TO BE ENACTED, in concurrence.",  # Maine
        "Delivered to Governor",  # Missouri, Kentucky
        "Enrolled Bill Signed",  # Mississippi
        "Enacted and laid before the Governor",  # Massachusetts
        "Ordered Enrolled",  # North Carolina
        "Reported correctly enrolled, signed by Speaker and President",  # Iowa
        "Speaker signed.",  # Oregon
        "Ready to Enroll",  # Alabama
        "Transmitted to Secretary of the State",  # Connecticut
        "Sent to the Governor",  # Texas, Ohio
    ],
)
def test_wording_that_the_legislature_finished_passed_the_legislature(description: str) -> None:
    actions = [*_actions(INTRODUCED), Action("2025-03-30", (), 5, "House", description)]

    outcome = _status(actions, as_of=AFTER_SESSION, kinds=("bill",))

    assert outcome == Outcome(Status.PASSED_LEGISLATURE, "finished-in-words", date(2025, 3, 30))


def test_a_refused_concurrence_is_not_the_legislature_finishing() -> None:
    actions = [*_actions(INTRODUCED), Action("2025-03-30", (), 5, "House", "House Refuses Concur")]

    assert _status(actions, as_of=AFTER_SESSION, kinds=("bill",)).status is Status.DIED


@pytest.mark.parametrize("step", ["enrolled", "executive-receipt"])
def test_an_enrolled_bill_or_one_the_governor_received_passed_the_legislature(step: str) -> None:
    actions = _actions(INTRODUCED, ADOPTED_HOUSE, ("2025-03-30", step, "Senate"))

    outcome = _status(actions, as_of=DURING_SESSION, kinds=("bill",))

    assert outcome == Outcome(Status.PASSED_LEGISLATURE, step, date(2025, 3, 30))


def test_a_bill_one_chamber_passed_died_with_the_session() -> None:
    actions = _actions(INTRODUCED, ADOPTED_HOUSE)

    assert _status(actions, as_of=AFTER_SESSION, kinds=("bill",)).status is Status.DIED


def test_a_bill_both_chambers_passed_then_vetoed_died() -> None:
    actions = _actions(INTRODUCED, ADOPTED_HOUSE, ADOPTED_SENATE, ("2025-04-10", "executive-veto"))

    assert _status(actions, as_of=AFTER_SESSION, kinds=("bill",)).status is Status.DIED


def test_an_enrolled_resolution_was_adopted() -> None:
    # Louisiana records a resolution's adoption in words ("Read by title, rules suspended,
    # adopted."); the enrollment that follows is classified.
    actions = _actions(("2023-04-24", "", "House"), ("2023-04-25", "enrolled", "House"))

    outcome = _status(actions, as_of=AFTER_SESSION, kinds=("resolution",))

    assert outcome == Outcome(Status.PASSED, "passage", date(2023, 4, 25))


def test_a_record_listed_newest_first_is_read_in_date_order() -> None:
    # Louisiana, Nebraska, North Carolina and others list actions newest first. Read in export
    # order, the veto came before the passages and the bill looked alive again.
    newest_first = [
        Action("2023-06-20", ("executive-veto",), 0, "Executive"),
        Action("2023-06-01", ("passage",), 1, "House"),
        Action("2023-06-01", ("reading-3",), 2, "House"),
        Action("2023-05-16", ("passage",), 3, "Senate"),
    ]

    outcome = _status(newest_first, as_of=AFTER_SESSION, kinds=("bill",))

    assert outcome == Outcome(Status.DIED, "executive-veto", date(2023, 6, 20))


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


def _worded(description: str) -> list[Action]:
    passed = _actions(INTRODUCED, PASSED_HOUSE, PASSED_SENATE)
    return [*passed, Action("2025-05-20", (), len(passed), "Executive", description)]


@pytest.mark.parametrize(
    "description",
    [
        "Becomes law without Governor's signature 05/20/2025",  # Oklahoma
        "Approved by the Governor",  # Arkansas
        "Law Without Signature 05/20/2025; Chapter 112; eff. 07/19/2025",  # New Hampshire
        "Enacted in accordance with Article 44 Part II of the N.H. Constitution without the "
        "signature of the governor.",
        "PUBLIC ACT 25-61",  # Connecticut
        "Signed by the Lieutenant Governor, Chapter 81 of the Acts of 2024",  # Massachusetts
        "Chapter No. 2025-104",  # Florida
        "Chapter 393, (2025 Laws): Effective date January 1, 2026.",  # Oregon
        "Law without signature - March 28, 2025    Session Law Chapter 140",  # Idaho
        "Item vetoed, signed by Governor.",  # Iowa
    ],
)
def test_an_enactment_recorded_only_in_words_passed(description: str) -> None:
    assert _status(_worded(description), as_of=AFTER_SESSION) == Outcome(
        Status.PASSED, "enacted-in-words", date(2025, 5, 20)
    )


@pytest.mark.parametrize(
    "description",
    [
        # "Delivered to Governor" is not an enactment either: the legislature finished (above).
        "Laid before the Governor",
        # A motion to override, not the override itself: the vote is classified when it passes.
        "Rountree MO120 That the bill becomes law notwithstanding the objections of the Governor",
        "Taken from the files on motion and referred to Subcommittee on chapter 3 of the acts "
        "of 2024",
    ],
)
def test_other_wording_about_the_governor_or_a_chapter_is_not_an_enactment(
    description: str,
) -> None:
    assert _status(_worded(description), as_of=AFTER_SESSION).status == Status.DIED


def test_a_classified_outcome_is_read_before_the_wording() -> None:
    actions = [*_worded("Approved by the Governor")]
    actions[-1] = Action("2025-05-20", ("executive-signature",), 3, "Executive", "Approved")

    assert _status(actions, as_of=AFTER_SESSION).reason == "executive-signature"
