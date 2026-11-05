"""Bill status (passed, died, introduced) from Open States action classifications.

Passed means signed, became law without a signature, or a veto was overridden. Died means
vetoed, failed, or the session (for carryover legislatures, the biennium) ended without
passage. Anything else is introduced. Status is judged as of the archive's Generated At, never
today. See DESIGN.md#bill-status.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo


class Status(StrEnum):
    PASSED = "passed"
    DIED = "died"
    INTRODUCED = "introduced"


@dataclass(frozen=True, slots=True)
class Action:
    date: str  # verbatim: a plain date or a zoned ISO timestamp
    classification: tuple[str, ...]
    order: int


@dataclass(frozen=True, slots=True)
class Outcome:
    status: Status
    reason: str | None  # the deciding action's classification, or "session-ended"
    on: date | None


# Enactment is final: nothing after it in the record undoes it in this model, and a conditional
# veto followed by a signature (New Jersey) is a law.
ENACTED = frozenset({"executive-signature", "became-law", "veto-override-passage"})
KILLED = frozenset({"executive-veto", "failure", "veto-override-failure"})
SESSION_ENDED = "session-ended"
# Every classification `bill_status` reads; other actions cannot change a bill's status.
OUTCOME_CLASSIFICATIONS = ENACTED | KILLED | {"passage"}

# The zone each state's export uses for zoned timestamps. Most write a plain date as local
# midnight, so the local date is the true one. The capital's zone is right for every state but
# Alabama, whose export writes midnight Eastern (98% of its zoned timestamps).
EXPORT_TIMEZONES: dict[str, str] = {
    "AL": "America/New_York",
    "AK": "America/Juneau",
    "AZ": "America/Phoenix",
    "AR": "America/Chicago",
    "CA": "America/Los_Angeles",
    "CO": "America/Denver",
    "CT": "America/New_York",
    "DE": "America/New_York",
    "FL": "America/New_York",
    "GA": "America/New_York",
    "HI": "Pacific/Honolulu",
    "ID": "America/Boise",
    "IL": "America/Chicago",
    "IN": "America/Indiana/Indianapolis",
    "IA": "America/Chicago",
    "KS": "America/Chicago",
    "KY": "America/New_York",
    "LA": "America/Chicago",
    "ME": "America/New_York",
    "MD": "America/New_York",
    "MA": "America/New_York",
    "MI": "America/Detroit",
    "MN": "America/Chicago",
    "MS": "America/Chicago",
    "MO": "America/Chicago",
    "MT": "America/Denver",
    "NE": "America/Chicago",
    "NV": "America/Los_Angeles",
    "NH": "America/New_York",
    "NJ": "America/New_York",
    "NM": "America/Denver",
    "NY": "America/New_York",
    "NC": "America/New_York",
    "ND": "America/Chicago",
    "OH": "America/New_York",
    "OK": "America/Chicago",
    "OR": "America/Los_Angeles",
    "PA": "America/New_York",
    "RI": "America/New_York",
    "SC": "America/New_York",
    "SD": "America/Chicago",
    "TN": "America/Chicago",
    "TX": "America/Chicago",
    "UT": "America/Denver",
    "VT": "America/New_York",
    "VA": "America/New_York",
    "WA": "America/Los_Angeles",
    "WV": "America/New_York",
    "WI": "America/Chicago",
    "WY": "America/Denver",
}


def local_date(value: str, state: str) -> date:
    """The calendar date of an action as the state's export meant it."""
    if len(value) == 10:
        return date.fromisoformat(value)
    return datetime.fromisoformat(value).astimezone(ZoneInfo(EXPORT_TIMEZONES[state])).date()


def bill_status(
    actions: Sequence[Action], *, state: str, session_end: date | None, as_of: date
) -> Outcome:
    """Explicit outcomes first, then the session's end; `as_of` is the archive's Generated At."""
    ordered = sorted(actions, key=lambda a: a.order)

    enacted = [a for a in ordered if ENACTED.intersection(a.classification)]
    if enacted:
        last = enacted[-1]
        return Outcome(Status.PASSED, _deciding(last, ENACTED), local_date(last.date, state))

    killed = [a for a in ordered if KILLED.intersection(a.classification)]
    # A chamber can reconsider a failed vote; a later passage means the bill is alive again.
    if killed and not any(
        "passage" in a.classification and a.order > killed[-1].order for a in ordered
    ):
        last = killed[-1]
        return Outcome(Status.DIED, _deciding(last, KILLED), local_date(last.date, state))

    # The session is still sitting on its last day.
    if session_end is not None and as_of > session_end:
        return Outcome(Status.DIED, SESSION_ENDED, session_end)
    return Outcome(Status.INTRODUCED, None, None)


def _deciding(action: Action, kinds: frozenset[str]) -> str:
    return next(c for c in action.classification if c in kinds)
