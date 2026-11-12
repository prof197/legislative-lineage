"""Bill status (passed, passed_legislature, died, introduced) from action classifications.

Passed means signed, became law without a signature, or a veto was overridden; a resolution
passes when it is adopted. Passed_legislature means both chambers passed the bill (or it was
enrolled, or the governor received it) and the record holds no outcome after that: the
governor's action is missing, pending, or recorded nowhere the export reaches. Died means
vetoed, failed, or the session (for carryover legislatures, the biennium) ended without
passage. Anything else is introduced. Status is judged as of the archive's Generated At, never
today. See DESIGN.md#bill-status.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo


class Status(StrEnum):
    PASSED = "passed"
    PASSED_LEGISLATURE = "passed_legislature"
    DIED = "died"
    INTRODUCED = "introduced"


@dataclass(frozen=True, slots=True)
class Action:
    date: str  # verbatim: a plain date or a zoned ISO timestamp
    classification: tuple[str, ...]
    order: int
    organization: str  # the body that acted, as the export names it: "Senate", "House", ...
    description: str = ""


@dataclass(frozen=True, slots=True)
class Outcome:
    status: Status
    reason: str | None  # the deciding action's classification, or "session-ended"
    on: date | None


# A chapter or act number, or an override, is final: nothing after it in the record undoes it
# in this model. A signature is final unless a full veto falls on its day or later: Georgia
# classifies "Date Signed by Governor" as a signature on a bill the governor vetoed that day,
# and Texas classifies "Transmitted to the Governor" as one. A conditional veto followed by a
# signature (New Jersey) is a law.
ENACTED = frozenset({"executive-signature", "became-law", "veto-override-passage"})
FINAL = frozenset({"became-law", "veto-override-passage"})
# A veto of items, or one that sends the bill back with changes, leaves a law if it is signed.
PARTIAL_VETO_WORDING = r"(?i)(line[- ]item|item veto|in part|conditional|amendatory)"
_PARTIAL_VETO_WORDING = re.compile(PARTIAL_VETO_WORDING)
KILLED = frozenset({"executive-veto", "failure", "veto-override-failure"})
SESSION_ENDED = "session-ended"
ENACTED_IN_WORDS = "enacted-in-words"
# Some exports record an enactment in words only, with no classification: Oklahoma's "Becomes
# law without Governor's signature", Arkansas's "Approved by the Governor", Connecticut's
# "PUBLIC ACT 25-61", a Massachusetts or Florida chapter number. Read as unclassified, those laws
# died with their session. The wording is matched only on actions that carry no outcome
# classification; a motion to override names the governor's objections and does not match.
# RE2 (DuckDB) and Python's re both accept these patterns; the SQL pre-filter uses them too.
ENACTED_WORDING = (
    r"(?i)(\bwithout (the )?(governor'?s )?signature"
    r"|\b(approved|signed) by (the )?(lieutenant |acting )?governor\b"
    r"|^public act (no\. )?\d+-\d+"
    r"|^(signed\b[^,]*, )?chapter \d+ of the acts of \d{4}"
    r"|^chapter no\. \d+-\d+"
    r"|^chapter \d+, \(?\d{4} laws"
    r"|\bsession law chapter \d+"
    r"|\bassigned a chapter number)"
)
_ENACTED_WORDING = re.compile(ENACTED_WORDING)
FINISHED_IN_WORDS = "finished-in-words"
# The step that ends the legislature's part, where an export records it only in words: each
# is on nearly every enacted bill of its state and on almost no bill that died (measured per
# state over the snapshot; see DESIGN.md#bill-status). Votes in both chambers are not enough:
# Hawaii's conference deaths and Maryland's refused concurrences passed both in some form.
FINISHED_WORDING = (
    r"(?i)(^concurred in (the )?(house|senate) amendments"  # Ohio
    r"|^passed to be enacted"  # Maine
    r"|^delivered to (the )?governor"  # Missouri, Kentucky
    r"|^enrolled bill signed"  # Mississippi
    r"|^enacted and laid before the governor"  # Massachusetts
    r"|^ordered enrolled|^ratified\b"  # North Carolina, South Carolina
    r"|^reported correctly enrolled"  # Iowa
    r"|^(speaker|president) signed\b"  # Oregon
    r"|^signed by (the )?(house )?speaker\b|^signed by (the )?president pro tem"  # Indiana
    r"|^enrolled\b|^ready to enroll"  # Alabama, Kentucky, California
    r"|^(sent|presented) to (the )?governor|^pres\. to gov\."  # Texas, Ohio, North Carolina
    r"|^reported enrolled"  # Texas
    r"|^transmitted to secretary of the state"  # Connecticut
    r"|^truly agreed to and finally passed)"  # Missouri
)
_FINISHED_WORDING = re.compile(FINISHED_WORDING)
# What the SQL pre-filter keeps of the unclassified actions.
STATUS_WORDING = f"{ENACTED_WORDING}|{FINISHED_WORDING}"
# Steps that come only after both chambers have passed a measure.
LEGISLATURE_DONE = frozenset({"enrolled", "executive-receipt"})
# Every classification `bill_status` reads; other actions cannot change a bill's status.
OUTCOME_CLASSIFICATIONS = ENACTED | KILLED | LEGISLATURE_DONE | {"passage"}

# Measures that are adopted rather than enacted. A simple resolution needs its own chamber; the
# others need both. Classified as a constitutional amendment, a measure is a joint resolution
# by another name (California's ACA and SCA, for example).
ADOPTED_BY_ONE_CHAMBER = frozenset({"resolution", "memorial"})
ADOPTED_BY_BOTH_CHAMBERS = frozenset(
    {
        "joint resolution",
        "concurrent resolution",
        "joint memorial",
        "concurrent memorial",
        "constitutional amendment",
    }
)
# Passage actions name a chamber, or a body that acts only once every chamber has: Nebraska's
# unicameral legislature, Utah's enrolling step, Ohio's "Sent To The Governor".
CHAMBERS = {"Senate": "upper", "House": "lower", "Assembly": "lower"}

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
    actions: Sequence[Action],
    *,
    kinds: Sequence[str],
    state: str,
    session_end: date | None,
    as_of: date,
) -> Outcome:
    """Explicit outcomes first, classified then in words, then adoption, then the session's end.

    `kinds` is the record's classification; `as_of` is the archive's Generated At.
    """
    ordered = chronological(actions, state)
    vetoed = [local_date(a.date, state) for a in ordered if _full_veto(a)]

    def undone(action: Action) -> bool:
        """A signature or worded enactment that a full veto on its day or later undoes."""
        return bool(vetoed) and max(vetoed) >= local_date(action.date, state)

    enacted = [a for a in ordered if ENACTED.intersection(a.classification)]
    if enacted and (
        any(FINAL.intersection(a.classification) for a in enacted) or not undone(enacted[-1])
    ):
        last = enacted[-1]
        return Outcome(Status.PASSED, _deciding(last, ENACTED), local_date(last.date, state))
    worded = [a for a in ordered if _enacted_in_words(a)]
    if worded and not undone(worded[-1]):
        return Outcome(Status.PASSED, ENACTED_IN_WORDS, local_date(worded[-1].date, state))

    killed = [a for a in ordered if KILLED.intersection(a.classification)]
    # A chamber can reconsider a failed vote; a later passage means the bill is alive again.
    if killed and not any(
        "passage" in a.classification for a in ordered[ordered.index(killed[-1]) + 1 :]
    ):
        last = killed[-1]
        return Outcome(Status.DIED, _deciding(last, KILLED), local_date(last.date, state))

    adoption = _adoption(ordered, kinds)
    if adoption is not None:
        return Outcome(Status.PASSED, "passage", local_date(adoption.date, state))

    done = _legislature_done(ordered)
    if done is not None:
        reason = next((c for c in done.classification if c in LEGISLATURE_DONE), FINISHED_IN_WORDS)
        return Outcome(Status.PASSED_LEGISLATURE, reason, local_date(done.date, state))

    # The session is still sitting on its last day.
    if session_end is not None and as_of > session_end:
        return Outcome(Status.DIED, SESSION_ENDED, session_end)
    return Outcome(Status.INTRODUCED, None, None)


def chronological(actions: Sequence[Action], state: str) -> list[Action]:
    """The actions by date, same-day actions in the direction the record lists them.

    Most exports list a bill's actions oldest first; Louisiana, Nebraska, North Carolina,
    Wyoming, Colorado and Hawaii list them newest first, and Kansas, Alabama and Montana vary
    by session. Export order alone would put a veto before the votes that preceded it.
    """
    by_order = sorted(actions, key=lambda a: a.order)
    if not by_order:
        return by_order
    days = {id(a): local_date(a.date, state) for a in by_order}
    newest_first = days[id(by_order[0])] > days[id(by_order[-1])]
    direction = -1 if newest_first else 1
    return sorted(by_order, key=lambda a: (days[id(a)], direction * a.order))


def _full_veto(action: Action) -> bool:
    """A veto of the whole bill: not of items, and not one returned with changes."""
    return (
        "executive-veto" in action.classification
        and "executive-veto-line-item" not in action.classification
        and _PARTIAL_VETO_WORDING.search(action.description) is None
    )


def _enacted_in_words(action: Action) -> bool:
    if OUTCOME_CLASSIFICATIONS.intersection(action.classification) - {"passage"}:
        return False
    return _ENACTED_WORDING.search(action.description) is not None


def _adoption(ordered: Sequence[Action], kinds: Sequence[str]) -> Action | None:
    """The passage that completed a resolution's adoption, if it was adopted."""
    if ADOPTED_BY_BOTH_CHAMBERS.intersection(kinds):
        needed = {"upper", "lower"}
    elif ADOPTED_BY_ONE_CHAMBER.intersection(kinds):
        needed = None
    else:
        return None
    # A measure is enrolled only once it is adopted. Louisiana records the adoption itself in
    # words ("Read by title, rules suspended, adopted.") and classifies the enrollment.
    return _both_chambers(ordered, needed) or next(
        (a for a in ordered if "enrolled" in a.classification), None
    )


def _both_chambers(ordered: Sequence[Action], needed: set[str] | None) -> Action | None:
    """The passage that completed the chambers `needed` (None: any one chamber)."""
    passed: set[str] = set()
    for action in ordered:
        if "passage" not in action.classification:
            continue
        chamber = CHAMBERS.get(action.organization)
        if needed is None or chamber is None:
            return action
        passed.add(chamber)
        if passed >= needed:
            return action
    return None


def _legislature_done(ordered: Sequence[Action]) -> Action | None:
    """The first step showing the legislature finished with the bill: enrollment, the
    governor's receipt, or a state's wording for either."""
    return next(
        (
            a
            for a in ordered
            if LEGISLATURE_DONE.intersection(a.classification)
            or (
                not KILLED.intersection(a.classification)
                and _FINISHED_WORDING.search(a.description) is not None
            )
        ),
        None,
    )


def _deciding(action: Action, kinds: frozenset[str]) -> str:
    return next(c for c in action.classification if c in kinds)
