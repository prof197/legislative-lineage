"""The deploy's rules (rules/deploy.toml), and the circuit breaker they set: a day's counts
against the days deployed without a hold (lineage.live.deploy).

See DESIGN.md#updates, DR-13.
"""

import statistics
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from lineage.export.release import Checksums, changed_files
from lineage.live.deploy_gates import Candidate
from lineage.live.deploy_ledger import DeployPaths, records
from lineage.live.feed import listed_origin
from lineage.live.updates import Change, OriginGained

RULES = Path("rules/deploy.toml")
KINDS = ("origin_claims", "changed_files")


# The rules.


@dataclass(frozen=True, slots=True)
class Limits:
    floor: int
    ceiling: int


@dataclass(frozen=True, slots=True)
class Breaker:
    window: int
    min_history: int
    multiple: float
    limits: Mapping[str, Limits]


@dataclass(frozen=True, slots=True)
class DeployRules:
    first_day: Mapping[str, date]
    confirmed: Mapping[str, bool]
    breaker: Breaker
    max_share: float  # of the Pages limit, for the size gate


def load_rules(path: Path = RULES) -> DeployRules:
    doc = tomllib.loads(path.read_text())
    sessions: dict[str, Any] = doc.get("session", {})
    b = doc["breaker"]
    return DeployRules(
        first_day={st: s["first_day"] for st, s in sessions.items()},
        confirmed={st: bool(s.get("confirmed", False)) for st, s in sessions.items()},
        breaker=Breaker(
            window=int(b["window"]),
            min_history=int(b["min_history"]),
            multiple=float(b["multiple"]),
            limits={k: Limits(int(b[k]["floor"]), int(b[k]["ceiling"])) for k in KINDS},
        ),
        max_share=float(doc["size"]["max_share_of_pages_limit"]),
    )


def in_session(rules: DeployRules, state: str, day: date) -> bool:
    """Whether `day` is on or after the state's first day of session, once the maintainer has
    confirmed that date; a state with none recorded, or with its date unconfirmed, never
    deploys by itself, since deploying without review is what the date turns on."""
    first = rules.first_day.get(state)
    return first is not None and rules.confirmed.get(state, False) and day >= first


# The circuit breaker.


def listed_origin_claims(
    changes: Iterable[Change], days: Iterable[date], withheld: frozenset[str] = frozenset()
) -> list[OriginGained]:
    """The origin matches the given days gained that a page or the feed lists: by the larger
    of the two shares, at the listing floor, their document not withheld, as lineage.live.feed lists
    them."""
    wanted = set(days)
    return [c for c in changes if isinstance(c, OriginGained) and c.as_of in wanted and
            listed_origin(c, withheld)]  # fmt: skip


def day_counts(
    candidate: Candidate,
    reference: Checksums | None,
    changes: Sequence[Change],
    days: Iterable[date],
    withheld: frozenset[str] = frozenset(),
) -> dict[str, int]:
    """A day's counts: its listed new origin claims, and the files its release adds, changes or
    removes against the one before (every file, without one)."""
    if reference is None:
        files = len(candidate.checksums.files)
    else:
        diff = changed_files(reference, candidate.checksums)
        files = sum(len(v) for v in diff.values())
    return {
        "origin_claims": len(listed_origin_claims(changes, days, withheld)),
        "changed_files": files,
    }


@dataclass(frozen=True, slots=True)
class Decision:
    hold: bool
    reasons: list[str]
    baseline: dict[str, Any]


def breaker(
    counts: Mapping[str, int], history: Sequence[Mapping[str, int]], b: Breaker
) -> Decision:
    """Hold when a count passes its limit: with at least `min_history` normal days, both
    `multiple` times the median of the last `window` and the floor; with fewer, the ceiling."""
    recent = list(history)[-b.window :]
    reasons: list[str] = []
    baseline: dict[str, Any] = {"days": len(recent)}
    for kind in KINDS:
        limits = b.limits[kind]
        if len(recent) < b.min_history:
            limit = float(limits.ceiling)
            baseline[kind] = {"median": None, "limit": limit, "rule": "ceiling"}
            why = (
                f"its ceiling of {limits.ceiling} while fewer than {b.min_history} days are "
                "deployed"
            )
        else:
            median = float(statistics.median(h[kind] for h in recent))
            limit = max(b.multiple * median, float(limits.floor))
            baseline[kind] = {"median": median, "limit": limit, "rule": "median"}
            why = (
                f"{b.multiple:g} times the median of {median:g} over {len(recent)} days and the "
                f"floor of {limits.floor}"
            )
        if counts[kind] > limit:
            reasons.append(f"{counts[kind]} {kind.replace('_', ' ')}, past {why}")
    return Decision(hold=bool(reasons), reasons=reasons, baseline=baseline)


def normal_history(paths: DeployPaths, before: date) -> list[dict[str, int]]:
    """The counts of the days deployed without a hold before `before`, oldest first. A day held
    and then released was a jump by definition, so it does not set what normal is."""
    return [r.counts for r in records(paths) if r.day < before and r.state == "deployed"]
