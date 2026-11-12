"""A corpus bill's relatives in other states, read from the build's stored pairs.

A new text is searched (lineage.query). A bill already in the corpus has its matches stored:
the near-duplicate stage's cross-state section pairs (lineage.pairs) and the semantic stage's
(lineage.semantic_pairs), so its relatives are read from them with no search, as a page lists
them (lineage.listing). They are scored by
the query's own function, `lineage.query.group_matches`: the share of the bill's words in the
sections a family matches, of the words of every section either stage searched, each family
counted once. So a bill and a query of its text score alike.

Every related family is counted, and the RELATIVES_SHOWN best are kept for display. Whether an
enactment counts as found never depends on that cap (DESIGN.md#commission-recall): a uniform
act's real relatives can outnumber any fixed rank. See DESIGN.md#bill-scoring.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.listing import ListedPairs, listed_pairs
from lineage.query import Match, group_matches
from lineage.semantic import MODEL, THRESHOLD
from lineage.semantic_pairs import section_keys

RELATIVES_SHOWN = 50  # a bill page's relatives; DESIGN.md#bill-scoring
STAGES = ("near_duplicate", "semantic")

Hit = tuple[int, str, float]  # (the bill's section ordinal, the other bill's id, its score)


@dataclass(frozen=True, slots=True)
class BillRelatives:
    words_searched: int
    families: int  # every related family in another state, however many are shown
    matches: list[Match]  # the best `shown` of them, best first


def _hits(
    left: NDArray[np.uint32],
    right: NDArray[np.uint32],
    scores: NDArray[np.generic],
    own: NDArray[np.bool_],
    bills: Sequence[str],
    ordinals: NDArray[np.int64],
) -> dict[str, list[Hit]]:
    """Each wanted bill's pairs, from its own side: a pair of two wanted bills counts for both."""
    out: dict[str, list[Hit]] = defaultdict(list)
    left, right = np.asarray(left), np.asarray(right)
    for mine, theirs in ((left, right), (right, left)):
        at = np.flatnonzero(own[mine])
        ms: list[int] = mine[at].tolist()
        ts: list[int] = theirs[at].tolist()
        values: list[float] = np.asarray(scores)[at].tolist()
        for m, t, score in zip(ms, ts, values, strict=True):
            out[bills[m]].append((int(ordinals[m]), bills[t], score))
    return out


def bill_relatives(
    tables_dir: Path,
    bill_ids: Sequence[str],
    *,
    model: str = MODEL,
    stages: Sequence[str] = STAGES,
    shown: int = RELATIVES_SHOWN,
    threshold: float = THRESHOLD,
    flag: NDArray[np.bool_] | None = None,
    exempt_both: bool = True,
    listed: ListedPairs | None = None,
) -> dict[str, BillRelatives]:
    """Each wanted bill's relatives in other states, best first, the best `shown` kept, as a
    page lists them (lineage.listing): by meaning at or above `threshold`, and without the pairs
    of a section `flag` marks that one stage alone found. `listed` passes pairs already read."""
    wanted = sorted(set(bill_ids))
    bills, ordinals = section_keys(tables_dir)
    wanted_set = set(wanted)
    own = np.fromiter((b in wanted_set for b in bills), dtype=bool, count=len(bills))
    signed: dict[str, set[int]] = defaultdict(set)
    for number in np.flatnonzero(own).tolist():
        signed[bills[number]].add(int(ordinals[number]))
    with queries.connect_quiet() as con:
        rows = queries.matchable_section_normalized(
            con,
            sections_glob=str(tables_dir / "sections" / "*.parquet"),
            eligible_path=tables_dir / "eligible.parquet",
            ids=wanted,
        )
    words: dict[str, dict[int, int]] = defaultdict(dict)
    for bill_id, ordinal, normalized in rows:
        words[bill_id][ordinal] = len(normalized.split())

    if listed is None:
        listed = listed_pairs(
            tables_dir, model=model, threshold=threshold, flag=flag, exempt_both=exempt_both
        )
    near: dict[str, list[Hit]] = {}
    if "near_duplicate" in stages:
        pairs = listed.near
        near = _hits(pairs.left, pairs.right, pairs.agreements, own, bills, ordinals)
    by_meaning: dict[str, list[Hit]] = {}
    if "semantic" in stages:
        found = listed.meaning
        by_meaning = _hits(found.left, found.right, found.cosine, own, bills, ordinals)

    out: dict[str, BillRelatives] = {}
    for bill_id in wanted:
        # The sections a query of the bill's text would search: the signed ones, and with the
        # semantic stage every matchable section with a word.
        searched = set(signed[bill_id])
        if "semantic" in stages:
            searched |= {o for o, n in words[bill_id].items() if n}
        words_searched = sum(words[bill_id].get(o, 0) for o in searched)
        matches = group_matches(
            [(o, b, int(s)) for o, b, s in near.get(bill_id, [])],
            [(o, b, float(s)) for o, b, s in by_meaning.get(bill_id, [])],
            words[bill_id],
            words_searched,
            tables_dir,
        )
        out[bill_id] = BillRelatives(words_searched, len(matches), matches[:shown])
    return out
