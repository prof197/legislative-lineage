"""`lineage featured`: the featured stories' scope, the section pairs `lineage explain` explains.

Each story has an anchor bill and a rule for the relatives it is compared with (STORIES), as
the M6 review ruled them. A pair is every section pair a page lists between the anchor and a
relative (lineage.listing: both stages, by meaning at the stage's threshold, the state
boilerplate flag applied), the anchor's section as the origin. "Origin" only names the text the
comparison starts from: no story's anchor is a documented source, so no explanation may say
which text came first (DESIGN.md#explanations).

- Anti-SLAPP: Utah SB 18 of 2023, the earliest enacted state version of the commission's act
  in the corpus, against the bills the commission's lists name as enacting it
  (reports/answer_key.csv). The act's own text stays a local eval query and is never shown.
  An enactment with no listed pair with Utah is reported, and gets no explanation.
- Deepfakes: Washington SB 5152 of 2023-24, against the relatives its page lists that were
  introduced after it.
- Georgia: SB 351 of 2023-24, the Protecting Georgia's Children on Social Media Act, against
  every relative its page lists.

The pairs file names sections by bill id and ordinal and holds no text; the report lists each
story's relatives with their titles and pair counts.
"""

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lineage import queries
from lineage.answer_key import ANSWER_KEY, read_answer_key
from lineage.listing import ListedPairs, listed_pairs, shipped_flag
from lineage.relatives import bill_relatives
from lineage.semantic_pairs import section_keys

PAIRS = Path("evals/featured_pairs.jsonl")
REPORT = Path("reports/featured_pairs.json")
COMMAND = "uv run lineage featured"


class FeaturedError(Exception):
    pass


class Relatives(StrEnum):
    ENACTMENTS = "the bills the commission's lists name as enacting its act"
    LISTED_LATER = "the relatives its page lists that were introduced after it"
    LISTED = "every relative its page lists"


@dataclass(frozen=True, slots=True)
class Story:
    name: str
    state: str
    session: str
    identifier: str
    relatives: Relatives
    act: str | None = None  # for ENACTMENTS: the act whose enactments are the relatives


STORIES = (
    Story(
        "anti-slapp", "UT", "2023", "SB 18", Relatives.ENACTMENTS, "public-expression-protection"
    ),
    Story("deepfakes", "WA", "2023-2024", "SB 5152", Relatives.LISTED_LATER),
    Story("georgia-social-media", "GA", "2023_24", "SB 351", Relatives.LISTED),
)

SectionPair = tuple[int, str, int]  # (anchor's ordinal, relative's bill id, relative's ordinal)


def pairs_with(
    listed: ListedPairs,
    bills: Sequence[str],
    ordinals: NDArray[np.int64],
    anchor: str,
    relatives: Collection[str],
) -> list[SectionPair]:
    """Every listed section pair between `anchor` and one of `relatives`, either stage, once,
    from the anchor's side."""
    wanted = set(relatives)
    mine = np.fromiter((b == anchor for b in bills), dtype=bool, count=len(bills))
    out: set[SectionPair] = set()
    for left, right in (
        (listed.near.left, listed.near.right),
        (listed.meaning.left, listed.meaning.right),
    ):
        a, b = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
        for x, y in ((a, b), (b, a)):
            at = np.flatnonzero(mine[x])
            for s, t in zip(x[at].tolist(), y[at].tolist(), strict=True):
                other = str(bills[int(t)])
                if other in wanted:
                    out.add((int(ordinals[s]), other, int(ordinals[t])))
    return sorted(out, key=lambda p: (p[1], p[0], p[2]))


def introduced_after(
    relatives: Sequence[str], first_action: dict[str, str | None], anchor_first: str | None
) -> list[str]:
    """The relatives whose first recorded action falls after the anchor's."""
    if anchor_first is None:
        raise FeaturedError("the anchor has no dated action to order its relatives by")
    return [r for r in relatives if (first_action.get(r) or "") > anchor_first]


def _enactments(act: str, anchor: str, answer_key: Path) -> list[str]:
    _, rows = read_answer_key(answer_key)
    ids = [r["bill_id"] for r in rows if r["act"] == act and r["counted"].startswith("yes")]
    return [b for b in dict.fromkeys(ids) if b and b != anchor]


Details = dict[str, tuple[str, str, str, str, str | None, str | None]]


def _bill(b: str, details: Details) -> dict[str, Any]:
    state, session, identifier, title, status, first = details[b]
    return {
        "bill_id": b,
        "state": state,
        "session": session,
        "identifier": identifier,
        "title": title,
        "status": status,
        "first_action": first,
    }


def build_featured(
    tables_dir: Path, *, answer_key: Path = ANSWER_KEY, stories: Sequence[Story] = STORIES
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The featured pairs, one row per section pair, and the report of each story's relatives."""
    flag = shipped_flag(tables_dir)
    if flag is None:
        raise FeaturedError("no state boilerplate counts; run `uv run lineage state-boilerplate`")
    listed = listed_pairs(tables_dir, flag=flag)
    bills, ordinals = section_keys(tables_dir)
    bills_glob = str(tables_dir / "bills" / "*.parquet")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, int]] = set()
    report: dict[str, Any] = {"command": COMMAND, "stories": {}}
    with queries.connect_quiet() as con:
        for story in stories:
            anchor = queries.bill_id_of(
                con,
                bills_glob=bills_glob,
                state=story.state,
                session=story.session,
                identifier=story.identifier,
            )
            if anchor is None:
                raise FeaturedError(
                    f"{story.name}: no bill {story.state} {story.session} {story.identifier}"
                )
            if story.relatives is Relatives.ENACTMENTS:
                if story.act is None:
                    raise FeaturedError(f"{story.name}: an enactments story names its act")
                relatives = _enactments(story.act, anchor, answer_key)
            else:
                page = bill_relatives(tables_dir, [anchor], listed=listed)[anchor]
                relatives = [m.bill_id for m in page.matches]
            details = queries.bill_details(con, bills_glob=bills_glob, ids=[anchor, *relatives])
            if story.relatives is Relatives.LISTED_LATER:
                first = {b: d[5] for b, d in details.items()}
                relatives = introduced_after(relatives, first, first[anchor])
            pairs = pairs_with(listed, bills, ordinals, anchor, relatives)
            by_relative: dict[str, int] = {}
            for origin, relative, ordinal in pairs:
                by_relative[relative] = by_relative.get(relative, 0) + 1
                key = (anchor, origin, relative, ordinal)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "story": story.name,
                        "origin": {"bill_id": anchor, "ordinal": origin},
                        "descendant": {"bill_id": relative, "ordinal": ordinal},
                    }
                )

            report["stories"][story.name] = {
                "anchor": _bill(anchor, details),
                "relatives_rule": str(story.relatives),
                "relatives": [
                    _bill(r, details) | {"section_pairs": by_relative.get(r, 0)} for r in relatives
                ],
                "relatives_without_a_pair": sum(1 for r in relatives if r not in by_relative),
                "section_pairs": len(pairs),
            }
    report["section_pairs"] = len(rows)
    return rows, report


def write_featured(
    rows: Sequence[dict[str, Any]],
    report: dict[str, Any],
    *,
    pairs_path: Path = PAIRS,
    report_path: Path = REPORT,
) -> None:
    pairs_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
