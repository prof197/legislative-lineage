"""Recorded days for the live track's tests, built from the commission fixture at test time,
and the reference rebuild an update is checked against (DESIGN.md#updates).

Day 1 brings a Georgia record and the commission fixture's records from states before N; day 2
the rest, Utah's among them; day 3 replaces the Georgia record's text and records a governor's
signature on a day-1 record without changing its text. The Georgia record is synthetic, the only
one here: no Georgia bill enacts these acts, so it carries Utah SB 18's text, the Public
Expression Protection Act, and then the Special Deposits Act's query text. It arrives before
Utah's version, so Utah SB 18 joins its relatives only on day 2; day 1 already brings the act's
enactments in the states before N, its first relatives.
"""

import copy
import json
from pathlib import Path
from typing import Any

from lineage.live.updates import fold, load_days, write_archives
from lineage.match.semantic.pairs import build_semantic_pairs
from tests.conftest import COMMISSION, FAKE_MODEL, embed_with_stand_in, tables_from_raw

GEORGIA_ID = "ocd-bill/fixture-ga-0001"  # not an Open States id: the record is synthetic
UTAH_SB_18 = "ocd-bill/7968b9d9-25ee-4684-a92e-c76167d6f917"
SPECIAL_DEPOSITS = "ocd-bill/a3bec6ea-4c68-4251-b36c-60f0e4053031"
DAYS = ("2026-01-12", "2026-01-13", "2026-01-14")
# Georgia's 2025-26 session, copied from snapshot/sessions.csv.
GEORGIA_SESSION = "GA,2025_26,2025-2026 Regular Session,2025-01-13,2026-04-02,\n"


def _archives() -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for fixture in sorted(p for p in COMMISSION.iterdir() if p.is_dir()):
        state, _, session = fixture.name.partition("_")
        out[(state, session)] = json.loads(next(fixture.glob("*_bills.json")).read_text())
    return out


def _georgia(text_of: dict[str, Any], note: str) -> dict[str, Any]:
    return {
        "id": GEORGIA_ID,
        "legislative_session": "2025_26",
        "identifier": "SB 9001",
        "title": f"Synthetic fixture record: {note}",
        "chamber": "upper",
        "classification": ["bill"],
        "actions": [
            {
                "organization__name": "Senate",
                "description": "Read and Referred",
                "date": DAYS[0],
                "classification": ["introduction"],
                "order": 1,
            }
        ],
        "versions": [],
        "raw_text": text_of["raw_text"],
        "raw_text_url": None,
    }


def _write_day(
    root: Path, as_of: str, archives: dict[tuple[str, str], list[dict[str, Any]]]
) -> None:
    day = root / as_of
    day.mkdir(parents=True)
    (day / "day.json").write_text(json.dumps({"as_of": as_of, "sources": ["fixture"]}))
    for (state, session), rows in archives.items():
        (day / f"{state}_{session}_bills.json").write_text(json.dumps(rows))


def write_days(root: Path) -> Path:
    """The three recorded days under `root`; returns it."""
    archives = _archives()
    by_id = {r["id"]: r for rows in archives.values() for r in rows}
    early = {k: v for k, v in archives.items() if k[0] < "N"}
    late = {k: v for k, v in archives.items() if k[0] >= "N"}
    utah = _georgia(by_id[UTAH_SB_18], "Public Expression Protection Act (Utah SB 18's text)")
    _write_day(root, DAYS[0], {**early, ("GA", "2025_26"): [utah]})
    _write_day(root, DAYS[1], late)
    deposits = _georgia(by_id[SPECIAL_DEPOSITS], "Special Deposits Act (the act query's text)")
    (state, session), rows = next(iter(early.items()))
    signed = copy.deepcopy(rows[0])
    signed["actions"].append(
        {
            "organization__name": "Executive",
            "description": "Signed by the Governor (fixture)",
            "date": DAYS[2],
            "classification": ["executive-signature"],
            "order": 9999,
        }
    )
    _write_day(root, DAYS[2], {("GA", "2025_26"): [deposits], (state, session): [signed]})
    return root


def sessions_csv(root: Path) -> Path:
    path = root / "sessions.csv"
    path.write_text((COMMISSION / "sessions.csv").read_text() + GEORGIA_SESSION)
    return path


def rebuild(days_root: Path, upto: int, root: Path) -> Path:
    """The reference: a full build over the first `upto` days, the semantic stage's pairs
    included; returns its tables dir."""
    root.mkdir(parents=True)
    folded = fold(load_days(days_root)[:upto])
    raw = write_archives(folded, root / "raw")
    built = tables_from_raw(root, raw, sessions_csv(root))
    tables = embed_with_stand_in(built, root / "embedded")
    build_semantic_pairs(tables, model=FAKE_MODEL, floor_bytes=0)
    return tables
