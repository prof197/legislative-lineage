"""Build the commission recall fixture: the enacting bills among distractors, in archive layout.

Dev only: it reads the local snapshot tables, which CI does not have. The pipeline regression
test (`tests/test_commission.py`) runs every pipeline command on what this writes. Run with
`uv run python -m evals.fixture`, after `lineage index`.

- Queries: per act, the earliest enacted state version with text, since act text cannot be
  committed (DESIGN.md#commission-recall).
- Targets: every bill an in-window enactment names, text or not, except bills over
  `MAX_TARGET_CHARS`, which are omnibus bills too large for a committed fixture.
- Distractors: per archive, the `DISTRACTORS` bills titled as some other uniform act, chosen by
  the hash of their id, so a rebuild picks the same ones. Uniform acts share drafting
  conventions and closing sections, which makes them hard negatives, and they are
  low-controversy.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import duckdb

from lineage import queries
from lineage.commission import FIRST_YEAR
from lineage.sessions import load_sessions
from lineage.stats import AMENDMENT_CHARS, AMENDMENT_OPENING, AMENDMENT_SUBSTITUTE
from lineage.ulc import BillRecord, find_bills, load_enactments

TABLES = Path("data/tables")
SESSIONS = Path("snapshot/sessions.csv")
OUT = Path("tests/fixtures/commission")
MAX_TARGET_CHARS = 40_000
DISTRACTORS = 1
DISTRACTOR_CHARS = (2_000, 20_000)
_ACT_TITLES = ("%public expression%", "%special deposit%", "%estate planning%")
# Committed examples stay low-controversy (CLAUDE.md conventions): some laws named "uniform"
# are criminal, labor or staffing policy rather than commission acts on commercial or civil
# procedure. Each pick is also read by hand when the fixture is rebuilt.
_CONTESTED = (
    "%substance%",
    "%labor%",
    "%worker%",
    "%staffing%",
    "%firearm%",
    "%weapon%",
    "%abortion%",
    "%election%",
    "%immigra%",
    "%police%",
    "%crime%",
    "%custod%",
)
_FIELDS = (
    "id, session AS legislative_session, identifier, title, chamber, classification, "
    "actions, versions, raw_text, raw_text_url"
)


def _hash(bill_id: str) -> str:
    return hashlib.sha256(bill_id.encode()).hexdigest()


def main() -> None:
    snapshot = load_enactments()
    sessions = load_sessions(SESSIONS)
    years = {
        k: (s.start_date.year, s.end_date.year)
        for k, s in sessions.items()
        if s.start_date is not None and s.end_date is not None
    }
    counted = [
        r
        for r in snapshot.rows
        if r.stratum is not None and r.state is not None and r.year >= FIRST_YEAR
    ]
    bills_glob = str(TABLES / "bills" / "*.parquet")
    with duckdb.connect() as con:
        bills = queries.enactment_bills(
            con,
            bills_glob=bills_glob,
            eligible_path=TABLES / "eligible.parquet",
            families_path=TABLES / "families" / "families.parquet",
            states=sorted({r.state for r in counted if r.state is not None}),
            opening=AMENDMENT_OPENING,
            chars=AMENDMENT_CHARS,
            substitute=AMENDMENT_SUBSTITUTE,
        )
        records = [
            BillRecord(b.id, b.state, b.session, b.identifier, b.status, b.status_date,
                       b.passage_chambers)
            for b in bills
        ]  # fmt: skip
        by_id = {b.id: b for b in bills}
        named = {r: find_bills(r, records, years).bills for r in counted}
        targets = {b.id for found in named.values() for b in found}
        kept = {i for i in targets if by_id[i].text_chars <= MAX_TARGET_CHARS}
        omitted = sorted(targets - kept)

        query_bills: dict[str, str] = {}
        for act in sorted({r.act for r in counted}):
            passed = [
                by_id[b.id]
                for r, found in named.items()
                if r.act == act
                for b in found
                if b.id in kept and b.status == "passed" and by_id[b.id].text_chars > 0
            ]
            first = min(passed, key=lambda b: (b.status_date, b.id))
            query_bills[act] = first.id

        archives = sorted(
            {
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT archive FROM read_parquet($g) WHERE id IN (SELECT unnest($i))",
                    {"g": bills_glob, "i": sorted(kept)},
                ).fetchall()
            }
        )
        title_filter = " AND ".join(f"title NOT ILIKE '{t}'" for t in _ACT_TITLES + _CONTESTED)
        candidates: list[tuple[str, str, str]] = con.execute(
            f"""
            SELECT archive, id, title FROM read_parquet($g)
            WHERE archive IN (SELECT unnest($a)) AND title ILIKE '%uniform%act%' AND {title_filter}
              AND length(trim(coalesce(raw_text, ''))) BETWEEN $lo AND $hi
            """,
            {"g": bills_glob, "a": archives, "lo": DISTRACTOR_CHARS[0], "hi": DISTRACTOR_CHARS[1]},
        ).fetchall()
        distractors: set[str] = set()
        for archive in archives:
            pool = sorted((c for c in candidates if c[0] == archive and c[1] not in targets),
                          key=lambda c: _hash(c[1]))  # fmt: skip
            for _, bill_id, title in pool[:DISTRACTORS]:
                distractors.add(bill_id)
                print(f"distractor {archive}: {title[:90]}")

        wanted = sorted(kept | distractors)
        cursor = con.execute(
            f"SELECT archive, generated_at, state, {_FIELDS} FROM read_parquet($g) "
            "WHERE id IN (SELECT unnest($i)) ORDER BY archive, id",
            {"g": bills_glob, "i": wanted},
        )
        columns = [d[0] for d in cursor.description]
        rows: list[dict[str, Any]] = [dict(zip(columns, r, strict=True)) for r in cursor.fetchall()]

    if OUT.exists():
        shutil.rmtree(OUT)
    by_archive: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_archive.setdefault(row["archive"], []).append(row)
    used_sessions: set[tuple[str, str]] = set()
    for archive, members in by_archive.items():
        state, session = members[0]["state"], members[0]["legislative_session"]
        used_sessions.add((state, session))
        folder = OUT / f"{state}_{session}"
        folder.mkdir(parents=True)
        (folder / "README").write_text(
            "Open States Data Export\n\n"
            f"State: {state}\nSession: {session}\n"
            f"Generated At: {members[0]['generated_at']}\nJSON Format Version: 1.0\n"
        )
        records_out = [
            {k: v for k, v in m.items() if k not in ("archive", "generated_at", "state")}
            for m in members
        ]
        (folder / f"{state}_{session}_bills.json").write_text(
            json.dumps(records_out, indent=1, ensure_ascii=False, default=str) + "\n"
        )
        print(f"{archive}: {len(members)} bills")

    source = SESSIONS.read_text().splitlines()
    header = next(i for i, line in enumerate(source) if line.startswith("state,"))
    keep = [line for line in source[header + 1 :] if tuple(line.split(",", 2)[:2]) in used_sessions]
    (OUT / "sessions.csv").write_text(
        "# Session dates for the commission fixture, copied from snapshot/sessions.csv.\n"
        + source[header] + "\n" + "\n".join(keep) + "\n"
    )  # fmt: skip
    (OUT / "build.json").write_text(
        json.dumps(
            {
                "command": "uv run python -m evals.fixture",
                "queries": query_bills,
                "targets": len(kept),
                "omitted_over_max_chars": omitted,
                "max_target_chars": MAX_TARGET_CHARS,
                "distractors": len(distractors),
            },
            indent=1,
        )
        + "\n"
    )
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(kept)} targets, {len(distractors)} distractors, {len(omitted)} omitted, "
          f"{size:,} bytes -> {OUT}")  # fmt: skip


if __name__ == "__main__":
    main()
