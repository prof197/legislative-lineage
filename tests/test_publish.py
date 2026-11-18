"""The incremental export (DESIGN.md#updates, Publishing): after each recorded day, the export
the day's change log rewrote equals a full export of the same tables, and only the files that
differ, with the pages the log names, were written; every other file is the same file."""

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from lineage.core import db
from lineage.export.pages import bill_refs
from lineage.export.schema import BillPage
from lineage.live.publish import (
    PublishError,
    changes_on,
    export_changes,
    promote,
    stage,
    staging_dir,
    touched,
)
from lineage.live.updates import RecordUpdated, RelativeGained, RelativeLost
from tests.days import DAYS
from tests.live import Recorded, export_inputs, files_seen, full_export, normalized, recorded

D1, D2, D3 = (date.fromisoformat(d) for d in DAYS)


@pytest.fixture(scope="module")
def rec(tmp_path_factory: pytest.TempPathFactory) -> Recorded:
    return recorded(tmp_path_factory)


def _named_pages(rec: Recorded, n: int, export: Path) -> set[str]:
    """The pages day n's log names: the bill's own, or one listing a record the day changed."""
    log = changes_on(rec.log, date.fromisoformat(DAYS[n - 1]))
    named = {c.bill_id for c in log} | {
        c.relative_id for c in log if isinstance(c, RelativeGained | RelativeLost)
    }
    records = {c.bill_id for c in log if c.kind in ("added", "updated", "replaced")}
    with db.connect_quiet() as con:
        refs, _ = bill_refs(con, rec.tables[n], sorted(named))
    named_slugs = {refs[b].slug for b in named}
    record_slugs = {refs[b].slug for b in records}
    out: set[str] = set()
    for page in sorted(export.glob("bills/*.json")):
        doc = BillPage.model_validate_json(page.read_text())
        if doc.bill.slug in named_slugs or any(r.bill.slug in record_slugs for r in doc.relatives):
            out.add(f"bills/{page.name}")
    return out


@pytest.mark.parametrize(("state", "first"), [("GA", 2), ("UT", 3)])
def test_only_the_files_a_day_changes_are_rewritten_and_the_export_equals_a_full_one(
    rec: Recorded, tmp_path: Path, state: str, first: int
) -> None:
    inputs = export_inputs(tmp_path / "inputs")
    full = {
        n: full_export(rec.tables[n], tmp_path / f"full{n}" / state, inputs, state)
        for n in range(first - 1, 4)
    }
    live = tmp_path / "export" / state
    shutil.copytree(full[first - 1], live)

    for n in range(first, 4):
        day = date.fromisoformat(DAYS[n - 1])
        before = files_seen(live)
        report = export_changes(
            rec.tables[n],
            state=state,
            export=live,
            changes_dir=rec.log_through(n, tmp_path / f"log{n}"),
            since=day if n == first else None,  # a full export carries no as-of date
            models_dir=None,
            floor=0,
            **inputs,
        )
        assert files_seen(live) == before  # nothing reaches the live export before the swap
        promote(live)
        after = files_seen(live)

        assert normalized(live) == normalized(full[n])
        changed = {k for k in after if k not in before or after[k].data != before[k].data}
        assert changed == set(report["written"])
        assert set(report["removed"]) == set(before) - set(after)
        for name in set(after) - changed:  # untouched: the same file, not a copy of it
            assert (after[name].inode, after[name].mtime_ns) == (
                before[name].inode,
                before[name].mtime_ns,
            )
        old, new = normalized(full[n - 1]), normalized(full[n])
        differ = {k for k in new if old.get(k) != new[k]}
        named = _named_pages(rec, n, live)
        assert changed == differ | named | {"meta.json"}
        assert report["beyond_log"] == []
        for name in after:
            if name.startswith("bills/"):
                stamp = json.loads(after[name].data)["updated_on"]
                assert (stamp == day.isoformat()) == (name in named)
        assert json.loads(after["meta.json"].data)["as_of"] == day.isoformat()
        if state == "GA":  # Utah SB 18 joins the hosted page on day 2 and leaves it on day 3
            utah = "compare/ga-2025_26-sb-9001--ut-2023-sb-18.json"
            assert utah in (report["written"] if n == 2 else report["removed"])

    if state == "GA":  # the recorded days do what they say on the hosted page
        page = next(iter(live.glob("bills/*.json")))
        assert json.loads(page.read_text())["updated_on"] == DAYS[2]


def test_the_log_names_both_sides_of_a_relative_and_each_record_it_changed() -> None:
    log = [
        RelativeGained(as_of=D2, bill_id="a", relative_id="b", stages=["semantic"]),
        RecordUpdated(as_of=D2, bill_id="c"),
        RelativeLost(as_of=D3, bill_id="a", relative_id="d"),
    ]
    t = touched(log)
    assert t.bills == {"a": D3, "b": D2, "c": D2, "d": D3}
    assert t.records == {"c": D2}


def test_a_page_a_relative_moved_without_a_log_entry_is_rebuilt_and_reported(
    rec: Recorded, tmp_path: Path
) -> None:
    # A file the log does not name but the tables no longer give is still made right: here a
    # page and a comparison edited as if a bucket had pushed one section pair out.
    inputs = export_inputs(tmp_path / "inputs")
    live = full_export(rec.tables[2], tmp_path / "export" / "UT", inputs, "UT")
    target = full_export(rec.tables[3], tmp_path / "full3", inputs, "UT")
    named = _named_pages(rec, 3, live)
    utah = next(p for p in sorted(live.glob("bills/*.json")) if f"bills/{p.name}" not in named)
    doc = json.loads(utah.read_text())
    doc["words_searched"] += 1
    utah.write_text(json.dumps(doc))
    compared = live / "compare" / f"{doc['relatives'][0]['comparison']}.json"
    cmp = json.loads(compared.read_text())
    cmp["pairs"] = cmp["pairs"][1:]
    compared.write_text(json.dumps(cmp))

    report = export_changes(
        rec.tables[3],
        state="UT",
        export=live,
        changes_dir=rec.log_through(3, tmp_path / "log"),
        since=D3,
        models_dir=None,
        floor=0,
        **inputs,
    )
    promote(live)

    assert report["beyond_log"] == sorted([f"bills/{utah.name}", f"compare/{compared.name}"])
    assert normalized(live) == normalized(target)


def test_an_export_ahead_of_or_behind_the_tables_is_refused(rec: Recorded, tmp_path: Path) -> None:
    inputs = export_inputs(tmp_path / "inputs")
    live = full_export(rec.tables[1], tmp_path / "export" / "GA", inputs)
    log = rec.log_through(3, tmp_path / "log")
    with pytest.raises(PublishError, match="not had 2026-01-14 applied"):
        export_changes(
            rec.tables[2], state="GA", export=live, changes_dir=log, models_dir=None, **inputs
        )
    with pytest.raises(PublishError, match="not in"):
        export_changes(
            rec.tables[3],
            state="GA",
            export=live,
            changes_dir=rec.log_through(2, tmp_path / "short"),
            models_dir=None,
            **inputs,
        )


def test_staging_links_the_live_files_and_a_stopped_promote_is_finished(tmp_path: Path) -> None:
    live = tmp_path / "GA"
    (live / "bills").mkdir(parents=True)
    (live / "meta.json").write_text("{}")
    (live / "bills" / "a.json").write_text("{}")
    staged = stage(live)
    assert staged == staging_dir(live)
    assert (staged / "bills" / "a.json").stat().st_ino == (live / "bills" / "a.json").stat().st_ino

    live.rename(live.with_name("GA.old"))  # stopped between promote's two renames
    stage(live)
    assert (live / "meta.json").exists()
    promote(live)
    assert not staging_dir(live).exists()
    assert not live.with_name("GA.old").exists()
    with pytest.raises(PublishError, match="nothing to promote"):
        promote(live)
