"""`lineage text merge`: each record's text, and where it came from, in the `texts` table.

A record takes the text the text layer fetched for it when the fetch kept usable text and the
current rules still call for it (lineage.text_run.plan_jobs); otherwise the text the source
exported. `texts/<name>.parquet` mirrors `bills/<name>.parquet`, one row per record, so the
splitter keeps its one worker per file. Later stages read text only from here, and each row
names its source, so coverage and recall can be counted both ways. See DESIGN.md#dr-6.
"""

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from lineage import queries
from lineage.ca_bulk import HOST as BULK_HOST
from lineage.disk import DEFAULT_FLOOR_BYTES, require_free_space
from lineage.text_inventory import PLAN
from lineage.text_rules import RULES, load_rules, replaced_states
from lineage.text_run import QUALITY, STORE, plan_jobs

# Values of texts.text_source.
EXPORTED = "openstates"
FETCHED = "legislature"
BULK = "ca_bulk"


def _latest_ok(store: Path, wanted: set[str]) -> Iterator[dict[str, str | None]]:
    """The latest stored row per wanted record, where that row kept usable text."""
    latest: dict[str, tuple[Path, int]] = {}
    for path in sorted(store.glob("*.jsonl")):
        with path.open() as f:
            for n, line in enumerate(f):
                row = json.loads(line)
                if row["id"] in wanted:
                    latest[row["id"]] = (path, n)
    by_file: dict[Path, set[int]] = {}
    for path, n in latest.values():
        by_file.setdefault(path, set()).add(n)
    for path, lines in sorted(by_file.items()):
        source = BULK if path.stem == BULK_HOST else FETCHED
        with path.open() as f:
            for n, line in enumerate(f):
                if n not in lines:
                    continue
                row = json.loads(line)
                if row["outcome"] == "ok":
                    yield {
                        "id": row["id"], "text": row["text"], "source": source,
                        "url": row["fetched_url"], "version": row["note"],
                        "date": row["date"] or None,
                    }  # fmt: skip


def merge_texts(
    tables_dir: Path,
    *,
    plan_path: Path = PLAN,
    store: Path = STORE,
    rules_path: Path = RULES,
    quality_path: Path = QUALITY,
    exported_only: bool = False,
    floor_bytes: int = DEFAULT_FLOOR_BYTES,
) -> dict[str, int]:
    """Rebuild `<tables>/texts/`; return the records per text source."""
    bills = sorted((tables_dir / "bills").glob("*.parquet"))
    if not bills:
        raise FileNotFoundError(f"no bills table in {tables_dir}; run `lineage ingest` first")
    building = Path(tempfile.mkdtemp(prefix="texts.", dir=tables_dir))
    final = tables_dir / "texts"
    try:
        fetched = building / "_fetched.jsonl"
        fetched.touch()
        if not exported_only and plan_path.exists() and store.exists():
            rules = load_rules(rules_path)
            replace = set(replaced_states(json.loads(quality_path.read_text()), rules))
            with queries.connect_quiet() as con:
                plan = queries.fetch_plan(con, path=plan_path)
            jobs, bulk, _ = plan_jobs(plan, rules, replace)
            with fetched.open("w") as out:
                for row in _latest_ok(store, {j.id for j in jobs} | set(bulk)):
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        with queries.connect_quiet() as con:
            text = queries.text_bytes(con, bills_glob=str(tables_dir / "bills" / "*.parquet"))
            require_free_space(tables_dir, needed=text + fetched.stat().st_size, floor=floor_bytes)
            for path in bills:
                queries.write_texts(
                    con, bills_path=path, fetched_path=fetched, out_path=building / path.name
                )
            fetched.unlink()
            counts = queries.text_sources(con, texts_glob=str(building / "*.parquet"))
        if final.exists():
            shutil.rmtree(final)
        building.rename(final)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return counts
