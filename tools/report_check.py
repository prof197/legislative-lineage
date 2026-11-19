"""Dev check: the offline evals, run again, leave every committed report byte for byte as it is.

A change meant to move code and not behavior (M13's restructure) is held to this after every
commit: each eval below reads the built tables and the committed labels, asks no model and
touches no network, and rewrites its report. Any diff in the tracked outputs is a behavior
change. Left out: the sealed keys (`--final` is never passed, and `lineage eval origins`
runs only the development key unless asked); `eval cost`, which times queries; and the two
readings taken before generic sections left origin matching and kept as read
(DESIGN.md#origin-matches), `eval origin-threshold`'s proposal and `eval origin-census`'s EPPC
census, which a rerun against today's matches is meant to change.

    uv run python -m tools.report_check            # every eval
    uv run python -m tools.report_check --from 8   # from the ninth, after a fix

Exits 1 and names the files that moved; `git checkout -- <file>` restores them.
"""

import subprocess
import sys
import time

# The outputs the evals write. Any change under them after a run is a failure.
OUTPUTS = ("reports", "README.md", "evals")

EVALS: tuple[tuple[str, ...], ...] = (
    ("eval", "commission"),
    ("eval", "origins"),
    ("eval", "origins", "--key", "v1_held_out"),
    ("eval", "precision"),
    ("eval", "relatives"),
    ("eval", "relatives", "--part", "tuning_093"),
    ("eval", "threshold"),
    ("eval", "agreement"),
    ("eval", "faithfulness", "evals/featured_pairs.jsonl"),
    ("eval", "pair-data"),
    ("eval", "answer-key"),
    ("eval", "answer-key", "--held-out"),
    ("eval", "answer-key", "--held-out", "--origins"),
    ("eval", "readme"),
)


def changed(paths: tuple[str, ...] = OUTPUTS) -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], capture_output=True, text=True, check=True
    ).stdout
    return [line[3:] for line in out.splitlines()]


def main(argv: list[str]) -> int:
    start_at = int(argv[argv.index("--from") + 1]) if "--from" in argv else 0
    before = changed()
    if before:
        print(f"outputs already differ from the commit: {', '.join(before)}", file=sys.stderr)
        return 2
    for args in EVALS[start_at:]:
        start = time.monotonic()
        run = subprocess.run(["uv", "run", "lineage", *args], capture_output=True, text=True)
        took = time.monotonic() - start
        if run.returncode != 0:
            print(f"lineage {' '.join(args)} failed ({run.returncode}):", file=sys.stderr)
            print(run.stderr[-2000:], file=sys.stderr)
            return 1
        moved = changed()
        note = f"  MOVED: {', '.join(moved)}" if moved else ""
        print(f"{took:6.1f}s  lineage {' '.join(args)}{note}", flush=True)
        if moved:
            return 1
    print(f"{len(EVALS) - start_at} evals rerun; every report unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
