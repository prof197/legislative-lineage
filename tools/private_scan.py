"""Dev and CI check: no tracked file and no commit names a private person or a home directory.

The names to refuse come from the PRIVATE_NAMES environment variable, comma-separated, and are
matched without regard to case; in CI they come from a repository secret, so this file never
contains them. Home-directory paths (/Users/<name>/, /home/<name>/) are matched as written,
because a URL such as `/Home/GetObject` is not one. With `--history`, every commit reachable
from any ref is scanned too: its diff, its message, and its author and committer. A match is
reported by place and kind only, never by the text matched, so the log does not repeat it.

    PRIVATE_NAMES=... uv run python -m tools.private_scan --history
"""

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

HOME = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")


def names_pattern(names: Iterable[str]) -> re.Pattern[str] | None:
    words = [n.strip() for n in names if n.strip()]
    if not words:
        return None
    return re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)


def findings(text: str, names: re.Pattern[str] | None) -> Iterator[tuple[int, str]]:
    """(line number, kind) for each line that names a private person or a home directory."""
    for n, line in enumerate(text.splitlines(), start=1):
        if names is not None and names.search(line):
            yield n, "private name"
        if HOME.search(line):
            yield n, "home-directory path"


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], check=True, capture_output=True)
    return out.stdout.decode("utf-8", errors="replace")


def scan_tree(names: re.Pattern[str] | None) -> list[str]:
    problems: list[str] = []
    for name in filter(None, _git("ls-files", "-z").split("\0")):
        if (names is not None and names.search(name)) or HOME.search(name):
            problems.append(f"{name}: the path itself")
        path = Path(name)
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        problems += [f"{name}:{n}: {kind}" for n, kind in findings(text, names)]
    return problems


def scan_history(names: re.Pattern[str] | None) -> list[str]:
    problems: list[str] = []
    log = _git("log", "--all", "-p", "--format=commit %H%n%an <%ae>%n%cn <%ce>%n%B")
    commit = ""
    for line in log.splitlines():
        if line.startswith("commit ") and len(line) == 47:
            commit = line[7:19]
            continue
        problems += [f"commit {commit}: {kind}" for _, kind in findings(line, names)]
    return sorted(set(problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan for private names and home paths.")
    parser.add_argument("--history", action="store_true", help="scan every commit too")
    parser.add_argument(
        "--require-names", action="store_true", help="fail when PRIVATE_NAMES is not set"
    )
    args = parser.parse_args(argv)
    names = names_pattern(os.environ.get("PRIVATE_NAMES", "").split(","))
    if names is None and args.require_names:
        print("PRIVATE_NAMES is not set, so names cannot be checked", file=sys.stderr)
        return 2
    problems = scan_tree(names) + (scan_history(names) if args.history else [])
    for problem in problems:
        print(problem, file=sys.stderr)
    checked = "names and home paths" if names is not None else "home paths only"
    print(f"{len(problems)} problems ({checked}; history: {'yes' if args.history else 'no'})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
