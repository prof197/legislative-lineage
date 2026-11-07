"""Dev and CI check: no tracked file or commit carries a local path or a deny-listed term.

The pipeline writes reports that are committed, and a report that records where it read or
wrote a file can carry the path of the machine that ran it. A home directory (/Users/<user>/,
/home/<user>/) or a temporary directory (tmp at the root, or macOS's private/tmp and var/folders)
is matched as written, because a URL such as `/Home/GetObject` is not one.

Deny-listed terms come from the LEAK_CHECK_TERMS environment variable, comma-separated, and
are matched without regard to case; in CI they come from a repository secret, so this file
never lists them. With `--history`, every commit reachable from any ref is checked too: its
diff, its message, and its author and committer. A match is reported by place and kind only,
never by the text matched, so the log does not repeat it.

    LEAK_CHECK_TERMS=... uv run python -m tools.leak_check --history
"""

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

LOCAL_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+|(?:/private)?/(?:tmp|var/folders)/")


def terms_pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    words = [t.strip() for t in terms if t.strip()]
    if not words:
        return None
    return re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)


def findings(text: str, terms: re.Pattern[str] | None) -> Iterator[tuple[int, str]]:
    """(line number, kind) for each line that holds a deny-listed term or a local path."""
    for n, line in enumerate(text.splitlines(), start=1):
        if terms is not None and terms.search(line):
            yield n, "deny-listed term"
        if LOCAL_PATH.search(line):
            yield n, "local path"


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], check=True, capture_output=True)
    return out.stdout.decode("utf-8", errors="replace")


def check_tree(terms: re.Pattern[str] | None) -> list[str]:
    problems: list[str] = []
    for name in filter(None, _git("ls-files", "-z").split("\0")):
        if (terms is not None and terms.search(name)) or LOCAL_PATH.search(name):
            problems.append(f"{name}: the path itself")
        path = Path(name)
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        problems += [f"{name}:{n}: {kind}" for n, kind in findings(text, terms)]
    return problems


def check_history(terms: re.Pattern[str] | None) -> list[str]:
    problems: list[str] = []
    log = _git("log", "--all", "-p", "--format=commit %H%n%an <%ae>%n%cn <%ce>%n%B")
    commit = ""
    for line in log.splitlines():
        if line.startswith("commit ") and len(line) == 47:
            commit = line[7:19]
            continue
        problems += [f"commit {commit}: {kind}" for _, kind in findings(line, terms)]
    return sorted(set(problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check for local paths and deny-listed terms.")
    parser.add_argument("--history", action="store_true", help="check every commit too")
    parser.add_argument(
        "--require-terms", action="store_true", help="fail when LEAK_CHECK_TERMS is not set"
    )
    args = parser.parse_args(argv)
    terms = terms_pattern(os.environ.get("LEAK_CHECK_TERMS", "").split(","))
    if terms is None and args.require_terms:
        print("LEAK_CHECK_TERMS is not set, so the deny list cannot be checked", file=sys.stderr)
        return 2
    problems = check_tree(terms) + (check_history(terms) if args.history else [])
    for problem in problems:
        print(problem, file=sys.stderr)
    checked = "local paths and the deny list" if terms is not None else "local paths only"
    print(f"{len(problems)} problems ({checked}; history: {'yes' if args.history else 'no'})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
