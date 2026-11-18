"""CI check: no module in the package is longer than LIMIT lines, but those allow-listed here.

Ruled at M13's step 1 review: 800 lines. Incremental and deploy were allowed until step 5's
lanes split them; the list is empty now. An allowed module back under the limit is an error,
so the list only shrinks.

    uv run python -m tools.module_size
"""

import sys
from collections.abc import Mapping
from pathlib import Path

LIMIT = 800
ALLOWED: dict[str, str] = {}


def problems(
    root: Path, *, limit: int = LIMIT, allowed: Mapping[str, str] = ALLOWED, package: str = "src"
) -> list[str]:
    """Each module over the limit and not allowed, and each allowed module now within it."""
    out: list[str] = []
    for path in sorted((root / package).rglob("*.py")):
        name = path.relative_to(root).as_posix()
        lines = len(path.read_text().splitlines())
        if lines > limit and name not in allowed:
            out.append(f"{name}: {lines} lines, over the limit of {limit}")
        elif lines <= limit and name in allowed:
            out.append(f"{name}: {lines} lines, within the limit; take it off ALLOWED")
    missing = [n for n in allowed if not (root / n).exists()]
    out += [f"{n}: allowed but not found; take it off ALLOWED" for n in missing]
    return out


def main() -> int:
    found = problems(Path.cwd())
    for p in found:
        print(p, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
