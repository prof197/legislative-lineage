"""CI check: every top-level name the package defines is used by code outside its definition.

A name counts as used when other code reads it: a name, an attribute, an import, or a string
that is exactly the name (the collector registry looks names up with getattr). The package's
own modules, the benchmarks, evals/ and tools/ are the code read; the tests are not, so a
function only its tests call is reported. Commands and callbacks registered by a decorator,
names in an `__all__` and dunders are used by definition. A name kept on purpose goes in
dead_code_allow.txt as `module:name  why`; an allowed name that is used, or gone, is an error.

    uv run python -m tools.dead_code
"""

import ast
import sys
from collections import Counter
from pathlib import Path

ALLOW = Path(__file__).with_name("dead_code_allow.txt")
READ = ("src", "benchmarks", "evals", "tools")


def _defined(tree: ast.Module) -> list[tuple[str, bool]]:
    """Each top-level name and whether an assignment (a Name, so a use to discount) made it."""
    exported = {
        e.value
        for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in n.targets)
        for e in getattr(n.value, "elts", [])
        if isinstance(e, ast.Constant)
    }
    out: list[tuple[str, bool]] = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef | ast.ClassDef):
            decorators = [ast.unparse(d) for d in n.decorator_list]
            if not any(".command" in d or ".callback" in d for d in decorators):
                out.append((n.name, False))
        elif isinstance(n, ast.Assign):
            out += [(t.id, True) for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.append((n.target.id, True))
    return [(x, a) for x, a in out if not x.startswith("__") and x not in exported]


def _uses(tree: ast.AST) -> Counter[str]:
    used: Counter[str] = Counter()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            used[n.id] += 1
        elif isinstance(n, ast.Attribute):
            used[n.attr] += 1
        elif isinstance(n, ast.alias):
            used[n.name.rpartition(".")[2]] += 1
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
            used[n.value] += 1
    return used


def unused(root: Path, package: str = "src/lineage") -> list[str]:
    """`module:name` for each top-level name nothing reads, by its module's path."""
    used: Counter[str] = Counter()
    defined: list[tuple[str, str, bool]] = []
    for where in READ:
        for path in sorted((root / where).rglob("*.py")):
            tree = ast.parse(path.read_text())
            used += _uses(tree)
            if path.is_relative_to(root / package):
                module = path.relative_to(root / "src").with_suffix("").as_posix()
                defined += [(module.replace("/", "."), n, a) for n, a in _defined(tree)]
    # A definition is not a use: an assignment's target is itself one Name.
    return [f"{m}:{n}" for m, n, assigned in defined if used[n] - assigned < 1]


def allowed(path: Path = ALLOW) -> dict[str, str]:
    rows = [line.split(None, 1) for line in path.read_text().splitlines() if line.strip()]
    return {r[0]: r[1] if len(r) > 1 else "" for r in rows if not r[0].startswith("#")}


def problems(root: Path, allow: dict[str, str]) -> list[str]:
    found = set(unused(root))
    out = [
        f"{x}: nothing reads it; remove it or allow it in {ALLOW.name}"
        for x in sorted(found - allow.keys())
    ]
    return out + [
        f"{x}: allowed but read, or gone; take it off {ALLOW.name}"
        for x in sorted(allow.keys() - found)
    ]


def main() -> int:
    found = problems(Path.cwd(), allowed())
    for p in found:
        print(p, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
