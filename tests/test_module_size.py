"""tools.module_size: a module over the limit fails unless allowed, and an allowed one back
within it, or gone, fails too, so the allow-list only shrinks."""

from pathlib import Path

from tools.module_size import problems


def _module(root: Path, name: str, lines: int) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines)


def test_over_the_limit_fails_unless_allowed(tmp_path: Path) -> None:
    _module(tmp_path, "src/pkg/big.py", 11)
    _module(tmp_path, "src/pkg/ok.py", 10)
    assert problems(tmp_path, limit=10, allowed={}) == [
        "src/pkg/big.py: 11 lines, over the limit of 10"
    ]
    assert problems(tmp_path, limit=10, allowed={"src/pkg/big.py": "why"}) == []


def test_an_allowed_module_within_the_limit_or_gone_fails(tmp_path: Path) -> None:
    _module(tmp_path, "src/pkg/small.py", 5)
    got = problems(tmp_path, limit=10, allowed={"src/pkg/small.py": "", "src/pkg/gone.py": ""})
    assert got == [
        "src/pkg/small.py: 5 lines, within the limit; take it off ALLOWED",
        "src/pkg/gone.py: allowed but not found; take it off ALLOWED",
    ]


def test_the_package_as_committed_passes() -> None:
    assert problems(Path.cwd()) == []
