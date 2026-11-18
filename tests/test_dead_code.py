"""tools.dead_code: a top-level name nothing outside its definition reads is reported, the tests'
reads do not count, and the allow-list neither hides a used name nor outlives a removed one."""

from pathlib import Path

from tools.dead_code import problems, unused


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


PACKAGE = {
    "src/lineage/__init__.py": "",
    "src/lineage/a.py": (
        "import typer\n"
        "app = typer.Typer()\n"
        "LIMIT = 3\n"
        "UNUSED = 4\n"
        "SOURCE = 'a source'\n"
        "__all__ = ['exported']\n"
        "def used() -> int:\n    return LIMIT\n"
        "def only_tests() -> None: ...\n"
        "def exported() -> None: ...\n"
        "@app.command()\ndef cmd() -> None: ...\n"
    ),
    "src/lineage/b.py": "from lineage.a import used\nx = getattr(object(), 'SOURCE', used())\n",
    "tests/test_a.py": "from lineage.a import only_tests\nonly_tests()\n",
}


def test_names_nothing_reads_are_reported_and_tests_do_not_count(tmp_path: Path) -> None:
    root = _tree(tmp_path, PACKAGE)
    assert sorted(unused(root)) == ["lineage.a:UNUSED", "lineage.a:only_tests", "lineage.b:x"]


def test_the_allow_list_keeps_a_name_and_refuses_one_read_or_gone(tmp_path: Path) -> None:
    root = _tree(tmp_path, PACKAGE)
    allow = {
        "lineage.a:only_tests": "a test hook",
        "lineage.b:x": "",
        "lineage.a:used": "",
        "lineage.a:gone": "",
    }
    got = problems(root, allow)
    assert any(p.startswith("lineage.a:UNUSED: nothing reads it") for p in got)
    assert any(p.startswith("lineage.a:used: allowed but read") for p in got)
    assert any(p.startswith("lineage.a:gone: allowed but read, or gone") for p in got)
    assert not any(p.startswith(("lineage.a:only_tests", "lineage.b:x")) for p in got)


def test_the_package_as_committed_passes() -> None:
    from tools.dead_code import allowed

    assert problems(Path.cwd(), allowed()) == []
