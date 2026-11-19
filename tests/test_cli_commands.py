"""The CLI's command tree, pinned: every command path and each of its parameters (names, kind,
type, whether required, flag, count, help and default). Splitting or moving the CLI's modules
must leave it as it is; a command that changes on purpose regenerates the fixture with
LINEAGE_REGENERATE=1.
"""

import json
import os
from pathlib import Path
from typing import Any

import typer

from lineage.cli import app
from lineage.core.paths import DATA

FIXTURE = Path(__file__).parent / "fixtures" / "cli_commands.json"


def _default(value: Any) -> str | None:
    """A default as text, a path under the data directory or the repository written relative to
    it, so the fixture is the same whether LINEAGE_DATA is set or not, and on any machine."""
    if value is None or callable(value):
        return None
    if isinstance(value, Path):
        path = (value if value.is_absolute() else Path.cwd() / value).resolve()
        for root, name in ((DATA.resolve(), "<data>"), (Path.cwd().resolve(), "<repo>")):
            if path.is_relative_to(root):
                return f"{name}/{path.relative_to(root).as_posix()}"
        return str(value)
    return str(value)


def _params(command: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": p.name,
            "kind": type(p).__name__,
            "opts": [*p.opts, *p.secondary_opts],
            "type": p.type.name,
            "required": p.required,
            "flag": bool(getattr(p, "is_flag", False)),
            "multiple": p.multiple,
            "help": getattr(p, "help", None),
            "default": _default(p.default),
        }
        for p in command.params  # Typer's vendored click: read by duck typing
    ]


def tree(command: Any, path: str = "lineage") -> dict[str, Any]:
    out: dict[str, Any] = {path: {"help": command.help, "params": _params(command)}}
    for name, sub in getattr(command, "commands", {}).items():
        out |= tree(sub, f"{path} {name}")
    return out


def test_the_command_tree_is_pinned() -> None:
    got = tree(typer.main.get_command(app))
    if os.environ.get("LINEAGE_REGENERATE") == "1":
        FIXTURE.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    assert got == json.loads(FIXTURE.read_text())
