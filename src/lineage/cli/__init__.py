"""`lineage`, the command line: thin wrappers over library functions. The Typer apps are in app;
each layer's commands are in a module named for it, imported here so they register.
See DESIGN.md#module-map.
"""

from lineage.cli.app import app

# Registered in pipeline order, which is the order --help lists the commands in.
# isort: off
from lineage.cli import data, text, match, explain, evals, origins, export, live
# isort: on

__all__ = ["app", "data", "evals", "explain", "export", "live", "match", "origins", "text"]
