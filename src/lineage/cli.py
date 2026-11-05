"""The `lineage` command line. Each subcommand is a thin wrapper over a library function."""

from typing import Annotated

import typer

from lineage import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"lineage {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            # Eager so `--version` answers before Typer demands a subcommand.
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Trace provisions shared between state bills."""
