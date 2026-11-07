# Contributing

Before every commit, run the four checks CI runs:

```sh
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

## Leak check

Several pipeline commands write reports under `reports/` that are committed. A report that records where it read or wrote a file can carry the path of the machine that ran it. One report written from a temporary directory did, and the path was caught by hand before it was committed. `tools/leak_check.py` catches it automatically. It fails on:

- **local paths:** home directories (`/Users/…`, `/home/…`) and temporary directories (`tmp` at the root, and macOS's `private/tmp` and `var/folders`), matched as written;
- **deny-listed terms:** a comma-separated list in `LEAK_CHECK_TERMS`, matched in any case.

It checks every tracked file and, with `--history`, every commit's diff, message, author and committer. It reports a match by file or commit and kind, never by the matched text. Run it before every push:

```sh
LEAK_CHECK_TERMS=... uv run python -m tools.leak_check --history
```

CI runs the same check on every push and pull request, with the deny list from the `LEAK_CHECK_TERMS` repository secret. GitHub gives no secrets to runs in a fork or to pull requests from one, so there the step is skipped with a notice, and the check runs once the change reaches this repository. Here the step fails while the secret is unset, so a missing secret can't pass the check on local paths alone.
