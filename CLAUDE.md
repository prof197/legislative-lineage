# CLAUDE.md

## Commands

```sh
uv sync                      # install the locked environment
uv run ruff check            # lint
uv run ruff format --check   # formatting (drop --check to apply)
uv run pyright               # type check, strict
uv run pytest                # tests
uv run lineage --help        # the CLI

uv run lineage fetch-raw     # download the archives the manifest pins into data/raw/
uv run lineage snapshot      # pin data/raw/*/*.zip into snapshot/manifest.json
uv run lineage ingest        # verify against the manifest, write data/tables/bills/*.parquet
```

All four checks must pass before every commit. CI runs the same four commands.

## Layout

```
src/lineage/      the package (import name `lineage`; distribution `legislative-lineage`)
  cli.py          Typer app; subcommands stay thin wrappers over library functions
  snapshot.py     archive manifest: scan, write, load, verify
  fetch.py        download pinned archives; hash-checked, paced, disk-guarded
  ingest.py       raw archives -> bills Parquet, one file per archive
  queries.py      every SQL query the pipeline runs (DR-4)
  disk.py         free-space floor checked before large writes
tests/            pytest; small committed fixtures only
  fixtures/openstates/  trimmed real records in archive layout (README + *_bills.json)
snapshot/manifest.json  the pinned snapshot: per-archive size, SHA-256, Generated At
DESIGN.md         design, alternatives, scaling limits, failure modes, decision records
data/             local snapshots and derived tables; gitignored, never committed
  raw/<ST>/*.zip  Open States session archives, never edited
  tables/         Parquet tables written by the pipeline
```

## Conventions

- Python 3.13, pyright strict, ruff. Dependencies are pinned in `uv.lock`. Add a dependency with `uv add`, and only after it is agreed.
- Small modules with explicit interfaces. Use plain functions, dataclasses and Pydantic models, not class hierarchies or metaprogramming.
- A module that implements a decision record links to it in its docstring, e.g. `See DESIGN.md#dr-3`.
- Comments explain why, not what.
- Write tests first wherever DESIGN.md or the build plan specifies the behavior.
- Never weaken a test, threshold or eval to get to green.
- Every number in README.md or DESIGN.md comes from a committed eval or benchmark output, with the command that reproduces it. Never hard-code a result.
- Fix random seeds. The snapshot is pinned by `snapshot/manifest.json`; an archive's as-of date is its README's Generated At.
- Before a large write, call `disk.require_free_space` with an upper bound on the bytes written. The default floor is 20 GiB free.
- Secrets come only from environment variables. `.env` is gitignored.
- Model calls happen only in the explanation step and the provisional precision labels. Retrieval runs with no API key.
- Commits use conventional messages, one build step per commit. The body gives the engineering reason: the tradeoff, the measurement, the alternative rejected.
