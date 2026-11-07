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
uv run lineage sessions --commit <sha>  # pin session dates into snapshot/sessions.csv
uv run lineage ingest        # verify against the manifest, write data/tables/bills/*.parquet
uv run lineage text merge    # each record's text, fetched or exported -> data/tables/texts/*.parquet (--exported-only: the export alone)
uv run lineage split         # split the texts table into data/tables/sections/*.parquet
uv run lineage eligible      # document-type rules -> data/tables/eligible.parquet, reports/document_types.json
uv run lineage boilerplate   # shingle stoplist -> data/tables/stoplist.parquet, reports/boilerplate.json
uv run lineage stats         # coverage, sections, types, status per state/session -> reports/coverage.json
uv run lineage stats --samples  # also: sample titles per document-type rule -> data/reports/ (local; names private people, never committed)
uv run lineage index         # signatures, LSH bands, verified pairs, families -> data/tables/{signatures,bands,pairs,families}/, reports/index.json
uv run lineage query FILE     # a bill's near-duplicate families, offline, from the built index
uv run lineage eval commission  # commission recall, act texts from data/acts/ -> reports/results.json (--results reports/results_exported_only.json after an --exported-only merge)
uv run lineage eval answer-key  # each commission list row and the bills it resolved to -> reports/answer_key.csv
uv run lineage eval readme   # regenerate README.md's results table from reports/results.json
uv run lineage text inventory  # version to fetch per record, by host, robots.txt read -> data/text/plan.parquet, reports/text_inventory.json
uv run lineage text sample   # quality test: fetch a seeded sample per state and format -> data/text/sample/, reports/text_quality.json
uv run lineage text fetch    # fetch every planned record as rules/text_layer.toml says, California from its bulk download; resumable -> data/text/full/, reports/text_fetch.json
uv run python -m evals.fixture  # rebuild tests/fixtures/commission/ from the local snapshot, dev only
uv run python -m benchmarks.minhash  # NumPy vs datasketch: signing and LSH index time and memory (DR-3), dev only
```

All four checks must pass before every commit. CI runs the same four commands.

## Layout

```
src/lineage/      the package (import name `lineage`; distribution `legislative-lineage`)
  cli.py          Typer app; subcommands stay thin wrappers over library functions
  snapshot.py     archive manifest: scan, write, load, verify
  fetch.py        HTTP as this project (User-Agent, TLS intermediates, retired hosts); download pinned archives
  sessions.py     session dates from openstates-scrapers, parsed as source, never run
  ingest.py       raw archives -> bills Parquet, one file per archive, with status
  status.py       passed / died / introduced from action classifications
  sections.py     clean, split at headings or into windows, normalize (pure functions)
  sectioning.py   build the sections table, one worker process per bills file
  doctypes.py     document-type rules (DR-5): which records are matched
  boilerplate.py  pattern list that flags boilerplate-only sections
  shingles.py     word 5-gram shingles, 32-bit hashes (NumPy; values pinned by a test)
  minhash.py      MinHash signatures, exact (a·x + b) mod p (pure functions; values pinned)
  signatures.py   build the signatures table, one worker process per sections file
  lsh.py          LSH banding: band keys, sorted-array buckets, the cap (pure functions)
  bands.py        build the band index from the signatures; cap counter and cap sweep
  verify.py       re-estimate candidates from full signatures; the verification threshold
  families.py     same-state families: bill match shares, connected components (pure functions)
  pairs.py        verify candidates, store cross-state pairs, write families
  index.py        `lineage index`: signatures, bands, pairs and families, then reports/index.json
  query.py        `lineage query`: sign a bill's text, look up its bands, verify, group by family
  acts.py         uniform act texts as eval queries: keep the sections, drop comments and notes
  ulc.py          the commission's enactment lists, and the bills each enactment names
  answer_key.py   `lineage eval answer-key`: the resolved list rows, for checking by eye
  commission.py   `lineage eval commission`: recall, a miss reason per enactment, the results JSON
  results_table.py  the README results table, rendered from reports/results.json
  stoplist.py     document-frequency stoplist and its report
  stats.py        coverage report: the measurements that replace the design's estimates
  versions.py     which bill version the text layer fetches (pure functions)
  robots.py       robots.txt rules for this project's User-Agent
  text_inventory.py  `lineage text inventory`: unusable text per record, the fetch plan, per-host counts
  certs/intermediates.pem  intermediate certificates some legislature servers omit
  extract.py      text from HTML or PDF (pypdf), lines kept, struck text dropped
  text_fetch.py   polite per-host fetching: robots, pacing, backoff, resumable JSON-lines store; extraction in worker processes
  text_rules.py   rules/text_layer.toml: the usable-text bar, per-state formats and readings, the replacement rule
  text_quality.py  `lineage text sample`: the quality test, fetched text against Open States'
  text_run.py     `lineage text fetch`: the full fetch's jobs from the plan and the rules, and its report
  ca_bulk.py      California's text from its official bulk download, one biennium at a time
  text_merge.py   `lineage text merge`: the texts table, each record's text and its source; later stages read text only from it
  queries.py      every SQL query the pipeline runs (DR-4)
  disk.py         free-space floor checked before large writes
benchmarks/       dev-only measurements that need dev dependencies (datasketch)
  memory.py       current and peak memory as the OS charges it (macOS physical footprint)
evals/            eval inputs: ulc_enactments.csv (hand copy of the commission's lists, dated)
  fixture.py      dev-only: builds tests/fixtures/commission/ from the snapshot
tests/            pytest; small committed fixtures only
  fixtures/openstates/  trimmed real records in archive layout (README + *_bills.json)
  fixtures/commission/  enacting bills among uniform-act distractors, same layout, + sessions.csv
snapshot/manifest.json  the pinned snapshot: per-archive size, SHA-256, Generated At
snapshot/sessions.csv   session start/end dates and carryover, from a pinned scrapers commit
rules/document_types.csv  which document types are matched, per state; first fitting rule wins
rules/text_layer.toml  the text layer's rules: usable bar, per-state format and PDF reading, content elements, replacement
reports/          committed outputs of pipeline commands; README and DESIGN numbers cite them
  results.json    eval results, one section per eval; the README results table reads it
  results_exported_only.json  commission recall on the export's text alone, for the comparison in DR-6
  answer_key.csv  each commission list row and the corpus bills it resolved to
  text_inventory.json  records without usable text, the versions planned, hosts and their robots.txt
  text_quality.json  the quality test per state and format: outcomes, usable and clean rates per reading, comparison
  text_fetch.json  the full fetch per state: outcomes, and records left out with the reason
DESIGN.md         design, alternatives, scaling limits, failure modes, decision records
data/             local snapshots and derived tables; gitignored, never committed
  raw/<ST>/*.zip  Open States session archives, never edited
  acts/<act>.txt  uniform act texts, downloaded from the ULC under its terms; never committed
  tables/         Parquet tables written by the pipeline; bills/ holds the source's records under the project's names, texts/ each record's chosen text; signatures/, bands/ and pairs/ hold .npy arrays
  text/           the text layer's fetch plan and fetched text (JSON lines per host)
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
- A script that fetches through `text_fetch.fetch_all` runs under `if __name__ == "__main__":`; its extraction workers are spawned and re-import the script.
- Open DuckDB only through `queries.connect_quiet` or `queries.connect`, which cap a query's spill to disk at the same floor. This holds for tests and one-off probes too; a test fails on any other `duckdb.*()` call.
- Secrets come only from environment variables. `.env` is gitignored.
- Model calls happen only in the explanation step and the provisional precision labels. Retrieval runs with no API key.
- Commits use conventional messages, one build step per commit. The body gives the engineering reason: the tradeoff, the measurement, the alternative rejected.
