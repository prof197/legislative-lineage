# Legislative Lineage

Give it one state bill, and it finds bills in other states with similar provisions, matched section by section. It runs on a laptop with no API key.

Status: early development. The near-duplicate stage works end to end; the semantic stage, the evaluation and the site are not built yet.

## Data

The corpus is Open States' per-session JSON archives for the 50 states, 2023–2026. `snapshot/manifest.json` pins each archive by size and SHA-256.

```sh
uv run lineage fetch-raw   # download the pinned archives into data/raw/
uv run lineage ingest      # verify them and build data/tables/
```

The archives are public: downloading them needs no account, API key or other credential. If Open States has since regenerated an archive, its hash no longer matches and `fetch-raw` refuses it. `--repin` keeps the new file and updates the manifest. Both commands refuse to run if they would leave less than 20 GiB of free disk (`--min-free-gib`).

## Finding a bill's relatives

Build the tables and the index once, then query with any bill's text saved as a plain-text file:

```sh
uv run lineage split && uv run lineage eligible && uv run lineage boilerplate && uv run lineage index
uv run lineage query bill.txt
```

A query runs offline against the index in `data/tables/`. It lists the families of bills that share near-identical sections with the query. A family is one state's near-identical bills, such as companion bills in each chamber, counted once. For each family it shows the share of the query's sections with a match and the highest estimated similarity of a matched section. Bills that share a provision in different words are not found yet: that is the semantic stage's job.
