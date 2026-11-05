# Legislative Lineage

Give it one state bill, and it finds bills in other states with similar provisions, matched section by section. It runs on a laptop with no API key.

Status: early development. Nothing here is usable yet.

## Data

The corpus is Open States' per-session JSON archives for the 50 states, 2023–2026. `snapshot/manifest.json` pins each archive by size and SHA-256.

```sh
uv run lineage fetch-raw   # download the pinned archives into data/raw/
uv run lineage ingest      # verify them and build data/tables/
```

The archives are public: downloading them needs no account, API key or other credential. If Open States has since regenerated an archive, its hash no longer matches and `fetch-raw` refuses it. `--repin` keeps the new file and updates the manifest. Both commands refuse to run if they would leave less than 20 GiB of free disk (`--min-free-gib`).
