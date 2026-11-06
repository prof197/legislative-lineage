# Legislative Lineage

Give it one state bill, and it finds bills in other states with similar provisions, matched section by section. It runs on a laptop with no API key.

Status: early development. The near-duplicate stage and the commission recall eval work end to end; the semantic stage, the other evals and the site are not built yet.

## Data

The corpus is Open States' per-session JSON archives for the 50 states, 2023–2026. `snapshot/manifest.json` pins each archive by size and SHA-256.

```sh
uv run lineage fetch-raw   # download the pinned archives into data/raw/
uv run lineage ingest      # verify them and build data/tables/
```

The archives are public: downloading them needs no account, API key or other credential. If Open States has since regenerated an archive, its hash no longer matches and `fetch-raw` refuses it. `--repin` keeps the new file and updates the manifest. Both commands refuse to run if they would leave less than 20 GiB of free disk (`--min-free-gib`).

### Uniform act texts

The commission recall eval starts from three uniform acts. Their text is the Uniform Law Commission's copyright, so it is not in this repository (DESIGN.md, [Commission recall](DESIGN.md#commission-recall)). From each act's page on [uniformlaws.org](https://www.uniformlaws.org), download the final act without prefatory note and comments, a Word file, into `data/acts/`. Then convert each one once to UTF-8 plain text under the name the eval reads. `textutil` is built into macOS:

```sh
textutil -convert txt -encoding UTF-8 -output data/acts/public-expression-protection.txt "data/acts/UPEPA_Final Act_NC_2020.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/special-deposits.txt "data/acts/Uniform Special Deposits Act_Final Act_No Comments_2025jan13.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/electronic-estate-planning-documents.txt "data/acts/Uniform Electronic Estate Planning Documents Act_Final Act_No Comments_2022.docx"
```

This is a one-time local step, not part of the pipeline. The eval records each text file's SHA-256 in its results, so a different edition or conversion shows up there. The enactments it checks are in `evals/ulc_enactments.csv`, copied from the commission's bill list for each act, which loads only in a browser. The file's first line gives the date it was read.

## Results

<!-- results:commission -->
Commission recall: of the bills that enacted three uniform acts, the share that the act's own text finds among its top 50 relatives. The lists were read 2026-11-06. Reproduce with `uv run lineage eval commission` → `reports/results.json`.

| | All enactments | With text in the source |
|---|---|---|
| Pooled, three acts | 20 / 36 (55.6%) | 20 / 27 (74.1%) |
| Electronic Estate Planning Documents | 6 / 9 (66.7%) | 6 / 8 (75.0%) |
| Public Expression Protection | 5 / 14 (35.7%) | 5 / 9 (55.6%) |
| Special Deposits | 9 / 13 (69.2%) | 9 / 10 (90.0%) |
| Enacted | 20 / 35 (57.1%) | 20 / 26 (76.9%) |
| Substantially similar | 0 / 1 (0.0%) | 0 / 1 (0.0%) |

Misses: no text in the source 9, never proposed 5, verification 1, below the cutoff 1. The reasons are defined in DESIGN.md, [Commission recall](DESIGN.md#commission-recall).
<!-- /results:commission -->

These are the near-duplicate stage's numbers alone: an enactment is found only where a state kept the act's wording closely enough, section by section. The results file lists every enactment's bills, with each bill's signed sections and the stage that lost it. The commission lists no failed bills, so this measures recall only.

## Finding a bill's relatives

Build the tables and the index once, then query with any bill's text saved as a plain-text file:

```sh
uv run lineage split && uv run lineage eligible && uv run lineage boilerplate && uv run lineage index
uv run lineage query bill.txt
```

A query runs offline against the index in `data/tables/`. It lists the families of bills that share near-identical sections with the query. A family is one state's near-identical bills, such as companion bills in each chamber, counted once. For each family it shows the share of the query's sections with a match and the highest estimated similarity of a matched section. Bills that share a provision in different words are not found yet: that is the semantic stage's job.
