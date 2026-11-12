# Legislative Lineage

Give it one state bill, and it finds bills in other states with similar provisions, matched section by section. It runs on a laptop with no API key.

Status: early development. The near-duplicate stage and the commission recall eval work end to end. The semantic stage is built and its first corpus run is under way; the other evals and the site are not built yet.

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

A held-out eval, run once at the end of M5, uses four more acts that were never examined while tuning (DESIGN.md, [Commission recall](DESIGN.md#commission-recall)). Download the same edition of each into `data/acts/held-out/` and convert it the same way:

```sh
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/assignment-for-benefit-of-creditors.txt "data/acts/held-out/Assignment for Benefit of Creditors_Final Act_No Comments_2025dec11.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/easement-relocation.txt "data/acts/held-out/UERA_Final Act_NC_2020.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/mortgage-modification.txt "data/acts/held-out/Mortgage Modifcation Act_Final Act_No Comments_2024.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/directed-trust.txt "data/acts/held-out/UDTA_Final_NC_2021sep21.docx"
```

Their enactments are in `evals/ulc_enactments_heldout.csv`.

## Results

<!-- results:commission -->
Commission recall: of the bills that enacted three uniform acts, the share that the act's own text finds, matching at least one of its sections by wording or by meaning, not counting the commission's standard closing sections. The lists were read 2026-11-06. Reproduce with `uv run lineage eval commission` → `reports/results.json`. The last column counts the same enactments with Open States' exported text alone, before the project fetched any: `uv run lineage text merge --exported-only`, the pipeline, then `uv run lineage eval commission --results reports/results_exported_only.json`.

| | Both stages | Near-duplicate stage | Near-duplicate stage, Open States' text alone |
|---|---|---|---|
| Pooled, three acts | 33 / 36 (91.7%) | 31 / 36 (86.1%) | 22 / 36 (61.1%) |
| Electronic Estate Planning Documents | 9 / 9 (100.0%) | 8 / 9 (88.9%) | 7 / 9 (77.8%) |
| Public Expression Protection | 11 / 14 (78.6%) | 10 / 14 (71.4%) | 5 / 14 (35.7%) |
| Special Deposits | 13 / 13 (100.0%) | 13 / 13 (100.0%) | 10 / 13 (76.9%) |
| Enacted | 33 / 35 (94.3%) | 31 / 35 (88.6%) | 22 / 35 (62.9%) |
| Substantially similar | 0 / 1 (0.0%) | 0 / 1 (0.0%) | 0 / 1 (0.0%) |

Misses by both stages, at the step where the near-duplicate stage lost them: never proposed 2, verification 1. The reasons are defined in DESIGN.md, [Commission recall](DESIGN.md#commission-recall).
<!-- /results:commission -->

The near-duplicate stage finds an enactment only where a state kept the act's wording closely enough, section by section. The semantic stage adds enactments that put the act in other words. The results file lists every enactment's bills, with each bill's signed sections, the stages that found it, its best similarity to the act by each, and the step that lost it. The commission lists no failed bills, so this measures recall only. The precision of what each stage finds is not yet measured ([DESIGN.md](DESIGN.md#stage-ablation)).

<!-- results:held-out -->
Held-out recall: the same measure on four uniform acts never examined while matching was tuned, run once, at the end of M5, with the semantic stage's threshold at 0.93 and with the state boilerplate flag. The lists were read 2026-11-07. Reproduce with `uv run lineage eval commission --held-out --final` → `reports/results.json`, `commission_held_out`.

| | Both stages | Near-duplicate stage | Semantic stage |
|---|---|---|---|
| Pooled, four acts | 17 / 19 (89.5%) | 15 / 19 (78.9%) | 17 / 19 (89.5%) |
| Assignment for Benefit of Creditors | 5 / 6 (83.3%) | 5 / 6 (83.3%) | 5 / 6 (83.3%) |
| Directed Trust | 4 / 4 (100.0%) | 3 / 4 (75.0%) | 4 / 4 (100.0%) |
| Easement Relocation | 4 / 5 (80.0%) | 4 / 5 (80.0%) | 4 / 5 (80.0%) |
| Mortgage Modification | 4 / 4 (100.0%) | 3 / 4 (75.0%) | 4 / 4 (100.0%) |
| Enacted | 17 / 18 (94.4%) | 15 / 18 (83.3%) | 17 / 18 (94.4%) |
| Substantially similar | 0 / 1 (0.0%) | 0 / 1 (0.0%) | 0 / 1 (0.0%) |

Misses by both stages, at the step where the near-duplicate stage lost them: never proposed 2.
<!-- /results:held-out -->

## Finding a bill's relatives

Build the tables and the index once, embed the corpus once, then query with any bill's text saved as a plain-text file:

```sh
uv run lineage split && uv run lineage eligible && uv run lineage boilerplate && uv run lineage index
uv run lineage model fetch && uv run lineage embed
uv run lineage query bill.txt
```

Embedding every distinct window takes about 25 hours on the reference laptop ([DESIGN.md](DESIGN.md#semantic-stage)); `lineage embed` resumes where it stopped. Until it has finished, `uv run lineage query bill.txt --no-semantic` searches the near-duplicate stage alone.

A query runs offline against the index and the vectors in `data/tables/`. It lists the families of bills that share sections with the query, found by near-identical text, by similar meaning, or both, and says which. A family is one state's near-identical bills, such as companion bills in each chamber, counted once. For each family it shows the share of the query's sections with a match, the highest estimated similarity of a near-identical section, and the best cosine of a match by meaning.
