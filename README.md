# Legislative Lineage

Find bills in other states that share provisions with a state bill, matched section by section, and see how each one's text differs. Find, too, the published model legislation the bill shares wording with. It shows that texts are alike and when each appeared, not who wrote them or why.

**See it.** [The Georgia site](https://prof197.github.io/legislative-lineage/) lists, for each of Georgia's bills from the 2023–2026 sessions, the bills in other states that share its provisions, with side-by-side comparisons and <!-- stories -->
three featured stories, and one on a passed bill that shares wording with a published model, once a model has judged such a match correct.
<!-- /stories -->

**Documented origins.** A bill's page also lists the model bills it shares wording with, from any organization that publishes model legislation in a public collection: the Uniform Law Commission, advocacy groups, think tanks, and trade and lobbying groups. Each origin gives both shares, how much of the bill's wording is found in the model and how much of the model the bill covers, and links to the document. The model's text is never shown, and a shared passage is not a claim that the organization drafted the bill or asked for it. Each organization is described by what kind of organization it is and its tax status, from public filings, and documents that became public through a leak are labeled leaked. The sources page lists every library matched, whether or not any bill matches it, with its terms and how many documents were collected ([DESIGN.md](DESIGN.md#model-documents)).

**Kept current.** A live track reads each day's new and changed bills from Open States' API and matches each new text against the whole corpus in both directions, so a published bill gains a relative when another state introduces one. A national Atom feed carries the events from every state: a hosted bill with new relatives, a bill that gains a relative or a match to a published model, and a family whose first versions reach several states in a short span, stated as dates and shared text, never as who drew on whom. Each update must give what a full rebuild over the same days gives, and a weekly rebuild during session checks it ([DESIGN.md](DESIGN.md#updates)). How long a new bill's matches take to surface will be reported weekly once it runs in session ([OUTCOMES.md](OUTCOMES.md)).

**Does it work?**

<!-- results:summary -->
| Question | Result | Reproduce |
|---|---|---|
| Does it find the bills that enacted a uniform act? (three acts) | 33 of 36 (91.7%) | `uv run lineage eval commission` |
| The same, on four acts never used while tuning | 17 of 19 (89.5%) | `uv run lineage eval commission --held-out --final` |
| Judged by a model: do the related bills a page lists share the provision? (100 sampled from Georgia's pages) | 0.83 [0.74, 0.89] | `uv run lineage eval relatives` |
| Judged by a model: do the explanations describe the text they cite? (359 claims) | 0.87 [0.83, 0.90] | `uv run lineage eval faithfulness` |
| Matched as published model legislation, does a uniform act find the bills that enacted it? (three acts, at the listing floor) | 31 of 36 (86.1%) | `uv run lineage eval origins` |
| Judged by a model: does the model legislation a page lists as an origin share the provision? (300 sampled matches) | 0.91 [0.87, 0.94] | `uv run lineage eval origin-precision --part reporting_after_compact_fix` |
<!-- /results:summary -->

**Where it falls short.**

- **It misses three of the development enactments.** Pennsylvania's and Oregon's put the act in their own words, and neither stage scores them close enough to it; Oregon's is listed as substantially similar, not enacted. Maine's is missed because the text stored is the bill as first printed ([DESIGN.md](DESIGN.md#stage-ablation)).
- **Maine's amended bills are not in the corpus as amended.** For a bill of Maine's 131st Legislature rewritten in committee, the enacted wording is in a separate document of adopted amendments, which the text layer does not fetch ([DESIGN.md](DESIGN.md#parked)).
- **Minnesota's text includes the language a bill deletes.** Comparisons strike it through, but it is still matched as if it were law the bill makes ([DESIGN.md](DESIGN.md#known-failure-modes)).
- **It covers the 2023–2026 sessions only.** A relative introduced before 2023 is not found ([DESIGN.md](DESIGN.md#data-snapshot)).
- **An origin can be missed, by design, to keep wrong ones out.** An origin names an organization, so a wrong one costs more than a missed one: a bill lists a model only when one of the two shares reaches a floor, and the standard sections that models and bills share, such as definitions, findings, severability and effective dates, are left out on both sides. Some enactments of the commission's acts fall under the floor ([Results](#results), [DESIGN.md](DESIGN.md#origin-matches)).
- **Only published collections are matched.** A model that no public library or archive holds cannot be found; the sources page lists the libraries that are matched.
- **No evaluation has expert labels.** Precision, origin precision and faithfulness are a model's judgments. A person checked a sample of the model's precision labels, but not with legal training, which labels that settle the question would need ([DESIGN.md](DESIGN.md#precision-sample)).

[Results](#results) has the detail and [DESIGN.md](DESIGN.md) how each figure was measured.

**Run it.** Everything builds from source on a laptop with no API key: [Building it](#building-it). A packaged install and a download of the prebuilt index are not available yet.

**Other limits.** [Known failure modes](#known-failure-modes) lists the rest.

**Used it on real work?** [OUTCOMES.md](OUTCOMES.md) records where it has been used.

**Data.** Bill data from [Open States](https://open.pluralpolicy.com/), by [Plural Policy](https://pluralpolicy.com/) (public domain); bill text from each legislature's website. Model legislation from each source's public library or archive, held locally for matching and never committed, redistributed or shown; the sources page states each source's terms as read.

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

The near-duplicate stage finds an enactment only where a state kept the act's wording closely enough, section by section. The semantic stage adds enactments that put the act in other words. The results file lists every enactment's bills, with each bill's signed sections, the stages that found it, its best similarity to the act by each, and the step that lost it. The commission lists no failed bills, so this measures recall only.

Precision is measured on what a bill's page lists, a model's judgment of a random sample of Georgia's pages, drawn once the semantic stage's threshold was final at 0.94: 0.83 [0.74, 0.89] of the relatives listed state the same provision as the bill, 0.90 [0.79, 0.96] of those sharing wording and 0.75 [0.61, 0.85] of those found by meaning alone, which a page lists apart (`uv run lineage eval relatives` → `reports/results.json`, `relatives_precision`). On 50 pairs a person also labeled, the model's labels gave 0.65 where the person's gave 0.75, so the model's numbers more likely run low than high, though the question asked of these pages was not itself checked against a person. [DESIGN.md](DESIGN.md#precision-sample) has how the sample was drawn, the samples the threshold was chosen on, and what the check found.

<!-- results:held-out -->
Held-out recall: the same measure on four uniform acts never examined while matching was tuned, run at the end of M5 with the semantic stage's threshold at 0.94 and with the state boilerplate flag. It was first run at a threshold of 0.93, which found 17 of 19; the threshold then moved on tuning data alone ([DESIGN.md](DESIGN.md#commission-recall)). The lists were read 2026-11-07. Reproduce with `uv run lineage eval commission --held-out --final` → `reports/results.json`, `commission_held_out`.

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

Documented origins are measured the same two ways. Recall comes from the commission's lists, with each uniform act's published text matched as a model document, and counts an enactment found only where its page would list the act. Its sealed key is the held-out lists' reserve acts, which v1 never ran, run once with `--final`; v1's held-out acts are reported beside it but apart, since v1's final run had already looked at their matches. Precision is a model's judgment of a seeded sample of the origins pages list, asked the same passage-level question as the relatives, and read for each source and for one source against the rest as well as overall, since errors concentrated on one organization would be unfair to it ([DESIGN.md](DESIGN.md#origin-matches)).

<!-- results:origins -->
Origin recall: each uniform act's published text is matched against the corpus as a model document, through both stages and the rule a page lists by, and an enactment is found when the act's document matches one of its bills with the larger of the two shares, of the model's words or of the bill's, at the listing floor of 0.25 or above. Origin precision: a model's judgment of a seeded sample of the matches pages list at the floor, drawn by stage and weighted by each stage's matches, each labeled on the section pair it rests on most with v1's passage-level question. One source's matches are read against every other source's together, each weighted by stage as the whole is (`compared`), and each source's own figure is in the same section (`by_source`). Each result carries its 95% Wilson interval, and each command writes `reports/results.json` ([DESIGN.md](DESIGN.md#origin-matches)).

| | Result | Reproduce |
|---|---|---|
| Recall on the development key, three acts | 31 of 36, 0.86 [0.71, 0.94] | `uv run lineage eval origins` |
| Recall on the sealed key, two acts v1 never ran, run once with `--final`[^sealed] | 6 of 6, 1.00 [0.61, 1.00] | `uv run lineage eval origins --key held_out --final` |
| Precision, model-judged, all stages (300 labels of 2,751 matches) | 0.91 [0.87, 0.94] | `uv run lineage eval origin-precision --part reporting_after_compact_fix` |
| Precision, found by the near-duplicate stage alone (100 labels of 526 matches) | 0.87 [0.79, 0.92] | `uv run lineage eval origin-precision --part reporting_after_compact_fix` |
| Precision, found by meaning alone (100 labels of 824 matches) | 0.98 [0.93, 0.99] | `uv run lineage eval origin-precision --part reporting_after_compact_fix` |
| Precision, found by both stages (100 labels of 1,401 matches) | 0.89 [0.81, 0.94] | `uv run lineage eval origin-precision --part reporting_after_compact_fix` |
| Precision, American Legislative Exchange Council (ALEC)'s matches (158 labels of 1,505 matches) | 0.84 [0.77, 0.89] | `uv run lineage eval origin-precision --part reporting_after_compact_fix` |
| Precision, every other source's matches (142 labels of 1,246 matches) | 0.99 [0.95, 1.00] | `uv run lineage eval origin-precision --part reporting_after_compact_fix` |

[^sealed]: Run once, before generic sections were left out of origin matches; a sealed key is not run again, so it stands at the configuration it records ([DESIGN.md](DESIGN.md#origin-matches)).

Apart from the sealed key, v1's held-out acts: v1's final run looked at their matches, so they test nothing sealed.

| | Result | Reproduce |
|---|---|---|
| Recall on v1's four held-out acts: opened in v1; not a sealed test | 16 of 19, 0.84 [0.62, 0.94] | `uv run lineage eval origins --key v1_held_out` |
<!-- /results:origins -->

## Known failure modes

Each has a mitigation or a plain disclosure; [DESIGN.md](DESIGN.md#known-failure-modes) has the full list.

- **Stored text is often the introduced version**, so amendments made before passage may not show. Every comparison names the version it uses.
- **Shared language is not authorship or direction.** Two bills can share text because both drew on a model act, a federal law, an older statute or each other.
- **Status is legislative, not legal.** A law struck down in court still shows as passed.
- **The data is a dated snapshot.** Every page carries the dates of the archives it draws on.
- **Some bills cannot be matched, or match only in part:** those with no usable text, and those whose stored text is an amendment rather than the bill.
- **Matches by meaning alone are right less often** than matches that share wording, so a page lists them apart.
- **Shared boilerplate can make false matches.** A stoplist, a cap on common clauses and a per-state boilerplate flag keep most of them out.
- **A short provision inside a long section can be missed.**
- **Explanations come from a model and can be wrong.** Each claim cites the spans it rests on, and the faithfulness eval measures how often a claim describes them correctly.
- **Minnesota's text includes the language a bill deletes.** Comparisons show it struck through.
- **An origin is shared wording, not authorship.** A model and a bill can share text because the bill drew on the model, the model on a bill, or both on another text. A page gives both shares and a link, never the model's text.
- **A model's standard sections can make false origins.** A long severability clause that recites case law matches many bills; sections of the standard kinds are left out of origin matches on both sides.
- **A model reworded many ways is found by meaning only in its closest matches,** since each of its windows keeps only its nearest corpus texts, as a query does. Matches by wording have no such limit.

## Building it

### The data

The corpus is the per-session JSON archives that [Open States](https://open.pluralpolicy.com/) publishes for the 50 states ([Session JSON Archives](https://open.pluralpolicy.com/data/session-json/)), 2023–2026. `snapshot/manifest.json` pins each archive by size and SHA-256.

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

A held-out eval, run at the end of M5, uses four more acts that were never examined while tuning (DESIGN.md, [Commission recall](DESIGN.md#commission-recall)). Download the same edition of each into `data/acts/held-out/` and convert it the same way:

```sh
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/assignment-for-benefit-of-creditors.txt "data/acts/held-out/Assignment for Benefit of Creditors_Final Act_No Comments_2025dec11.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/easement-relocation.txt "data/acts/held-out/UERA_Final Act_NC_2020.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/mortgage-modification.txt "data/acts/held-out/Mortgage Modifcation Act_Final Act_No Comments_2024.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/directed-trust.txt "data/acts/held-out/UDTA_Final_NC_2021sep21.docx"
```

Their enactments are in `evals/ulc_enactments_heldout.csv`.

Origin recall's sealed test, run once with `--final` in v2, uses that file's two reserve acts, which v1 never ran (DESIGN.md, [Origin matches](DESIGN.md#origin-matches)). Download and convert them the same way:

```sh
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/health-care-decisions.txt "data/acts/held-out/Health-Care Decisions Act (2023)_Final Act_No Comments.docx"
textutil -convert txt -encoding UTF-8 -output data/acts/held-out/antitrust-pre-merger-notification.txt "data/acts/held-out/Antitrust Pre_merger Notification Act_Final Act_No Comments_2026apr20.docx"
```

### Finding a bill's relatives

Build the tables and the index once, embed the corpus once, then query with any bill's text saved as a plain-text file:

```sh
uv run lineage split && uv run lineage eligible && uv run lineage boilerplate && uv run lineage index
uv run lineage model fetch && uv run lineage embed
uv run lineage query bill.txt
```

Embedding every distinct window takes about a day on the reference laptop (`reports/embedding.json`, `seconds`; [DESIGN.md](DESIGN.md#cost-and-runtime)); `lineage embed` resumes where it stopped. Until it has finished, `uv run lineage query bill.txt --no-semantic` searches the near-duplicate stage alone.

A query runs offline against the index and the vectors in `data/tables/`. It lists the families of bills that share sections with the query, found by near-identical text, by similar meaning, or both, and says which. Families found by meaning alone are listed apart from those that share wording, since they are right less often. A family is one state's near-identical bills, such as companion bills in each chamber, counted once. For each family it shows the share of the query's sections with a match, the highest estimated similarity of a near-identical section, and the best cosine of a match by meaning.

### Documented origins

Each source's library is collected by its own collector, one source at a time, through the same fetcher as bill text, which honors robots.txt and paces each host. The collected documents are then matched against the built index and vectors, each one as `lineage query` treats a bill:

```sh
uv run lineage collect-models commission   # one source -> data/modeldocs/commission/, reports/modeldocs/commission.json
uv run lineage origins                     # every collected document against the corpus
```

The commission's collector reads the act texts already in `data/acts/` and asks the commission's site only for its robots.txt, since fetching from the site is off until its terms are ruled on. Collected texts stay in `data/modeldocs/`, which is gitignored; each source's coverage report, which counts the documents found, collected, unchanged, blocked and failed, is committed.

### The site

`lineage export` writes one state's site data, and the site builds from it with Node 22:

```sh
uv run lineage export --state GA            # data/export/GA/
rm -rf site/public/data && cp -R data/export/GA site/public/data
cd site && npm ci && SITE_BASE_PATH= npm run build && python3 -m http.server 8000 -d out
```

The public site builds from the data release that `site/data-release.json` pins: `uv run lineage export --package TAG` packs the export and pins its SHA-256 there, and the Pages workflow refuses an asset that does not match ([DESIGN.md](DESIGN.md#export)).
