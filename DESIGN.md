# Design

Legislative Lineage takes one state bill and finds bills in other states that carry a similar provision. For each match it shows where the bill was introduced, what happened to it, and how its text differs. It indexes every state bill from the 2023–2026 sessions on a single laptop, matches them section by section in two stages that need no API key, and publishes the results for one state as a static site.

Sections marked **To be measured** fill in as the build reaches them. Every number here names the command and the committed output it comes from.

## Contents

- [Problem](#problem)
- [Constraints](#constraints)
- [Pipeline overview](#pipeline-overview)
- [Data snapshot](#data-snapshot)
- [Bill status](#bill-status)
- [Sections](#sections)
- [Boilerplate](#boilerplate)
- [Near-duplicate stage](#near-duplicate-stage)
- [Commission recall](#commission-recall)
- [Alternatives considered](#alternatives-considered)
- [Scaling limits](#scaling-limits)
- [Known failure modes](#known-failure-modes)
- [Measured facts](#measured-facts)
- [Decision records](#decision-records)

## Problem

Model legislation spreads by copying. An interest group, a uniform-law body or another state's statute supplies text, and a legislature introduces it, sometimes verbatim and sometimes with changes that matter: a narrowed definition, or a dropped enforcement clause. Nothing on a bill's face says where its text came from, so anyone who wants to know where else a provision appeared has to search for it.

Three properties of the corpus make this hard:

- **Size.** Every bill from all 50 states over four years. Comparing every pair of sections is quadratic: *N* sections make *N*(*N* − 1)/2 pairs, so the method has to find candidates without enumerating pairs. The snapshot has *N* = 2,340,883 sections (`uv run lineage stats`, `reports/coverage.json`), which is about 2.7 × 10¹² pairs.
- **The unit is the section, not the bill.** A copied provision often sits inside a much larger bill, such as a code revision or a budget. Similarity between whole bills dilutes it below any useful threshold.
- **Copying is not always verbatim.** Reordered clauses and substituted terms break exact-text methods, while generic legal phrasing ("the department shall…") makes meaning-based methods over-match.

The tool reports shared text and timing. It does not claim who wrote a bill, or which of two similar bills came first in any causal sense.

## Constraints

| Constraint | Consequence for the design |
|---|---|
| Retrieval needs no API key | Both matching stages run locally: hashing, then a local embedding model. A model is called only to write the change explanations, and those replies are cached so a keyless run can replay them. |
| Hosted output is static | All results are precomputed. The site serves JSON files and has no query server or hosted vector database. |
| Runs on one laptop | Data structures must fit in memory or stream from disk. Hot loops are vectorized or run in compiled code, never per-element Python. |
| Queries run offline | `lineage query` runs on a user's laptop after one download of the prebuilt index, published as [release assets](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas) of under 2 GiB each. The index must be files that memory-map, not a structure rebuilt in memory on each run. |
| Reproducible | The data snapshot date is pinned, dependencies are locked in `uv.lock`, and random seeds are fixed. |
| Redistributable data | Bill text comes from [Open States](https://open.pluralpolicy.com/data/) bulk data, which is dedicated to the public domain. Legislators' official work is not copyrightable ([*Georgia v. Public.Resource.Org*, 2020](https://www.supremecourt.gov/opinions/19pdf/18-1150_7m58.pdf)). |
| Explanations are grounded | Every change an explanation claims must cite an aligned span that exists, and code checks this. A claim that fails the check is dropped. |

## Pipeline overview

```
Open States snapshot ─► ingest ─► sections ─┬─► near-duplicate stage (MinHash + LSH) ─┐
   (pinned JSON)       (DuckDB)  (split,    │                                          ├─► merge ─► bill scoring ─► alignment ─► explanations ─► export
                                 normalize, └─► semantic stage (embeddings + ANN) ─────┘                              (difflib)    (only model call,  (static JSON)
                                 boilerplate)                                                                                       cached)
```

1. **Ingest.** Verify the raw archives against the committed manifest, then read the per-session JSON into Parquet tables without editing the raw files ([DR-4](#dr-4)). See [Data snapshot](#data-snapshot).
2. **Sections.** Restrict matching to the document types that can carry law ([DR-5](#dr-5)), then split each bill on its section headings, falling back to fixed paragraph windows. Normalize the text and flag legal boilerplate such as enacting clauses, severability and effective dates, which would otherwise dominate the matches.
3. **Near-duplicate stage.** Word shingles, MinHash signatures and LSH banding generate candidate pairs that share text, and each candidate is verified against its full signature ([DR-3](#dr-3)).
4. **Semantic stage.** A local sentence-embedding model and an approximate nearest-neighbor index generate candidates that share meaning but not wording. This is a second candidate generator over every section, not a reranker of the first stage's output, so it can find pairs the first stage never proposed.
5. **Merge and score.** Each match records which stage found it. A bill's score against another bill is the share of the *query* bill's text that falls in matched sections, so a provision inside a large omnibus still scores high.
6. **Alignment.** Word-level alignment marks the spans that were copied exactly, the spans that were paraphrased, and the unmatched text.
7. **Explanations.** For featured matches, a model summarizes what changed, citing span ids. The citations are validated in code.
8. **Export.** Per-bill and per-story JSON for the static site ([DR-1](#dr-1), [DR-2](#dr-2)).

## Data snapshot

**Source.** Open States publishes one JSON archive per state legislative session ([Session JSON Archives](https://open.pluralpolicy.com/data/session-json/)). The corpus is every archive for the 50 states whose session overlaps 2023–2026. Each archive holds a README and one file of bill records, and each record carries at most one extracted text (`raw_text`) together with the URL it came from.

**As-of dates.** The archives are generated independently, so the snapshot has no single date. Each archive's README states when it was generated, and that *Generated At* timestamp is the as-of date of its records everywhere: in the tables, the reports and the site's data stamp. The listing page also shows an "updated" date for each archive. It does not match Generated At, so it isn't used.

**Pinning.** `snapshot/manifest.json` (committed) lists every archive with its size, SHA-256 and Generated At. Ingest refuses to run if the files on disk differ from the manifest in any way: a changed file, a missing one, or an extra one. A rebuild therefore either reproduces the same inputs or fails loudly.

**Rebuilding the raw files.** `lineage fetch-raw` downloads every pinned archive that is missing locally. Open States serves them publicly, with no account or key. A file is kept only if its SHA-256 matches the manifest. `--repin` is the deliberate way to move the snapshot forward: it keeps the changed file and writes its new hash and Generated At to the manifest, so the change shows up in review as a diff of `snapshot/manifest.json`. Requests go out one at a time, a second apart, under a User-Agent that names the project.

**Session dates.** Bill status needs to know whether a session has ended, and the export has no session dates. `snapshot/sessions.csv` (committed) holds each snapshot session's start and end dates, read from the session lists in Open States' own scraper repository at a pinned commit, which the file's header names. `lineage sessions --commit <sha>` regenerates it. The command parses those Python modules as source and never imports or runs them, and it keeps only names and dates, not code. Some legislatures sit for two years while Open States publishes each year separately, and their bills carry over from the first year to the second. For those, the CSV names the session a bill carries over to, so a first-year bill stays alive until the biennium ends. Special sessions keep their own dates.

**Extraction failures.** The export has no error flag. A record whose `raw_text` is missing or blank counts as a failed extraction: it stays in the `bills` table but cannot match anything. Blank means no character that Python's `str.isspace()` rejects. `lineage stats` reports the rate per state and session.

## Bill status

Every bill carries a status, `passed`, `died` or `introduced`, derived from Open States' action classifications as of its archive's Generated At. The order of evidence is as follows.

1. **Enacted.** A signature, becoming law without one, or a veto override means `passed`. Enactment is final, whatever follows it in the record. Georgia files a line-item veto as a veto action after the signature, and New Jersey's conditional veto is followed by a signature once the legislature adopts the governor's changes. In both cases the bill is law. Some exports record an enactment only in words, on an action with no classification: Oklahoma's "Becomes law without Governor's signature", Arkansas's "Approved by the Governor", Connecticut's "PUBLIC ACT 25-61", or a chapter number in Massachusetts, Florida, Oregon or Idaho. That wording counts as an enactment, with the reason `enacted-in-words`, but only on an action that carries no outcome classification, and a motion to override a veto does not match. `lineage stats` counts these per state and type (`reports/coverage.json`, `status_gaps.enacted_in_words`).
2. **Killed.** Otherwise, a veto, a failed vote or a failed override means `died`. The exception is a failed vote followed by a later passage, because chambers can reconsider. In that case the bill is alive again.
3. **Adopted.** Otherwise, a resolution that has been adopted is `passed`. Resolutions are adopted, not signed, so under the enactment rule alone an adopted resolution would show as died once its session ended. A simple resolution or memorial is adopted by its own chamber. A joint or concurrent resolution, a joint or concurrent memorial, and a measure classified as a constitutional amendment need both chambers. Passage actions name the chamber that acted, "Senate", "House" or "Assembly". A passage by a body named for the whole legislature counts for both chambers, because the exports use one only once every chamber has acted: Nebraska's unicameral legislature, Utah's enrolling step and Ohio's "Sent To The Governor". The deciding action is the adoption that completed the measure, so a page for a constitutional amendment names the vote that sent it to the ballot.
4. **Session ended.** Otherwise, a bill whose session ended before the archive was generated `died` with its session. A session is still sitting on its last day. Where bills carry over within a two-year term, the biennium's end applies ([Data snapshot](#data-snapshot)). A session with no known end date leaves the bill `introduced`, which avoids guessing that it died.
5. Anything else is `introduced`.

The table keeps the deciding classification (`status_reason`, or `enacted-in-words` or `session-ended`) and its date (`status_date`), so a bill page can name the action, for example a veto.

Explicit outcomes still come first. A joint resolution that goes to the governor and is vetoed `died`, and one that is signed names the signature. Joint and concurrent measures that only one chamber adopted die with their session; `lineage stats` counts them per state and type (`reports/coverage.json`, `status_gaps.resolutions_adopted_by_one_chamber`).

Actions are sequenced by Open States' `order` field. Dates arrive either as plain dates or as zoned timestamps. A timestamp is converted to a calendar date in the time zone the state's export uses. Most exports write a plain date as local midnight. Alabama's writes midnight Eastern although the state keeps Central time, and converting it with Central time would move each date a day earlier.

## Sections

The unit of matching is the section, not the bill. Provisions travel between states one section at a time, often inside omnibus bills, so whole-bill similarity would bury a shared provision under unrelated text. `lineage split` writes the `sections` table: bill id, ordinal, heading, display text, normalized text, and whether the bill was split at headings.

**Cleaning.** Before splitting, the text is cleaned:
- Line endings are unified.
- Style sheets left by the source's conversion from HTML are removed. Every Texas and West Virginia record with text carries style rules before the bill, and Texas's Word exports carry a whole style sheet inside an HTML comment. HTML comments are dropped. A style rule is dropped only when its line holds nothing but rules whose bodies are lowercase `property: value` declarations, because Kansas marks amendments with braces and other states brace parts of forms.
- Text is NFKC-normalized, so non-breaking spaces and ligatures become plain characters.
- Format, private-use and control characters are dropped. Kentucky's export, for example, puts a private-use glyph before each heading.
- Runs of spaces collapse to one.
- A printed line-number gutter is removed. A gutter is recognized only when most lines of the bill start with a bare number, so New Jersey's "1." section numbers and a line such as "2023 Regular Session" survive.
- Arizona's export markers are removed. Arizona's export wraps each code section in `START_STATUTE` and `END_STATUTE`, glued to the citation (`START_STATUTE6-251. Short title`), which would hide the citation from the heading rule and add a junk token to the shingles.
- Running page headers are removed. A header is a line that repeats at least three times in the bill, has at most eight words, contains a digit and has no lowercase letter, such as "LC 47 3375" or "STATE OF WYOMING 25LSO-0123". Repeated lines that read as text, such as "is amended to read as follows:", stay, and the document-frequency stoplist handles them.

**Headings.** A section starts at a line that opens with `Section 1.`, `SECTION 1.`, `Sec. 1.`, `SEC. 1.` or `§ 1.`, with a colon accepted in place of the period. The number can be code-style (`1-101`, `2a`, New York's `680-a`). Alaska's leading asterisk and Washington's `NEW SECTION.` are skipped. A heading must start its line, so a line opening with a quotation mark is not a heading. That keeps the sections of an act being amended, quoted inside the bill, from splitting it. Lowercase "section" at the start of a line is ignored, because there it is a wrapped cross-reference far more often than a heading. Text before the first heading becomes section 0.

**Citation headings.** A bill that enacts a new chapter often does so under one `SECTION 1.`, heading each code section with its citation alone (`19-35-1. Short title.`). Amending bills print the amended section the same way after the amending clause (`Section 4. Section 27-71-7, Mississippi Code of 1972, is amended as follows:` then `27-71-7. (1) ...`). Such a citation is a heading when it starts a line and is followed by a period, under these conditions:

- It has at least one hyphen, and every part after a hyphen starts with a digit (`6-3901.`, `9-33.2-10.`, `39-4A-2.`). That leaves out enumerators (`2.`), dotted outline numbers (`31.150.`), year ranges (`2025-2026.`) and New York's subdivision labels (`2-b.`). New York's sections themselves start with `§`.
- The line before does not end on a word that introduces a citation (`section`, `§`, `of`, `by`, `under`, `and`, `law`, `RSA` and a few more), on a hyphen that breaks the citation itself, or on a struck citation in brackets that Utah replaces with the new one. One blank line between them changes nothing. In those cases the citation is a wrapped cross-reference.
- The bill already splits at section headings. A bill with none keeps the window fallback, because there a citation starting a line was a wrapped reference ("Public Law\n116-260."), and taking it turned one Ohio budget bill's windows into two sections.

**Fallback.** A bill with no heading is cut into windows of at least 200 words on paragraph boundaries. A paragraph longer than a window is cut by words, and a short remainder joins the window before it. Every bill with text therefore yields sections, and the `by_heading` flag records which rule produced them. The exception is text that cleaning reduces to nothing, or to Texas's notice that a version's text is not available, which is treated as no text and counted as text without sections. `lineage stats` reports the heading hit rate per state.

**Normalized text** is case-folded words and numbers separated by single spaces, with the section sign spelled out so that "§ 2" and "Section 2" normalize alike. Shingling and hashing read it. The display text is kept for alignment and the site.

## Boilerplate

Enacting clauses, effective dates, severability clauses, repealers and amending formulas are near-identical across thousands of bills. Left in, they would make unrelated bills look alike. Two mechanisms remove them, and `lineage boilerplate` reports what each one caught in `reports/boilerplate.json` (committed).

**Patterns.** A short list of patterns runs on normalized text and recognizes enacting clauses, effective dates, severability clauses and repealers, along with two of the Uniform Law Commission's standard closing sections: the uniformity clause ("In applying and construing this [act], ..."), which closes every uniform act, and the statement of an act's relation to the federal E-SIGN Act, which closes its acts on electronic records. Left in, those two made each uniform act's text match the enactments of dozens of other uniform acts (see [Commission recall](#commission-recall)). A section is flagged as boilerplate only when, once its heading and every pattern match are removed, at most four words remain, which is room for a catchline such as "effective date". A provision that ends with an effective date therefore stays matchable. So does a preamble that carries the bill's title along with its enacting clause. The `boilerplate` column records which kind a flagged section is.

**Document-frequency stoplist.** The patterns can't anticipate standard definitions, amending formulas ("is amended to read as follows"), or markup that survives extraction, such as Minnesota's "new text begin". The stoplist catches them from the data. Every matchable section's set of word 5-gram shingles is counted, and a shingle found in more than a set share of those sections is dropped before MinHash. Matchable means the bill's document type is eligible ([DR-5](#dr-5)) and the section is not flagged by the patterns, so the list reflects the text that will actually be compared: excluded resolutions cannot push their formulas onto it, and sections that are dropped anyway don't count twice. The list adapts to each state's habits without upkeep and is rebuilt with the corpus. The proposed share is 0.1% (`--max-share`). The report tabulates how many shingles, and what share of all shingle occurrences, each of several thresholds would remove, and lists the most frequent stoplisted shingles. The ablation sets the final value against recall.

**Shingle hashing.** Each word is hashed with 64-bit FNV-1a over its UTF-8 bytes. Five consecutive word hashes are combined positionally and finished with MurmurHash3's 64-bit mixer, and the top 32 bits are kept. The computation is whole-array NumPy, and a test checks it against a one-shingle-at-a-time reference and pins its values. The stoplist and every later signature depend on those values, so changing the hash means rebuilding both.

## Near-duplicate stage

Comparing every pair of sections is out of reach, so this stage summarizes each section in a short signature from which the similarity of any two can be estimated, and proposes for comparison only pairs whose signatures suggest they are alike. `lineage index` builds the signatures and the band index that proposes the pairs ([DR-3](#dr-3)).

**Shingle sets.** A section is represented by its set of distinct word 5-gram hashes, less the stoplist. A section shorter than five words has no shingle, and one whose every shingle is stoplisted has none left. Neither gets a signature, and `reports/index.json` counts both.

**MinHash.** Each section's signature holds 128 values. Value *k* is the minimum over the section's shingles of *h*<sub>*k*</sub>(*x*) = ((*a*<sub>*k*</sub>·*x* + *b*<sub>*k*</sub>) mod *p*) mod 2<sup>32</sup>, with *p* = 2<sup>61</sup> − 1. Two sections' values at position *k* agree when the shingle that minimizes *h*<sub>*k*</sub> over both sets lies in both. Under an ideal random permutation that happens with probability equal to the sets' Jaccard similarity *J*, so the share of agreeing positions estimates *J* with standard error √(*J*(1 − *J*)/128), at most 0.044. A linear hash family only approximates an ideal permutation. The test oracle described below checks that the approximation costs neither bias nor spread.

**Exact arithmetic.** The product *a*<sub>*k*</sub>·*x* reaches 2<sup>93</sup>. Computed in NumPy's 64-bit integers it would wrap around before the remainder is taken, and the result would no longer belong to the family. The implementation splits *a*<sub>*k*</sub> at bit 32 and reduces each partial product using 2<sup>61</sup> ≡ 1 (mod *p*), so every value is exact. A test compares it with Python's arbitrary-precision integers on edge and random inputs.

**Keeping 32 bits.** The final mod 2<sup>32</sup> halves the signature's size, but two different shingles can now share a value. For two sets of *n* shingles each, a false agreement at one position has probability about *n*/2<sup>33</sup>. The largest measured set has 261,326 shingles (`reports/index.json`), which puts the bias at about 3 × 10<sup>−5</sup>, against a standard error of up to 0.044.

**Fixed parameters.** The 128 pairs (*a*<sub>*k*</sub>, *b*<sub>*k*</sub>) are drawn with SplitMix64 from a fixed seed rather than from a NumPy generator, because NumPy does not promise that its generators' streams stay the same across versions, and every stored signature depends on these numbers. A test pins them, along with sample signature values.

**Computation.** Each permutation runs over every shingle of a batch in one vectorized pass, and a reduction takes each section's minimum. Batches are cut into runs of about 131,000 shingles, so the working arrays stay in cache and are reused across permutations instead of being reallocated. Sections files are signed in parallel, one worker process per file.

**Storage.** Each sections file's signatures form one dense `uint32` matrix, stored as `.npy` so it memory-maps as it is. Next to it, a Parquet file holds each row's bill id and section ordinal. A signature takes 512 bytes.

**Test oracle.** datasketch, a development-only dependency, signs the same pairs of sets as the NumPy implementation: 200 pairs at each of five Jaccard levels from 0.1 to 0.9, with the exact similarity known by construction. For both engines, the mean error is within four standard errors of zero, and the two engines' means agree within the same bound. Each engine's root-mean-square error is within 20% of the theoretical standard error, since an error that is too small would point to correlated permutations just as surely as one that is too large. The two use different hash families, so the test compares estimates, not values.

**LSH banding.** Each signature is cut into 32 bands of 4 values. Two sections become candidates when all four values of at least one band agree. A pair with Jaccard similarity *s* agrees on a given band with probability *s*<sup>4</sup>, so it becomes a candidate with probability 1 − (1 − *s*<sup>4</sup>)<sup>32</sup>. That S-curve is 0.05 at *s* = 0.2, 0.56 at 0.4, 0.87 at 0.5 and 0.99 at 0.6, and it rises most steeply near (1/32)<sup>1/4</sup> ≈ 0.42. The setting leans toward recall, because verification discards a false candidate cheaply while a pair that is never proposed is lost for good. 32 × 4 is the proposed start, and the ablation sweeps it.

**S-curve test.** The curve assumes each band's four values come from independent permutations, which a linear hash family only approximates. So the test bands real signatures rather than simulated agreements: 500 pairs of sets at each of five similarities from 0.2 to 0.6, with the exact Jaccard similarity known by construction. At every level, the share of pairs that become candidates is within four binomial standard errors of the curve.

**Band keys and buckets.** Each band's four values are hashed to one 64-bit key with SplitMix64's finalizer. Equal keys from unequal bands would only add a candidate that verification removes, and with two million sections a band expects fewer than 10<sup>−6</sup> of them. Each band's keys are sorted together with the row behind each key, and runs of equal keys are the buckets, so the index is two dense matrices of 12 bytes per section per band rather than millions of dictionary entries. It is stored in `bands/` beside the signatures and memory-mapped by queries. Pairs from all the buckets of one size are formed together from one matrix, and a pair that shares several bands is kept once.

**Bucket cap.** A bucket larger than the cap yields no pairs, and `reports/index.json` counts the buckets, section memberships and pairs skipped. The large buckets are boilerplate that the stoplist misses because it is common only within one state, such as a state's own wording of effective-date, appropriation and transition clauses. A cap set too low would also cut real copying: interstate compacts put the same section into bills in dozens of states. So the report sweeps a range of caps and, for each, counts the buckets skipped, those spanning more than one state, and the cross-state candidates the cap removes for good, after checking every band a pair shares. The cap is 1,000, the smallest cap in the sweep that removes cross-state candidates from no more than one bucket. It is a performance guard, not a precision control: verification and the boilerplate rules decide what counts as a match. Raising it from 500 to 1,000 brought back 2,005 cross-state candidates, and none of them verified. Every build reports the cross-state candidates its cap drops (`lsh.capped.cross_state_pairs_lost`), and when the recall eval loses an enactment to the cap, its miss reason says so. The numbers are under [Measured facts](#measured-facts).

**Verification.** Each candidate's similarity is re-estimated from all 128 values and kept as the count of agreeing positions, one byte per pair. Pairs estimated below 0.5, fewer than 64 agreements, are dropped. The threshold of 0.5 is proposed. At 0.5 the S-curve already proposes 87% of pairs, so what verification keeps is what the bands find reliably, while below the curve's midpoint whether a pair is proposed at all is largely chance. The candidates' estimates show no valley to set the threshold by: cross-state candidates spread from 0.1 to 1 (`reports/index.json`, `pairs.estimate_histogram`). Partial or reworded copying below the threshold is the semantic stage's to find, and the ablation sweeps the threshold. Verified cross-state pairs are stored in `pairs/` by signature row. Same-state pairs serve only to form families.

**Families.** A state files the same text many times: companion bills in each chamber, a bill refiled in a later session, or one template for many resolutions. Lineage counts the other states that carry a provision, so each state's copies are grouped into a family that counts once. Two bills of one state are linked when, in each direction, at least 80% of the bill's signed sections have a verified match among the other's, and families are the connected components of the links. Requiring a share, not any one match, keeps a clause both bills happen to share from linking them. Requiring it in both directions keeps an omnibus bill from absorbing each small bill it contains. The share of 0.8 is proposed and was set from the measured distribution (`reports/index.json`, `families.min_share_histogram`). Shares spike at the fractions of small bills, 1/3, 1/2, 2/3 and 3/4, where a short bill shares one or two clauses with another, and again at 1. Between 0.70 and 0.95 they are sparse, and 0.8 sits there, past the 3/4 spike. The report's sweep shows what a lower share costs: at 0.5 the largest family grows to 1,514 bills and at 2/3 to 815, chained through shared clauses, while from 0.75 up it stays at 764. The largest families are one template filed many times, such as Georgia's notices of local bills, New Mexico's placeholder bills and Massachusetts's study orders, which is what a family should collapse. A family is named by its first bill id in sort order and stored in `families/`.

**Query.** `lineage query FILE` treats a bill's text exactly as the corpus was treated: it is cleaned, split and normalized, boilerplate-only sections are set aside, and the rest are shingled, stoplisted and signed. Each signature's band keys are found in the memory-mapped band index by binary search, and a bucket over the cap yields nothing, as in the build. Each candidate is verified against its stored signature at the build's threshold. Matches are grouped by family and ranked by the share of the query's signed sections that the family matches, scored from the query's side so a provision carried inside a larger bill still ranks high. A query reads only the index pages its keys land on and needs no network or API key. Results are shown through each family's best-matching bill, with the family's size.

## Commission recall

The external ground truth is the Uniform Law Commission's list of enactments. Starting from a uniform act's text, the eval asks what share of the bills that enacted it the pipeline finds.

**Act text.** The ULC holds the copyright in its acts. Its [terms of use](https://www.uniformlaws.org/termsofuse) let anyone view, copy, print and distribute the site's content for informational, noncommercial purposes, unmodified and with the copyright notice or other attribution. Modifying it is allowed only as part of a legislative process. The terms say nothing specific about showing an act on another website. Downloading an act's final text shows no separate notice or terms. Each act prints its own notice in its front matter:

| Act | Edition used | Notice printed in the act |
|---|---|---|
| Uniform Public Expression Protection Act | Final act without prefatory note and comments, October 2, 2020 | Copyright © 2020 By National Conference of Commissioners on Uniform State Laws |
| Uniform Special Deposits Act | Final act without prefatory note and comments, January 13, 2025 | Copyright © 2023 By National Conference of Commissioners on Uniform State Laws |
| Uniform Electronic Estate Planning Documents Act | Final act without prefatory note and comments, September 28, 2022 | Copyright © 2022 By National Conference of Commissioners on Uniform State Laws |

So:

- The repository never contains act text. Its code is MIT-licensed, and an act committed beside it would read as offered under that license, which includes the right to modify. Act texts live in `data/acts/`, which is gitignored, and each user downloads them from the ULC under its terms (README, [Data](README.md#data)).
- The eval uses them locally, as query documents.
- The site may show an act's text unmodified, with the act's own copyright notice above and a link to the act's page, as informational, noncommercial use. No narrower notice came with the download, so the terms of use are the whole of the permission.

**Act texts as queries.** An act as the commission publishes it is mostly not law: a title page and committee list, a table of contents, a prefatory note, and an official comment or legislative note after many sections. States enact the sections. Scoring is from the query's side, the share of the query's sections a bill matches, so every comment left in would lower every enacted bill's score. The editions above already leave out the prefatory note and comments, but each still carries legislative notes, which tell a state how to adapt a section and can run several paragraphs. `lineage.acts` keeps the numbered sections, including bracketed optional ones, and the article headings of an act divided into articles. It drops the front matter up to the first section or article heading that is not a contents entry, and each comment or legislative note up to the next section or article heading. In the commission's final acts, both close a section. The words each act keeps, and each file's SHA-256, are recorded in the eval's results, so a different edition or conversion is visible.

**Finding an enactment's bills.** The ground truth is `evals/ulc_enactments.csv`, every row of each act's bill list on the day it was read. Rows the list shows only as introduced are not enactments and are skipped. The District of Columbia is excluded, since the corpus is the 50 states, and so are enactments listed before 2023, where the corpus begins. Enacted and Substantially Similar are reported as separate strata. A row's cell can name several bills, usually companions in each chamber, so bill numbers are compared in one form: Colorado's year prefix ("HB 24-1232") is dropped, and "H 40" and "HB 40" compare equal. Each enactment's bills are found by state, number and session year, in three steps:

1. Every bill the cell names in a session whose years contain the listed year. A bill shown passed after that year is a later bill reusing the number and is dropped. If any of the rest passed, these are the enactment's bills.
2. Otherwise, a named bill that passed in the session ending the year before. The list's year can be the signing year: Ohio's act passed both chambers in December 2024, the last month of its session, and is listed under 2025.
3. Otherwise, the bills of step 1, recorded as matched by number only.

"Passed" is a passed status, or a passage action recorded in every chamber, since a bill signed after its session ends can still carry a status of died. `lineage.ulc` implements the rule, and each enactment's results name the step that matched it.

The resolved mapping is committed as `reports/answer_key.csv` (`uv run lineage eval answer-key`) so every row can be checked by eye. Each bill a row resolved to is listed with its Open States id, title, chambers passed and last passage date, along with the step that found it and the listed number it matched. Rows that are not counted are listed with the reason. The key needs only the bills table, so it can be checked before the index is built. A test fails if the committed key and the committed results name different bills.

**Found.** An enactment is found when any of its bills belongs to a family among the act text's relatives with a score above the threshold. The score is the family's share of the act's signed sections, the share that query-side scoring ranks by. Rank is reported but does not decide. A uniform act's real relatives can outnumber any fixed rank once each state's introduced, failed and enacted versions count, so a rank cutoff would cap recall as an act spreads. The site's display cap of 50 relatives is a separate choice.

The threshold is proposed and comes from the measured scores. An earlier rank cutoff of 50 was examined first. With the commission's standard closing sections still matchable, most of each act's relatives were bills of other uniform acts matched through the uniformity clause alone, or through the E-SIGN statement, and they outranked some enactments. Those two sections are now boilerplate ([Boilerplate](#boilerplate)). Every relative left for each of the three acts carries part of that act's text. Each is an enactment, another state's introduced or failed bill, a companion, an omnibus bill that includes the act, or a bill that adapts a few of its sections. The results list them (`queries.<act>.relatives`). No score separates unrelated bills from relatives, so the threshold is one verified section (`threshold.min_share` 0). The results record each found enactment's rank and share, and recall at higher minimum shares (`recall_by_min_share`), so another threshold can be read from them without rerunning. The threshold is re-examined whenever a relative turns out not to carry the act. An enactment is "with text" when at least one of its bills has stored text, and recall is reported over all enactments and over those with text.

**Miss reasons.** Each miss is classified automatically. Each bill an enactment names gets the first reason that applies, and the enactment takes the reason of the bill that got furthest through the pipeline. The stage reasons are measured by replaying LSH and verification on the bill's own stored signatures, with the query's band keys, the bucket cap and the verification threshold.

- `not_in_corpus`: no bill the cell names is in the snapshot in the session years.
- `no_text_in_source`: the export has no text for the bill, as for every Connecticut, Maine and South Dakota record.
- `amendment_only`: the stored text is an amendment that does not replace the whole bill (see [Known failure modes](#known-failure-modes)).
- `excluded_type`: the document-type rules exclude the record ([DR-5](#dr-5--match-only-document-types-that-can-carry-law)).
- `no_usable_text`: the text yields no signed section, because every section is boilerplate or stoplisted.
- `not_proposed`: no section of the bill shares a band key with any section of the act.
- `bucket_cap`: a section pair would have verified, but every band it shares is in a bucket over the cap.
- `verification`: a section pair was proposed, but none reached the verification threshold.
- `below_threshold`: the bill's family is among the relatives, with a score at or below the threshold.

**Pipeline regression test.** The act texts cannot be committed, so CI cannot compute the headline number. Instead, `tests/test_commission.py` runs every pipeline command on a committed fixture: the bills each enactment names, among distractors that are bills of other uniform acts from the same archives. They share the commission's drafting conventions and closing sections, which makes them hard negatives. Each act is queried with its earliest enacted state version, and that version's own enactment is left out. The test fails if any enactment it found when written is lost. `uv run python -m evals.fixture` rebuilds the fixture from the snapshot. It leaves out omnibus bills over 40,000 characters. Its number is not the recall the README reports, because a state's version differs from the act.

**Results.** `uv run lineage eval commission` writes the `commission` section of `reports/results.json`: recall pooled, per act and per stratum, the miss-reason counts, and each enactment with its bills, their stages and their best similarity to the act. It also records the SHA-256 of the enactment file and of each act's text, before and after the comments and notes are dropped. The README's results table is generated from this file.

## Alternatives considered

| Alternative | Reason rejected | Evidence |
|---|---|---|
| Compare all pairs | Quadratic in the number of sections: about 2.4 × 10¹² pairs at the measured *N* | Arithmetic on the measured section count |
| Embeddings only | No exact-span evidence to highlight, a higher cost per section, and weaker on legal boilerplate that sounds alike | To be measured by the stage ablation (M4) |
| Embeddings as a reranker of LSH candidates | A reranker can only reorder pairs LSH already proposed, so it can never recover a paraphrase LSH missed | Follows from the design |
| Whole bills as the unit | Dilutes provisions carried inside omnibus bills | Commission-recall misses show the cost at a smaller scale: a bill split into a few chapter-sized sections is never proposed against the act's short ones (`reports/results.json`, each bill's `signed_sections` and `stage`) |
| Scope the corpus by topic keywords | Makes recall circular, since only bills that match the keywords can be found, and it removes the hard negatives | Follows from the design |
| LLM calls or a hosted vector database in retrieval | Breaks the keyless and static constraints | Constraint |
| A query server behind the site | Breaks the static constraint | Constraint |
| Smith-Waterman alignment for highlighting | Gap scoring adds nothing to marking shared blocks; `difflib` finds them from the standard library | Follows from the design |
| LegiScan as the text source | Its [terms of service](https://legiscan.com/terms-of-service) restrict copying, which conflicts with publishing excerpts | Sourced |

## Scaling limits

The section count sets the size of everything downstream. The 2,058,452 signed sections' MinHash signatures take 1.05 GB, and their band index 0.79 GB (`uv run lineage index` → `reports/index.json`, `signature_bytes` and `lsh.index_bytes`). Ten times the corpus is about 18 GB, which no longer fits comfortably in a laptop's memory.

| Pressure | What breaks first at 10× | Response |
|---|---|---|
| Index size | Signatures and embeddings outgrow memory | Shard by state and session, memory-map signatures, and merge the sorted LSH keys per shard |
| New sessions | A full rebuild for every update | Append only: hash and embed the new sections, then recompute neighbors only for bills that share a bucket or neighbor with them |
| Formatting noise | More state-specific markup, scanned images, and struck-and-inserted amendment text | Per-state normalizers behind one interface, each with fixture tests, and extraction errors tracked per state |
| Duplicate versions | With full version history, a bill's own versions crowd out other states | Collapse versions into one family before scoring, as companion bills already are |
| Boilerplate | The shared-clause load grows with the corpus | Recompute the document-frequency stoplist each build, and watch the LSH bucket-cap counter |
| Hosting | One file per bill for all states exceeds static-host limits | One export per state, or a sharded layout on object storage |

## Known failure modes

These carry into the README's known-failure-modes section. Each has a mitigation or a plain disclosure.

| Failure | Effect | Mitigation or disclosure |
|---|---|---|
| The stored text is often the introduced version ([Open States stores one version's text per bill](https://github.com/openstates/openstates-core/blob/main/openstates/data/models/bill.py)) | Amendments made during passage are invisible to "what changed" | Fetch enacted versions for featured matches; measure the share of enacted bills stored as introduced text and state it |
| Text extraction failed, or the source is a scanned image | No text, so no matches. This includes enacted bills in states that otherwise have text, and 1,392 Texas records whose text is only a notice to see the PDF | Measured per state and session (`reports/coverage.json`). The recall eval names "no text in source" as a miss reason and reports recall both over all enactments and over enactments with text |
| Amendment-style bills | New text sits inside old statute text, so matches can land on the old law | Drop struck text where its markup survives extraction; section-level scoring limits the damage |
| Omnibus bills | The provision is a small share of the bill | Score from the query side; display matches by section |
| Boilerplate leakage | False matches on shared clauses | Document-frequency stoplist, the LSH bucket cap, and the ablation's false-match rate |
| A state's own boilerplate that stays under both the bucket cap and the corpus-wide stoplist | A query section carrying such a clause, such as a state's standard request for a federal waiver, matches hundreds of that state's bills, each listed as a family with a small share | Results rank by the share of the query matched, so whole-bill relatives lead. A per-state stoplist, counted over families, is tested in the ablation (M4) |
| The stored text is an amendment, not the bill | An amendment that changes a few lines can match only on those lines, so the bill it amends is missed or scores low. 5,737 records in 13 states are such partial amendments (`reports/coverage.json`, `amendment_only`) | Counted per state and session from each text's opening. A substitute that replaces the whole bill is counted too but matches like a bill. Disclosed |
| A Georgia local bill whose stored text is only its notice | Georgia publishes a notice of intention before a local bill is introduced, and for 774 of its bills the export stores that notice and the sponsor's affidavit in place of the bill: 318 in 2023–24, 3 in the 2023 special session, 355 in 2025–26 and 98 in the 2026 special session (`uv run lineage stats` → `reports/coverage.json`, `georgia_local_notices`). Georgia is the hosted state, so these bill pages have no bill text to compare, and 764 of them form one family through the shared notice wording (`reports/index.json`, `families.largest`) | They can match no other state through the notice, so cross-state counts are unaffected. Disclosed as a known gap. How the site shows these bills is decided when the site is built |
| A citation heading the splitter misses, or one it takes wrongly | A missed citation leaves a chapter as one section, which cannot match a copied provision's sections. A wrong one splits a section at a cross-reference or a table of contents | Citation headings are taken only in bills that already split at section headings, and only when the line before does not lead into a citation (see [Sections](#sections)). A New York article's table of contents lists its sections by citation, and each listed line becomes a section holding only that caption |
| A copied section that sits in an LSH bucket over the cap in every band it shares | The pair is never proposed | Each build counts the cross-state pairs the cap drops (`reports/index.json`), and the recall eval names "bucket cap" as a miss reason |
| Legal phrasing that sounds alike | The semantic stage pairs unrelated sections | Label meaning-only matches as such, tune the cosine threshold, and run the precision sample |
| An explanation with valid citations but a wrong summary | A claim cites a real difference but describes it wrongly | A hand-labeled faithfulness eval, and the cited spans shown beside each claim |
| Similarity has no direction | An earlier bill is not necessarily the source | Copy states timing only, never "copied from" |
| Legislative status is not legal standing | A law struck down in court still shows as passed | A legend line wherever status is shown |
| A bill awaiting the governor when the session ended, whose outcome the export never recorded | Shows as died | `lineage stats` counts these per state and type (`reports/coverage.json`, `status_gaps`) |
| A state publishes no extracted text | Its bills cannot match: Connecticut, Maine and South Dakota have no text for any eligible record (`reports/document_types.json`, `states_without_text`) | Excluded and disclosed per state. Their version links point to PDFs. A short test of Open States' own per-state conversion functions on a sample of documents decides whether to extract them, after checking that library's license and each state site's robots.txt |
| A stale snapshot | Bills filed after an archive was generated are missing, and archives were generated at different times | Every page carries the Generated At dates of the archives it draws on |
| A substantive measure filed under an excluded document type | It can never be found | Excluded records stay in `bills`, and the recall eval names "excluded type" as a miss reason ([DR-5](#dr-5)) |
| A state whose resolution titles mark constitutional amendments in a way no rule reads yet | Its amendments are excluded with its resolutions. Nebraska's, Michigan's and Rhode Island's resolutions carry constitution titles whose conventions are not yet checked | Listed as a known gap; each is admitted by a rule once its convention is checked and counted ([DR-5](#dr-5)) |

## Measured facts

Each entry names the command that produces it and the committed file it reads from. Snapshot-wide figures are below. Per-state and per-session figures are in the files.

**Snapshot** (`uv run lineage snapshot` → `snapshot/manifest.json`). 220 archives, 6.42 GiB, for the 50 states' sessions overlapping 2023–2026. Generated At dates run from 2023-03-01 to 2026-11-05. `reports/coverage.json` gives the range per state.

**Text coverage** (`uv run lineage stats` → `reports/coverage.json`).
- Of 525,633 records, 11,675 have no text and 141,770 have blank text. 1,394 have text that cleaning reduces to nothing, 1,392 of them Texas notices that a version's text is not available and the PDF should be consulted. 370,794 (70.5%) yield sections.
- In Georgia, 774 records hold only a local bill's notice of intention and its affidavit, with no bill text (`georgia_local_notices`, per session). They yield sections from the notice, so they count as text here; the known failure modes below say what that costs.
- 7,588 records store an amendment in place of the bill (`totals.amendment_only`, and `amendment_only` per session). 1,851 of them replace everything after the enacting clause, so they carry a full substitute text. The other 5,737 hold only the lines they change. By state: Louisiana 3,242, Illinois 1,807 (1,201 substitutes), Pennsylvania 376, Virginia 358 (all substitutes), Oklahoma 293, Vermont 276, Wisconsin 256, Delaware 233, Oregon 229, New Hampshire 220, Georgia 207, Nebraska 54, Mississippi 31 and Alabama 6.
- Connecticut, Maine and South Dakota publish blank text for every record. Maryland, Nebraska, New Hampshire, Montana and Indiana have text for under a third of their records.

**Sections** (same command and file).
- 2,340,883 sections.
- The heading rule splits 257,108 of the 370,794 bills with text (69.3%), and the window fallback splits the rest.
- Ohio, Washington, New Hampshire and New Jersey split almost entirely by windows (99.8% or more), and Missouri nearly so (97.9%). Their exports lack line breaks or section numbers, or number sections without the word "Section".

**Document types** (same command and file). Record counts by classification per state, the title-pattern count of likely commemorative measures within each type, and status by type. These set DR-5's type rules.

**Eligibility** (`uv run lineage eligible` → `reports/document_types.json`). 447,389 of 525,633 records are of an eligible type, and 307,659 of those yield sections. Every rule's count, per state, is in the file, and for a rule with an `unless` pattern, the count per state of the titles it spared. The titles themselves are not committed, because commemorations name private people. `uv run lineage stats --samples` writes a sample of each title rule's titles and every spared title to `data/reports/document_type_samples.json`, a local file anyone can reproduce from the pinned snapshot.

**Boilerplate** (`uv run lineage boilerplate` → `reports/boilerplate.json`). Sections flagged by each pattern, and the shingles and shingle occurrences that each candidate stoplist threshold removes. 2,081,174 sections are matchable. At the proposed 0.1% share, 3,359 shingles are stoplisted, removing 3.6% of shingle occurrences.

**Signatures** (`uv run lineage index` → `reports/index.json`). 2,058,452 of the 2,081,174 matchable sections are signed. 22,720 are shorter than five words, and 2 lose every shingle to the stoplist. After the stoplist, the median section has 146 distinct shingles, the 99th percentile 1,933 and the largest 261,326. The signatures take 1.05 GB.

**LSH bands** (same command and file, `lsh`). The band index takes 0.79 GB. At 32 × 4 with the cap at 1,000, the bands propose 20,934,941 distinct candidate pairs, of which 452,713 (2.2%) cross state lines. The largest bucket holds 1,847 sections. Summed over bands, the cap skips 37 buckets and the 28,317,289 pairs inside them, none spanning two states, so no cross-state candidate is lost. The cap sweep:

| Cap | Buckets skipped | Of those, spanning more than one state | Most states in a skipped bucket | Cross-state candidates lost |
|---|---|---|---|---|
| 100 | 3,710 | 60 | 35 | 92,783 |
| 200 | 1,750 | 33 | 35 | 55,231 |
| 500 | 357 | 2 | 2 | 2,005 |
| 1,000 | 37 | 0 | 1 | 0 |
| 2,000 | 0 | 0 | 0 | 0 |

**Verification and families** (same command and file, `pairs` and `families`). 14,361,827 of the 20,934,941 candidates verify at an estimate of 0.5 or more, 197,548 of them across state lines, and 6,573,114 are dropped. The stored cross-state pairs take 1.8 MB. 9,643,736 pairs of same-state bills share a verified section. 433,189 of them reach a share of 0.8 in both directions, and they form 63,424 families of two or more bills, 165,933 bills in all. The largest holds 764. The share sweep:

| Share | Linked bill pairs | Families of two or more | Bills in them | Largest family |
|---|---|---|---|---|
| 0.5 | 995,973 | 66,213 | 205,682 | 1,514 |
| 2/3 | 640,544 | 66,223 | 183,402 | 815 |
| 0.75 | 452,152 | 64,796 | 171,337 | 764 |
| 0.8 | 433,189 | 63,424 | 165,933 | 764 |
| 0.9 | 413,642 | 61,306 | 158,136 | 764 |
| 1.0 | 406,501 | 60,208 | 154,480 | 764 |

**MinHash engines** (`uv run python -m benchmarks.minhash` → `reports/minhash_benchmark.json`). On one core of an Apple M1 Max, signing every section took 271 s with the NumPy implementation and 105 s with datasketch's batch path. Signatures hold 512 and 785 bytes per section. Building the LSH index over the same stored signatures took 11.5 s with NumPy, adding 0.70 GiB, at a peak of 2.2 GiB with the signatures included. datasketch's insertion session reached the 16 GiB bound after 2,025,000 of the 2,058,712 sections, holding 14.9 GiB of index, and the benchmark stopped it. See [DR-3](#dr-3).

**To be measured.**
- Commission recall, pooled and by act (`lineage eval commission`, M3).
- Stage ablation: recall and precision for each stage alone and both together (M4).
- Cost and runtime per stage (M5).

## Decision records

Each record gives the context, the options weighed, the decision, its consequences, and what would reverse it. The module that implements a decision links back to its record.

<a id="dr-1"></a>
### DR-1 · Front end: Next.js static export

**Status:** accepted, 2026-11-05.

**Context.** The hosted site is a set of read-only views over precomputed JSON: a landing page, featured stories with a state tile map and a timeline, a side-by-side comparison with highlighted spans, a browse page that filters the hosted state's bill list in the browser, a page per bill, and validation and baseline pages. Nothing may run server-side. Several views are interactive, so some client-side code ships whatever the framework. The data contract is the JSON that `lineage export` writes, with its schema emitted by the Pydantic models that produce it.

**Options.**

- **Next.js with `output: "export"`.** React components, file-based routing, and per-bill pages generated at build time from the exported bill list. Static export gives up server features (route handlers, on-demand revalidation, image optimization) that this site has no use for. The cost is that every page ships the React runtime.
- **Astro.** Ships no JavaScript by default and hydrates interactive islands on their own. Pages would be lighter. But the comparison and browse views still need a component framework inside their islands, so the project would carry two component models instead of one.
- **Python templating (Jinja) into plain HTML.** Keeps one language. The interactive views would become hand-written JavaScript with nothing checking them against the JSON contract, which is where schema drift between pipeline and site would surface as runtime bugs.
- **Streamlit or a dashboard framework.** Needs a running server, which breaks the static constraint, and presents an internal-tool look to a public audience.

**Decision.** Next.js static export in TypeScript, for three technical reasons:

- **Pre-rendering at scale.** Static export renders every hosted bill's page at build time from the exported bill list, so thousands of pages ship as plain files with nothing running server-side.
- **One data contract.** The Pydantic models that write the export also emit its JSON Schema, and the site's TypeScript types are generated from that schema. The pipeline and the site therefore share one definition of every field.
- **Interactive views in one component model.** React components carry the state tile map, the timeline, the highlighted comparison and the browse filter.

Existing team expertise with Next.js is a secondary, maintainability factor: it lowers the cost of keeping the front end current.

**Consequences.**

- The Python side owns the schema, and the site consumes it as generated types. A breaking change to the export fails the site's type check, not a user's browser.
- The repo carries two toolchains, uv and npm. CI gains a front-end job in M7.
- Build time and output size grow with the number of hosted bills. Both are measured in M7.
- Every page pays for the React runtime. Page weight is measured in M7 and reported.

**Would reverse it.** If page weight on the bill or comparison page breaks the budget set for phones on slow connections in M7, the presentation layer moves to Astro. The JSON contract and the generated types carry over unchanged, so only the presentation layer would be rewritten.

<a id="dr-2"></a>
### DR-2 · Host: GitHub Pages

**Status:** accepted, 2026-11-05.

**Context.** The site is static files, one JSON file per hosted bill plus a few aggregate files. File count and total size are the binding constraints. The build runs in this repository's CI, so the deploy should need no credentials beyond what that CI already has.

**Options.**

- **GitHub Pages**, deployed by this repo's Actions with `actions/deploy-pages`. The deploy authenticates with the workflow's own short-lived token, so the project holds no secrets or external accounts. GitHub [documents](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits) a 1 GB limit on a published site and a soft bandwidth limit. Response headers cannot be customized.
- **Cloudflare Pages.** A faster edge and configurable headers. Its free plan caps a site at 20,000 files ([limits](https://developers.cloudflare.com/pages/platform/limits/)), so one file per bill leaves little headroom even for one state, and it needs an external account plus a deploy token stored as a repo secret.
- **Netlify or Vercel.** Comparable to Cloudflare, with the same external-account and secret cost.
- **Object storage (R2 or S3) behind a CDN.** No practical file or size limits, but it needs a custom domain, bucket configuration and credentials, which is more moving parts than a single-state site needs.

**Decision.** GitHub Pages, deployed from this repository's workflow.

**Consequences.**

- No secrets: the deploy job requests the `pages: write` and `id-token: write` permissions on its own job and nothing more.
- The 1 GB limit holds the hosted site to one state. Hosting more states is out of scope for this host.
- The site is served under `/<repo>/`, so the Next.js `basePath` must match.
- Caching headers are GitHub's defaults. JSON is small and per-bill, so this costs little.

**Would reverse it.** If the hosted state's export exceeds about half the 1 GB limit (measured in M7), which leaves too little room for growth, or if hosting more than one state becomes a goal. The response is the sharded object-storage layout in [Scaling limits](#scaling-limits).

<a id="dr-3"></a>
### DR-3 · MinHash engine: an own NumPy implementation, with datasketch as the test oracle

**Status:** decided, 2026-11-05. The step 11 measurement met the reversal criterion as first written. That criterion left out two requirements, and against the completed criterion the decision stands. Both versions of the criterion are below.

**Context.** The near-duplicate stage signs every section in the corpus and buckets the signatures by LSH band. The hot paths are hashing each shingle under every permutation, taking the minimum per permutation, hashing each band to a key, and grouping equal keys. At corpus scale, any per-element Python work in those loops dominates runtime and memory. The implementation also has to be simple enough to verify by reading.

**Options.**

- **[datasketch](https://github.com/ekzhu/datasketch).** Mature and widely used. Its batch path (`MinHash.bulk`, `update_batch`, and `MinHashLSH.insertion_session` for buffered inserts) avoids per-shingle Python loops. It still represents each set as a Python object and stores LSH buckets in dict-backed storage, so per-object overhead and the memory of millions of buckets are the costs to measure.
- **An own NumPy implementation.** Signatures form one dense unsigned-integer matrix with one row per section. Shingles are hashed once to 32 bits, and each permutation is a universal hash (*a*·*x* + *b*) mod *p*, applied to all shingles in a single vectorized operation. Band keys are hashed to 64 bits, and candidate pairs come from sorting the keys and reading off runs of equal values, not from a dict.
- **A compiled third-party MinHash library** with Rust or C++ bindings. Likely the fastest, but it adds a binary dependency and hides the algorithm behind bindings that can't be read or tested line by line.
- **A distributed engine (Spark).** Solves a scale this corpus doesn't have, and needs a cluster.
- **A hybrid:** datasketch's signing feeding this project's band index. Weighed after step 11, below.

**Decision.** An own NumPy implementation. datasketch stays as a development-only dependency and serves as the test oracle: on the same inputs and parameters, both must produce agreeing similarity estimates.

**The criterion as first written.** It named two binding constraints. The full corpus must be signed within an overnight run, and the LSH index must fit in memory, both on the reference laptop. That is the machine the benchmarks run on, and each benchmark output records its CPU, core count and memory. Fitting in memory means that a full index build, signatures plus LSH index, peaks within 16 GiB, half of the reference laptop's 32 GiB, leaving the rest for the operating system and the later stages. The bound was fixed before the index was measured. The criterion read:

> If, on the reference laptop, datasketch builds signatures for the full corpus within an overnight run (eight hours of wall-clock time) *and* its LSH index fits in memory, then datasketch meets the constraints that justified owning the code, and it becomes the engine. A raw speed ratio is not the test: a slower library that meets both constraints is still the cheaper thing to maintain.

The benchmark puts both engines through the same work. It runs datasketch through its batch path, never a per-shingle loop, so the comparison is with the library at its best. It records wall-clock time and peak memory for building the signatures and the LSH index, on the full corpus.

**Measured in step 10** (`uv run python -m benchmarks.minhash` → `reports/minhash_benchmark.json`; Apple M1 Max, 10 cores, 32 GiB). The comparison concerns cost per section, so it is re-run when the corpus changes by more than 5%. Step 10's run gave each engine one core and the same 1,916,250 stoplisted sections. Splitting at citation headings later raised the signed sections to 2,058,712, 7.4% more, so the benchmark was re-run on them, 475,914,210 shingles in all (`engines.numpy.sections` in the report). Making the commission's standard sections boilerplate then left 2,058,452 (`reports/index.json`), 0.01% fewer. The figures below are from the re-run. Step 10's report is in git history (`git show 714f085:reports/minhash_benchmark.json`), and its signing times differ from the re-run's by under 4%.

- **Signing time.** The NumPy implementation took 271 s and datasketch 105 s. The context above expected per-element Python work to dominate datasketch's time. On this corpus it does not.
- **Signature memory.** The NumPy matrix holds 512 bytes per section and datasketch's objects 785 (`tracemalloc`, on a 20,000-section sample), or 1.05 GB against 1.62 GB for the corpus.

**Why signing is slower here.** The cost is the permutation arithmetic, not data layout. datasketch 2.0's default scheme computes *a*·*x* + *b* in wrapping 32-bit integers: two array operations on 4-byte values for each shingle and permutation. The exact (*a*·*x* + *b*) mod *p* takes about sixteen operations on 8-byte values (`minhash._hash_into`), and NumPy runs each as a separate pass over the run's arrays. A profile on one core found that hash to be nearly all of the signing time, and changing the run length or removing the two temporary arrays it allocates did not change the time materially. No small fix remains. Fusing the passes needs compiled code, which the options above rejected. A multiply-add modulo 2<sup>32</sup> would change every stored signature, and its accuracy would rest on empirical tests rather than on the pairwise independence of the mod-*p* family. Signing runs once per snapshot, in parallel, so the gap is left as it is.

**Measured in step 11** (same command; the report as of `714f085`). Each engine built its index of 32 bands × 4 rows in a fresh process from the stored signatures table, loaded into memory. datasketch inserted every row of the 1,916,250 through `MinHashLSH.insertion_session`, under a watchdog that would have stopped the build at the memory bound or the disk floor. It did not stop.

- **Peak memory.** NumPy peaked at 2.1 GiB and datasketch at 15.3 GiB, 0.7 GiB inside the 16 GiB bound. Peak memory is the process's physical footprint, which on macOS includes compressed pages that the resident set size leaves out.
- **Index memory.** The index added 0.66 GiB with NumPy, its two dense matrices, and 14.3 GiB with datasketch, about 8 KB per section in dictionaries of byte strings and sets.
- **Build time.** 10 s and 571 s.

datasketch therefore met both conditions of the criterion as first written, with 4% of the memory bound to spare.

**The re-run** (`reports/minhash_benchmark.json`, `lsh`). On the 2,058,712 sections, datasketch reached the 16 GiB bound after inserting 2,025,000, holding 14.9 GiB of index, and the watchdog stopped it. NumPy built the whole index in 11.5 s, adding 0.70 GiB, at a peak of 2.2 GiB. On the current corpus, datasketch fails even the criterion as first written.

**The criterion was incomplete.** It left out two requirements that the project's plan had set before the benchmark ran, though this document did not state them until now ([Constraints](#constraints)). The prebuilt index ships as release assets, each under GitHub's 2 GiB per-file limit. `lineage query` runs offline on a user's laptop after one download of those assets. Both were added to the criterion after the step 11 result was known. That is the kind of after-the-fact change that fixing the memory bound in advance was meant to rule out, so it is stated here for the reader to weigh.

**Against the full criterion, datasketch fails.**

- Its index is 14.3 GiB of Python dictionaries inside the building process. It has no file format and cannot be memory-mapped, so every query would first have to bring all of it into memory, whether by loading a serialized copy split across release assets or by rebuilding it.
- Rebuilding it on the user's machine from the signatures took 571 s on the reference laptop in step 11 and peaked at 15.3 GiB, more than a laptop with 16 GiB of memory can hold.
- The 4% of headroom that step 11 left on the reference laptop was gone after 7.4% more sections.

This project's index is two `.npy` matrices, 0.79 GB in all (`reports/index.json`, `lsh.index_bytes`), each under the limit. A query memory-maps them and reads only the pages its keys land on.

**The hybrid, rejected.** datasketch's signing feeding this project's band index would meet the full criterion. It would save at most the 165 s of single-core signing time that the re-run measured, less once spread over the worker processes of a build that runs once per snapshot. In exchange, datasketch would become a runtime dependency rather than a test oracle, and the signatures would depend on its hash scheme staying fixed across releases.

**Consequences.**

- Correctness is this project's to own. The oracle test and the S-curve test of the empirical candidate rate are what make that acceptable.
- A dense signature matrix memory-maps and shards without conversion, which is the first response to [index growth](#scaling-limits).
- Sorting to bucket costs *O*(*n* log *n*) with predictable memory, where dict buckets fragment the heap.
- Time and peak memory for both engines are a recorded measurement, not an assumption. The benchmark stays, so the comparison can be rerun on a new snapshot or a new datasketch release.

**Would reverse it.** Revisit this decision if a maintained library offers a persistent LSH index that memory-maps and that a full build produces within the 16 GiB bound, or if the offline query path is dropped. If the NumPy implementation itself misses the overnight or memory bound, a compiled library replaces it. In either case the oracle tests carry over.

<a id="dr-4"></a>
### DR-4 · Data tooling: DuckDB over Parquet

**Status:** accepted, 2026-11-05.

**Context.** The input is Open States' per-session JSON for 50 states across four years: 220 archives, 6.42 GiB (`uv run lineage snapshot`, `snapshot/manifest.json`). The transformations are relational: flatten nested actions and versions, derive status, filter, and aggregate per state and session for coverage statistics. Their outputs feed NumPy, for hashing and embedding, and the exporter. Everything runs on a laptop and has to work when the data does not fit in memory.

**Options.**

- **pandas.** Ubiquitous, but eager and in-memory. Normalizing nested JSON is awkward, and object-dtype columns are weakly typed under a strict type checker.
- **Polars.** Fast, lazy, able to stream larger-than-memory data, and well typed. A close second.
- **DuckDB.** Reads nested JSON in place with SQL (`read_json`, `unnest`), spills to disk when memory runs short, writes Parquet natively, and hands columns to NumPy through Arrow without copying. Its CLI makes the intermediate tables easy to inspect directly.
- **SQLite.** A row store with no native Parquet or JSON-array handling. Aggregations over the full corpus would be slow.
- **The standard library's `json` module with Python loops.** The simplest option, but slow and memory-heavy at corpus scale.

**Decision.** DuckDB reads the raw JSON and writes Parquet tables (`bills`, `sections`). Relational work stays in SQL, and numeric work moves to NumPy through Arrow.

**Consequences.**

- Parquet is the interchange format between pipeline stages, readable by any tool. A stage can be rewritten without touching its neighbors.
- SQL inside Python strings is invisible to pyright. Queries live in one module, run against committed fixtures in tests, and their result schemas are checked where they enter typed code.
- The choice over Polars is a judgment, not a measurement: reading nested JSON with SQL and inspecting tables in the CLI outweighed Polars' typed expression API.

**Would reverse it.** If DuckDB's JSON reader cannot handle the snapshot's shape or size without per-file Python preprocessing, which step 4 finds out, a streaming Python reader writes the Parquet and DuckDB keeps the queries. If most transforms turn out to need row-wise Python functions, Polars replaces DuckDB.

<a id="dr-5"></a>
### DR-5 · Match only document types that can carry law

**Status:** accepted, 2026-11-05. The type lists and per-state rules were set from the step 8 counts.

**Context.** An Open States session archive holds every measure a legislature filed, not just bills. It includes resolutions, proclamations, appointments, nominations, memorials and commemorations. Commemorative measures are numerous and highly formulaic ("commending", "recognizing", "expressing condolences"), so they flood the near-duplicate stage with matches on text no one is tracing. Type labels also vary by state, and a generic label can hide both kinds of record. Georgia, for example, files proposed constitutional amendments and commendations alike as "resolution", and its titles carry the distinction instead: a trailing "; commend" or "; recognize" against "; provide - CA". So a single global allowlist of types would either drop substantive measures or admit commemorations.

**Options.**

- **Match everything.** No record is lost, but commemorative boilerplate dominates the candidate pairs and the precision sample, and the stoplist ends up working around document types instead of legal boilerplate.
- **Match only records labeled `bill`.** Simple, but it drops substantive measures that states file under other labels, such as constitutional amendments and joint resolutions with the force of law.
- **Per-state rules over document type.** A default list of types eligible for matching, with per-state overrides where a state's labels don't separate law from ceremony. An override may use that state's own title convention, as in Georgia. Commemorative and memorial measures are always excluded.
- **A topic or keyword filter.** Rejected: it would make commission recall circular, since only bills that match the keywords could be found, and it would remove the unrelated bills that test precision.
- **A model classifying each record.** Rejected: it puts a model call in the retrieval path, which breaks the keyless constraint.

**Decision.** Per-state rules over document type, in one committed table, `rules/document_types.csv`. Rules are read top to bottom and the first that fits a record decides it. `lineage eligible` applies them and counts every rule's records per state in `reports/document_types.json`.

- **Matched by default:** bills, appropriation bills, proposed bills, joint resolutions, and records classified as constitutional amendments. Joint resolutions stay in because many states use them to propose constitutional amendments, and some give them the force of law.
- **Excluded by default:** simple and concurrent resolutions, memorials, commemorations, proclamations, appointments, nominations and study requests. They are formula-heavy: commendations, condolences, requests to Congress, and procedural motions written to a few templates. Matched, they would crowd every bill page with relatives that share only those templates, and their formulas would claim stoplist slots meant for legal boilerplate.
- **Per-state exceptions** admit constitutional amendments that a state files under an excluded type, by that state's title convention: New Jersey's and Mississippi's concurrent resolutions that propose amending the constitution, and Georgia's resolutions whose titles end in "-CA". Each is a row of the table, and the report gives its count.
- **Commemorative joint resolutions** are excluded by a title pattern. The title must open, after any type prefix, with commending, commemorating, congratulating, honoring, paying tribute, recognizing, celebrating or mourning. A title that names a constitution, an amendment, a proposal, a right or a law is spared. The report counts the titles each rule excluded and spared, per state, and `lineage stats --samples` lists them locally, so false positives in both directions can be checked. Tennessee files every memorial as a joint resolution titled "Memorials, <occasion> - <honoree> -", for recognitions, deaths, retirements, academic and sports achievements, interns and petitions to Congress alike. Those titles name the occasion rather than opening with a verb, and an honoree's name or cause can carry a word the guard reads as law, so a Tennessee row excludes the prefix with no guard. It was found when 830 graduation memorials formed the corpus's largest family. "Memorials, Congress" titles call on Congress, so they are kept.
- **Commemorative and procedural joint resolutions** are excluded whatever their wording, by one row per kind: namings of highways, bridges and roads; designations of a day, week, month or year; confirmations of appointees; adjournments, recesses, joint sessions and conventions, joint rules and joint orders; and claims against the state. A joint resolution that expresses a position or calls on Congress is kept, and so is one that proposes changing the process it names, such as a constitutional amendment on confirmations. Each row's guard says what it spares.
- **Title rules apply only to joint resolutions and to the per-state exceptions.** Bills are never excluded by title: a bill naming a bridge or designating a day is still law.

**Counts** (`uv run lineage eligible` → `reports/document_types.json`). 78,244 of 525,633 records are excluded: 53,165 simple resolutions, 9,470 concurrent resolutions, 5,783 Tennessee memorials filed as joint resolutions, 2,265 proclamations, 1,635 appointments, 1,546 other commemorative joint resolutions, 780 memorials of the three kinds, 423 nominations, 188 commemorations and 24 study requests. Of joint resolutions, 1,074 designate an observance, 701 decide claims against the state, 602 confirm appointees, 357 are procedural and 231 name a road or bridge. The exceptions admit 451 New Jersey concurrent resolutions, 122 Mississippi concurrent resolutions and 102 Georgia resolutions. No record has a type the table doesn't list.

The filter does not bias the recall eval. It selects on what kind of document a record is, independent of its subject, so it removes records that aren't law without regard to any story or act. The unrelated bills on every topic, which are the hard negatives that test precision, all remain.

**Consequences.**

- Excluded records stay in `bills` with their classification, and the filter is applied downstream (`data/tables/eligible.parquet`). Changing a rule means editing the table and rerunning `lineage eligible` and `lineage boilerplate`, with no re-ingest. The choice is reversible by configuration.
- The stoplist is counted over matchable sections only, so the excluded types can't shift it.
- A substantive measure filed under an excluded type can't be found. The recall eval reports "excluded type" as a distinct miss reason, so such a loss shows up rather than hiding inside "not found".
- Known gap: Nebraska's, Michigan's and Rhode Island's resolutions include constitution-titled records whose conventions are not yet checked, so they stay excluded. Each is admitted by a rule once its convention is checked and its count measured.

**Would reverse it.** If the recall eval shows any listed enactment lost to "excluded type", the rule that excluded it is revised. If the precision sample shows commemorative text still producing matches, the rules widen. The exclusion of resolutions as a whole is reopened when a user wants to trace resolution campaigns, such as the same request to Congress filed in many states. That use needs the excluded types matched as their own corpus, not mixed into bill pages.
