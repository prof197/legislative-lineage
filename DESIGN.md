# Design

Legislative Lineage takes one state bill and finds bills in other states that carry a similar provision. For each match it shows where the bill was introduced, what happened to it, and how its text differs. It indexes every state bill from the 2023–2026 sessions on a single laptop, matches them section by section in two stages that need no API key, and publishes the results for one state as a static site.

This document is a skeleton. Sections marked **To be measured** fill in as the build reaches them. Every number added here will name the command and the committed output it comes from.

## Contents

- [Problem](#problem)
- [Constraints](#constraints)
- [Pipeline overview](#pipeline-overview)
- [Data snapshot](#data-snapshot)
- [Alternatives considered](#alternatives-considered)
- [Scaling limits](#scaling-limits)
- [Known failure modes](#known-failure-modes)
- [Measured facts](#measured-facts)
- [Decision records](#decision-records)

## Problem

Model legislation spreads by copying. An interest group, a uniform-law body or another state's statute supplies text, and a legislature introduces it, sometimes verbatim and sometimes with changes that matter: a narrowed definition, or a dropped enforcement clause. Nothing on a bill's face says where its text came from, so anyone who wants to know where else a provision appeared has to search for it.

Three properties of the corpus make this hard:

- **Size.** Every bill from all 50 states over four years. Comparing every pair of sections is quadratic: *N* sections make *N*(*N* − 1)/2 pairs, so the method has to find candidates without enumerating pairs. *N* is measured in M1.
- **The unit is the section, not the bill.** A copied provision often sits inside a much larger bill, such as an elections omnibus or a budget. Similarity between whole bills dilutes it below any useful threshold.
- **Copying is not always verbatim.** Reordered clauses and substituted terms break exact-text methods, while generic legal phrasing ("the department shall…") makes meaning-based methods over-match.

The tool reports shared text and timing. It does not claim who wrote a bill, or which of two similar bills came first in any causal sense.

## Constraints

| Constraint | Consequence for the design |
|---|---|
| Retrieval needs no API key | Both matching stages run locally: hashing, then a local embedding model. A model is called only to write the change explanations, and those replies are cached so a keyless run can replay them. |
| Hosted output is static | All results are precomputed. The site serves JSON files and has no query server or hosted vector database. |
| Runs on one laptop | Data structures must fit in memory or stream from disk. Hot loops are vectorized or run in compiled code, never per-element Python. |
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

**Extraction failures.** The export has no error flag. A record whose `raw_text` is missing or blank counts as a failed extraction: it stays in the `bills` table but cannot match anything. Step 8 reports the rate per state and session.

## Alternatives considered

| Alternative | Reason rejected | Evidence |
|---|---|---|
| Compare all pairs | Quadratic in the number of sections | Arithmetic, once *N* is measured (M1) |
| Embeddings only | No exact-span evidence to highlight, a higher cost per section, and weaker on legal boilerplate that sounds alike | To be measured by the stage ablation (M4) |
| Embeddings as a reranker of LSH candidates | A reranker can only reorder pairs LSH already proposed, so it can never recover a paraphrase LSH missed | Follows from the design |
| Whole bills as the unit | Dilutes provisions carried inside omnibus bills | To be shown by commission-recall misses (M3) |
| Scope the corpus by topic keywords | Makes recall circular, since only bills that match the keywords can be found, and it removes the hard negatives | Follows from the design |
| LLM calls or a hosted vector database in retrieval | Breaks the keyless and static constraints | Constraint |
| A query server behind the site | Breaks the static constraint | Constraint |
| Smith-Waterman alignment for highlighting | Gap scoring adds nothing to marking shared blocks; `difflib` finds them from the standard library | Follows from the design |
| LegiScan as the text source | Its [terms of service](https://legiscan.com/terms-of-service) restrict copying, which conflicts with publishing excerpts | Sourced |

## Scaling limits

**To be measured.** M1 measures the section count, which sets the size of the signature matrix and the embedding index. This table gets numbers then.

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
| Text extraction failed, or the source is a scanned image | No text, so no matches | Measure the rate per state; report it as a named miss reason |
| Amendment-style bills | New text sits inside old statute text, so matches can land on the old law | Drop struck text where its markup survives extraction; section-level scoring limits the damage |
| Omnibus bills | The provision is a small share of the bill | Score from the query side; display matches by section |
| Boilerplate leakage | False matches on shared clauses | Document-frequency stoplist, the LSH bucket cap, and the ablation's false-match rate |
| Legal phrasing that sounds alike | The semantic stage pairs unrelated sections | Label meaning-only matches as such, tune the cosine threshold, and run the precision sample |
| An explanation with valid citations but a wrong summary | A claim cites a real difference but describes it wrongly | A hand-labeled faithfulness eval, and the cited spans shown beside each claim |
| Similarity has no direction | An earlier bill is not necessarily the source | Copy states timing only, never "copied from" |
| Legislative status is not legal standing | A law struck down in court still shows as passed | A legend line wherever status is shown |
| A stale snapshot | Bills filed after an archive was generated are missing, and archives were generated at different times | Every page carries the Generated At dates of the archives it draws on |
| A substantive measure filed under an excluded document type | It can never be found | Excluded records stay in `bills`, and the recall eval names "excluded type" as a miss reason ([DR-5](#dr-5)) |

## Measured facts

**To be measured.** Each entry will name the command that produced it and the committed file it reads from.

- Records, and the share with missing or blank text, by state and session (`lineage stats`, M1).
- Record counts by document classification, per state (`lineage stats`, M1).
- The range of archive Generated At dates, per state (`lineage stats`, M1).
- Section-heading hit rate, and sections per bill (`lineage stats`, M1).
- Signature and LSH build time and peak memory on the reference laptop, for the own NumPy MinHash and for datasketch's batch path, on the same inputs (step 10).
- Candidate rate against the theoretical LSH S-curve (step 11).
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

**Status:** accepted, 2026-11-05. The supporting measurement is pending (step 10).

**Context.** The near-duplicate stage signs every section in the corpus and buckets the signatures by LSH band. The hot paths are hashing each shingle under every permutation, taking the minimum per permutation, hashing each band to a key, and grouping equal keys. At corpus scale, any per-element Python work in those loops dominates runtime and memory. The implementation also has to be simple enough to verify by reading.

The binding constraints are two. The full corpus must be signed within an overnight run, and the LSH index must fit in memory, both on the reference laptop. That is the machine the benchmarks run on, and each benchmark output records its CPU, core count and memory.

**Options.**

- **[datasketch](https://github.com/ekzhu/datasketch).** Mature and widely used. Its batch path (`MinHash.bulk`, `update_batch`, and `MinHashLSH.insertion_session` for buffered inserts) avoids per-shingle Python loops. It still represents each set as a Python object and stores LSH buckets in dict-backed storage, so per-object overhead and the memory of millions of buckets are the costs to measure.
- **An own NumPy implementation.** Signatures form one dense unsigned-integer matrix with one row per section. Shingles are hashed once to 32 bits, and each permutation is a universal hash (*a*·*x* + *b*) mod *p*, applied to all shingles in a single vectorized operation. Band keys are hashed to 64 bits, and candidate pairs come from sorting the keys and reading off runs of equal values, not from a dict.
- **A compiled third-party MinHash library** with Rust or C++ bindings. Likely the fastest, but it adds a binary dependency and hides the algorithm behind bindings that can't be read or tested line by line.
- **A distributed engine (Spark).** Solves a scale this corpus doesn't have, and needs a cluster.

**Decision.** An own NumPy implementation. datasketch stays as a development-only dependency and serves as the test oracle: on the same inputs and parameters, both must produce agreeing similarity estimates.

The step 10 benchmark puts both engines through the same work. It runs datasketch through its batch path, never a per-shingle loop, so the comparison is with the library at its best. It records wall-clock time and peak memory for building the signatures and the LSH index. Where a run completes on the full corpus, the full-corpus result is reported. Otherwise the result is extrapolated from a measured sample, and the report states the extrapolation.

**Consequences.**

- Correctness is this project's to own. The oracle test, plus a test of the empirical candidate rate against the theoretical S-curve (step 11), are what make that acceptable.
- A dense signature matrix memory-maps and shards without conversion, which is the first response to [index growth](#scaling-limits).
- Sorting to bucket costs *O*(*n* log *n*) with predictable memory, where dict buckets fragment the heap.
- Time and peak memory for both engines become a recorded measurement, not an assumption.

**Would reverse it.** If, on the reference laptop, datasketch builds signatures for the full corpus within an overnight run (eight hours of wall-clock time) *and* its LSH index fits in memory, then datasketch meets the constraints that justified owning the code, and it becomes the engine. A raw speed ratio is not the test: a slower library that meets both constraints is still the cheaper thing to maintain. If the NumPy implementation itself misses either bound, a compiled library replaces it, and the oracle tests carry over.

<a id="dr-4"></a>
### DR-4 · Data tooling: DuckDB over Parquet

**Status:** accepted, 2026-11-05.

**Context.** The input is Open States' per-session JSON for 50 states across four years. Its total size is measured in M1. The transformations are relational: flatten nested actions and versions, derive status, filter, and aggregate per state and session for coverage statistics. Their outputs feed NumPy, for hashing and embedding, and the exporter. Everything runs on a laptop and has to work when the data does not fit in memory.

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

**Status:** accepted, 2026-11-05. The per-state type lists are set from the step 8 counts.

**Context.** An Open States session archive holds every measure a legislature filed, not just bills. It includes resolutions, proclamations, appointments, nominations, memorials and commemorations. Commemorative measures are numerous and highly formulaic ("commending", "recognizing", "expressing condolences"), so they flood the near-duplicate stage with matches on text no one is tracing. Type labels also vary by state, and a generic label can hide both kinds of record. Georgia, for example, files proposed constitutional amendments and commendations alike as "resolution", and its titles carry the distinction instead: a trailing "; commend" or "; recognize" against "; provide - CA". So a single global allowlist of types would either drop substantive measures or admit commemorations.

**Options.**

- **Match everything.** No record is lost, but commemorative boilerplate dominates the candidate pairs and the precision sample, and the stoplist ends up working around document types instead of legal boilerplate.
- **Match only records labeled `bill`.** Simple, but it drops substantive measures that states file under other labels, such as constitutional amendments and joint resolutions with the force of law.
- **Per-state rules over document type.** A default list of types eligible for matching, with per-state overrides where a state's labels don't separate law from ceremony. An override may use that state's own title convention, as in Georgia. Commemorative and memorial measures are always excluded.
- **A topic or keyword filter.** Rejected: it would make commission recall circular, since only bills that match the keywords could be found, and it would remove the unrelated bills that test precision.
- **A model classifying each record.** Rejected: it puts a model call in the retrieval path, which breaks the keyless constraint.

**Decision.** Per-state rules over document type. Step 8 reports record counts by classification for each state. The review sets the default eligible types and any per-state overrides from those counts, and commemorative and memorial measures stay out regardless. The rules live in one committed table, so each exclusion can be reviewed.

The filter does not bias the recall eval. It selects on what kind of document a record is, independent of its subject, so it removes records that aren't law without regard to any story or act. The unrelated bills on every topic, which are the hard negatives that test precision, all remain.

**Consequences.**

- Excluded records stay in `bills` with their classification, and the filter is applied downstream. Changing the rules needs no re-ingest.
- Every rule is visible in one table, and step 8 reports how many records each rule excludes per state.
- A substantive measure filed under an excluded type can't be found. The recall eval reports "excluded type" as a distinct miss reason, so such a loss shows up rather than hiding inside "not found".

**Would reverse it.** If the recall eval shows any listed enactment lost to "excluded type", the rule that excluded it is revised. If the precision sample shows commemorative text still producing matches, the per-state rules widen.
