# CLAUDE.md

## Commands

```sh
uv sync --extra build        # install the locked environment, with faiss for the index build
uv run ruff check            # lint
uv run ruff format --check   # formatting (drop --check to apply)
uv run pyright               # type check, strict
uv run pytest                # tests
uv run lint-imports          # the layering rule, as import-linter contracts in pyproject.toml
uv run python -m tools.module_size  # no module over 800 lines but those allowed
uv run python -m tools.dead_code    # no top-level name nothing reads but those in tools/dead_code_allow.txt
uv run python -m tools.report_check  # the offline evals rerun, every committed report unchanged (a move or refactor, dev only)
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
uv run lineage index         # signatures, LSH bands, verified pairs, families -> data/tables/{signatures,bands,pairs,same_state_pairs,families}/, reports/index.json
uv run lineage embed         # every distinct 94-word window embedded once, float16, resumable -> data/tables/semantic/, reports/embedding.json
uv run lineage neighbors     # each distinct window's 20 nearest by meaning, HNSW over float16, recall checked exactly -> data/tables/semantic/<model>/neighbors/, reports/neighbors.json
uv run lineage neighbors --check-at-stage  # the built neighbors' recall at the stage's threshold, no build -> reports/neighbors.json
uv run lineage semantic-pairs  # the neighbors as cross-state section pairs, best cosine and window pair each -> data/tables/semantic/<model>/pairs/, reports/semantic_pairs.json
uv run lineage precision-sample  # the precision sample, stratified by stage, from both stages' pairs -> evals/precision_sample.jsonl, reports/precision_sample.json
uv run lineage precision-sample --flagged  # a seeded sample of the pairs the state boilerplate flag drops -> evals/precision_sample_flagged.jsonl, reports/precision_sample_flagged.json
uv run lineage state-boilerplate  # each section's count of other same-state families its near-duplicates reach; flagged at 50 -> data/tables/state_boilerplate/, reports/state_boilerplate.json
uv run lineage query FILE     # a bill's relatives by both stages, offline, from the built index and vectors (--no-semantic: near-duplicates only)
uv run lineage eval commission  # commission recall, act texts from data/acts/ -> reports/results.json (--results reports/results_exported_only.json after an --exported-only merge); with the stage ablation unless --no-semantic
uv run lineage eval threshold  # each semantic threshold option's page precision on the tuning sample of relatives, beside development recall -> reports/threshold_proposal.json
uv run lineage eval answer-key  # each commission list row and the bills it resolved to -> reports/answer_key.csv
uv run lineage eval answer-key --held-out  # the held-out lists resolved, development-set bills skipped -> reports/answer_key_heldout.csv
uv run lineage eval answer-key --held-out --origins  # the reserve acts' lists resolved, origin recall's sealed key -> reports/answer_key_origins_heldout.csv
uv run lineage eval commission --held-out --final  # held-out recall: run at the end of M5 (2026-11-12), at 0.93 and then at the final configuration; refuses a configuration already run
uv run lineage eval readme   # regenerate README.md's summary and results tables from reports/results.json, results_exported_only.json and relatives_reporting.json
uv run lineage label SAMPLE --labeler HANDLE  # label sampled matches by hand, 0 unrelated to 4 identical, blind to the stage, resumable -> evals/precision_labels.jsonl
uv run lineage label-page SAMPLE --labeler HANDLE  # the same labeling as a phone page: pending pairs, aligned, blind -> data/label_page/ (local), published as a private artifact
uv run lineage label-import SAVED --sample SAMPLE --labeler HANDLE  # the page's saved labels, checked against the sample -> evals/precision_labels.jsonl
uv run lineage featured     # the featured stories' section pairs: each anchor against its listed relatives -> evals/featured_pairs.jsonl, reports/featured_pairs.json
uv run lineage explain PAIRS  # grounded explanations from the cache -> explanations/cache/ (--live asks the model for uncached pairs within the budget; ANTHROPIC_API_KEY)
uv run lineage eval faithfulness PAIRS  # model-judged: do the kept claims describe their cited spans correctly? with the hard check's result -> reports/results.json (--live asks the judge for unjudged claims within the budget -> evals/faithfulness_judgments.jsonl)
uv run lineage pair-data fetch  # the labeled pair dataset at its pinned commit -> data/pair_data/ (local, evaluation only, never committed)
uv run lineage eval pair-data  # both stages scored on the pair dataset; each model's threshold set on train+val, measured on test -> reports/pair_data.json
uv run lineage eval cost  # cost and runtime per stage: the build's from its reports, each development act's query timed by stage -> reports/cost.json
uv run lineage relatives-sample [--part tuning_093|reporting]  # 100 Georgia bills at random, one listed relative each: the threshold's first tuning sample at 0.8905 (tuning_093: the second, read at 0.93 with the flag; reporting: drawn once at the final threshold, from bills no earlier sample drew) -> evals/relatives_<part>.jsonl, reports/relatives_<part>.json
uv run lineage eval relatives [--part tuning_093]  # product precision, from the reporting sample's labels (or a tuning part's, recorded apart) -> reports/results.json
uv run lineage label-spend --batch-id ID  # record an ended label batch's tokens and cost -> reports/label_spend.json
uv run lineage agreement-sample  # 50 of the precision sample's pairs, stratified by stage, for a person to label -> evals/agreement_sample.jsonl
uv run lineage eval agreement  # a person's labels against the model's on that subsample: weighted kappa, cutoff agreement -> reports/results.json
uv run lineage label-model SAMPLE  # provisional labels from a model, kept apart from people's; asks nothing without --live
uv run lineage eval precision  # precision per stratum, stage and overall, people's and the model's apart -> reports/results.json
uv run lineage collect-models SOURCE  # one source's model legislation -> data/modeldocs/<source>/, reports/modeldocs/<source>.json
uv run lineage origins       # collected model documents against the corpus -> data/tables/origins/matches.jsonl, reports/origins.json (needs the built index and semantic vectors; --report-only: count the table as it stands)
uv run lineage eval origins  # origin recall on the development answer key -> reports/results.json (origins_development); --key v1_held_out: v1's held-out acts, opened in v1, extra evidence; --key held_out --final: the sealed key, once per configuration
uv run lineage origin-sample  # seeded origin precision sample (--part tuning), label cost projected, nothing asked -> evals/origin_precision_tuning.jsonl, reports/origin_precision_sample.json; --part reporting --final once matching is final (--part reporting_after_fix: the fresh sample after the generic-section fix)
uv run lineage origin-label  # the sample's matches labeled by a model on each one's strongest section pair, v1's passage question; asks only with --live, through the Batches API, under v2's $40 -> evals/origin_precision_labels.jsonl, reports/origin_label_spend.json
uv run lineage withhold     # model documents whose labeled listed matches are mostly wrong (at least 3, under half correct), withheld from pages, the feed and stories -> reports/withheld_documents.json (--doc ID --note TEXT: withheld by hand after a held day's review, kept on every rewrite)
uv run lineage eval origin-threshold  # each candidate share floor's precision on the tuning sample, beside its development recall -> reports/origin_threshold_proposal.json
uv run lineage origin-sample --part census_eppc  # every EPPC match a page lists, a census, label cost projected -> evals/origin_precision_census_eppc.jsonl
uv run lineage eval origin-census  # the census read: precision as a count, each wrong match's section pairs by generic kind -> reports/results.json (origins_census_eppc)
uv run lineage eval origin-precision  # the reporting sample after the generic-section fix read once (--part reporting: the reading before it, kept as origins_precision_before_fix): overall, by stage, by source, ALEC against the rest -> reports/results.json (origins_precision); --intervals-only: regenerate a stored reading's intervals, refused if anything else moves
uv run lineage intake       # one day of new and changed bills from Open States' API v3 (key from OPENSTATES_API_KEY), text through the text layer -> data/days/<day>/, data/intake/; --tier default|bronze|silver
uv run lineage intake-estimate  # the API requests a January day needs, against each tier, from the snapshot; no network -> reports/intake_estimate.json
uv run lineage update --day DIR  # apply one day's new and changed records -> data/changes/<as_of>.jsonl and the tables in place, held to a full rebuild over the same days
uv run lineage rebuild --out DIR [--pinned-from TABLES]  # the folded days laid over the snapshot's archives, the whole pipeline run again, statistics pinned from tables at the same day or counted afresh, vectors reused for the same texts -> DIR, DIR.build/rebuild.json
uv run lineage text check-plan  # fail unless the stored fetch plan was built from the code checked out (or none is stored); the weekly job's first step
uv run lineage text inventory  # version to fetch per record, by host, robots.txt read -> data/text/plan.parquet, reports/text_inventory.json
uv run lineage text sample   # quality test: fetch a seeded sample per state and format -> data/text/sample/, reports/text_quality.json
uv run lineage text fetch    # fetch every planned record as rules/text_layer.toml says, California from its bulk download; resumable -> data/text/full/, reports/text_fetch.json
uv run lineage export --state GA  # the static site's JSON for one state -> data/export/GA/ (local), reports/export.json
uv run lineage export --schema   # the export's JSON Schema, the site's contract -> schema/export.schema.json
uv run lineage export --state GA --package data-ga-YYYY-MM-DD  # the export as a release asset, checksums.json inside, refused unless every file is one the contract names -> data/export/, pinned in site/data-release.json
uv run lineage publish pages --state GA  # the live track: the logged days the export lacks, only the files they touch rewritten, into a staging copy -> data/export/GA.staging/, data/publish/GA.json
uv run lineage publish feed --site-url URL  # the national Atom feed of the live track's events, checked against RFC 4287 -> <staging>/feed.xml
uv run lineage publish promote --state GA  # swap the staging copy in as the live export; deploys nothing
uv run lineage publish weekly --pinned DIR  # the weekly rebuild's check: agreement with the updated tables, failing below 98%, and the drift -> data/schedule/weekly.json
uv run lineage publish deploy --state GA  # in session: the live export's day packed as a release, gated (contract, local paths, copy, size), the circuit breaker, the pin committed and leak-checked, then the GitHub release, push and Pages dispatch; exits 3 when held -> data/deploy/days/<day>.json
uv run lineage publish release --day D  # a day the circuit breaker held, approved after review (--repack: the live export as it stands, after withholding the documents behind bad claims; RUNBOOK.md)
uv run lineage publish reject --day D  # a held day rejected for an infrastructure problem: never published, its data kept, the next day publishes as usual
uv run lineage publish candidate --site-url URL  # before session: the live export packed and gated as a release candidate, publishing nothing -> data/deploy/candidates/<tag>/review.txt
uv run lineage publish review --site-url URL  # in session: the week's new origin claims for the maintainer's review -> data/deploy/reviews/<day>.txt
uv run lineage schedule daily --site-url URL  # the daily job: intake, update, pages, feed, promote, and from the first day of session (rules/deploy.toml) the deploy; a failed step or held day stops it and alerts -> data/schedule/status.json
uv run lineage schedule weekly --site-url URL  # the weekly job: two rebuilds and the agreement check, then a candidate before session or the origin-claims review during it
uv run lineage schedule plist daily --out FILE --site-url URL  # fill a launchd template from scheduling/ at the ruled times (daily Monday to Saturday 05:30, weekly Saturday 20:00, local); loading it is by hand
uv run python -m evals.fixture  # rebuild tests/fixtures/commission/ from the local snapshot, dev only
uv run python -m benchmarks.minhash  # NumPy vs datasketch: signing and LSH index time and memory (DR-3), dev only
uv run lineage model fetch   # the pinned embedding models' files, hash-checked -> data/models/<name>/
uv run python -m benchmarks.semantic_units  # semantic stage sizing per window scheme: units, tokens, rate, bytes, dev recall -> reports/semantic_sizing.json, dev only
uv run python -m benchmarks.embed_throughput  # embedding rate per batch order, thread count and worker processes -> reports/embed_throughput.json, dev only
uv run python -m benchmarks.ann  # IVF-PQ and HNSW against exact search on a sample of the embedded corpus -> reports/ann_benchmark.json, dev only
LEAK_CHECK_TERMS=... uv run python -m tools.leak_check --history  # no local path or deny-listed term (any case) in any file or commit; CI runs it
```

All seven checks must pass before every commit. CI runs the same seven commands, the site's checks over both fixtures, then the leak check, whose deny list comes from the `LEAK_CHECK_TERMS` repository secret. Run the leak check before every push. On pull requests to main, data.yml downloads the pinned data release, verifies it, builds the site from it and runs the copy check.

## Layout

```
src/lineage/      the package (import name `lineage`; distribution `legislative-lineage`); layers and the layering rule in DESIGN.md#module-map
  cli/            Typer apps (app.py) and one module of commands per layer, the labeling commands apart (labels.py); subcommands stay thin wrappers over library functions
  align.py        word-level alignment of a section pair: copied, paraphrased and unmatched spans, with ids and offsets (pure functions)
  core/           the bottom layer: paths, the disk floor, memory, intervals, digests, DuckDB, HTTP
    paths.py        DATA: LINEAGE_DATA or data/, the root of every default local path (worktrees share it)
    disk.py         free-space floor checked before large writes
    memory.py       current and peak memory as the OS charges it (macOS physical footprint)
    intervals.py    the Wilson interval
    hashing.py      SHA-256 of a file, as reports and manifests record it
    dirs.py         a table directory written aside and swapped in, for every layer's tables; two ways a failed write is left
    db.py           DuckDB connections, each with its spill cap; the SQL literal helpers (DR-4)
    http.py         HTTP as this project (User-Agent, TLS intermediates, moved hosts, retries); open_url, streamed downloads
    pace.py         one request at a time per host, its interval apart, across every process (locked files)
    robots.py       robots.txt rules for this project's User-Agent; temporary refusals asked again
    certs/intermediates.pem intermediate certificates some legislature servers omit
  data/           Open States' bill data: the snapshot, sessions, ingest, status, document types, stats
    snapshot.py     archive manifest: scan, write, load, verify
    archives.py     download the archives the manifest pins, byte for byte
    sessions.py     session dates from openstates-scrapers, parsed as source, never run
    ingest.py       raw archives -> bills Parquet, one file per archive, with status
    status.py       passed / died / introduced from action classifications
    doctypes.py     document-type rules (DR-5): which records are matched
    stats.py        coverage report: the measurements that replace the design's estimates
    queries.py      this package's SQL, and only its own (DR-4)
  text/           bill text and sections: versions, the text layer, sections, boilerplate, windows
    versions.py     which bill version the text layer fetches (pure functions)
    inventory.py    `lineage text inventory`: unusable text per record, the fetch plan, per-host counts
    extract.py      text from HTML or PDF (pypdf), lines kept, struck text dropped
    fetch.py        polite per-host fetching: robots, pacing, backoff, resumable JSON-lines store; extraction in worker processes
    rules.py        rules/text_layer.toml: the usable-text bar, per-state formats and readings, the replacement rule
    quality.py      `lineage text sample`: the quality test, fetched text against Open States'
    run.py          `lineage text fetch`: the full fetch's jobs from the plan and the rules, and its report
    ca_bulk.py      California's text from its official bulk download, one biennium at a time
    merge.py        `lineage text merge`: the texts table, each record's text and its source; later stages read text only from it
    sections.py     clean, split at headings (with per-state heading rules) or into fallback windows, normalize (pure functions)
    sectioning.py   build the sections table, one worker process per bills file
    boilerplate.py  pattern list that flags boilerplate-only sections
    generic_sections.py a section's generic kind by its heading or opening, within its kind's bound: definitions, short title, findings, severability and the rest, or lineage.text.boilerplate's kinds
    windows.py      the matching unit: a section of up to 94 words whole, a longer one in overlapping 94-word windows (pure functions)
    queries.py      this package's SQL, and only its own (DR-4)
  match/          matching: near below semantic, below the modules that combine both stages
    query.py        `lineage query`: sign a bill's text, look up its bands, verify, group by family, score by the share of its words matched
    relatives.py    a corpus bill's relatives from both stages' stored pairs, scored as a query is, the 50 best kept
    listing.py      what a page lists: matches at the stage's threshold, less the state boilerplate flag's but for pairs both stages found
    state_boilerplate.py `lineage state-boilerplate`: a section flagged as its state's boilerplate by the other families it reaches
    strata.py       both stages' stored cross-state pairs by stratum, which the samples draw from
    queries.py      this package's SQL, and only its own (DR-4)
  match/near/     the near-duplicate stage (DR-3)
      shingles.py     word 5-gram shingles, 32-bit hashes (NumPy; values pinned by a test)
      stoplist.py     document-frequency stoplist and its report
      minhash.py      MinHash signatures, exact (a·x + b) mod p (pure functions; values pinned)
      signatures.py   build the signatures table, one row per window, one worker process per sections file
      lsh.py          LSH banding: band keys, sorted-array buckets, the cap (pure functions)
      bands.py        build the band index from the signatures; cap counter and cap sweep
      verify.py       re-estimate candidates from full signatures; the verification threshold
      families.py     same-state families: bill match shares, connected components (pure functions)
      pairs.py        verify window candidates, store cross-state and same-state section pairs, write families
      index.py        `lineage index`: signatures, bands, pairs and families, then reports/index.json
      queries.py      this package's SQL, and only its own (DR-4)
  match/semantic/ the semantic stage
      models.py       the embedding models: pinned repository revision and file hashes; fetched through lineage.core.http.download
      embed.py        unit-length vectors from a pinned model, offline (fastembed, ONNX), batched by token count
      vectors.py      `lineage embed`: plan the distinct windows, embed them in shards by worker processes, resume where it stopped
      ann.py          nearest neighbors over unit vectors: exact search, and faiss indexes (build extra), typed
      neighbors.py    `lineage neighbors`: every distinct window's nearest neighbors for the build, HNSW over the float16 vectors
      stage.py        the semantic stage: its model, the build's threshold (the pair evaluation's) and the stage's (the tuning sample's), and a query's search by meaning
      pairs.py        `lineage semantic-pairs`: the semantic stage's matches as cross-state section pairs, fingerprinted by model, K and threshold
      queries.py      this package's SQL, and only its own (DR-4)
  explain/        the explanation step and the spend meter
    explanations.py the model's reply schema (Pydantic) and the span-grounding check that drops unbacked claims
    explain.py      `lineage explain`: the prompt from aligned spans, the SDK call, retry and withholding, the cache, and the cache's check
    structured.py   one structured-output request: the schema sent without the bounds the API ignores, the reply validated in code, every reply metered
    spend.py        M6's model calls metered against their $30 cap, projected before a run and checked before each call -> reports/explanation_spend.json
    queries.py      this package's SQL, and only its own (DR-4)
  evals/          the evaluations of matching
    acts.py         uniform act texts as eval queries: keep the sections, drop comments and notes
    ulc.py          the commission's enactment lists, and the bills each enactment names
    answer_key.py   `lineage eval answer-key`: the resolved list rows, for checking by eye
    commission.py   `lineage eval commission`: recall, a miss reason per enactment, the results JSON
    threshold.py    `lineage eval threshold`: a page's precision under each threshold option and with the flag; the pair-level diagnostic at the stage's threshold
    pair_data.py    the labeled pair dataset (Kim et al., EMNLP 2021): pinned files, fetch, load; no license, so local only
    pair_eval.py    `lineage eval pair-data`: the stages scored on labeled pairs, metrics only
    precision.py    the precision sample's seeded draw, and the estimate weighted by stratum size
    labels.py       `lineage label`: a person labels the precision sample's pairs, aligned; labels keyed to survive a re-split
    label_page.py   `lineage label-page` and `label-import`: the labeling session as a phone page (pages/label.html, beside it), and its saved labels checked back in
    agreement.py    the subsample a person labels to check the model's labels, and the agreement: weighted kappa, cutoff agreement, the bars
    model_labels.py provisional precision labels from a model, under its id and prompt version; each batch's spend against the budget
    relatives_sample.py the precision of what a page lists: random bills' listed relatives, the pair each rests on, its aligned passages
    faithfulness.py `lineage eval faithfulness`: a model judges each kept claim against its cited spans, verdicts cached without text, the Wilson share
    flag_precision.py the state boilerplate flag's precision: the sample with the flag on, the dropped pairs' sample
    cost.py         `lineage eval cost`: each stage's build time, memory and size from its report, and a query's seconds by stage
    queries.py      this package's SQL, and only its own (DR-4)
  origins/        documented origins: model legislation collected, matched and measured
    modeldocs.py    model legislation: each source's entry, each collected document, texts per version under data/modeldocs/
    modeltext.py    model texts as matching reads them: sections kept, commentary and drafting instructions dropped; Word text
    collect.py      the collector registry: one module per source under collectors/, found by name; the coverage report
    collect_base.py what collectors share: robots.txt per host, paced requests, provenance, storing, the report
    collectors/     one collector module per source of model legislation, each owned by the lane that builds it
    collectors/commission.py the commission's acts: those on disk, then its site's (off, SITE_FETCH, until its terms are ruled)
    matches.py      OriginMatch, both shares and the bill's spans, and the origins table
    matching.py     `lineage origins`' body: each model document queried as `lineage query` queries a bill, through both stages and the page listing rule
    recall.py       `lineage eval origins`: model-to-bill recall from the resolved answer keys; a sealed key only with --final
    precision.py    `lineage origin-sample` and `eval origin-precision`: seeded samples of origin matches, the tuning part by share band, the reporting part by stage at the floor; v2's label budget; the reporting reading by source
    labels.py       `lineage origin-label`: a match's strongest section pair, both sections and their passages, asked in one batch; rows by text hash
    threshold.py    `lineage eval origin-threshold`: each candidate floor's weighted precision and development recall
    withheld.py     `lineage withhold`: the documents withheld from pages because their labeled matches are mostly wrong, as a committed report the export, the live track and the feed read
    queries.py      this package's SQL, and only its own (DR-4)
  export/         what leaves the pipeline: the data contract, the export, stories, the release
    schema.py       the site's data contract: every exported file as a Pydantic model, which emits the JSON Schema
    build.py        `lineage export`: bill pages, comparisons, section texts, stories and the data stamp for one state
    validation.py   the validation page's figures, read from the committed results and the withheld report
    featured.py     `lineage featured`: the three featured stories, their anchors and relatives rules, and the section pairs they explain
    origin_story.py the origin story's rule, which names no organization, and its file (DESIGN.md#export)
    release.py      the data release: the export contract's files, packing with its checksum manifest, pinning (DESIGN.md#export)
    atom.py         Atom (RFC 4287): the namespace and each way a document breaks the rules
    results_table.py the README's summary and results tables, rendered from the committed reports
    queries.py      this package's SQL, and only its own (DR-4)
  live/           the live track: intake, updates, rebuild, publishing, the schedule, the deploy
    openstates_api.py API v3's bills request and its bills mapped into the bulk archive's record shape
    intake.py       `lineage intake`: paged by updated_at, pages recorded, text through the text layer, days written once; the January estimate
    updates.py      the live track's contracts (DR-7): days, records over time, the change log, the snapshot an update must equal
    incremental.py  `lineage update`'s body: apply_day and update, the journal, the embedder registry, the free-space bound
    incremental_records.py  the day's bills, eligibility, texts and sections written into the staging view
    incremental_near.py  signatures renumbered, the band index, both pairs tables, families, the carried boilerplate counts
    incremental_semantic.py  the day's windows, vectors, neighbors and semantic section pairs
    incremental_relatives.py  the bills whose relatives can have changed, and their relatives and origins before and after
    staging.py      the view an update builds the tables in before swapping it in
    rebuild.py      `lineage rebuild`: the folded days over the snapshot's archives, the pipeline run again with pinned or refreshed statistics (DR-7)
    publish.py      `lineage publish pages|promote`: the pages a day's changes touch, rewritten in a staging copy, and the swap (DESIGN.md#updates)
    feed.py         `lineage publish feed`: the national Atom feed of the live track's events, checked against RFC 4287
    schedule.py     `lineage schedule`: the daily and weekly jobs under launchd, their status (held included) and alerts; the weekly agreement check and drift
    deploy.py       `lineage publish deploy|release|reject|candidate|review`: the day's release, a held day's release or rejection, the candidate and the origin-claims review (DR-13)
    deploy_ledger.py  the deploy's day records and paths
    deploy_shell.py  every external command through an injectable shell; the keychain token and github_shell
    deploy_gates.py  packing the release and its gates: contract, local paths, the deny list, the site's checks
    deploy_rules.py  rules/deploy.toml and the circuit breaker
    deploy_publish.py  the publish: commit, leak check, release, push, Pages dispatch
    queries.py      this package's SQL, and only its own (DR-4)
benchmarks/       dev-only measurements that need dev dependencies (datasketch)
scheduling/       the live track's launchd templates and run.sh; nothing loads them
tools/            dev and CI checks
  leak_check.py   local paths and deny-listed terms in tracked files and every commit
evals/            eval inputs: ulc_enactments.csv (hand copy of the commission's lists, dated), ulc_enactments_heldout.csv (the held-out acts' lists, reserves included), precision_labels.jsonl (the precision sample's labels, keyed by Open States bill id and section text hash; labelers by GitHub handle or model id; no text)
  fixture.py      dev-only: builds tests/fixtures/commission/ from the snapshot
tests/            pytest; small committed fixtures only
  fixtures/openstates/  trimmed real records in archive layout (README + *_bills.json)
  fixtures/commission/  enacting bills among uniform-act distractors, same layout, + sessions.csv
  fixtures/collect/  synthetic responses in the commission site's shape, for its collector's tests
  fixtures/intake/  API v3 responses constructed in the documented shape, synthetic bills and sponsors
  days.py         recorded days for the live track, built at test time from fixtures/commission/, and the reference rebuild
  live.py         the live track built once per session over the recorded days: updated tables, a pinned and a refreshed rebuild, for the publish, feed and schedule tests
  fixture_export.py  builds fixtures/export/: Utah's bills, synthetic model documents, hand-made origins (LINEAGE_REGENERATE=1 rewrites it)
  fixtures/export/  the fixture export every v2 lane builds and tests against (committed, generated)
  collector_fixtures/  two synthetic sources, one leaked, for the registry and the fixture export
snapshot/manifest.json  the pinned snapshot: per-archive size, SHA-256, Generated At
snapshot/sessions.csv   session start/end dates and carryover, from a pinned scrapers commit
rules/document_types.csv  which document types are matched, per state; first fitting rule wins
rules/text_layer.toml  the text layer's rules: usable bar, per-state format and PDF reading, content elements, replacement
rules/stories.toml  which featured stories each hosted state's site carries
rules/deploy.toml  each hosted state's first day of session, from which the daily job deploys, and the circuit breaker's and size gate's numbers
schema/export.schema.json  the export's JSON Schema, generated from lineage.export.schema; a test fails when it drifts
reports/          committed outputs of pipeline commands; README and DESIGN numbers cite them
  results.json    eval results, one section per eval; the README results table reads it
  results_exported_only.json  commission recall on the export's text alone, for the comparison in DR-6
  answer_key.csv  each commission list row and the corpus bills it resolved to
  answer_key_heldout.csv  the held-out lists resolved to bills; no matching is run on them before the final evaluation
  answer_key_origins_heldout.csv  the reserve acts' lists resolved to bills, origin recall's sealed key
  pair_data.json  the stages on the labeled pair dataset: precision, recall, per label; metrics only
  text_inventory.json  records without usable text, the versions planned, hosts and their robots.txt
  text_quality.json  the quality test per state and format: outcomes, usable and clean rates per reading, comparison
  text_fetch.json  the full fetch per state: outcomes, and records left out with the reason
explanations/cache/  model replies, one JSON file per (model, prompt version, spans) hash; replayed without a key
DESIGN.md         design, alternatives, scaling limits, failure modes, decision records
CONTRIBUTING.md   the checks to run before a commit and a push, and the leak check
RUNBOOK.md        the live track's setup, schedule, and what to do when a job fails or holds a day
OUTCOMES.md       where the tool has been used on real work, with permission
data/             local snapshots and derived tables; gitignored, never committed
  raw/<ST>/*.zip  Open States session archives, never edited
  acts/<act>.txt  uniform act texts, downloaded from the ULC under its terms; never committed
  acts/held-out/<act>.txt  the held-out acts' texts, read only by the parser check and the final run
  tables/         Parquet tables written by the pipeline; bills/ holds the source's records under the project's names, texts/ each record's chosen text; signatures/, bands/, pairs/ and same_state_pairs/ hold .npy arrays
  text/           the text layer's fetch plan and fetched text (JSON lines per host)
  changes/        the live track's change log, one JSON-lines file per day
  modeldocs/<source>/  model legislation as collected: docs.jsonl, versions.jsonl, raw/ and texts/ by SHA-256; never committed
  tables/semantic/  the semantic stage's windows (units/), distinct texts (texts/), and each model's float16 vector shards
  pair_data/      the labeled pair dataset's split files, from `lineage pair-data fetch`; never committed
  models/<name>/  embedding model files at their pinned revision, from `lineage model fetch`
  export/<ST>/    the static site's JSON for one state, from `lineage export`
  rebuild/        the weekly job's two rebuilds, pinned/ and refreshed/, each with its .build/ archives' manifest and reports
  deploy/         the deploy's day records, release candidates, weekly reviews, and checkout/, the job's own clone of the repository (set up by hand)
```

## Conventions

- Python 3.13, pyright strict, ruff. Dependencies are pinned in `uv.lock`. Add a dependency with `uv add`, and only after it is agreed. faiss-cpu is the `build` extra: the index build needs it, a query does not.
- A module imports only from its own layer and the layers below it (DESIGN.md#module-map); a new module goes in the lowest layer that can hold it.
- Small modules with explicit interfaces. Use plain functions, dataclasses and Pydantic models, not class hierarchies or metaprogramming.
- A module that implements a decision record links to it in its docstring, e.g. `See DESIGN.md#dr-3`.
- Comments explain why, not what.
- Write tests first wherever DESIGN.md or the build plan specifies the behavior.
- Never weaken a test, threshold or eval to get to green.
- Every number in README.md or DESIGN.md comes from a committed eval or benchmark output, with the command that reproduces it. Never hard-code a result.
- Docs drafted ahead of a measurement mark each number still to come with the one placeholder token `tests/test_placeholders.py` defines; that test fails while any remain, so a draft number cannot be committed or ship.
- Fix random seeds. The snapshot is pinned by `snapshot/manifest.json`; an archive's as-of date is its README's Generated At.
- Before a large write, call `lineage.core.disk.require_free_space` with an upper bound on the bytes written. The default floor is 20 GiB free.
- Every run that touches the network goes through the `lineage` CLI, never a one-off script or `python -c`: archives, robots.txt and bill text alike. If no command does what is needed, add one to the CLI first. The CLI's spawned extraction workers never re-run a fetch, and `fetch_all` refuses to run inside one.
- Every HTTP request goes through `lineage.core.http.open_url`, which waits for the host's turn in `lineage.core.pace`. Never pace requests any other way.
- Open DuckDB only through `lineage.core.db.connect_quiet` or `connect`, which cap a query's spill to disk at the same floor. This holds for tests and one-off probes too; a test fails on any other `duckdb.*()` call.
- Secrets come only from environment variables. `.env` is gitignored.
- Default paths to local data start at `lineage.core.paths.DATA`, never a literal `data/...`. Set `LINEAGE_DATA` to an absolute directory in `.claude/settings.local.json` (gitignored) so worktrees share the data.
- In v2 lanes, edit only the files your lane owns (DESIGN.md#lanes); add SQL only in your package's own `queries.py` (DR-4).
- Model calls happen only in the explanation step and the provisional precision labels. Retrieval runs with no API key.
- Commits use conventional messages, one build step per commit. The body gives the engineering reason: the tradeoff, the measurement, the alternative rejected.
