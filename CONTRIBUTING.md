# Contributing

Before every commit, run the four checks CI runs:

```sh
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

## Leak check

Several pipeline commands write reports under `reports/` that are committed. A report that records where it read or wrote a file can carry the path of the machine that ran it. One report written from a temporary directory did, and the path was caught by hand before it was committed. `tools/leak_check.py` catches it automatically. It fails on:

- **local paths:** home directories (`/Users/…`, `/home/…`) and temporary directories (`tmp` at the root, and macOS's `private/tmp` and `var/folders`), matched as written;
- **deny-listed terms:** a comma-separated list in `LEAK_CHECK_TERMS`, matched in any case.

It checks every tracked file and, with `--history`, every commit's diff, message, author and committer. It reports a match by file or commit and kind, never by the matched text. Run it before every push:

```sh
LEAK_CHECK_TERMS=... uv run python -m tools.leak_check --history
```

CI runs the same check on every push and pull request, with the deny list from the `LEAK_CHECK_TERMS` repository secret. GitHub gives no secrets to runs in a fork or to pull requests from one, so there the step is skipped with a notice, and the check runs once the change reaches this repository. Here the step fails while the secret is unset, so a missing secret can't pass the check on local paths alone.

## Labeling matches

Precision, the share of matches that are right, has no answer key, so it is measured on labeled samples ([DESIGN.md](DESIGN.md#precision-sample)). The labels reported so far are a model's, checked against one person's on 50 pairs. Labels from people with legal training are the stronger evidence, and they are kept and estimated apart from the model's, never pooled with them.

**What you need.** The samples name each section by its Open States bill id and section number and hold no bill text, so labeling reads the sections from a local build of the corpus (`data/tables/`, see the commands in [CLAUDE.md](CLAUDE.md#commands)). With the semantic stage's model files (`uv run lineage model fetch`), paraphrased sentences are marked too.

**Labeling a sample.**

```sh
uv run lineage label evals/precision_sample.jsonl --labeler YOUR_GITHUB_HANDLE
uv run lineage label evals/relatives_tuning.jsonl --labeler YOUR_GITHUB_HANDLE --labels evals/relatives_labels.jsonl
```

The first is the pair-level sample: cross-state section pairs drawn per stage. The second holds one listed relative of each of 100 random Georgia bills, the measure of what a bill's page shows. `evals/relatives_reporting.jsonl` is its twin, kept for reporting.

Each pair is shown as two sections, with copied text in green, paraphrased sentences in yellow and unmatched text dimmed. Give it one label:

| Label | Meaning |
|---|---|
| 0 unrelated | different subjects |
| 1 partially related | a shared subject, but different provisions |
| 2 related | the same provision or purpose, worded or detailed differently |
| 3 almost identical | the same provision in mostly the same words, with small edits |
| 4 identical | the same text, apart from formatting, numbering or place names |

A match counts as correct from 2 up. Judge the text alone. You are not told which stage found a pair, so don't look the bills up. `s` skips a pair and `q` stops. Each label is saved as you give it, so the next run starts where you stopped.

**What a label records.** Each side's bill id and the SHA-256 of its section's normalized text, so a label outlives a re-split of the corpus, the label, your handle as `github:HANDLE` and the date. Never your name: the tool refuses anything that isn't a GitHub handle. Commit the labels file in a pull request.

**What would help most.** The model's labels on the relatives samples answer a narrower question than the scale above: do the passages the alignment matched state the same provision, beyond boilerplate? That question has not been checked against a person. Labels on `evals/relatives_tuning.jsonl` and `evals/relatives_reporting.jsonl` would check it, label 2 or above taken as yes.
