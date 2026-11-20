# Runbook: the live track

How the scheduled jobs run on the maintainer's Mac, how to set them up, and what to do when one stops or holds a day. Why each piece is built as it is: [DESIGN.md](DESIGN.md#updates) and [DR-13](DESIGN.md#dr-13).

## What runs when

| Job | When (Eastern, the Mac's zone) | What it does |
|---|---|---|
| Daily | Monday to Saturday, 05:30 | Intake, update, the incremental export and feed, promote; from Georgia's first day of session (2027-01-11, `rules/deploy.toml`) the deploy, behind its gates and circuit breaker |
| Weekly | Saturday, 20:00 | The fetch plan checked against the code (fails at once on a stale plan; run `uv run lineage text inventory`), two rebuilds and the agreement check (fails below 98%); before session a release candidate for review, during session the week's new origin claims |

The two never run at once: a job that finds the other running stops on the lock and alerts. A day without a daily run loses nothing, since intake resumes each state from its own cursor.

Every run writes `<data>/schedule/status.json` (`<data>` is `LINEAGE_DATA`) and its log to `<data>/schedule/daily.log` or `weekly.log`. A failed step or a held day raises a macOS notification.

## One-time setup

1. **The job's GitHub token.** On GitHub: Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token. Repository access: only this repository. Repository permissions: Contents, read and write; Actions, read and write (Metadata, read-only, is added by itself). Set an expiry and a reminder before it.
2. **The keychain.** Each command prompts for its value, so none reaches the shell's history:

   ```sh
   security add-generic-password -a "$USER" -s legislative-lineage-gh-token -w
   security add-generic-password -a "$USER" -s legislative-lineage-leak-terms -w
   security add-generic-password -a "$USER" -s legislative-lineage-openstates -w
   ```

   The deploy reads the token before every git or gh call and stops without it, so it never uses your own gh login (`lineage.live.deploy.github_shell`). To replace the token later, run the first command again with `-U` after `add-generic-password`.
3. **The job's own clone,** on main, where it builds the site, commits the pin and pushes:

   ```sh
   GH_TOKEN="$(security find-generic-password -w -s legislative-lineage-gh-token)" GH_CONFIG_DIR="$(mktemp -d)" \
     gh repo clone prof197/legislative-lineage "$LINEAGE_DATA/deploy/checkout" -- --branch main
   cd "$LINEAGE_DATA/deploy/checkout"
   git config user.name <your GitHub handle>
   git config user.email <your GitHub noreply address>
   uv sync --frozen && (cd site && npm ci)
   ```

4. **The launchd jobs,** filled at the ruled times, from your working checkout:

   ```sh
   uv run lineage schedule plist daily  --out ~/Library/LaunchAgents/local.legislative-lineage.daily.plist  --site-url URL
   uv run lineage schedule plist weekly --out ~/Library/LaunchAgents/local.legislative-lineage.weekly.plist --site-url URL
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.legislative-lineage.daily.plist
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.legislative-lineage.weekly.plist
   ```

5. **Waking the Mac** five minutes before each run. `pmset repeat` holds one pattern, so the Saturday evenings are scheduled one by one, a year at a time:

   ```sh
   sudo pmset repeat cancel
   sudo pmset schedule cancelall
   sudo pmset repeat wakeorpoweron MTWRFS 05:25:00
   for i in $(seq 0 51); do sudo pmset schedule wake "$(date -v+sat -v+${i}w +%m/%d/%y) 19:55:00"; done
   pmset -g sched
   ```

   Keep the Mac logged in, asleep rather than shut down, and on power: the jobs are your login's agents. The runner holds off idle sleep for the length of a job (`caffeinate -i`).

## Run the daily job once by hand

As launchd runs it, with the plist's environment:

```sh
launchctl kickstart -k gui/$(id -u)/local.legislative-lineage.daily
tail -f "$LINEAGE_DATA/schedule/daily.log"
```

Then read `$LINEAGE_DATA/schedule/status.json`: `"state": "ok"` with every step listed. Before Georgia's first day of session the job ends at the local export and publishes nothing.

## When a day is held

The notification and `status.json` name the day, its counts against the normal days, and the reasons; the day's record is `<data>/deploy/days/<day>.json`. Nothing was published. Decide which of these it is.

**A busy day, nothing wrong.** Approve it:

```sh
uv run lineage publish release --day YYYY-MM-DD
```

**Bad claims: origin matches that should not be listed.** Do not fix the tables. Withhold the documents behind the bad claims in the committed withhold report, then approve the day:

```sh
uv run lineage withhold --doc <source>/<slug> --note "what the review found"   # once per document
git add reports/withheld_documents.json && git commit -m "data(withhold): <document> after the <day> review" && git push
uv run lineage publish pages --state GA
uv run lineage publish feed --site-url URL
uv run lineage publish promote --state GA
uv run lineage publish release --day YYYY-MM-DD --repack
```

The sources page lists each such document as withheld after review. `--repack` packs the live export as it now stands, rewritten under the report, and refuses until it is; a plain release refuses once the report has changed, since the day's own release still lists the withheld documents. Days held behind it go out with it. A document withheld by review stays withheld when `lineage withhold` writes the report again.

**An infrastructure problem:** a broken fetch, a day written half way, a gate that failed for a reason outside the data. Reject the day:

```sh
uv run lineage publish reject --day YYYY-MM-DD
```

Its release is never published. Intake keeps its data, so the next good day's release carries it and publishes as usual. Reject is not a way to keep a day's data off the site.

**A deny-list hit.** The record names the files, never the term. If it is public record, approve the day. If it is not, stop the daily job before the next run carries it (`launchctl bootout gui/$(id -u)/local.legislative-lineage.daily`) and find where it came from.

The breaker's numbers are expected to hold days in the first week of session; those holds are reported, with adjustments proposed after week one.

## Weekly

During session, read the week's new origin claims in `<data>/deploy/reviews/<day>.txt`. Before session, `<data>/deploy/candidates/<tag>/review.txt` holds the release candidate's summary and the commands that would publish it.

## When a step fails

The notification names the step; `status.json` holds its command, exit code and the tail of its output. The live export stays the last good one, since nothing before the promote touches it. Fix the cause, then run the job again by hand (above). A deploy stopped part way resumes from its record without publishing twice.

## The release order

Every data release goes out in one order: **the release, then the pin PR, then the merge, then the deploy.**

1. Pack the export (`uv run lineage export --state GA --package data-ga-YYYY-MM-DD`), which pins it in `site/data-release.json`, and run its gates.
2. Create the GitHub release with the asset (`gh release create`) before the pin leaves the machine. The PR's data check (`data.yml`) downloads the release the pin names and verifies its digest, so the release has to exist first.
3. Commit the pin, run the leak check, push the branch and open the PR to main.
4. Merge the PR once its checks pass, with a merge commit.
5. Deploy: dispatch the Pages workflow on main (`gh workflow run pages.yml --ref main`), then check that the live `data/release.json` names the new tag.

If the PR does not merge, delete its release (`gh release delete data-ga-YYYY-MM-DD --cleanup-tag`), so no release stands that main does not pin.

The live track's deploy (`lineage publish deploy`, `lineage.live.deploy_publish`) keeps the same order: it creates the release, then pushes the pin, then dispatches Pages. In session the job's checkout pushes the pin to main itself, so the push stands in for the PR and the merge. A push that fails stops the day with the release up, and rerunning the job resumes at the push. If the pin cannot land (for example, main moved and the commit cannot be pushed), delete that day's release as above before the next run packs a new one.
