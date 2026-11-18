#!/usr/bin/env bash
# The live track's scheduled job as launchd starts it: `daily` or `weekly` (DESIGN.md#updates).
# Every step is a lineage CLI command that `lineage schedule` runs in order, writing its status
# under the data directory and raising a notification when a step fails or a day is held for
# review. This script only sets the environment up, and alerts when the job cannot start at all.
# From the first day of session (rules/deploy.toml) the daily job deploys; this script does not.
# Usage: scheduling/run.sh daily|weekly
set -uo pipefail
job="${1:?usage: scheduling/run.sh daily|weekly}"
cd "$(dirname "$0")/.." || exit 2
alert() {
  osascript -e "display notification \"$1\" with title \"Legislative Lineage\"" >/dev/null 2>&1
}
case "$job" in
  daily|weekly) needed=(LINEAGE_DATA LINEAGE_STATE LINEAGE_SITE_URL) ;;
  *) echo "usage: scheduling/run.sh daily|weekly" >&2; exit 2 ;;
esac
for name in "${needed[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    alert "The $job job did not start: $name is not set."
    exit 2
  fi
done

# The intake's key is read from the login keychain rather than kept in the launchd plist, so no
# file holds it; it reaches lineage only through the environment.
if [[ "$job" == daily && -z "${OPENSTATES_API_KEY:-}" ]]; then
  if ! OPENSTATES_API_KEY="$(security find-generic-password -w -s legislative-lineage-openstates 2>/dev/null)"; then
    alert "The daily job did not start: no Open States key in the keychain."
    exit 2
  fi
  export OPENSTATES_API_KEY
fi

# The leak check's deny list, as CI gets it from a repository secret, from the keychain here.
# Only the deploy's leak check reads it, and it refuses to pass without it, so a missing entry
# fails the deploy step, recorded and alerted, rather than the job's start.
if [[ -z "${LEAK_CHECK_TERMS:-}" ]]; then
  if LEAK_CHECK_TERMS="$(security find-generic-password -w -s legislative-lineage-leak-terms 2>/dev/null)"; then
    export LEAK_CHECK_TERMS
  fi
fi

# The job's GitHub access is not set here: the deploy reads its own token from the keychain
# (legislative-lineage-gh-token) before every git or gh call and stops without it, so it never
# falls back to the maintainer's login (lineage.deploy.github_shell).

# caffeinate holds off idle sleep while the job runs: the Mac is woken for it (pmset), and a
# rebuild runs for hours, longer than the Mac stays awake on its own.
if [[ "$job" == daily ]]; then
  caffeinate -i uv run --frozen lineage schedule daily --state "$LINEAGE_STATE" --site-url "$LINEAGE_SITE_URL"
else
  caffeinate -i uv run --frozen lineage schedule weekly --state "$LINEAGE_STATE" --site-url "$LINEAGE_SITE_URL"
fi
code=$?
# Exit code 1 is a step that failed and 3 a day held for review, which lineage has already
# recorded and alerted on.
if [[ $code -ne 0 && $code -ne 1 && $code -ne 3 ]]; then
  alert "The $job job could not run (exit code $code)."
fi
exit $code
