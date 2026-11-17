#!/usr/bin/env bash
# The live track's scheduled job as launchd starts it: `daily` or `weekly` (DESIGN.md#updates).
# Every step is a lineage CLI command that `lineage schedule` runs in order, writing its status
# under the data directory and raising a notification when a step fails. This script only sets
# the environment up, and alerts when the job cannot start at all. It deploys nothing.
# Usage: scheduling/run.sh daily|weekly
set -uo pipefail
job="${1:?usage: scheduling/run.sh daily|weekly}"
cd "$(dirname "$0")/.." || exit 2
alert() {
  osascript -e "display notification \"$1\" with title \"Legislative Lineage\"" >/dev/null 2>&1
}
case "$job" in
  daily) needed=(LINEAGE_DATA LINEAGE_STATE LINEAGE_SITE_URL) ;;
  weekly) needed=(LINEAGE_DATA) ;;
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

if [[ "$job" == daily ]]; then
  uv run --frozen lineage schedule daily --state "$LINEAGE_STATE" --site-url "$LINEAGE_SITE_URL"
else
  uv run --frozen lineage schedule weekly
fi
code=$?
# Exit code 1 is a step that failed, which lineage has already recorded and alerted on.
if [[ $code -ne 0 && $code -ne 1 ]]; then
  alert "The $job job could not run (exit code $code)."
fi
exit $code
