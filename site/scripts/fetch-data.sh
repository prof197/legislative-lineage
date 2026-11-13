#!/usr/bin/env bash
# Fetch the site's data from its GitHub Release, verify it against the published SHA-256, and
# unpack it into public/data with release.json for the data stamp (DESIGN.md#export).
# Usage: scripts/fetch-data.sh TAG [STATE]
set -euo pipefail
tag="$1"
state="${2:-GA}"
asset="export-${state}-${tag}.tar.gz"
here="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

gh release download "$tag" --pattern "$asset" --pattern "$asset.sha256" --dir "$work"
(cd "$work" && if command -v sha256sum >/dev/null; then sha256sum -c "$asset.sha256"; else shasum -a 256 -c "$asset.sha256"; fi)

rm -rf "$here/public/data"
mkdir -p "$here/public/data"
tar -xzf "$work/$asset" -C "$here/public/data"
sha="$(cut -d' ' -f1 "$work/$asset.sha256")"
printf '{"tag":"%s","asset":"%s","sha256":"%s"}\n' "$tag" "$asset" "$sha" > "$here/public/data/release.json"
echo "$asset verified -> public/data"
