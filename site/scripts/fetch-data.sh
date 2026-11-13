#!/usr/bin/env bash
# Fetch the site's data from the GitHub Release that data-release.json pins, verify it against
# the SHA-256 committed there, and unpack it into public/data with release.json for the data
# stamp (DESIGN.md#export). The digest comes from the reviewed commit, not from the release, so
# a replaced asset fails here; a data change reaches the site only as a commit to the manifest.
# Usage: scripts/fetch-data.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
manifest="$here/data-release.json"
field() { node -e 'process.stdout.write(String(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"))[process.argv[2]] ?? ""))' "$manifest" "$1"; }
tag="$(field tag)"
asset="$(field asset)"
sha="$(field sha256)"
# The fields reach a download pattern and file names, so only the release naming is accepted.
[[ "$tag" =~ ^data-[a-z]{2}-[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "not a data release tag: $tag" >&2; exit 2; }
[[ "$asset" =~ ^export-[A-Z]{2}-${tag}\.tar\.gz$ ]] || { echo "not this release's asset: $asset" >&2; exit 2; }
[[ "$sha" =~ ^[0-9a-f]{64}$ ]] || { echo "not a SHA-256: $sha" >&2; exit 2; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

gh release download "$tag" --pattern "$asset" --dir "$work"
printf '%s  %s\n' "$sha" "$asset" > "$work/expected.sha256"
(cd "$work" && if command -v sha256sum >/dev/null; then sha256sum -c expected.sha256; else shasum -a 256 -c expected.sha256; fi) \
  || { echo "$asset does not match the digest committed in data-release.json" >&2; exit 1; }

rm -rf "$here/public/data"
mkdir -p "$here/public/data"
tar -xzf "$work/$asset" -C "$here/public/data"
cp "$manifest" "$here/public/data/release.json"
echo "$asset verified against data-release.json -> public/data"
