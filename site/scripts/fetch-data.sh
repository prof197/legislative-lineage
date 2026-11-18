#!/usr/bin/env bash
# Fetch the site's data from the GitHub Release that data-release.json pins, verify it against
# the SHA-256 committed there, then verify every unpacked file against the checksum manifest
# inside it (checksums.json: each file's path, size and SHA-256), with no file missing and none
# extra, and only then put it in public/data with release.json for the data stamp
# (DESIGN.md#export). The digest comes from the reviewed commit, not from the release, so a
# replaced asset fails here; a data change reaches the site only as a commit to the manifest.
# Usage: scripts/fetch-data.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
manifest="$here/data-release.json"
checksums="checksums.json"
# Packed before releases carried checksums.json. The committed digest still pins every byte of
# it; the manifest adds a per-file account. Any other release without one is refused.
pre_manifest=(data-ga-2026-11-12)
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

# Every file on disk against the manifest, and the manifest against the files: a listed file
# missing, a file not listed, a size or a digest that differs. Problems are all named at once.
verify="$(cat <<'JS'
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const [root, name] = process.argv.slice(1);
const fail = (lines) => {
  for (const l of lines.slice(0, 20)) console.error(l);
  if (lines.length > 20) console.error(`... and ${lines.length - 20} more`);
  process.exit(1);
};
let files;
try {
  const doc = JSON.parse(fs.readFileSync(path.join(root, name), "utf8"));
  files = doc && doc.files;
  const entry = (f) => f && Object.keys(f).length === 3 && typeof f.path === "string"
    && Number.isSafeInteger(f.size) && f.size >= 0 && /^[0-9a-f]{64}$/.test(f.sha256);
  if (Object.keys(doc).length !== 1 || !Array.isArray(files) || !files.every(entry)) {
    throw new Error("not a list of files, each with only its path, size and sha256");
  }
} catch (e) {
  fail([`${name} is not a checksum manifest: ${e.message}`]);
}
const listed = new Map(files.map((f) => [f.path, f]));
if (listed.size !== files.length) fail([`${name} is not a checksum manifest: a path listed twice`]);
const found = [];
const walk = (dir, rel) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const r = rel ? `${rel}/${e.name}` : e.name;
    if (e.isDirectory()) walk(path.join(dir, e.name), r);
    else found.push([r, e]);
  }
};
walk(root, "");
const problems = [];
for (const [r, e] of found.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
  if (r === name) continue;
  const f = listed.get(r);
  if (!f) { problems.push(`${r}: not listed in ${name}`); continue; }
  listed.delete(r);
  if (!e.isFile()) { problems.push(`${r}: not a regular file`); continue; }
  const data = fs.readFileSync(path.join(root, r));
  if (data.length !== f.size) problems.push(`${r}: ${data.length} bytes, not ${f.size} as ${name} lists`);
  else if (crypto.createHash("sha256").update(data).digest("hex") !== f.sha256) problems.push(`${r}: does not match its SHA-256 in ${name}`);
}
for (const r of listed.keys()) problems.push(`${r}: missing, though listed in ${name}`);
if (problems.length) fail(problems);
console.log(`${files.length} files verified against ${name}`);
JS
)"

mkdir "$work/data"
tar -xzf "$work/$asset" -C "$work/data"
if [[ -f "$work/data/$checksums" ]]; then
  node -e "$verify" "$work/data" "$checksums" \
    || { echo "$asset does not match the checksum manifest inside it" >&2; exit 1; }
elif [[ " ${pre_manifest[*]} " == *" $tag "* ]]; then
  echo "$tag predates the checksum manifest: verified as a whole against data-release.json"
else
  echo "$asset holds no $checksums; only a release packed before it existed may lack one" >&2
  exit 1
fi

# Replaced only once verified, so a failed fetch leaves the data already there untouched.
rm -rf "$here/public/data"
mkdir -p "$here/public"
mv "$work/data" "$here/public/data"
cp "$manifest" "$here/public/data/release.json"
echo "$asset verified against data-release.json -> public/data"
