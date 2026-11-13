// The built site's size against GitHub Pages' 1 GB limit, by part: the bill pages Next renders,
// the export's data copied in, the shared scripts and styles, and the rest. Run after
// `npm run build`; writes ../reports/site.json (committed), which DESIGN.md#scaling-limits cites.
import { readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const OUT = path.resolve("out");
const LIMIT = 1_000_000_000;
const PART = { bills: "bill_pages", data: "data", _next: "scripts_and_styles" };

const parts = {};
let total = 0;
let files = 0;
function walk(dir, part) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, part);
    else {
      const bytes = statSync(full).size;
      const p = (parts[part] ??= { bytes: 0, files: 0, html: 0 });
      p.bytes += bytes;
      p.files += 1;
      if (entry.name.endsWith(".html")) p.html += 1;
      total += bytes;
      files += 1;
    }
  }
}
for (const entry of readdirSync(OUT, { withFileTypes: true })) {
  const part = PART[entry.name] ?? "other";
  if (entry.isDirectory()) walk(path.join(OUT, entry.name), part);
  else {
    const bytes = statSync(path.join(OUT, entry.name)).size;
    const p = (parts[part] ??= { bytes: 0, files: 0, html: 0 });
    p.bytes += bytes;
    p.files += 1;
    if (entry.name.endsWith(".html")) p.html += 1;
    total += bytes;
    files += 1;
  }
}
for (const p of Object.values(parts)) p.share_of_site = Number((p.bytes / total).toFixed(4));
const report = {
  command: "cd site && npm run build && npm run size",
  total_bytes: total,
  files,
  share_of_pages_limit: Number((total / LIMIT).toFixed(4)),
  parts: Object.fromEntries(Object.entries(parts).sort(([, a], [, b]) => b.bytes - a.bytes)),
};
writeFileSync(path.resolve("../reports/site.json"), JSON.stringify(report, null, 1) + "\n");
console.log(`${(total / 1e6).toFixed(1)} MB in ${files} files, ${(100 * report.share_of_pages_limit).toFixed(1)}% of the Pages limit -> reports/site.json`);
