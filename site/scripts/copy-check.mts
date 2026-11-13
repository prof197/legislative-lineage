// The copy rules, checked over the built site (`npm run build`, then `npm run test:copy`): no page,
// script string or displayed claim may use a phrase that states direction or authorship, except
// where ALLOWED says why the use is legitimate. Text that comes from the bills themselves (titles,
// headings, section text) is the legislatures' wording, not the site's, and is set aside first.
// See DESIGN.md#explanations and the copy rulings it records.
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { nameRoles } from "../src/lib/claims.ts";

const OUT = path.resolve("out");
const BANNED = ["descendant", "the origin", "changed", "copied", "copying", "written by", "who is really writing"];
const PATTERN = new RegExp(`\\b(${BANNED.map((b) => b.replace(/ /g, "\\s+")).join("|")})\\b`, "gi");

// Legitimate uses: where the phrase appears (a file name pattern) and the words around it.
const ALLOWED: { where: RegExp; context: RegExp; why: string }[] = [
  {
    where: /^_next\//,
    context: /Written by a language model from the aligned text/,
    why: "the claims panel discloses that a model wrote the claims; it names no author of a bill",
  },
];

// The check must be able to fail: every banned phrase matches itself.
for (const b of BANNED) {
  PATTERN.lastIndex = 0;
  if (!PATTERN.test(`a ${b} b`)) throw new Error(`the pattern misses "${b}"`);
}

type Hit = { file: string; phrase: string; context: string };
const hits: Hit[] = [];

function files(dir: string, keep: (f: string) => boolean, out: string[] = []): string[] {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) files(full, keep, out);
    else if (keep(full)) out.push(full);
  }
  return out;
}

const decode = (s: string) =>
  s.replace(/&#x27;|&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&nbsp;|&#160;/g, " ");

// Every string the export carries that holds a banned phrase: bill titles, headings, section text.
const fromBills = new Set<string>();
function collect(value: unknown): void {
  if (typeof value === "string") {
    PATTERN.lastIndex = 0;
    if (PATTERN.test(value)) fromBills.add(value);
  } else if (Array.isArray(value)) value.forEach(collect);
  else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) if (k !== "explanation") collect(v);
  }
}
for (const f of files(path.join(OUT, "data"), (f) => f.endsWith(".json"))) collect(JSON.parse(readFileSync(f, "utf8")));
const billStrings = [...fromBills].sort((a, b) => b.length - a.length);

function scan(file: string, text: string): void {
  for (const s of billStrings) if (text.includes(s)) text = text.split(s).join(" ");
  for (const m of text.matchAll(PATTERN)) {
    const context = text.slice(Math.max(0, m.index - 60), m.index + m[0].length + 60).replace(/\s+/g, " ").trim();
    const rel = path.relative(OUT, file);
    if (ALLOWED.some((a) => a.where.test(rel) && a.context.test(context))) continue;
    hits.push({ file: rel, phrase: m[0].toLowerCase(), context });
  }
}

// Pages: visible text and the attributes read aloud or shown on hover.
for (const f of files(OUT, (f) => f.endsWith(".html"))) {
  const html = readFileSync(f, "utf8").replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>/gi, " ");
  const attrs = [...html.matchAll(/\b(?:aria-label|title|alt|placeholder)="([^"]*)"/g)].map((m) => m[1]).join(" · ");
  scan(f, decode(`${html.replace(/<[^>]+>/g, " ")} · ${attrs}`));
}
// Scripts: the client components' strings, which the comparison page renders in the browser.
for (const f of files(path.join(OUT, "_next"), (f) => f.endsWith(".js"))) {
  const js = readFileSync(f, "utf8");
  const strings = [...js.matchAll(/"((?:[^"\\\n]|\\.){4,})"|`((?:[^`\\]|\\.){4,})`/g)].map((m) => m[1] ?? m[2]).filter((s) => / [a-z]/i.test(s!));
  scan(f, strings.join(" · "));
}
// Claims as displayed: the judged text with its role labels resolved to the bills' names.
for (const f of files(path.join(OUT, "data", "compare"), (f) => f.endsWith(".json"))) {
  const c = JSON.parse(readFileSync(f, "utf8"));
  const name = (b: { state: string; identifier: string }) => `${b.state} ${b.identifier}`;
  const shown = c.pairs.flatMap((p: { explanation?: { claims: { summary: string }[] } | null }) =>
    (p.explanation?.claims ?? []).map((x) => nameRoles(x.summary, name(c.a), name(c.b))),
  );
  if (shown.length) scan(f, shown.join(" · "));
}

const byPhrase = new Map<string, number>();
for (const h of hits) byPhrase.set(h.phrase, (byPhrase.get(h.phrase) ?? 0) + 1);
if (hits.length) {
  const seen = new Set<string>();
  for (const h of hits) {
    const key = `${h.phrase}|${h.context.slice(20, 100)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    console.error(`${h.file}: "${h.phrase}" in …${h.context}…`);
  }
  console.error(`\n${hits.length} uses of banned phrases (${[...byPhrase].map(([p, n]) => `${p} ${n}`).join(", ")}); fix the copy or allow the use in scripts/copy-check.ts`);
  process.exit(1);
}
console.log("copy rules: no banned phrase in the built site");
