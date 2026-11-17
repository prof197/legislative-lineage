// The copy rules, checked over the built site (`npm run build`, then `npm run test:copy`): no page,
// script string or displayed claim may use a phrase that states direction or authorship, except
// where ALLOWED says why the use is legitimate. Text that comes from the bills themselves (titles,
// headings, section text) is the legislatures' wording, not the site's, and is set aside first.
// No page may give an organization a political label, and every origin line must give both
// shares in the ruled form, exactly as its bill's data has them.
// See DESIGN.md#explanations, DESIGN.md#origin-matches and the copy rulings they record.
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { nameRoles } from "../src/lib/claims.ts";
import { ORIGIN_LINE, originText } from "../src/lib/origins.ts";

const OUT = path.resolve("out");
const BANNED = [
  "descendant", "the origin", "changed", "copied", "copying", "written by", "who is really writing",
  "wrote", "drafted by", "authored",
];
const PATTERN = new RegExp(`\\b(${BANNED.map((b) => b.replace(/ /g, "\\s+")).join("|")})\\b`, "gi");

// Legitimate uses: where the phrase appears (a file name pattern) and the words around it.
const ALLOWED: { where: RegExp; context: RegExp; why: string }[] = [
  {
    where: /^_next\//,
    context: /Written by a language model from the aligned text/,
    why: "the claims panel discloses that a model wrote the claims; it names no author of a bill",
  },
  {
    where: /./,
    context: /not who wrote (them|either one)/,
    why: "denies authorship: the site shows shared language, not who wrote a bill",
  },
];

// Political labels: the site describes an organization by its type and IRS status from public
// filings, never by a label (DESIGN.md#model-documents). Bill text and a document's own title are
// the legislatures' and the organizations' wording and are set aside; a source's name, descriptor,
// members line and notes are the site's account of it and are checked.
const POLITICAL = [
  "conservative", "liberal", "progressive", "libertarian", "right-wing", "left-wing", "far-right",
  "far-left", "partisan", "republican", "democrat", "democratic", "free-market", "extremist",
];
const POLITICAL_PATTERN = new RegExp(`\\b(${POLITICAL.join("|")})\\b`, "gi");
// Strings under these keys are the site's own account of a source, never set aside.
const ACCOUNT_KEYS = new Set(["organization", "co_publishers", "descriptor", "members", "leak_note", "terms_note"]);

// The check must be able to fail: every banned phrase and label matches itself, and the origin
// line's form rejects the line it replaced and a line without the model's share.
for (const b of BANNED) {
  PATTERN.lastIndex = 0;
  if (!PATTERN.test(`a ${b} b`)) throw new Error(`the pattern misses "${b}"`);
}
for (const b of POLITICAL) {
  POLITICAL_PATTERN.lastIndex = 0;
  if (!POLITICAL_PATTERN.test(`a ${b} b`)) throw new Error(`the label pattern misses "${b}"`);
}
for (const bad of ["Shares 34% of its wording with Model Act.", "34% of this bill's wording appears in Org's model Model Act."]) {
  if (ORIGIN_LINE.test(bad)) throw new Error(`the origin line's form accepts "${bad}"`);
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

// Every string the export carries that holds a banned phrase or label: bill titles, headings,
// section text, model documents' own titles. A source's account (ACCOUNT_KEYS) is not set aside.
const fromBills = new Set<string>();
// A source's terms note reports what its library's own terms allow, in their words ("viewing,
// copying, printing and distributing"): that is no claim about a bill, so it is set aside from the
// direction and authorship scan, and checked for political labels like the rest of the account.
const termsNotes = new Set<string>();
function flagged(s: string): boolean {
  PATTERN.lastIndex = 0;
  POLITICAL_PATTERN.lastIndex = 0;
  return PATTERN.test(s) || POLITICAL_PATTERN.test(s);
}
function collect(value: unknown, key = ""): void {
  if (key === "terms_note" && typeof value === "string") termsNotes.add(value);
  if (ACCOUNT_KEYS.has(key)) return;
  if (typeof value === "string") {
    if (flagged(value)) fromBills.add(value);
  } else if (Array.isArray(value)) value.forEach((v) => collect(v, key));
  else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) if (k !== "explanation") collect(v, k);
  }
}
for (const f of files(path.join(OUT, "data"), (f) => f.endsWith(".json"))) collect(JSON.parse(readFileSync(f, "utf8")));
const billStrings = [...fromBills].sort((a, b) => b.length - a.length);

function scan(file: string, text: string): void {
  for (const s of billStrings) if (text.includes(s)) text = text.split(s).join(" ");
  let claims = text;
  for (const s of termsNotes) if (claims.includes(s)) claims = claims.split(s).join(" ");
  for (const m of claims.matchAll(PATTERN)) {
    const context = claims.slice(Math.max(0, m.index - 60), m.index + m[0].length + 60).replace(/\s+/g, " ").trim();
    const rel = path.relative(OUT, file);
    if (ALLOWED.some((a) => a.where.test(rel) && a.context.test(context))) continue;
    hits.push({ file: rel, phrase: m[0].toLowerCase(), context });
  }
  for (const m of text.matchAll(POLITICAL_PATTERN)) {
    const context = text.slice(Math.max(0, m.index - 60), m.index + m[0].length + 60).replace(/\s+/g, " ").trim();
    hits.push({ file: path.relative(OUT, file), phrase: m[0].toLowerCase(), context });
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

// Origin lines (DESIGN.md#origin-matches): each one rendered gives both shares in the ruled form,
// "N% of this bill's wording appears in [title], a model published by [organization], covering M% of the model.",
// and is exactly the line its bill's data gives, so no model text, authorship or label rides in
// it. A bill page shows a line for every origin its data lists. A line is found by its
// data-origin-line key, the document's id; its bill is the page's, or a story's data-origin-bill.
type OriginData = Parameters<typeof originText>[0];
const DOC_KEYS = new Set(["id", "source", "organization", "title", "url", "published", "leaked", "co_publishers"]);
const ORIGIN_KEYS = new Set(["doc", "stages", "bill_share", "model_share", "sections"]);
const originProblems: string[] = [];
const billOrigins = new Map<string, OriginData[]>();
for (const f of files(path.join(OUT, "data", "bills"), (f) => f.endsWith(".json"))) {
  const origins = (JSON.parse(readFileSync(f, "utf8")).origins ?? []) as OriginData[];
  billOrigins.set(path.basename(f, ".json"), origins);
  for (const o of origins) {
    // The export carries no model text: an origin and its document hold these fields and no more.
    const extra = [...Object.keys(o).filter((k) => !ORIGIN_KEYS.has(k)), ...Object.keys(o.doc).filter((k) => !DOC_KEYS.has(k))];
    if (extra.length) originProblems.push(`data/bills/${path.basename(f)}: ${o.doc.id} carries ${extra.join(", ")}`);
  }
}
const attr = (tag: string, name: string): string | undefined => new RegExp(`\\b${name}="([^"]*)"`).exec(tag)?.[1];
for (const f of files(OUT, (f) => f.endsWith(".html"))) {
  const rel = path.relative(OUT, f);
  const html = readFileSync(f, "utf8").replace(/<script[\s\S]*?<\/script>/gi, " ");
  const pageBill = /^bills\/([^/]+)\/index\.html$/.exec(rel)?.[1];
  let shown = 0;
  for (const m of html.matchAll(/<p\b([^>]*\bdata-origin-line="[^"]*"[^>]*)>([\s\S]*?)<\/p>/g)) {
    shown += 1;
    const id = decode(attr(m[1]!, "data-origin-line")!);
    const bill = attr(m[1]!, "data-origin-bill") ?? pageBill;
    const text = decode(m[2]!.replace(/<!--[\s\S]*?-->/g, "").replace(/<[^>]+>/g, "")).replace(/\s+/g, " ").trim();
    if (!ORIGIN_LINE.test(text)) originProblems.push(`${rel}: origin line not in the ruled form: "${text}"`);
    const o = bill ? billOrigins.get(bill)?.find((x) => x.doc.id === id) : undefined;
    if (!o) originProblems.push(`${rel}: origin line for ${id} has no origin in ${bill ? `data/bills/${bill}.json` : "a bill's data"}`);
    else if (text !== originText(o)) originProblems.push(`${rel}: origin line "${text}" is not its data's "${originText(o)}"`);
  }
  const listed = pageBill ? (billOrigins.get(pageBill)?.length ?? 0) : shown;
  if (shown !== listed) originProblems.push(`${rel}: ${shown} origin lines for the ${listed} origins its data lists`);
}
if (originProblems.length) {
  for (const p of originProblems.slice(0, 20)) console.error(p);
  console.error(`\n${originProblems.length} origin line problems; every origin line gives both shares, as its data does`);
  process.exit(1);
}

// The data credit, on every page but the not-found pages: it travels in the shared footer, and a
// page built without that footer would drop it silently. stories/none is the not-found page too:
// a static export must render one route, so data with no featured stories renders it there.
const CREDIT = "Bill data from Open States, by Plural Policy (public domain); bill text from each legislature's website.";
const uncredited = files(OUT, (f) => f.endsWith(".html"))
  .filter((f) => !/(^|\/)(404|_not-found|stories\/none)(\/index)?\.html$/.test(path.relative(OUT, f)))
  .filter((f) => {
    const html = readFileSync(f, "utf8").replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>/gi, " ");
    return !decode(html.replace(/<[^>]+>/g, "")).replace(/\s+/g, " ").includes(CREDIT);
  });
if (uncredited.length) {
  for (const f of uncredited.slice(0, 20)) console.error(`${path.relative(OUT, f)}: no data credit`);
  console.error(`\n${uncredited.length} pages without the data credit; render <Footer /> on them`);
  process.exit(1);
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
console.log("copy rules: no banned phrase or political label in the built site, every origin line in the ruled form, and the data credit on every page");
