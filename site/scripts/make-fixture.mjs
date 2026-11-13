// Dev only: trims a real export into the committed fixture (fixture/), a handful of bills that
// exercise every kind of page: a bill with relatives in each list, a notice-only bill and a bill
// with none, each relative's comparison and texts, the stories and the data stamp.
// Usage: node scripts/make-fixture.mjs ../path/to/export/GA
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const src = path.resolve(process.argv[2] ?? "public/data");
const out = new URL("../fixture/", import.meta.url).pathname;
const read = async (name) => JSON.parse(await readFile(path.join(src, name), "utf8"));
const write = async (name, value) => {
  await mkdir(path.dirname(path.join(out, name)), { recursive: true });
  await writeFile(path.join(out, name), JSON.stringify(value));
};
const RELATIVES = 3; // kept per bill, so the fixture stays small

const list = await read("bills.json");
const pages = new Map();
for (const e of list.bills) pages.set(e.bill.slug, null);
const pick = [];
const want = (pred) => {
  for (const e of list.bills) if (!pick.includes(e.bill.slug) && pred(e)) return pick.push(e.bill.slug);
};
const page = async (slug) => pages.get(slug) ?? pages.set(slug, await read(`bills/${slug}.json`)).get(slug);

pick.push("ga-2023_24-sb-351");
for (const e of list.bills) {
  if (pick.length >= 4) break;
  if (e.families === 0) continue;
  const p = await page(e.bill.slug);
  if (p.relatives.some((r) => r.names_compact) && !pick.includes(e.bill.slug)) pick.push(e.bill.slug);
}
for (const e of list.bills) {
  const p = await page(e.bill.slug);
  if (p.text?.notice_only) { pick.push(e.bill.slug); break; }
}
want((e) => e.families === 0);

await rm(out, { recursive: true, force: true });
const texts = new Set();
for (const slug of pick) {
  const p = await page(slug);
  const keep = [
    ...p.relatives.filter((r) => r.found_by === "wording" && !r.names_compact).slice(0, RELATIVES),
    ...p.relatives.filter((r) => r.found_by === "meaning" && !r.names_compact).slice(0, RELATIVES),
    ...p.relatives.filter((r) => r.names_compact).slice(0, RELATIVES),
  ];
  await write(`bills/${slug}.json`, { ...p, relatives: keep });
  for (const r of keep) {
    const c = await read(`compare/${r.comparison}.json`);
    await write(`compare/${r.comparison}.json`, c);
    texts.add(c.a.slug).add(c.b.slug);
  }
}
const meta = await read("meta.json");
for (const s of meta.stories) {
  const story = await read(`stories/${s}.json`);
  await write(`stories/${s}.json`, story);
}
for (const t of texts) await cp(path.join(src, "texts", `${t}.json`), path.join(out, "texts", `${t}.json`));
await write("bills.json", { bills: list.bills.filter((e) => pick.includes(e.bill.slug)) });
await write("meta.json", { ...meta, hosted_bills: pick.length });
console.log(`${pick.length} bills, ${texts.size} texts -> fixture/`);
