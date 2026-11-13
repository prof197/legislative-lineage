// The export's files, read at build time. Every page is rendered from them; the comparison page
// also fetches its files in the browser (lib/fetch.ts), so a section's text is downloaded once.
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type { BillList, BillPage, Meta, Release, Story } from "@/types/export";

const DATA = path.join(process.cwd(), "public", "data");
const cache = new Map<string, unknown>();

function read<T>(name: string): T {
  const hit = cache.get(name);
  if (hit !== undefined) return hit as T;
  const file = path.join(DATA, name);
  if (!existsSync(file)) {
    throw new Error(`${file} is missing: run scripts/fetch-data.sh or npm run data:fixture`);
  }
  const value = JSON.parse(readFileSync(file, "utf8")) as T;
  cache.set(name, value);
  return value;
}

export const meta = (): Meta => read<Meta>("meta.json");
export const billList = (): BillList => read<BillList>("bills.json");
export const billPage = (slug: string): BillPage => read<BillPage>(`bills/${slug}.json`);
export const story = (slug: string): Story => read<Story>(`stories/${slug}.json`);

// The release the data came from, which only the deploy's fetch writes.
export function release(): Release | null {
  return existsSync(path.join(DATA, "release.json")) ? read<Release>("release.json") : null;
}
