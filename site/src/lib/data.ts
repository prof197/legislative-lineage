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

// The run page's figures, read from the pipeline's committed reports so none is typed by hand.
export type RunFigures = {
  indexSeconds: number;
  embedSeconds: number;
  neighborsSeconds: number;
  exportSeconds: number;
  exportState: string;
  archiveBytes: number;
  indexBytes: number;
  vectorBytes: number;
  peakMemoryBytes: number;
  machineMemoryBytes: number;
};

type Stage = { seconds: number; peak_memory_bytes: number; stored_bytes: number };

export function runFigures(): RunFigures {
  const repo = path.join(process.cwd(), "..");
  const json = <T,>(name: string): T => JSON.parse(readFileSync(path.join(repo, name), "utf8")) as T;
  const cost = json<{ build: Record<string, Stage> }>("reports/cost.json").build;
  const index = json<{ build: { machine: { memory_bytes: number } } }>("reports/index.json");
  const exported = json<{ seconds: number; state?: string }>("reports/export.json");
  const manifest = json<{ archives: { bytes: number }[] }>("snapshot/manifest.json");
  const stages = Object.values(cost);
  return {
    indexSeconds: cost.near_duplicate_index!.seconds,
    embedSeconds: cost.embedding!.seconds,
    neighborsSeconds: cost.neighbors!.seconds,
    exportSeconds: exported.seconds,
    exportState: exported.state ?? meta().state,
    archiveBytes: manifest.archives.reduce((n, a) => n + a.bytes, 0),
    indexBytes: cost.near_duplicate_index!.stored_bytes,
    vectorBytes: cost.embedding!.stored_bytes,
    peakMemoryBytes: Math.max(...stages.map((s) => s.peak_memory_bytes)),
    machineMemoryBytes: index.build.machine.memory_bytes,
  };
}
