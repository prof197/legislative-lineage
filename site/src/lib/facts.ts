// Counts the pages state, computed from the export so every number shown is the data's.
import type { BillRef, Story, StoryRelative } from "@/types/export";
import { STATUS_RANK, type Status } from "@/lib/format";
import type { Fill } from "@/components/Tiles";

function furthest<T extends { bill: BillRef }>(a: T, b: T, score: (x: T) => number): T {
  const ra = a.bill.status ? STATUS_RANK[a.bill.status] : 0;
  const rb = b.bill.status ? STATUS_RANK[b.bill.status] : 0;
  if (ra !== rb) return ra > rb ? a : b;
  return score(a) >= score(b) ? a : b;
}

// Each state's relative that went furthest, as a story's or a bill's map shows it.
export function bestByState<T extends { bill: BillRef }>(
  items: T[],
  score: (x: T) => number,
): Map<string, T> {
  const out = new Map<string, T>();
  for (const item of items) {
    const held = out.get(item.bill.state);
    out.set(item.bill.state, held ? furthest(held, item, score) : item);
  }
  return out;
}

export function fills<T extends { bill: BillRef }>(best: Map<string, T>, home?: string): Record<string, Fill> {
  const out: Record<string, Fill> = {};
  for (const [state, item] of best) out[state] = item.bill.status ?? "none";
  if (home) out[home] = "home";
  return out;
}

export type Tally = { states: number } & Record<Status, number>;

export function tally<T extends { bill: BillRef }>(best: Map<string, T>): Tally {
  const t: Tally = { states: best.size, passed: 0, died: 0, introduced: 0, passed_legislature: 0 };
  for (const item of best.values()) if (item.bill.status) t[item.bill.status] += 1;
  return t;
}

export const storyBest = (s: Story): Map<string, StoryRelative> =>
  bestByState(s.relatives, (r) => r.share ?? 0);
