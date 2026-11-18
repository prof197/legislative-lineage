// `node --test`: origin recall is read from the committed results, each key's pooled figures, and
// a results file without a key's section fails the build rather than showing nothing.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { originRecallFrom } from "./recall.ts";

const committed = JSON.parse(readFileSync(path.join(import.meta.dirname, "..", "..", "..", "reports", "results.json"), "utf8")) as Record<string, unknown>;

test("each key's pooled recall comes from its results section", () => {
  const r = originRecallFrom(committed);
  const pooled = (k: string) => (committed[k] as { recall: { pooled: { found: number; enactments: number } } }).recall.pooled;
  for (const [key, section] of [["development", "origins_development"], ["sealed", "origins_held_out"], ["opened", "origins_v1_held_out"]] as const) {
    assert.equal(r[key].found, pooled(section).found);
    assert.equal(r[key].enactments, pooled(section).enactments);
    assert.equal(r[key].interval_95.length, 2);
  }
  assert.equal(r.floor, (committed.origins_development as { threshold: { floor: number } }).threshold.floor);
});

test("a missing section is an error, not an empty figure", () => {
  const { origins_held_out: _, ...rest } = committed;
  assert.throws(() => originRecallFrom(rest), /origins_held_out/);
});
