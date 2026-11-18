// Origin recall as the committed results give it (reports/results.json): each uniform act's
// published text matched as a model document, counted found only where the enacting bill's page
// would list it (DESIGN.md#origin-matches). Read from the repository at build time, as the run
// page's figures are, so the validation page states it without a new data release.

export type KeyRecall = { enactments: number; found: number; recall: number; interval_95: [number, number] };

export type OriginRecall = {
  floor: number;
  development: KeyRecall; // the development answer key
  sealed: KeyRecall; // the held-out lists' reserve acts, run once with --final
  opened: KeyRecall; // v1's held-out acts: opened in v1, not a sealed test
};

type Section = { threshold: { floor: number }; recall: { pooled: KeyRecall } };

const SECTIONS = { development: "origins_development", sealed: "origins_held_out", opened: "origins_v1_held_out" } as const;

export function originRecallFrom(results: Record<string, unknown>): OriginRecall {
  const section = (key: string): Section => {
    const s = results[key] as Section | undefined;
    if (!s) throw new Error(`reports/results.json has no ${key}: run uv run lineage eval origins`);
    return s;
  };
  const pooled = (key: string): KeyRecall => {
    const { enactments, found, recall, interval_95 } = section(key).recall.pooled;
    return { enactments, found, recall, interval_95 };
  };
  return {
    floor: section(SECTIONS.development).threshold.floor,
    development: pooled(SECTIONS.development),
    sealed: pooled(SECTIONS.sealed),
    opened: pooled(SECTIONS.opened),
  };
}
