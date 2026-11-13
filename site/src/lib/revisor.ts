// Minnesota's bill text keeps the revisor's markers as words: "new text begin" ... "new text end"
// around inserted text and "deleted text begin" ... "deleted text end" around the language a bill
// deletes. The comparison hides the markers and strikes the deleted text through, at display only:
// the stored text, and so every span offset, is unchanged. See DESIGN.md#known-failure-modes.
export type Revisor = { hide: [number, number][]; struck: [number, number][] };
export type Segment = { from: number; to: number; struck: boolean };

// A marker and the spaces after it; the text may glue a marker to the word before it.
const MARKER = /(new|deleted) text (begin|end)[ \t]*/gi;

export function revisorMarks(text: string): Revisor | null {
  const hide: [number, number][] = [];
  const struck: [number, number][] = [];
  let open: number | null = null;
  for (const m of text.matchAll(MARKER)) {
    const at = m.index;
    hide.push([at, at + m[0].length]);
    if (m[1]!.toLowerCase() !== "deleted") continue;
    if (m[2]!.toLowerCase() === "begin") open ??= at + m[0].length;
    else if (open !== null) {
      struck.push([open, at]);
      open = null;
    }
  }
  if (open !== null) struck.push([open, text.length]);
  return hide.length ? { hide, struck } : null;
}

// The visible parts of text[from, to): markers dropped, deleted text flagged.
export function segments(from: number, to: number, r: Revisor | null): Segment[] {
  if (!r) return [{ from, to, struck: false }];
  const cuts = new Set([from, to]);
  for (const [a, b] of [...r.hide, ...r.struck]) {
    if (a > from && a < to) cuts.add(a);
    if (b > from && b < to) cuts.add(b);
  }
  const at = [...cuts].sort((x, y) => x - y);
  const inside = (ranges: [number, number][], x: number) => ranges.some(([a, b]) => x >= a && x < b);
  const out: Segment[] = [];
  for (let i = 0; i + 1 < at.length; i++) {
    const [a, b] = [at[i]!, at[i + 1]!];
    if (inside(r.hide, a)) continue;
    const struck = inside(r.struck, a);
    const last = out.at(-1);
    if (last && last.to === a && last.struck === struck) last.to = b;
    else out.push({ from: a, to: b, struck });
  }
  return out;
}
