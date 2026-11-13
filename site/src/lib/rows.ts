// A section pair's alignment as side-by-side clause rows, the way a code diff pairs lines.
//
// The export gives each span's character offsets in both texts, in reading order on both sides
// (DESIGN.md#alignment). A row is a run of spans that ends where both sides end a clause: at a
// sentence or clause end, or a line break. A copied span is the same words on both sides, word
// for word once case and punctuation are ignored, so it is cut into clause rows at the same word
// on both sides. A stretch neither side shares is read clause by clause, taking text from a side
// whose clause is still open until both close; a whole clause only one side has becomes its own
// row with a gap opposite. Spans are never reordered, so each text still reads top to bottom.
import type { Span } from "@/types/export";

export type Kind = "same" | "chg" | "rew" | "add" | "rem";
// copied and paraphrased text is highlighted; text that differs within a row is bold; numbering,
// and text in a row only one side has, is plain.
export type Mark = "copied" | "paraphrased" | "differs" | "plain";
export type Piece = { from: number; to: number; mark: Mark; span: string | null };
export type Row = {
  kind: Kind;
  a: Piece[];
  b: Piece[];
  words: number; // in a one-sided row, its words; else 0
  fold: boolean; // a long one-sided run with nothing beside it, shown folded
};

// A one-sided run at the start or end of a pair folds past this many words, one in the middle
// past MIDDLE_FOLD: the middle is where a clause one text adds or drops is read in context.
export const EDGE_FOLD = 40;
export const MIDDLE_FOLD = 150;
export const PREVIEW_WORDS = 24;
// Past this many characters on a side, a row also ends where only one side ends a clause, so
// texts punctuated differently throughout still come in rows of a few sentences.
export const LONG_ROW = 600;

type Side = "a" | "b";
type Atom = { kind: Span["kind"]; span: string; a: [number, number] | null; b: [number, number] | null; endA: boolean; endB: boolean };

const WORD = /\S+/g;
const KEYED = /[\p{L}\p{N}§]/u; // a word that normalizes to something (lineage.sections.normalize)
const STOP = /[.;:?!]["'”’)\]]*$/;
// Enumerators and abbreviations end with a stop but end no clause.
const ENUMERATOR = /^[(\[]?(\d+[a-z]?|[a-z]{1,2}|[ivxlc]{1,6})[.)\]]+$/i;
const ABBREVIATION =
  /^(sec|secs|no|nos|art|arts|ch|chs|subd|subds|par|para|paras|pp|inc|co|corp|ltd|jr|sr|st|vs|v|seq|al|cf|u\.s|e\.g|i\.e|o\.c\.g\.a|ann|stat|rev|gen)\.$/i;

type Word = { from: number; to: number; keyed: boolean };

function words(text: string, from: number, to: number): Word[] {
  const out: Word[] = [];
  for (const m of text.slice(from, to).matchAll(WORD)) {
    const at = from + (m.index ?? 0);
    out.push({ from: at, to: at + m[0].length, keyed: KEYED.test(m[0]) });
  }
  return out;
}

export function isEnumerator(word: string): boolean {
  return ENUMERATOR.test(word) || /^§+$/.test(word);
}

// Texts keep the printed page's line breaks. A break is structure only before a blank line or a
// line that opens with numbering or a section word; any other break is where a line wrapped.
const STRUCTURAL = /\n[ \t]*(?:\n|[(\[]?(?:\d+[a-z]?|[a-z]{1,2}|[ivxlc]{1,6})[.)\]]+\s|(?:section|sec\.|§|article|chapter|part)\s)/iy;

export function structural(text: string, at: number): boolean {
  STRUCTURAL.lastIndex = at;
  return STRUCTURAL.test(text);
}

/** Whether the word ending at `to` ends a clause: a stop that is not numbering or an
 * abbreviation, a structural line break after it, or the end of the text. */
export function endsClause(text: string, to: number): boolean {
  let i = to;
  while (i < text.length && (text[i] === " " || text[i] === "\t")) i++;
  if (i >= text.length) return true;
  const start = Math.max(text.lastIndexOf(" ", to - 1), text.lastIndexOf("\n", to - 1)) + 1;
  const word = text.slice(start, to);
  if (isEnumerator(word)) return false; // "(a)" on a line of its own opens the clause below it
  if (STOP.test(word) && !ABBREVIATION.test(word)) return true;
  return text[i] === "\n" && structural(text, i);
}

/** Characters from..to as displayed: a wrapped line's break read as a space. */
export function shown(text: string, from: number, to: number): string {
  let out = "";
  let pos = from;
  for (let i = text.indexOf("\n", from); i !== -1 && i < to; i = text.indexOf("\n", i + 1)) {
    if (structural(text, i) || (i > 0 && text[i - 1] === "\n")) continue;
    out += text.slice(pos, i) + " ";
    pos = i + 1;
  }
  return out + text.slice(pos, to);
}

// Some texts run a clause's stop into the next clause's numbering: "amendment.(b) Subject".
const GLUED = /[.;:]["'”’]*(?=\((?:\d+[a-z]?|[a-z]{1,3}|[ivxlc]{1,6})\))/gi;

// A one-sided stretch, cut after each clause end inside it.
function oneSided(text: string, range: [number, number], side: Side, kind: Span["kind"], span: string): Atom[] {
  const cuts: number[] = [];
  const forced = new Set<number>(); // where an unpunctuated run, such as a table, breaks a line
  let last = range[0];
  for (const w of words(text, range[0], range[1])) {
    for (const m of text.slice(w.from, w.to).matchAll(GLUED)) {
      const at = w.from + (m.index ?? 0) + m[0].length;
      if (at < w.to) cuts.push((last = at));
    }
    if (w.to >= range[1]) continue;
    if (endsClause(text, w.to)) cuts.push((last = w.to));
    else if (w.to - last > LONG_ROW && text[w.to] === "\n") {
      cuts.push((last = w.to));
      forced.add(w.to);
    }
  }
  cuts.push(range[1]);
  const out: Atom[] = [];
  let from = range[0];
  for (const to of cuts) {
    while (from < to && /\s/.test(text[from]!)) from++;
    if (from >= to) continue;
    const ends = forced.has(to) || endsClause(text, to);
    const r: [number, number] = [from, to];
    out.push({ kind, span, a: side === "a" ? r : null, b: side === "b" ? r : null, endA: side === "a" && ends, endB: side === "b" && ends });
    from = to;
  }
  return out;
}

// A copied span, cut at the same keyed word on both sides wherever either side ends a clause.
function copied(ta: string, tb: string, ra: [number, number], rb: [number, number], span: string): Atom[] {
  const wa = words(ta, ra[0], ra[1]);
  const wb = words(tb, rb[0], rb[1]);
  const ka = wa.flatMap((w, i) => (w.keyed ? [i] : []));
  const kb = wb.flatMap((w, i) => (w.keyed ? [i] : []));
  const whole: Atom = { kind: "copied", span, a: ra, b: rb, endA: endsClause(ta, ra[1]), endB: endsClause(tb, rb[1]) };
  if (ka.length !== kb.length || ka.length === 0) return [whole];
  const out: Atom[] = [];
  let sa = 0;
  let sb = 0;
  for (let k = 0; k < ka.length - 1; k++) {
    const ia = ka[k]!;
    const ib = kb[k]!;
    if (!endsClause(ta, wa[ia]!.to) && !endsClause(tb, wb[ib]!.to)) continue;
    // Inside a copy the words are the same on both sides, so a clause end on either side
    // closes the row on both.
    out.push({ kind: "copied", span, a: [wa[sa]!.from, wa[ia]!.to], b: [wb[sb]!.from, wb[ib]!.to], endA: true, endB: true });
    sa = ia + 1;
    sb = ib + 1;
  }
  out.push({ ...whole, a: [wa[sa]!.from, ra[1]], b: [wb[sb]!.from, rb[1]] });
  return out;
}

type Open = { atoms: Atom[]; openA: boolean; openB: boolean };

function has(row: Open, side: Side): boolean {
  return row.atoms.some((x) => x[side] !== null);
}

function classify(atoms: Atom[], ta: string, tb: string): Row {
  const onA = atoms.some((x) => x.a);
  const onB = atoms.some((x) => x.b);
  const numbering = (x: Atom) =>
    x.kind === "unmatched" && [x.a && ta.slice(...x.a), x.b && tb.slice(...x.b)].every((t) => !t || t.split(/\s+/).every(isEnumerator));
  const substantive = atoms.filter((x) => !numbering(x));
  let kind: Kind = !onA
    ? "add"
    : !onB
      ? "rem"
      : substantive.every((x) => x.kind === "copied")
        ? "same"
        : substantive.every((x) => x.kind === "paraphrased")
          ? "rew"
          : "chg";
  const mark = (x: Atom): Mark =>
    x.kind === "copied" ? "copied" : x.kind === "paraphrased" ? "paraphrased" : kind === "chg" && !numbering(x) ? "differs" : "plain";
  // In a row whose words differ, the words the two unshared runs still have in common, too few
  // to count as copying, are plain, so only what differs is bold.
  const common = kind === "chg" ? shared(atoms.filter((x) => mark(x) === "differs"), ta, tb) : null;
  // Unshared runs whose words all still match, differing only in punctuation or numbering, are
  // the same words by the comparison's own rule.
  if (common && substantive.every((x) => x.kind === "copied" || (x.kind === "unmatched" && allShared(x, common, ta, tb)))) kind = "same";
  const side = (s: Side, text: string): Piece[] => {
    const out: Piece[] = [];
    const add = (from: number, to: number, m: Mark, span: string | null) => {
      const prev = out.at(-1);
      if (prev && prev.mark === m && prev.span === span) {
        prev.to = to; // the space between two words marked alike is marked with them
        return;
      }
      if (prev && from > prev.to) out.push({ from: prev.to, to: from, mark: "plain", span: null });
      out.push({ from, to, mark: m, span });
    };
    for (const x of atoms) {
      const r = x[s];
      if (!r) continue;
      const m = mark(x);
      if (m === "differs" && common) {
        for (const w of words(text, r[0], r[1])) {
          const plain = common[s].has(w.from) || isEnumerator(text.slice(w.from, w.to));
          add(w.from, w.to, plain ? "plain" : "differs", x.span);
        }
      } else add(r[0], r[1], m, x.span);
    }
    return out.filter((p) => p.to > p.from);
  };
  const one = kind === "add" ? tb : kind === "rem" ? ta : null;
  const pieces = kind === "add" ? side("b", tb) : kind === "rem" ? side("a", ta) : [];
  const count = one && pieces.length ? one.slice(pieces[0]!.from, pieces.at(-1)!.to).split(/\s+/).length : 0;
  return { kind, a: side("a", ta), b: side("b", tb), words: count, fold: false };
}

function allShared(x: Atom, common: { a: Set<number>; b: Set<number> }, ta: string, tb: string): boolean {
  const each = (r: [number, number] | null, t: string, set: Set<number>) =>
    !r || words(t, ...r).every((w) => !w.keyed || set.has(w.from) || isEnumerator(t.slice(w.from, w.to)));
  return each(x.a, ta, common.a) && each(x.b, tb, common.b);
}

const LCS_CELLS = 250_000; // past this the row's differing words all stay bold

const key = (w: string) => w.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");

// The words of the row's unshared runs that a longest common subsequence keeps, by start offset.
function shared(atoms: Atom[], ta: string, tb: string): { a: Set<number>; b: Set<number> } {
  const wa = atoms.flatMap((x) => (x.a ? words(ta, ...x.a) : [])).filter((w) => w.keyed);
  const wb = atoms.flatMap((x) => (x.b ? words(tb, ...x.b) : [])).filter((w) => w.keyed);
  const out = { a: new Set<number>(), b: new Set<number>() };
  if (!wa.length || !wb.length || wa.length * wb.length > LCS_CELLS) return out;
  const ka = wa.map((w) => key(ta.slice(w.from, w.to)));
  const kb = wb.map((w) => key(tb.slice(w.from, w.to)));
  const n = kb.length + 1;
  const len = new Uint32Array((ka.length + 1) * n);
  for (let i = ka.length - 1; i >= 0; i--) {
    for (let j = kb.length - 1; j >= 0; j--) {
      len[i * n + j] = ka[i] === kb[j] ? len[(i + 1) * n + j + 1]! + 1 : Math.max(len[(i + 1) * n + j]!, len[i * n + j + 1]!);
    }
  }
  let i = 0;
  let j = 0;
  while (i < ka.length && j < kb.length) {
    if (ka[i] === kb[j]) {
      out.a.add(wa[i++]!.from);
      out.b.add(wb[j++]!.from);
    } else if (len[(i + 1) * n + j]! >= len[i * n + j + 1]!) i++;
    else j++;
  }
  return out;
}

/** The pair's rows, in reading order on both sides. */
export function rows(ta: string, tb: string, spans: Span[]): Row[] {
  const out: Row[] = [];
  let row: Open = { atoms: [], openA: false, openB: false };
  const push = (atoms: Atom[]) => out.push(classify(atoms, ta, tb));
  const span = (side: Side) => {
    const on = row.atoms.flatMap((x) => (x[side] ? [x[side]] : []));
    return on.length ? on.at(-1)![1] - on[0]![0] : 0;
  };
  const close = () => {
    const long = Math.max(span("a"), span("b")) > LONG_ROW && has(row, "a") && has(row, "b");
    if (row.atoms.length && ((!row.openA && !row.openB) || (long && (!row.openA || !row.openB)))) {
      push(row.atoms);
      row = { atoms: [], openA: false, openB: false };
    }
  };
  const take = (x: Atom) => {
    row.atoms.push(x);
    if (x.a) row.openA = !x.endA;
    if (x.b) row.openB = !x.endB;
    close();
  };

  // A stretch only one side or neither shares: clause by clause, from a side still open.
  const stretch = (as: Atom[], bs: Atom[]) => {
    let i = 0;
    let j = 0;
    while (i < as.length || j < bs.length) {
      if (row.openA && i < as.length) take(as[i++]!);
      else if (row.openB && j < bs.length) take(bs[j++]!);
      else if (!row.atoms.length) take(i < as.length ? as[i++]! : bs[j++]!);
      else {
        // The row is open on a side with nothing left here, so the other side's clauses fall
        // inside that side's sentence. A whole clause is a row of its own: before the open row
        // when the other side has no text in it yet, else after it, the open sentence going on
        // in the next row. A clause's opening joins the open row.
        const [side, next] = i < as.length ? (["a", as[i]!] as const) : (["b", bs[j]!] as const);
        if (side === "a") i++;
        else j++;
        const whole = side === "a" ? next.endA : next.endB;
        if (whole && !has(row, side)) push([next]);
        else if (whole) {
          push(row.atoms);
          row = { atoms: [], openA: false, openB: false };
          take(next);
        } else take(next);
      }
    }
  };

  let as: Atom[] = [];
  let bs: Atom[] = [];
  const flush = () => {
    if (as.length || bs.length) stretch(as, bs);
    as = [];
    bs = [];
  };
  for (const s of spans) {
    if (s.a && s.b) {
      flush();
      for (const x of s.kind === "copied" ? copied(ta, tb, s.a, s.b, s.id) : [{ kind: s.kind, span: s.id, a: s.a, b: s.b, endA: endsClause(ta, s.a[1]), endB: endsClause(tb, s.b[1]) }]) take(x);
    } else if (s.a) as.push(...oneSided(ta, s.a, "a", s.kind, s.id));
    else if (s.b) bs.push(...oneSided(tb, s.b, "b", s.kind, s.id));
  }
  flush();
  if (row.atoms.length) push(row.atoms);
  return fold(merge(out));
}

// Neighbouring rows only one side has, on the same side, read as one block.
function merge(rs: Row[]): Row[] {
  const out: Row[] = [];
  for (const r of rs) {
    const prev = out.at(-1);
    if (prev && (r.kind === "add" || r.kind === "rem") && prev.kind === r.kind) {
      const into = prev[r.kind === "add" ? "b" : "a"];
      const from = r[r.kind === "add" ? "b" : "a"];
      into.push({ from: into.at(-1)!.to, to: from[0]!.from, mark: "plain", span: null }, ...from);
      prev.words += r.words;
    } else out.push({ ...r, a: [...r.a], b: [...r.b] });
  }
  return out;
}

function fold(rs: Row[]): Row[] {
  const both = rs.flatMap((r, i) => (r.kind === "add" || r.kind === "rem" ? [] : [i]));
  const first = both[0] ?? rs.length;
  const last = both.at(-1) ?? -1;
  return rs.map((r, i) => {
    if (r.kind !== "add" && r.kind !== "rem") return r;
    const edge = i < first || i > last;
    return { ...r, fold: r.words > (edge ? EDGE_FOLD : MIDDLE_FOLD) };
  });
}

/** Where a folded row's preview ends: after its first PREVIEW_WORDS words. */
export function previewEnd(text: string, pieces: Piece[]): number {
  const ws = words(text, pieces[0]!.from, pieces.at(-1)!.to);
  return ws[Math.min(PREVIEW_WORDS, ws.length) - 1]!.to;
}
