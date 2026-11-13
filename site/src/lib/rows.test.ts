// `node --test`: rows from spans lineage.align produced for these texts (no embedder), pasted.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Span } from "@/types/export";
import { EDGE_FOLD, endsClause, LONG_ROW, previewEnd, rows, shown, type Row } from "./rows.ts";

function spans(a: string, b: string, parts: [Span["kind"], string | null, string | null][]): Span[] {
  let pa = 0;
  let pb = 0;
  return parts.map(([kind, x, y], i) => {
    const at = x === null ? null : ([a.indexOf(x, pa), a.indexOf(x, pa) + x.length] as [number, number]);
    const bt = y === null ? null : ([b.indexOf(y, pb), b.indexOf(y, pb) + y.length] as [number, number]);
    assert.ok(!at || at[0] >= 0, x ?? "");
    assert.ok(!bt || bt[0] >= 0, y ?? "");
    if (at) pa = at[1];
    if (bt) pb = bt[1];
    return { id: `s${i}`, kind, a: at, b: bt };
  });
}

const text = (t: string, r: Row, side: "a" | "b") => r[side].map((p) => t.slice(p.from, p.to)).join("");
const marked = (t: string, r: Row, side: "a" | "b", mark: string) =>
  r[side].filter((p) => p.mark === mark).map((p) => t.slice(p.from, p.to));

const A1 =
  "(a) Not later than 60 days after a party is served with a complaint, the party may file a special motion for expedited relief to dismiss the cause of action. (b) All other proceedings in the action are stayed while the motion is pending.";
const B1 =
  "1. Not later than 90 days after a party is served with a complaint, the party may file a special motion for expedited relief to dismiss the claim. 2. All other proceedings in the action are stayed while the motion is pending, except that the court may allow limited discovery for good cause shown.";
const MOTION = spans(A1, B1, [
  ["unmatched", "(a) Not later than 60", null],
  ["unmatched", null, "1. Not later than 90"],
  ["copied", "days after a party is served with a complaint, the party may file a special motion for expedited relief to dismiss the", "days after a party is served with a complaint, the party may file a special motion for expedited relief to dismiss the"],
  ["unmatched", "cause of action. (b)", null],
  ["unmatched", null, "claim. 2."],
  ["copied", "All other proceedings in the action are stayed while the motion is pending.", "All other proceedings in the action are stayed while the motion is pending,"],
  ["unmatched", null, "except that the court may allow limited discovery for good cause shown."],
]);

test("each clause pairs with its counterpart, cut where both sides end a clause", () => {
  const got = rows(A1, B1, MOTION);

  assert.deepEqual(got.map((r) => r.kind), ["chg", "chg"]);
  assert.equal(text(A1, got[0]!, "a"), A1.slice(0, A1.indexOf(" (b)")));
  assert.equal(text(B1, got[0]!, "b"), B1.slice(0, B1.indexOf(" 2.")));
  assert.equal(text(A1, got[1]!, "a"), A1.slice(A1.indexOf("(b)")));
  assert.equal(text(B1, got[1]!, "b"), B1.slice(B1.indexOf("2.")));
});

test("only the words that differ are bold; numbering and short shared words stay plain", () => {
  const [first, second] = rows(A1, B1, MOTION);

  assert.deepEqual(marked(A1, first!, "a", "differs"), ["60", "cause of action."]);
  assert.deepEqual(marked(B1, first!, "b", "differs"), ["90", "claim."]);
  assert.ok(marked(A1, first!, "a", "plain").some((t) => t.includes("Not later than")));
  assert.deepEqual(marked(B1, second!, "b", "differs"), ["except that the court may allow limited discovery for good cause shown."]);
  assert.equal(marked(A1, second!, "a", "copied").join(""), "All other proceedings in the action are stayed while the motion is pending.");
});

const A2 =
  "The board shall keep its minutes in a public register open to inspection by any resident of the county during ordinary business hours. A copy of each record shall be filed with the clerk of the superior court within ten days of the meeting at which it was adopted.";
const B2 =
  "Every agency shall publish its rules on its website in a format that can be searched. A copy of each record shall be filed with the clerk of the superior court within ten days of the meeting at which it was adopted. This Act shall become effective on July 1, 2025, and shall apply to all records created on or after that date.\nAll laws and parts of laws in conflict with this Act are repealed.";
const COPY = "A copy of each record shall be filed with the clerk of the superior court within ten days of the meeting at which it was adopted.";
const EDGES = spans(A2, B2, [
  ["unmatched", "The board shall keep its minutes in a public register open to inspection by any resident of the county during ordinary business hours.", null],
  ["unmatched", null, "Every agency shall publish its rules on its website in a format that can be searched."],
  ["copied", COPY, COPY],
  ["unmatched", null, "This Act shall become effective on July 1, 2025, and shall apply to all records created on or after that date.\nAll laws and parts of laws in conflict with this Act are repealed."],
]);

test("a clause one side has alone gets its own row, and neighbouring ones on a side merge", () => {
  const got = rows(A2, B2, EDGES);

  assert.deepEqual(got.map((r) => r.kind), ["rem", "add", "same", "add"]);
  assert.deepEqual(got[0]!.b, []);
  assert.deepEqual(got[1]!.a, []);
  assert.equal(text(B2, got[3]!, "b"), B2.slice(B2.indexOf("This Act")));
  assert.equal(got[3]!.words, 34);
  assert.ok(got.every((r) => !r.fold));
});

test("a long run with nothing beside it folds at the edges sooner than in the middle", () => {
  const tail = Array.from({ length: EDGE_FOLD + 1 }, (_, i) => `word${i}`).join(" ") + ".";
  const b = `${COPY} ${tail}`;
  const got = rows(COPY, b, spans(COPY, b, [["copied", COPY, COPY], ["unmatched", null, tail]]));

  assert.deepEqual(got.map((r) => [r.kind, r.fold]), [["same", false], ["add", true]]);
  assert.equal(b.slice(got[1]!.b[0]!.from, previewEnd(b, got[1]!.b)), tail.split(" ").slice(0, 24).join(" "));
});

test("a long copy is cut into clause rows at the same word on both sides", () => {
  const a = "(1) The fee is set by the board each year. (2) The board shall publish the fee schedule.";
  const b = "1. the fee is set by the board each year; 2. the board shall publish the fee schedule";
  const got = rows(a, b, spans(a, b, [["copied", "(1) The fee is set by the board each year. (2) The board shall publish the fee schedule.", "1. the fee is set by the board each year; 2. the board shall publish the fee schedule"]]));

  assert.deepEqual(got.map((r) => r.kind), ["same", "same"]);
  assert.equal(text(a, got[0]!, "a"), "(1) The fee is set by the board each year.");
  assert.equal(text(b, got[0]!, "b"), "1. the fee is set by the board each year;");
});

test("numbering, abbreviations and a wrapped line end no clause; a stop or new numbered line does", () => {
  const t = "See Sec. 4 and (a) the rule. 2.\nNext the board shall\nmeet monthly\n(b) The clerk";
  assert.equal(endsClause(t, t.indexOf("Sec.") + 4), false);
  assert.equal(endsClause(t, t.indexOf("rule.") + 5), true);
  assert.equal(endsClause(t, t.indexOf("2.") + 2), false);
  assert.equal(endsClause(t, t.indexOf("shall") + 5), false);
  assert.equal(endsClause(t, t.indexOf("monthly") + 7), true);
  assert.equal(endsClause(t, t.length), true);
});

test("a wrapped line shows as one line; numbered lines and paragraphs keep their breaks", () => {
  const t = "The board shall\nmeet monthly:\n(1) in public; and\n\nThe clerk";
  assert.equal(shown(t, 0, t.length), "The board shall meet monthly:\n(1) in public; and\n\nThe clerk");
});

test("whole clauses one side inserts inside the other's sentence get rows of their own", () => {
  const copyA = "The commission shall adopt rules for the program";
  const copyB = "and shall report to the General Assembly each year";
  const inserted = "The rules shall include fees. The rules shall include deadlines. The commission may waive either.";
  const a = `${copyA} ${copyB}.`;
  const b = `${copyA}. ${inserted} It ${copyB}.`;
  const got = rows(a, b, spans(a, b, [["copied", copyA, `${copyA}.`], ["unmatched", null, `${inserted} It`], ["copied", `${copyB}.`, `${copyB}.`]]));

  assert.deepEqual(got.map((r) => r.kind), ["same", "add", "chg"]);
  assert.equal(text(b, got[1]!, "b"), inserted);
  assert.equal(text(a, got[2]!, "a"), `${copyB}.`);
});

test("a row past the soft length ends where one side ends a clause", () => {
  const sa = "the clerk shall file the report with the board within thirty days one.";
  const sb = "the clerk shall file the report with the board within thirty days one";
  const a = Array(12).fill(sa).join(" ");
  const b = Array(12).fill(`${sb} extra`).join(" ");
  const parts: [Span["kind"], string | null, string | null][] = [];
  for (let k = 0; k < 12; k++) parts.push(["copied", sa, sb], ["unmatched", null, "extra"]);
  const got = rows(a, b, spans(a, b, parts));

  assert.ok(got.length > 1);
  for (const r of got) assert.ok(r.a.at(-1)!.to - r.a[0]!.from <= LONG_ROW + sa.length, `${r.a.at(-1)!.to - r.a[0]!.from}`);
  assert.equal(got.map((r) => text(a, r, "a")).join(" "), a);
});

test("a stop glued to the next clause's numbering still ends the clause", () => {
  const copy = "a communication by the sender is effective to cancel the order.";
  const extra = "(b) The bank may act on it.(c) After acceptance, it is not effective.(1) Unless agreed.";
  const b = `${copy}${extra}`;
  const head = copy.slice(0, copy.lastIndexOf(" "));
  const got = rows(copy, b, spans(copy, b, [["copied", head, head], ["unmatched", "order.", null], ["unmatched", null, `order.${extra}`]]));

  assert.deepEqual(got.map((r) => text(b, r, "b")), [copy, extra]);
  assert.deepEqual(got.map((r) => r.kind), ["same", "add"]);
});

test("an unpunctuated run, such as a table, breaks at a line past the soft length", () => {
  const copy = "The districts are composed of the following census blocks";
  const table = Array.from({ length: 60 }, (_, i) => `Block ${970000 + i}\n1001 1002 1003 1004 1005`).join("\n");
  const b = `${copy}\n${table}`;
  const got = rows(copy, b, spans(copy, b, [["copied", copy, copy], ["unmatched", null, table]]));

  // The two-sided row stops near the soft length; the rest is one block only one side has, folded.
  assert.deepEqual(got.map((r) => [r.kind, r.fold]), [["chg", false], ["add", true]]);
  assert.ok(got[0]!.b.at(-1)!.to - got[0]!.b[0]!.from <= LONG_ROW + 100);
  assert.equal(got.map((r) => text(b, r, "b")).join("\n"), b);
});
