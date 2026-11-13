// `node --test`: Minnesota's revisor markers hide, and the text a bill deletes is flagged struck.
import assert from "node:assert/strict";
import { test } from "node:test";
import { revisorMarks, segments } from "./revisor.ts";

const T = "(3) A private fund deleted text begin manager deleted text end new text begin advisernew text end , subject to";

const visible = (t: string, from: number, to: number) => {
  const r = revisorMarks(t);
  return segments(from, to, r).map((s) => (s.struck ? `~${t.slice(s.from, s.to)}~` : t.slice(s.from, s.to))).join("");
};

test("markers disappear, glued ones too, and deleted text is struck", () => {
  assert.equal(visible(T, 0, T.length), "(3) A private fund ~manager ~adviser, subject to");
});

test("a slice that starts inside deleted text is struck from its start", () => {
  const at = T.indexOf("ager");
  assert.equal(visible(T, at, T.indexOf("adviser")), "~ager ~");
});

test("text with no markers is left as it is", () => {
  assert.equal(revisorMarks("The board shall meet."), null);
  assert.deepEqual(segments(0, 5, null), [{ from: 0, to: 5, struck: false }]);
});
