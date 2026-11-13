// `node --test`: section labels read one way, keep the bill's order, and tell repeats apart.
import assert from "node:assert/strict";
import { test } from "node:test";
import { sectionLabel, sectionLabels } from "./format.ts";

test("headings are written one way for display", () => {
  assert.equal(sectionLabel("SECTION 2:", 3), "Section 2");
  assert.equal(sectionLabel("SECTION 2-2.", 3), "Section 2-2");
  assert.equal(sectionLabel("Section 19-3-30.1 .", 3), "Section 19-3-30.1");
  assert.equal(sectionLabel("Sec. 4.", 3), "Section 4");
  assert.equal(sectionLabel("37-1-130.", 3), "Code section 37-1-130");
  assert.equal(sectionLabel("ARTICLE I", 3), "ARTICLE I");
  assert.equal(sectionLabel(null, 0), "Opening text");
  assert.equal(sectionLabel(null, 4), "Unheaded text, part 5");
});

test("a repeated number is told apart by its position, in the bill's order", () => {
  const got = sectionLabels([
    { ordinal: 0, heading: null },
    { ordinal: 1, heading: "SECTION 1." },
    { ordinal: 2, heading: "SECTION 2." },
    { ordinal: 3, heading: "Section 1." },
  ]);
  assert.deepEqual([...got.values()], ["Opening text", "Section 1 (first)", "Section 2", "Section 1 (second)"]);
});
