// `node --test`: section labels read one way, keep the bill's order, and tell repeats apart.
import assert from "node:assert/strict";
import { test } from "node:test";
import { aState, leadSession, sectionLabel, sectionLabels, session, sessionPhrase } from "./format.ts";

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

test("sessions read the same for any state, a special one by its own name", () => {
  assert.equal(session("AK", "33"), "2023–24");
  assert.equal(session("GA", "2025_26"), "2025–26");
  assert.equal(sessionPhrase("GA", "2025_26"), "2025–26 session");
  assert.equal(sessionPhrase("GA", "2026_ss"), "2026 Special Session");
  assert.equal(session("CA", "20232024"), "2023–24"); // it began the December before
  assert.equal(session("ZZ", "x1"), "x1");
});

test("a site leads with its latest regular session, not a short special one inside it", () => {
  assert.equal(leadSession("GA", new Map([["2023_24", 2000], ["2025_26", 2300], ["2026_ss", 40]])), "2025_26");
  assert.equal(leadSession("AK", new Map([["33", 800], ["34", 700]])), "34");
});

test("a state's name takes the article its sound needs", () => {
  assert.equal(aState("Alaska"), "an Alaska");
  assert.equal(aState("Ohio"), "an Ohio");
  assert.equal(aState("Georgia"), "a Georgia");
  assert.equal(aState("Utah"), "a Utah");
});
