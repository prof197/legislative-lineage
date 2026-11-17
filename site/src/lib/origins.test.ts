// `node --test`: origin lines give both shares, "N% of this bill's wording appears in [organization]'s
// model [title], covering M% of the model", most of the bill first, and a share never rounds to all
// or none of the wording.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Origin, SourceEntry } from "@/types/export";
import { byBillShare, bySource, docDate, documents, foundBy, isoDate, ORIGIN_LINE, originLine, originText, percent, possessive, publishers } from "./origins.ts";

function origin(title: string, bill_share: number, model_share: number, published: string | null = null): Origin {
  return {
    doc: { id: `lib/${title}`, source: "lib", organization: "Org", title, url: `https://example.org/${title}`, published, leaked: false },
    stages: ["near_duplicate"],
    bill_share,
    model_share,
    sections: [0],
  };
}

test("a share reads as a whole percent, never all or none unless it is", () => {
  assert.equal(percent(0.3418), "34%");
  assert.equal(percent(0.4463), "45%");
  assert.equal(percent(1), "100%");
  assert.equal(percent(0.996), "99%");
  assert.equal(percent(0.004), "under 1%");
  assert.equal(percent(0), "0%");
});

test("every origin line gives both shares: the bill's, then the model's", () => {
  const o = origin("Model Act", 0.3418, 0.25);
  assert.deepEqual(originLine(o), {
    before: "34% of this bill's wording appears in Org's model",
    title: "Model Act",
    after: ", covering 25% of the model.",
  });
  assert.equal(originText(o), "34% of this bill's wording appears in Org's model Model Act, covering 25% of the model.");
  assert.equal(originText(origin("Short Act", 0.12, 1)), "12% of this bill's wording appears in Org's model Short Act, covering 100% of the model.");
});

test("a share under 1% opens the line capitalized", () => {
  assert.equal(originLine(origin("Long Act", 0.004, 0.9)).before, "Under 1% of this bill's wording appears in Org's model");
  assert.equal(originLine(origin("Long Act", 0.9, 0.004)).after, ", covering under 1% of the model.");
});

test("the organization is named as the data names it, with its possessive", () => {
  assert.equal(possessive("American Legislative Exchange Council (ALEC)"), "American Legislative Exchange Council (ALEC)'s");
  assert.equal(possessive("Public Citizen"), "Public Citizen's");
  assert.equal(possessive("Institute for Family Studies"), "Institute for Family Studies'");
  const o = { ...origin("A", 0.5, 0.5), doc: { ...origin("A", 0.5, 0.5).doc, organization: "The States Project", co_publishers: ["Second Institute"] } };
  assert.equal(originLine(o).before, "50% of this bill's wording appears in The States Project's model");
});

test("the copy check's form accepts every line and rejects the old one", () => {
  for (const [b, m] of [[0.3418, 0.25], [0.004, 1], [1, 0.004], [0.5, 0.996]] as const) {
    assert.match(originText(origin("Model, with a comma (2019)", b, m)), ORIGIN_LINE);
  }
  assert.doesNotMatch("Shares 34% of its wording with Model Act.", ORIGIN_LINE);
  assert.doesNotMatch("34% of this bill's wording appears in Org's model Model Act.", ORIGIN_LINE);
  assert.doesNotMatch("34% of this bill's wording was written by Org's model Model Act, covering 25% of the model.", ORIGIN_LINE);
});

test("most of the bill's words first, ties by the document's share, then title", () => {
  const got = byBillShare([origin("c", 0.2, 0.5), origin("b", 0.6, 0.1), origin("a", 0.2, 0.9), origin("d", 0.2, 0.5)]);
  assert.deepEqual(got.map((o) => o.doc.title), ["b", "a", "c", "d"]);
});

test("dates keep the precision the organization gave", () => {
  assert.equal(isoDate("2019"), "2019");
  assert.equal(isoDate("2019-05"), "May 2019");
  assert.equal(isoDate("2019-05-03"), "May 3, 2019");
  assert.equal(isoDate("2026-11-15"), "Nov 15, 2026");
  assert.equal(isoDate("someday"), "someday");
  assert.equal(docDate(origin("a", 0.1, 0.1)), "Undated");
  assert.equal(docDate(origin("a", 0.1, 0.1, "2019-05")), "Dated May 2019");
});

test("stages and counts", () => {
  const o = { ...origin("a", 0.1, 0.1), stages: ["near_duplicate", "semantic", "semantic"] as Origin["stages"] };
  assert.equal(foundBy(o), "Found by shared wording and similar meaning");
  assert.equal(documents(1), "1 document");
  assert.equal(documents(1200), "1,200 documents");
});

test("sources by organization", () => {
  const s = (key: string, organization: string) => ({ key, organization }) as SourceEntry;
  assert.deepEqual(bySource([s("b", "Beta"), s("a2", "Alpha"), s("a1", "Alpha")]).map((x) => x.key), ["a1", "a2", "b"]);
});

test("a joint document names every publisher, its library's first", () => {
  const solo = origin("joint", 0.5, 0.4);
  assert.equal(publishers(solo), solo.doc.organization);
  const one = { ...solo, doc: { ...solo.doc, co_publishers: ["Second Institute"] } };
  assert.equal(publishers(one), `${solo.doc.organization}, with Second Institute`);
  const two = { ...solo, doc: { ...solo.doc, co_publishers: ["Second Institute", "Third Center"] } };
  assert.equal(publishers(two), `${solo.doc.organization}, with Second Institute and Third Center`);
});
