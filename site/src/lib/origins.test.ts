// `node --test`: origin lines read "Shares N% of its wording with …", most of the bill first, and
// a share never rounds to all or none of the wording.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Origin, SourceEntry } from "@/types/export";
import { byBillShare, bySource, docDate, documents, foundBy, isoDate, modelShareLine, originLead, percent } from "./origins.ts";

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

test("every origin line reads Shares N% of its wording with", () => {
  assert.equal(originLead(origin("a", 0.3418, 1)), "Shares 34% of its wording with");
  assert.equal(modelShareLine(origin("a", 0.3418, 1)), "The shared wording is all of the document's text.");
  assert.equal(modelShareLine(origin("a", 0.3418, 0.25)), "The shared wording is 25% of the document's text.");
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
