// `node --test`: origin lines give both shares, "N% of this bill's wording appears in [title], a
// model published by [organization], covering M% of the model", most of the bill first, and a
// share never rounds to all or none of the wording.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Origin, SourceEntry } from "@/types/export";
import { byBillShare, bySource, docDate, documents, foundBy, isoDate, ORIGIN_LINE, originLine, originText, percent, possessive, publishers, shownTitle, withheldNote, sentence, termsRead, updatedLine } from "./origins.ts";

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
    before: "34% of this bill's wording appears in",
    title: "Model Act",
    after: ", a model published by Org, covering 25% of the model.",
  });
  assert.equal(originText(o), "34% of this bill's wording appears in Model Act, a model published by Org, covering 25% of the model.");
  assert.equal(originText(origin("Short Act", 0.12, 1)), "12% of this bill's wording appears in Short Act, a model published by Org, covering 100% of the model.");
});

test("a share under 1% opens the line capitalized", () => {
  assert.equal(originLine(origin("Long Act", 0.004, 0.9)).before, "Under 1% of this bill's wording appears in");
  assert.equal(originLine(origin("Long Act", 0.9, 0.004)).after, ", a model published by Org, covering under 1% of the model.");
});

test("the organization is named as the data names it, after the title", () => {
  assert.equal(possessive("American Legislative Exchange Council (ALEC)"), "American Legislative Exchange Council (ALEC)'s");
  assert.equal(possessive("Public Citizen"), "Public Citizen's");
  assert.equal(possessive("Institute for Family Studies"), "Institute for Family Studies'");
  const o = { ...origin("A", 0.5, 0.5), doc: { ...origin("A", 0.5, 0.5).doc, organization: "The States Project", co_publishers: ["Second Institute"] } };
  assert.equal(originLine(o).after, ", a model published by The States Project, covering 50% of the model.");
});

test("a leaked document names who drafted it and the archive that published the leak", () => {
  const alec = "American Legislative Exchange Council (ALEC)";
  const leaked = { ...origin("ALEC Model Act", 0.4, 0.5), doc: { ...origin("A", 0, 0).doc, title: "ALEC Model Act", organization: "Example Archive Project", leaked: true, drafted_by: alec } };
  const text = originText(leaked);
  assert.equal(text, `40% of this bill's wording appears in Model Act, an ${alec} model from a leaked document published by Example Archive Project, covering 50% of the model.`);
  assert.match(text, ORIGIN_LINE);
  const ncoil = { ...leaked, doc: { ...leaked.doc, drafted_by: "National Council of Insurance Legislators (NCOIL)" } };
  assert.match(originText(ncoil), /, a National Council of Insurance Legislators \(NCOIL\) model from a leaked document/);
  const uniform = { ...leaked, doc: { ...leaked.doc, drafted_by: "Uniform Law Commission" } };
  assert.match(originText(uniform), /, a Uniform Law Commission model/);
});

test("a title that opens with its organization's name or abbreviation does not repeat it", () => {
  const ncoil = "National Council of Insurance Legislators (NCOIL)";
  assert.equal(shownTitle("NCOIL Insurance Business Transfer Model Act", ncoil), "Insurance Business Transfer Model Act");
  assert.equal(shownTitle("National Council of Insurance Legislators Model Act", ncoil), "Model Act");
  assert.equal(shownTitle("Public Citizen Model State Law: A Consumer's Right To Know", "Public Citizen"), "Model State Law: A Consumer's Right To Know");
  assert.equal(shownTitle("ALEC's Model Resolution on Federalism", "American Legislative Exchange Council (ALEC)"), "Model Resolution on Federalism");
  assert.equal(shownTitle("States Project: Right to Read Act", "The States Project (policy work conducted by Future Now Action)"), "Right to Read Act");
  // Only an opening name, and only a whole word of it.
  assert.equal(shownTitle("Renaming the ALEC Safeguard American Votes Act", "American Legislative Exchange Council (ALEC)"), "Renaming the ALEC Safeguard American Votes Act");
  assert.equal(shownTitle("NCOILS Act", ncoil), "NCOILS Act");
  assert.equal(shownTitle("Public Citizenship Act", "Public Citizen"), "Public Citizenship Act");
  // A title that is only the organization's name stays whole.
  assert.equal(shownTitle("NCOIL", ncoil), "NCOIL");
  const o = { ...origin("NCOIL Insurance Fraud Model Act", 0.82, 0.98), doc: { ...origin("A", 0, 0).doc, title: "NCOIL Insurance Fraud Model Act", organization: ncoil } };
  assert.equal(originText(o), `82% of this bill's wording appears in Insurance Fraud Model Act, a model published by ${ncoil}, covering 98% of the model.`);
});

test("the copy check's form accepts every line and rejects the old one", () => {
  for (const [b, m] of [[0.3418, 0.25], [0.004, 1], [1, 0.004], [0.5, 0.996]] as const) {
    assert.match(originText(origin("Model, with a comma (2019)", b, m)), ORIGIN_LINE);
  }
  assert.doesNotMatch("Shares 34% of its wording with Model Act.", ORIGIN_LINE);
  assert.doesNotMatch("34% of this bill's wording appears in Org's model Model Act.", ORIGIN_LINE);
  assert.doesNotMatch("34% of this bill's wording appears in Org's model Model Act, covering 25% of the model.", ORIGIN_LINE);
  assert.doesNotMatch("34% of this bill's wording was written by Model Act, a model published by Org, covering 25% of the model.", ORIGIN_LINE);
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

test("a source's terms note ends as a sentence, and the day it was read follows as its own", () => {
  assert.equal(sentence("viewing and copying with attribution"), "viewing and copying with attribution.");
  assert.equal(sentence(" robots.txt disallows nothing. "), "robots.txt disallows nothing.");
  assert.equal(sentence("Is copying allowed?"), "Is copying allowed?");
  assert.equal(termsRead("2026-11-15"), "Terms read Nov 15, 2026.");
});

test("a bill page dates its matches only when the live track set the day", () => {
  assert.equal(updatedLine("2026-11-16"), "Related bills and model documents updated Nov 16, 2026");
  assert.equal(updatedLine(null), null);
  assert.equal(updatedLine(undefined), null);
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

test("a library's withheld documents are named with why, as its count reads", () => {
  assert.match(withheldNote(1), /^This document is withheld from bill pages, the feed and stories until matching improves: most of the matches checked for it rest on generic text/);
  assert.match(withheldNote(2), /^These 2 documents are withheld .* checked for them rest on generic text/);
  assert.equal(withheldNote(1, "review"), "This document is withheld from bill pages, the feed and stories: a review of new matches found those listed for it did not share a provision with the bills.");
});
