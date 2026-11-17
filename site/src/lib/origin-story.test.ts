// `node --test`: the origin story's copy, on a constructed story. It names the organization only as
// the data does, says only what the two shares say, states the rule that chose the bill, and never
// says who wrote the bill.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { BillRef, Origin } from "@/types/export";
import { NO_ORIGIN_STORY, ORIGIN_STORY_CAVEAT, type OriginStory, originStoryCopy, originText } from "./origins.ts";

const bill: BillRef = {
  slug: "ga-2023_24-hb-1",
  state: "GA",
  session: "2023_24",
  identifier: "HB 1",
  title: "Example Act; enact",
  status: "passed",
  status_reason: "signed",
  status_date: "2024-05-01",
  first_action: "2023-01-10",
};

const origin: Origin = {
  doc: {
    id: "example-library/model-act",
    source: "example-library",
    organization: "Example Council",
    title: "Model Example Act",
    url: "https://example.org/models/model-act",
    published: "2019-05",
    leaked: false,
    co_publishers: ["Second Institute"],
  },
  stages: ["near_duplicate", "semantic"],
  bill_share: 0.62,
  model_share: 0.4,
  sections: [1, 2],
};

const story: OriginStory = { bill, origin, eligible: 14, passed_over: 0 };
const names = { bill: "Georgia HB 1", state: "Georgia" };
const copy = originStoryCopy(story, names);

test("the story is titled by its bill and a published model, naming no organization", () => {
  assert.equal(copy.title, "Georgia HB 1 and a published model");
  assert.equal(copy.headline, "A passed Georgia bill that shares wording with a published model");
  assert.doesNotMatch(`${copy.title} ${copy.headline}`, /Example Council/);
});

test("the lede names the bill; the line beside it gives both shares", () => {
  assert.equal(copy.lede, "Georgia HB 1, Example Act; enact, passed.");
  assert.equal(originText(origin), "62% of this bill's wording appears in Model Example Act, a model published by Example Council, covering 40% of the model.");
});

test("the shares are explained as what they count, nothing more", () => {
  assert.deepEqual(copy.shares, [
    "62% of Georgia HB 1's wording is in sections that match sections of the model, by shared wording or similar meaning.",
    "The model's matching sections hold 40% of its wording.",
    "Sections such as definitions, severability and effective dates are left out of both counts.",
  ]);
  const whole = originStoryCopy({ ...story, origin: { ...origin, bill_share: 0.004, model_share: 1 } }, names);
  assert.equal(whole.shares[0], "Under 1% of Georgia HB 1's wording is in sections that match sections of the model, by shared wording or similar meaning.");
  assert.equal(whole.shares[1], "The model's matching sections hold 100% of its wording.");
});

test("the rule that chose the bill is stated, with how many bills it read", () => {
  assert.equal(
    copy.rule,
    "Of the 14 passed Georgia bills with a listed model document that both stages found, this bill has the largest share of its wording in such a document. A tie goes to the larger share of the model, then the lower bill id. A match is featured only after a model has read the passages it rests on and judged them the same provision. The rule names no organization; the one shown is whichever publishes the document.",
  );
  assert.equal(
    originStoryCopy({ ...story, eligible: 1 }, names).rule,
    "It is the only passed Georgia bill with a listed model document that both stages found. Among several, the one with the largest share of its wording in such a document would be shown; a tie would go to the larger share of the model, then the lower bill id. A match is featured only after a model has read the passages it rests on and judged them the same provision. The rule names no organization; the one shown is whichever publishes the document.",
  );
  assert.equal(
    originStoryCopy({ ...story, passed_over: 2 }, names).rule,
    "Of the 14 passed Georgia bills with a listed model document that both stages found, 2 hold a larger share of their wording in such a document, but their matches have not been judged the same provision. Of the matches that have, this bill's holds the largest share; a tie goes to the larger share of the model, then the lower bill id. A match is featured only after a model has read the passages it rests on and judged them the same provision. The rule names no organization; the one shown is whichever publishes the document.",
  );
  assert.match(originStoryCopy({ ...story, passed_over: 1 }, names).rule, /, one holds a larger share of its wording in such a document, but its match has not been judged/);
  assert.match(originStoryCopy({ ...story, eligible: 1200 }, names).rule, /^Of the 1,200 passed Georgia bills with /);
});

test("no authorship, direction or political label in the story's own copy", () => {
  const all = [copy.title, copy.headline, copy.lede, ...copy.shares, copy.rule, ORIGIN_STORY_CAVEAT, NO_ORIGIN_STORY("Georgia")].join(" ");
  assert.doesNotMatch(all.replace("not who wrote either one", ""), /\b(written by|wrote|drafted|authored|copied|the origin|descendant|conservative|liberal|progressive|partisan)\b/i);
  assert.equal(ORIGIN_STORY_CAVEAT, "Matching shows where the texts share wording, not who wrote either one or which came first.");
});

test("with no story, the page says no bill met the rule", () => {
  assert.equal(NO_ORIGIN_STORY("Georgia"), "No passed Georgia bill in this data has a listed model document that both stages found and a model has judged the same provision on the passages it rests on.");
});
