// `node --test`: the role labels resolve to bill names, in any case and with possessives, and
// nothing else in a claim changes.
import assert from "node:assert/strict";
import { test } from "node:test";
import { nameRoles } from "./claims.ts";

const A = "Georgia SB 351";
const B = "South Carolina H 5209";

test("each role label becomes its bill's name, at a sentence start and mid-sentence", () => {
  assert.equal(
    nameRoles("The descendant sets the fee every biennium; the origin sets it each fiscal year.", A, B),
    "South Carolina H 5209 sets the fee every biennium; Georgia SB 351 sets it each fiscal year.",
  );
});

test("possessives keep their apostrophe, straight or curly", () => {
  assert.equal(nameRoles("The descendant's notice period is longer than the origin’s.", A, B), "South Carolina H 5209's notice period is longer than Georgia SB 351’s.");
});

test("any case resolves; other words that start alike are left alone", () => {
  assert.equal(nameRoles("THE ORIGIN keeps the original wording; the descendants of the originator differ.", A, B), "Georgia SB 351 keeps the original wording; the descendants of the originator differ.");
  assert.equal(nameRoles("A minor's account needs a parent's consent.", A, B), "A minor's account needs a parent's consent.");
});
