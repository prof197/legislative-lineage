// Copies the committed fixture export (a few bills, their comparisons and texts, the stories
// and the data stamp, trimmed from a real export) into public/data, so CI builds and type-checks
// the site with no release download.
import { cp, rm } from "node:fs/promises";

const FIXTURE = new URL("../fixture/", import.meta.url);
const DATA = new URL("../public/data/", import.meta.url);

await rm(DATA, { recursive: true, force: true });
await cp(FIXTURE, DATA, { recursive: true });
console.log("fixture -> public/data");
