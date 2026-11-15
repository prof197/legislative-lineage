// Dev only: copies the pipeline's fixture export (tests/fixtures/export/: Utah's three bills, two
// synthetic model documents, one from a source marked leaked) into public/data, so origins and
// the sources page can be built and checked locally against export schema version 2. The deploy
// never runs this: it fetches its pinned release (scripts/fetch-data.sh), and CI copies the
// site's own fixture (npm run data:fixture). See DESIGN.md#origin-matches.
import { cp, rm } from "node:fs/promises";

const FIXTURE = new URL("../../tests/fixtures/export/", import.meta.url);
const DATA = new URL("../public/data/", import.meta.url);

await rm(DATA, { recursive: true, force: true });
await cp(FIXTURE, DATA, { recursive: true });
console.log("tests/fixtures/export -> public/data");
