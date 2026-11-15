// Every session's display label, from the pinned session list (snapshot/sessions.csv), so the
// site names any state's sessions the same way. A regular session is its years, "2025" or
// "2023–24", taken from its official name where the name gives them (a session can start the
// December before its first year) and from its dates otherwise. A special, extraordinary or
// called session keeps its official name, which is how the state tells several in a year apart.
// `--check` fails when the committed file is stale.
import { readFile, writeFile } from "node:fs/promises";

const CSV = new URL("../../snapshot/sessions.csv", import.meta.url);
const OUT = new URL("../src/lib/sessions.json", import.meta.url);
const SPECIAL = /special|extraordinary|called/i;
const YEARS = /\b(\d{4})(?:\s*[-–]\s*(\d{4}))?\b/;

const lines = (await readFile(CSV, "utf8")).split("\n").filter((l) => l && !l.startsWith("#"));
const [head, ...rows] = lines;
if (head !== "state,session,name,start_date,end_date,carries_over_to") throw new Error(`unexpected header: ${head}`);

// A field may be quoted when it holds a comma ("2023-2024, Special Session 1").
const fields = (line) => [...line.matchAll(/("([^"]*)"|[^,]*)(,|$)/g)].slice(0, 6).map((m) => m[2] ?? m[1]);

const out = {};
for (const line of rows) {
  const [state, id, name, start, end] = fields(line);
  let label = name;
  if (!SPECIAL.test(name)) {
    const named = YEARS.exec(name);
    const from = named ? Number(named[1]) : Number(start.slice(0, 4));
    const to = named ? Number(named[2] ?? named[1]) : Number(end.slice(0, 4));
    label = to > from ? `${from}–${String(to).slice(2)}` : `${from}`;
  }
  (out[state] ??= {})[id] = { label, special: SPECIAL.test(name), start };
}
const text = JSON.stringify(out, null, 1) + "\n";
if (process.argv.includes("--check")) {
  const committed = await readFile(OUT, "utf8").catch(() => "");
  if (committed !== text) {
    console.error("src/lib/sessions.json is stale: run `npm run sessions`");
    process.exit(1);
  }
} else {
  await writeFile(OUT, text);
  console.log(`${rows.length} sessions -> src/lib/sessions.json`);
}
