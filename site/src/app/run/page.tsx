import type { Metadata } from "next";
import { Footer, Header, REPOSITORY } from "@/components/Chrome";
import { RunCommands } from "@/components/RunCommands";
import { meta, runFigures } from "@/lib/data";
import { asOf, stateName } from "@/lib/format";

export const metadata: Metadata = { title: "Run it for your state" };

const LIMITS: [string, string][] = [
  [
    "Stored text is often the introduced version",
    "Many states publish a bill's introduced text first and its final text late or not at all. A comparison may show language that was amended away before passage. Every comparison says which version it uses.",
  ],
  [
    "The data is a dated snapshot",
    "", // filled with the snapshot's date below
  ],
  [
    "Similarity is not authorship",
    "Two bills can share language because both drew on a model act, a federal law, an older statute or each other. The tool shows that texts are alike and when each appeared, not who wrote them or why.",
  ],
];

const hours = (s: number) => `${(s / 3600).toFixed(1)} hours`;
const minutes = (s: number) => `${Math.round(s / 60)} minutes`;
const gb = (b: number) => `${(b / 1e9).toFixed(1)} GB`;
const gib = (b: number) => `${(b / 2 ** 30).toFixed(0)} GiB`;

export default function RunPage() {
  const m = meta();
  const f = runFigures();
  const latest = m.archives.map((a) => a.generated_at).sort().at(-1);
  return (
    <>
      <Header current="run" />
      <main className="wrap" style={{ paddingTop: 56 }}>
        <h1 className="display">Run it for your state</h1>
        <p className="lede">
          This site shows {stateName(m.state)}. The open-source tool that built it runs on your own
          computer: it builds the 50-state index from Open States&apos; archives and the legislatures&apos;
          published text, then writes a state&apos;s pages. It has built {stateName(m.state)}&apos;s site;
          building the site for other states is being tested. For now it builds everything from source.
        </p>
        <section aria-labelledby="quick" style={{ marginTop: 48 }}>
          <h2 id="quick" className="section">Build it from source</h2>
          <RunCommands repository={REPOSITORY} initial={m.state} />
        </section>
        <section aria-labelledby="get" style={{ marginTop: 48 }}>
          <h2 id="get" className="section">What you get</h2>
          <ul className="body-serif" style={{ marginTop: 12, paddingLeft: 22 }}>
            <li>A copy of this site for the state you export, from its 2023–2026 bills: the browse list, a page for each bill and side-by-side comparisons.</li>
            <li><code>uv run lineage query bill.txt</code>, which lists the related bills across the 50-state index for any bill&apos;s text saved as a file, offline.</li>
            <li>The validation figures, rerun against your build with <code>uv run lineage eval commission</code> once you download the acts&apos; texts from the Uniform Law Commission&apos;s site, as the README says.</li>
            <li>Plain static files you can host anywhere, with no server or database.</li>
          </ul>
        </section>
        <section aria-labelledby="long" style={{ marginTop: 48 }}>
          <h2 id="long" className="section">How long it takes</h2>
          <p className="note" style={{ marginTop: 12 }}>Measured on the reference laptop, with {gib(f.machineMemoryBytes)} of memory:</p>
          <ul className="body-serif" style={{ marginTop: 8, paddingLeft: 22 }}>
            <li>Matching wording across the index: {minutes(f.indexSeconds)}</li>
            <li>Embedding every window of text by meaning: {hours(f.embedSeconds)}. It resumes where it stopped.</li>
            <li>Finding each window&apos;s nearest neighbors by meaning: {minutes(f.neighborsSeconds)}</li>
            <li>Writing {stateName(f.exportState)}&apos;s pages: {hours(f.exportSeconds)}</li>
          </ul>
          <p className="note">The downloads, the text fetch and the other steps were not timed. The text fetch asks each legislature&apos;s server for one page at a time.</p>
          <p className="note">
            The archives take {gb(f.archiveBytes)}, the wording index {gb(f.indexBytes)} and the meaning vectors {gb(f.vectorBytes)},
            besides the text tables; a large write stops if it would leave under 20 GiB free. Memory peaked at{" "}
            {gb(f.peakMemoryBytes)}. No account or API key.
          </p>
        </section>
        <section aria-labelledby="limits" style={{ marginTop: 48 }}>
          <h2 id="limits" className="section">Known limits</h2>
          <ol style={{ listStyle: "none", margin: "16px 0 0", padding: 0, display: "grid", gap: 20 }}>
            {LIMITS.map(([title, body], i) => (
              <li key={title} style={{ display: "grid", gridTemplateColumns: "36px 1fr", gap: 12 }}>
                <p className="claim-n" style={{ margin: 0, fontSize: 20 }}>{i + 1}</p>
                <div>
                  <h3 style={{ margin: 0, fontSize: 18 }}>{title}</h3>
                  <p className="note" style={{ margin: "6px 0 0" }}>
                    {body ||
                      `The index reflects bill data as of ${latest ? asOf(latest) : "the snapshot date"}. Statuses change after that. Check the legislature's own site before you publish.`}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </section>
        <p className="note" style={{ marginTop: 40 }}>
          Source code, documentation and issues: <a href={REPOSITORY}>{REPOSITORY.replace("https://", "")}</a> · License: MIT
        </p>
      </main>
      <Footer />
    </>
  );
}
