import type { Metadata } from "next";
import { Footer, Header, REPOSITORY } from "@/components/Chrome";
import { RunCommands } from "@/components/RunCommands";
import { meta } from "@/lib/data";
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

export default function RunPage() {
  const m = meta();
  const latest = m.archives.map((a) => a.generated_at).sort().at(-1);
  return (
    <>
      <Header current="run" />
      <main className="wrap" style={{ paddingTop: 56 }}>
        <h1 className="display">Run it for your state</h1>
        <p className="lede">
          This site shows {stateName(m.state)}. The same open-source tool runs on your own computer for any
          state, against the same 50-state index. It is also where you can paste in a bill of your own and
          see what shares its language.
        </p>
        <section aria-labelledby="quick" style={{ marginTop: 48 }}>
          <h2 id="quick" className="section">Quick start</h2>
          <RunCommands />
        </section>
        <section aria-labelledby="get" style={{ marginTop: 48 }}>
          <h2 id="get" className="section">What you get</h2>
          <ul className="body-serif" style={{ marginTop: 12, paddingLeft: 22 }}>
            <li>A copy of this site for your state&apos;s 2023–2026 bills: the browse list, a page for each bill, maps and side-by-side comparisons.</li>
            <li>A paste-a-bill box that works only on your computer. Paste any bill&apos;s text to see related bills across the 50-state index.</li>
            <li>The validation report, rerun against your build, so you can check the numbers yourself.</li>
            <li>Plain static files you can host anywhere, with no server or database.</li>
          </ul>
        </section>
        <section aria-labelledby="long" style={{ marginTop: 48 }}>
          <h2 id="long" className="section">How long it takes</h2>
          <p style={{ margin: "12px 0 0", fontFamily: "var(--serif)", fontSize: 28 }}>30–45 min <span className="small">first build, recent laptop</span></p>
          <p className="note">
            The first build downloads the 50-state index snapshot, about 4 GB. Most of the time goes to the
            similar-meaning stage. Building for a second state takes about 10 minutes.
          </p>
          <p className="note">Needs Python 3.13 or later and about 6 GB of free disk. No account or API key.</p>
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
