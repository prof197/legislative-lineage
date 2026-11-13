import type { Metadata } from "next";
import { Footer, Header } from "@/components/Chrome";
import { Enactments } from "@/components/Enactments";
import { meta } from "@/lib/data";

export const metadata: Metadata = { title: "How we checked" };

const ACT_NAME: Record<string, string> = {
  "public-expression-protection": "Uniform Public Expression Protection Act",
  "electronic-estate-planning-documents": "Uniform Electronic Estate Planning Documents Act",
  "special-deposits": "Uniform Special Deposits Act",
};

const STAGE: [string, string, string][] = [
  ["near_duplicate", "Exact-match stage alone", "Sections that share long runs of identical wording"],
  ["semantic", "Similar-meaning stage alone", "Sections that say the same thing in different words"],
  ["both", "Both stages together", "A match by either stage"],
];

const pct = (x: number) => `${Math.round(x * 100)}%`;

export default function ValidationPage() {
  const v = meta().validation;
  const r = v.development_recall;
  const before = v.development_excluded.filter((x) => /before the corpus/.test(x.why)).length;
  const acts = Object.keys(v.development_recall_by_act);
  const p = v.relatives_precision;
  return (
    <>
      <Header current="validation" />
      <main className="wrap" style={{ paddingTop: 56 }}>
        <p className="eyebrow">How we checked</p>
        <h1 className="display" style={{ marginTop: 10 }}>Does it find the bills it should?</h1>
        <p className="lede">
          The Uniform Law Commission publishes where each of its acts has been enacted. We used those
          lists as an answer key for three uniform acts: for each enactment between 2023 and 2026, did
          the tool find the enacted bill on its own, starting only from the act&apos;s text?
        </p>

        <section aria-labelledby="pooled" style={{ marginTop: 48 }}>
          <h2 id="pooled" className="section">We found {r.found} of {r.enactments} enactments the Uniform Law Commission lists</h2>
          <p className="note" style={{ marginTop: 8 }}>
            Pooled across three uniform acts, 2023–2026 ({pct(r.recall)}). The Commission lists {before} more
            enactments from before 2023; they are outside this corpus and not counted.
          </p>
        </section>

        <section aria-labelledby="by-act" style={{ marginTop: 40 }}>
          <h2 id="by-act" className="section" style={{ fontSize: 24 }}>By act</h2>
          <ul style={{ listStyle: "none", margin: "16px 0 0", padding: 0, display: "grid", gap: 16 }}>
            {acts.map((a) => {
              const rows = v.development_enactments.filter((e) => e.act === a);
              const x = v.development_recall_by_act[a]!;
              return (
                <li key={a} style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 20px" }}>
                  <span style={{ flex: "0 1 340px" }}>{ACT_NAME[a] ?? a}</span>
                  <ul className="squares" aria-label={`${ACT_NAME[a] ?? a}: found ${x.found} of ${x.enactments}`}>
                    {rows.map((e) => <li key={`${e.state}-${e.bill_cell}`} className={e.found ? "found" : "missed"} title={`${e.state} ${e.bill_cell} (${e.year}): ${e.found ? "found" : "missed"}`} />)}
                  </ul>
                  <strong style={{ whiteSpace: "nowrap" }}>{x.found} of {x.enactments}</strong>
                </li>
              );
            })}
          </ul>
          <p className="note">Found Missed One square per enactment the Commission lists.</p>
        </section>

        <section aria-labelledby="stages" style={{ marginTop: 48 }}>
          <h2 id="stages" className="section" style={{ fontSize: 24 }}>Each matching stage alone, and both together</h2>
          <p className="note" style={{ marginTop: 8 }}>
            The exact-match stage finds sections that share long runs of identical wording. The
            similar-meaning stage finds sections that say the same thing in different words. Together they
            find more, and the exact stage keeps precision up.
          </p>
          <div style={{ marginTop: 16, display: "grid", gap: 14, maxWidth: 720 }}>
            {STAGE.map(([key, title, def]) => {
              const x = v.development_recall_by_stage[key]!;
              return (
                <figure key={key} style={{ margin: 0 }}>
                  <figcaption><strong>{title}</strong> <span className="small">{def}</span></figcaption>
                  <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 6 }}>
                    <span style={{ flex: 1, height: 14, background: "#eceae4", display: "block" }} aria-hidden="true">
                      <span style={{ display: "block", height: 14, width: pct(x.recall), background: "var(--ink)" }} />
                    </span>
                    <span style={{ flex: "none", width: 200, whiteSpace: "nowrap" }}>Recall {pct(x.recall)} ({x.found} of {x.enactments})</span>
                  </div>
                </figure>
              );
            })}
          </div>
          <p className="note">
            Recall is measured against the {r.enactments} listed enactments. Precision comes from a sample of{" "}
            {p.labeled} relatives that Georgia bill pages list, each judged by a model on the matched
            passages: {p.correct} of {p.labeled} share the provision ({p.share === null ? "–" : pct(p.share)},
            95% interval {pct(p.interval_95[0])}–{pct(p.interval_95[1])}). Matches to bills the Commission
            does not list, such as non-uniform anti-SLAPP bills, are not counted as wrong if the provision
            is shared.
          </p>
        </section>

        <section aria-labelledby="every" style={{ marginTop: 48 }}>
          <h2 id="every" className="section" style={{ fontSize: 24 }}>Every listed enactment</h2>
          <p className="note" style={{ marginTop: 8 }}>Found or missed, with the reason for each miss.</p>
          <Enactments acts={acts.map((a) => ({ key: a, name: ACT_NAME[a] ?? a }))} enactments={v.development_enactments} />
          <p className="note">
            Found means the enacted bill appears among the tool&apos;s matches for the act&apos;s text, by
            either stage, the similar-meaning stage at a cosine of at least {v.semantic_threshold}. The answer
            key is the Commission&apos;s enactment lists as of the snapshot date.
          </p>
        </section>
      </main>
      <Footer />
    </>
  );
}
