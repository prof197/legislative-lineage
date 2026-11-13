import Link from "next/link";
import { Footer, Header } from "@/components/Chrome";
import { MiniMap } from "@/components/Tiles";
import { billList, meta, story } from "@/lib/data";
import { fills, storyBest, tally } from "@/lib/facts";
import { STATUS_LABEL, stateName } from "@/lib/format";
import { STORY_COPY, STORY_ORDER } from "@/lib/stories";

function Arrow() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 8h10M9 4l4 4-4 4" />
    </svg>
  );
}

export default function Home() {
  const m = meta();
  const v = m.validation;
  const state = stateName(m.state);
  const latest = billList().bills.filter((b) => b.bill.session === "2025_26");
  const related = latest.filter((b) => b.families > 0).length;
  return (
    <>
      <Header current="home" />
      <main>
        <section className="wrap" style={{ paddingTop: 80, paddingBottom: 64, display: "flex", flexWrap: "wrap", alignItems: "flex-start", gap: "48px 56px" }}>
          <div style={{ flex: "999 1 540px", minWidth: 0 }}>
            <p className="eyebrow" style={{ fontSize: 13, marginBottom: 20 }}>State bills, traced across states</p>
            {/* TODO(v2): once lineage (direction from dates, documented origins and shared quirks of
                wording) lands, swap back in the mocks' tagline:
                  headline "Which {state} bills share language with other states?"
                  lede "Pick a {state} bill from the 2023–2026 sessions to see bills in other states
                    with similar provisions: where they were introduced, what happened to each one,
                    and what each state changed. It shows shared language and timing. It does not
                    tell you who wrote a bill."
                  and the aside's close "…so you know when to start asking… who is really writing
                    our laws?"
                v1 states no direction or authorship, so it reads as below. */}
            <h1 className="hero">Why does that {state} bill look so familiar?</h1>
            <p className="lede" style={{ marginTop: 28, fontSize: 21, maxWidth: "34em", color: "var(--ink-3)" }}>
              Find bills in other states that share provisions with a {state} bill, and see how each
              one&apos;s text differs.
            </p>
            <form action="bills/" method="get" style={{ marginTop: 44, maxWidth: 680 }}>
              <label htmlFor="find" style={{ display: "block", fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
                Find a {state} bill
              </label>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
                <input id="find" name="q" type="search" className="input" placeholder="Bill number or title words, e.g. SB 120 or deepfake" style={{ flex: "1 1 320px", minWidth: 0, height: 52, fontSize: 17 }} />
                <button type="submit" className="button solid" style={{ height: 52 }}>
                  Browse {state} bills
                  <Arrow />
                </button>
              </div>
              <p className="note" style={{ marginTop: 14, maxWidth: "40em" }}>
                A 2019 investigation by USA TODAY, The Arizona Republic and the Center for Public
                Integrity examined nearly 1 million bills introduced from 2010 to October 2018, with
                comparisons run on the equivalent of more than 150 virtual machines, nonstop for
                months.{" "}
                <a href="https://publicintegrity.org/state-politics/copy-paste-legislate/how-we-uncovered-10000-times-lawmakers-introduced-copycat-model-bills-and-why-it-matters/">
                  How they did it
                </a>
              </p>
            </form>
          </div>
          <aside className="dark" aria-labelledby="why-h" style={{ flex: "1 1 340px", minWidth: 0, padding: "28px 28px 24px" }}>
            <p className="eyebrow" style={{ color: "var(--line)", fontWeight: 700 }}>Why it matters</p>
            <h2 id="why-h" style={{ margin: "12px 0 0", fontFamily: "var(--serif)", fontWeight: 500, fontSize: 30, lineHeight: 1.15 }}>
              A bill that looks local may not be.
            </h2>
            <p style={{ margin: "16px 0 0", fontFamily: "var(--serif)", fontSize: 18, lineHeight: 1.55, color: "#f1efea" }}>
              Lobbyists, industry groups and advocacy organizations write model bills and hand them to
              lawmakers in many states at once. A bill doesn&apos;t have to say who drafted it, so{" "}
              <mark>the same words</mark> can arrive in your statehouse looking homegrown.
            </p>
            <p style={{ margin: "14px 0 0", fontFamily: "var(--serif)", fontSize: 18, lineHeight: 1.55, color: "#f1efea" }}>
              Matching language across states is the trail. This site lines the versions up side by side
              and shows what happened to each one.
            </p>
          </aside>
        </section>

        <section aria-labelledby="reach" style={{ borderTop: "1px solid var(--line)", borderBottom: "1px solid var(--line)" }}>
          <div className="wrap" style={{ paddingTop: 36, paddingBottom: 36, display: "flex", flexWrap: "wrap", alignItems: "flex-end", justifyContent: "space-between", gap: "20px 40px" }}>
            <h2 id="reach" style={{ flex: "999 1 560px", minWidth: 0, margin: 0, fontWeight: 400 }}>
              <span style={{ display: "block", fontFamily: "var(--serif)", fontWeight: 500, fontSize: "clamp(52px, 7vw, 84px)", lineHeight: 0.95, letterSpacing: "-0.02em" }}>
                <mark style={{ padding: "0 10px", borderBottomWidth: 3 }}>{related.toLocaleString("en-US")}</mark> of {latest.length.toLocaleString("en-US")}
              </span>
              <span style={{ display: "block", marginTop: 12, maxWidth: "30em", fontFamily: "var(--serif)", fontSize: "clamp(20px, 2.2vw, 26px)", lineHeight: 1.35 }}>
                {state} bills from the 2025–26 session share provisions with bills in other states.
              </span>
            </h2>
            <Link href="/bills/?related=1" className="button">See those bills<Arrow /></Link>
          </div>
        </section>

        <section aria-labelledby="stories" id="stories" className="wrap" style={{ paddingTop: 80 }}>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "baseline", justifyContent: "space-between", gap: "12px 32px", marginBottom: 28 }}>
            <h2 id="stories" className="section" style={{ fontSize: 36 }}>Featured stories</h2>
            <ul aria-label="Map key" className="key">
              <li><span className="swatch passed" aria-hidden="true" />{STATUS_LABEL.passed}</li>
              <li><span className="swatch introduced" aria-hidden="true" />{STATUS_LABEL.introduced}</li>
              <li><span className="swatch died" aria-hidden="true" />Died (hatched)</li>
              <li><span className="swatch" aria-hidden="true" />No match</li>
            </ul>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(300px, 100%), 1fr))", gap: "48px 40px" }}>
            {STORY_ORDER.filter((s) => m.stories.includes(s)).map((slug) => {
              const s = story(slug);
              const copy = STORY_COPY[slug]!;
              const best = storyBest(s);
              const t = tally(best);
              return (
                <article key={slug} style={{ display: "flex", flexDirection: "column", gap: 14, borderTop: "2px solid var(--ink)", paddingTop: 20 }}>
                  <div style={{ padding: "8px 0 6px" }}><MiniMap fills={fills(best)} /></div>
                  <p className="eyebrow">{copy.eyebrow}</p>
                  <h3 className="card"><Link href={`/stories/${slug}/`} style={{ textDecoration: "none" }}>{copy.card}</Link></h3>
                  <p className="body-serif">{cardLine(slug, t.states, s)}</p>
                  <p style={{ margin: 0, fontSize: 14, color: "var(--ink-3)" }}>
                    {t.states} states · {t.passed} passed · {t.introduced} introduced · {t.died} died
                    {t.passed_legislature ? ` · ${t.passed_legislature} passed the legislature` : ""}
                  </p>
                </article>
              );
            })}
          </div>
        </section>

        <section aria-labelledby="check-h" className="wrap" style={{ paddingTop: 72 }}>
          <div className="panel" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "16px 40px" }}>
            <div style={{ flex: "999 1 520px", minWidth: 0, display: "flex", alignItems: "flex-start", gap: 14 }}>
              <svg width="26" height="26" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ flex: "none", marginTop: 2 }}>
                <circle cx="8" cy="8" r="6.5" /><path d="M5.2 8.3l2 1.9 3.6-3.9" />
              </svg>
              <div>
                <h2 id="check-h" className="eyebrow" style={{ fontWeight: 700 }}>Does it work?</h2>
                <p style={{ margin: "6px 0 0", fontFamily: "var(--serif)", fontSize: 21, lineHeight: 1.45 }}>
                  Checked against the Uniform Law Commission&apos;s records: the tool found{" "}
                  <strong style={{ fontWeight: 600 }}>
                    {v.development_recall.found} of {v.development_recall.enactments}
                  </strong>{" "}
                  enactments of three uniform acts, 2023–2026.
                </p>
              </div>
            </div>
            <Link href="/validation/" className="button solid">How we checked<Arrow /></Link>
          </div>
        </section>

        <section aria-labelledby="run" style={{ marginTop: 96, background: "var(--paper-2)", borderTop: "1px solid var(--line)", borderBottom: "1px solid var(--line)" }}>
          <div className="wrap" style={{ paddingTop: 56, paddingBottom: 56, display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "24px 48px" }}>
            <div style={{ flex: "999 1 520px", minWidth: 0 }}>
              <h2 id="run" className="section" style={{ fontSize: 34 }}>Not in {state}? Run it for your state.</h2>
              <p className="lede" style={{ marginTop: 12, maxWidth: "36em", fontSize: 19, color: "var(--ink-3)" }}>
                The index covers all 50 states. The open-source tool builds these same pages for your
                state&apos;s bills on your own computer, and that is where you can paste in a bill of your own.
              </p>
            </div>
            <Link href="/run/" className="button">Run it for your state<Arrow /></Link>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}

function cardLine(slug: string, states: number, s: ReturnType<typeof story>): string {
  if (slug === "anti-slapp") {
    return `The Uniform Law Commission drafted this anti-SLAPP act as a model for states. Bills that share its language appear in ${states} states in this corpus.`;
  }
  if (slug === "deepfakes") {
    return `Bills requiring a label on AI-altered campaign ads moved fast. The earliest enacted version in this corpus is Washington's 2023 SB 5152.`;
  }
  return `${s.anchor.title}: bills in ${states} other states share language with its social-media provisions.`;
}
