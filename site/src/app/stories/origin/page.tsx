import type { Metadata } from "next";
import Link from "next/link";
import { Footer, Header } from "@/components/Chrome";
import { StatusBadge } from "@/components/Status";
import { billPage, meta } from "@/lib/data";
import { asOf, billName, sectionLabels, sessionPhrase, stateName } from "@/lib/format";
import {
  docDate,
  foundBy,
  LEAKED_LABEL,
  NO_ORIGIN_STORY,
  ORIGIN_STORY_CAVEAT,
  originLine,
  originStoryCopy,
  publishers,
  type OriginStory,
} from "@/lib/origins";
import { originStory } from "./data";

// The origin story (DESIGN.md#export): one passed bill of the hosted state and the published model
// both stages found holding the most of its wording, chosen by lineage.origin_story's rule. The
// organization is named only where the data names it, in the origin line and its publishers; the
// copy says what the two shares count and how the bill was chosen, never who wrote it.

const TITLE = "Model legislation";

function names(s: OriginStory) {
  return { bill: billName(s.bill), state: stateName(s.bill.state) };
}

export function generateMetadata(): Metadata {
  const s = originStory();
  return { title: s ? originStoryCopy(s, names(s)).title : TITLE };
}

function Story({ s }: { s: OriginStory }) {
  const copy = originStoryCopy(s, names(s));
  const o = s.origin;
  const line = originLine(o);
  const page = billPage(s.bill.slug);
  const labels = sectionLabels(page.sections);
  const order = new Map(page.sections.map((x, i) => [x.ordinal, i]));
  const sections = [...new Set(o.sections)].sort((a, b) => (order.get(a) ?? a) - (order.get(b) ?? b));
  return (
    <>
      <h1 className="display" style={{ marginTop: 10, maxWidth: "var(--lede-width)", overflowWrap: "anywhere" }}>{copy.headline}</h1>
      <p className="lede" style={{ overflowWrap: "anywhere" }}>{copy.lede}</p>
      <div style={{ marginTop: 16, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 20px", fontSize: 14, color: "var(--ink-3)" }}>
        <StatusBadge status={s.bill.status} />
        <span>{sessionPhrase(s.bill.state, s.bill.session)}</span>
        <Link href={`/bills/${s.bill.slug}/`}>{s.bill.identifier}&apos;s page</Link>
      </div>

      <section aria-labelledby="model-h" style={{ marginTop: 48, borderTop: "1px solid var(--line)", paddingTop: 24, maxWidth: "var(--lede-width)" }}>
        <h2 id="model-h" className="eyebrow">The published model</h2>
        <p className="body-serif" data-origin-line={o.doc.id} data-origin-bill={s.bill.slug} style={{ marginTop: 12, fontSize: 22, color: "var(--ink)", overflowWrap: "anywhere" }}>
          {line.before} <a href={o.doc.url}>{line.title}</a>{line.after}
        </p>
        <p style={{ margin: "12px 0 0", display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 16px", fontSize: 14, color: "var(--ink-3)" }}>
          <span style={{ overflowWrap: "anywhere" }}>{publishers(o)}</span>
          {o.doc.leaked && (
            <Link href={`/sources/#${o.doc.source}`} className="chip-dashed" style={{ textDecoration: "none" }} aria-label={`${LEAKED_LABEL}: about this source`}>
              {LEAKED_LABEL}
            </Link>
          )}
          <span>{docDate(o)}</span>
          <span>{foundBy(o)}</span>
        </p>
        <p className="small" style={{ margin: "6px 0 0", overflowWrap: "anywhere" }}>
          In {sections.length === 1 ? "section" : "sections"}: {sections.map((x) => labels.get(x) ?? `Section ${x}`).join(", ")}.
        </p>
        <p className="note" style={{ marginTop: 14 }}>
          <Link href={`/sources/#${o.doc.source}`}>About this library</Link>
        </p>
      </section>

      <section aria-labelledby="shares-h" style={{ marginTop: 56 }}>
        <h2 id="shares-h" className="section">What the two shares count</h2>
        <div className="stack" style={{ marginTop: 12, maxWidth: "40em" }}>
          {copy.shares.map((t) => <p key={t} className="body-serif" style={{ overflowWrap: "anywhere" }}>{t}</p>)}
        </div>
      </section>

      <section aria-labelledby="rule-h" style={{ marginTop: 56 }}>
        <h2 id="rule-h" className="section">How this bill was chosen</h2>
        <p className="body-serif" style={{ marginTop: 12, maxWidth: "40em" }}>{copy.rule}</p>
        <p className="note" style={{ marginTop: 14 }}>{ORIGIN_STORY_CAVEAT}</p>
      </section>
    </>
  );
}

export default function OriginStoryPage() {
  const s = originStory();
  const m = meta();
  const latest = m.archives.map((a) => a.generated_at).sort().at(-1);
  const crumb = s ? originStoryCopy(s, names(s)).title : TITLE;
  return (
    <>
      <Header current="stories" />
      <main className="wrap" style={{ paddingTop: 40 }}>
        <p className="crumbs"><Link href="/#stories">Stories</Link><span aria-hidden="true">/</span><span style={{ overflowWrap: "anywhere" }}>{crumb}</span></p>
        <p className="eyebrow" style={{ marginTop: 28 }}>Featured story · Model legislation</p>
        {s ? (
          <Story s={s} />
        ) : (
          <>
            <h1 className="display" style={{ marginTop: 10 }}>{TITLE}</h1>
            <p className="lede">{NO_ORIGIN_STORY(stateName(m.state))}</p>
            <p className="note" style={{ marginTop: 14 }}>
              <Link href="/sources/">The libraries searched</Link>
            </p>
          </>
        )}
        <p className="note" style={{ marginTop: 40 }}>Based on the {latest ? asOf(latest) : ""} snapshot</p>
      </main>
      <Footer />
    </>
  );
}
