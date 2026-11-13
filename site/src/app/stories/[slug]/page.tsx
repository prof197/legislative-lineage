import type { Metadata } from "next";
import Link from "next/link";
import { Footer, Header } from "@/components/Chrome";
import { StoryView } from "@/components/StoryView";
import { meta, story } from "@/lib/data";
import { storyBest, tally } from "@/lib/facts";
import { asOf, billName } from "@/lib/format";
import { STORY_COPY } from "@/lib/stories";

export const dynamicParams = false;

export function generateStaticParams(): { slug: string }[] {
  return meta().stories.map((slug) => ({ slug }));
}

type Props = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  return { title: STORY_COPY[(await params).slug]?.title ?? "Story" };
}

export default async function StoryPage({ params }: Props) {
  const slug = (await params).slug;
  const s = story(slug);
  const copy = STORY_COPY[slug]!;
  const m = meta();
  const t = tally(storyBest(s));
  const latest = m.archives.map((a) => a.generated_at).sort().at(-1);
  const missed = m.validation.development_enactments.filter(
    (e) => e.act === "public-expression-protection" && !e.found,
  );
  return (
    <>
      <Header current="stories" />
      <main className="wrap" style={{ paddingTop: 40 }}>
        <p className="crumbs"><Link href="/#stories">Stories</Link><span aria-hidden="true">/</span><span>{copy.title}</span></p>
        <p className="eyebrow" style={{ marginTop: 28 }}>{copy.pageEyebrow}</p>
        <h1 className="display" style={{ marginTop: 10, maxWidth: "20ch" }}>{copy.headline}</h1>
        <p className="lede">
          {slug === "anti-slapp" ? (
            <>
              The Uniform Law Commission drafted the Uniform Public Expression Protection Act and approved
              it in 2020 as a model for states to adopt. It gives people sued over their speech on public
              matters a fast way to ask a court to dismiss the case. Because the Commission publishes the
              model text, it is a documented origin. In this corpus, bills in {t.states} states share
              language with it, and {t.passed} of them passed.
            </>
          ) : (
            <>
              {billName(s.anchor)}, {s.anchor.title}. In this corpus, bills in {t.states} other states
              share language with it, and {t.passed} of them passed.
            </>
          )}
        </p>
        {s.scope_note && <p className="note" style={{ marginTop: 14 }}>{s.scope_note}</p>}
        <p className="note" style={{ marginTop: 14 }}>
          {copy.origin && <>Origin: <strong style={{ color: "var(--ink)" }}>{copy.origin}</strong> · </>}
          Based on the {latest ? asOf(latest) : ""} snapshot
        </p>
        <StoryView story={s} slug={slug} compareLabel={copy.compareLabel} listNoun={copy.listNoun} />
        {slug === "anti-slapp" && missed.length > 0 && (
          <p className="note" style={{ marginTop: 20 }}>
            The Commission also lists{" "}
            {missed.map((e) => `a ${e.year} ${e.state} enactment (${e.bill_cell})`).join(", ").replace(/, ([^,]*)$/, " and $1")}{" "}
            that this index missed. <Link href="/validation/">Why it was missed</Link>
          </p>
        )}
      </main>
      <Footer />
    </>
  );
}
