import type { Metadata } from "next";
import Link from "next/link";
import { BillView } from "@/components/BillView";
import { Footer, Header } from "@/components/Chrome";
import { StatusBadge } from "@/components/Status";
import { billList, billPage } from "@/lib/data";
import { billName, count, relatedCount, session, stateName } from "@/lib/format";

export const dynamicParams = false;

export function generateStaticParams(): { slug: string }[] {
  return billList().bills.map((b) => ({ slug: b.bill.slug }));
}

type Props = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const page = billPage((await params).slug);
  return { title: `${page.bill.identifier}: ${page.bill.title}` };
}

export default async function BillPageView({ params }: Props) {
  const page = billPage((await params).slug);
  const b = page.bill;
  const state = stateName(b.state);
  const states = new Set(page.relatives.map((r) => r.bill.state)).size;
  const earliest = page.relatives
    .filter((r) => r.bill.status === "passed" && r.bill.status_date)
    .sort((x, y) => (x.bill.status_date ?? "").localeCompare(y.bill.status_date ?? ""))[0];
  const notice = page.text?.notice_only ?? false;
  return (
    <>
      <Header current="bills" />
      <main className="wrap" style={{ paddingTop: 40 }}>
        <p className="crumbs"><Link href="/bills/">{state} bills</Link><span aria-hidden="true">/</span><span>{b.identifier}</span></p>
        <p className="eyebrow" style={{ marginTop: 28 }}>{state} · {session(b.session)} session</p>
        <h1 className="display" style={{ marginTop: 8 }}>{b.identifier}</h1>
        <p className="lede" style={{ marginTop: 10 }}>{b.title}</p>
        <div style={{ marginTop: 16, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 20px", fontSize: 14, color: "var(--ink-3)" }}>
          <StatusBadge status={b.status} />
          {page.text?.version && <span>Text stored: {page.text.version}</span>}
          {page.text?.url && <a href={page.text.url}>Bill on the {state} General Assembly site</a>}
        </div>

        {notice ? (
          <p className="body-serif" style={{ marginTop: 32, maxWidth: "40em" }}>
            Only the notice of intent to introduce this bill is in the source data, so it wasn&apos;t compared.
          </p>
        ) : page.relatives.length === 0 ? (
          <p className="body-serif" style={{ marginTop: 32, maxWidth: "40em" }}>
            No related bills found in other states&apos; 2023–2026 sessions.{" "}
            <Link href="/validation/">What the search can miss</Link>
          </p>
        ) : (
          <>
            <p className="body-serif" style={{ marginTop: 32, maxWidth: "40em", fontSize: 20 }}>
              {relatedCount(page.families)}, share at least one provision with {b.identifier}.{" "}
              {page.families > page.relatives.length
                ? `The ${page.relatives.length} closest, listed below, are in ${count(states, "other state", "other states")}.`
                : `They are in ${count(states, "other state", "other states")}.`}
              {earliest && ` The earliest enacted version in this corpus is ${billName(earliest.bill)}, ${earliest.bill.status_date?.slice(0, 4)}.`}
            </p>
            <BillView page={page} />
          </>
        )}
      </main>
      <Footer />
    </>
  );
}
