import type { Metadata } from "next";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { Footer, Header } from "@/components/Chrome";
import { bySource, documents, INCLUSION_RULE, isoDate, LEAKED_LABEL } from "@/lib/origins";
import type { SourceEntry, Sources } from "@/types/export";

export const metadata: Metadata = { title: "Sources" };

// sources.json lists every library the matching reads (DESIGN.md#model-documents). Data from
// before export schema version 2 has none, and the page says so rather than failing the build.
function sources(): SourceEntry[] | null {
  const file = path.join(process.cwd(), "public", "data", "sources.json");
  if (!existsSync(file)) return null;
  return bySource((JSON.parse(readFileSync(file, "utf8")) as Sources).sources);
}

// A note from the source's entry, ended as a sentence.
const sentence = (s: string): string => (/[.!?]$/.test(s.trim()) ? s.trim() : `${s.trim()}.`);

function Field({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div style={{ flex: "1 1 220px", minWidth: 0 }}>
      <dt className="eyebrow">{name}</dt>
      <dd style={{ margin: "6px 0 0", fontSize: 15, lineHeight: 1.5, color: "var(--ink-2)", overflowWrap: "anywhere" }}>{children}</dd>
    </div>
  );
}

function Source({ s }: { s: SourceEntry }) {
  return (
    <article id={s.key} aria-labelledby={`${s.key}-h`} style={{ borderTop: "1px solid var(--line)", padding: "24px 0", scrollMarginTop: 96 }}>
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px 14px" }}>
        <h3 id={`${s.key}-h`} className="card" style={{ overflowWrap: "anywhere", minWidth: 0 }}>{s.organization}</h3>
        {s.leaked && <span className="chip-dashed">{LEAKED_LABEL}</span>}
      </div>
      <p className="body-serif" style={{ marginTop: 8, overflowWrap: "anywhere" }}>{s.descriptor}</p>
      {s.leaked && (
        <p className="note" style={{ marginTop: 8, overflowWrap: "anywhere" }}>
          <strong style={{ color: "var(--ink)" }}>{LEAKED_LABEL} archive:</strong> {sentence(s.leak_note ?? "its documents became public through a leak")}
        </p>
      )}
      <dl style={{ margin: "20px 0 0", display: "flex", flexWrap: "wrap", gap: "18px 32px" }}>
        <Field name="Library">
          <a href={s.library_url}>{s.library_url.replace(/^https?:\/\//, "")}</a>
        </Field>
        <Field name="Documents">
          {documents(s.documents)}
          {s.collected_on ? `, collected as of ${isoDate(s.collected_on)}` : ", not yet collected"}
        </Field>
        <Field name="Terms">
          {s.terms_url ? <a href={s.terms_url}>{s.terms_note}</a> : s.terms_note}; read {isoDate(s.terms_read_on)}
        </Field>
      </dl>
    </article>
  );
}

export default function SourcesPage() {
  const list = sources();
  return (
    <>
      {/* The header's sections name no sources page yet, so no nav link is marked current here. */}
      <Header current={"sources" as never} />
      <main className="wrap" style={{ paddingTop: 56 }}>
        <p className="eyebrow">Model legislation</p>
        <h1 className="display" style={{ marginTop: 10 }}>Sources</h1>
        <p className="lede">
          Bills are matched against published model legislation from these libraries. A bill page lists
          each document it shares wording with and links to it in its library; the documents&apos; text is
          not shown here.
        </p>

        <section aria-labelledby="rule" style={{ marginTop: 48 }}>
          <h2 id="rule" className="section">Which libraries are included</h2>
          <p className="body-serif" style={{ marginTop: 12, maxWidth: "40em" }}>{INCLUSION_RULE}</p>
          <p className="note" style={{ marginTop: 12 }}>
            Each organization is described by its type and IRS status from public filings.
          </p>
        </section>

        <section aria-labelledby="libraries" style={{ marginTop: 56 }}>
          <h2 id="libraries" className="section">Libraries</h2>
          {list === null || list.length === 0 ? (
            <p className="body-serif" style={{ marginTop: 12 }}>This data release lists no libraries.</p>
          ) : (
            <>
              <p className="note" style={{ marginTop: 8 }}>
                {list.length} {list.length === 1 ? "library" : "libraries"}, {documents(list.reduce((n, s) => n + s.documents, 0))}{" "}
                in all, whether or not any bill shares wording with them.
              </p>
              <div style={{ marginTop: 20 }}>
                {list.map((s) => <Source key={s.key} s={s} />)}
              </div>
            </>
          )}
        </section>
      </main>
      <Footer />
    </>
  );
}
