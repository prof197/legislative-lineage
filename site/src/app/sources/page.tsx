import type { Metadata } from "next";
import { Footer, Header } from "@/components/Chrome";
import { sources } from "@/lib/data";
import { bySource, documents, INCLUSION_RULE, isoDate, LEAKED_LABEL, sentence, termsRead, withheldNote } from "@/lib/origins";
import type { SourceEntry } from "@/types/export";

export const metadata: Metadata = { title: "Sources" };

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
      {s.members && s.members_source && (
        <p className="note" style={{ marginTop: 8, overflowWrap: "anywhere" }}>
          {sentence(s.members)} <a href={s.members_source}>Source</a>
        </p>
      )}
      {s.leaked && (
        <p className="note" style={{ marginTop: 8, overflowWrap: "anywhere" }}>
          <strong style={{ color: "var(--ink)" }}>{LEAKED_LABEL} archive:</strong> {sentence(s.leak_note ?? "its documents became public through a leak")}
        </p>
      )}
      {/* Data from before the withholding rule has no list. */}
      {(["generic_text", "review"] as const).map((reason) => {
        const held = (s.withheld ?? []).filter((w) => w.reason === reason);
        if (!held.length) return null;
        return (
          <div key={reason} style={{ marginTop: 12 }}>
            <p className="note" style={{ overflowWrap: "anywhere" }}>
              <strong style={{ color: "var(--ink)" }}>Withheld:</strong> {withheldNote(held.length, reason)}
            </p>
            <ul className="note" style={{ margin: "6px 0 0", paddingLeft: 20, overflowWrap: "anywhere" }}>
              {held.map((w) => <li key={w.id}><a href={w.url}>{w.title}</a></li>)}
            </ul>
          </div>
        );
      })}
      <dl style={{ margin: "20px 0 0", display: "flex", flexWrap: "wrap", gap: "18px 32px" }}>
        <Field name="Library">
          <a href={s.library_url}>{s.library_url.replace(/^https?:\/\//, "")}</a>
        </Field>
        <Field name="Documents">
          {documents(s.documents)}
          {s.collected_on ? `, collected as of ${isoDate(s.collected_on)}` : ", not yet collected"}
        </Field>
        <Field name="Terms">
          {/* A note is often several sentences, so the date follows it as its own sentence. */}
          {sentence(s.terms_note)}{" "}
          {s.terms_url ? <a href={s.terms_url}>{termsRead(s.terms_read_on)}</a> : termsRead(s.terms_read_on)}
        </Field>
      </dl>
    </article>
  );
}

export default function SourcesPage() {
  // Data from before export schema version 2 lists no libraries, and the page says so.
  const found = sources();
  const list = found ? bySource(found.sources) : null;
  return (
    <>
      <Header current="sources" />
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
              {/* With every library collected the entries run long, so a short index leads to each. */}
              <ul aria-label="Libraries listed" style={{ listStyle: "none", padding: 0, margin: "16px 0 0", display: "flex", flexDirection: "column", gap: 6, fontSize: 15 }}>
                {list.map((s) => (
                  <li key={s.key} style={{ overflowWrap: "anywhere" }}>
                    <a href={`#${s.key}`}>{s.organization}</a>{" "}
                    <span className="small">· {documents(s.documents)}{s.leaked ? ` · ${LEAKED_LABEL}` : ""}{(s.withheld ?? []).length ? ` · ${s.withheld.length} withheld` : ""}</span>
                  </li>
                ))}
              </ul>
              <div style={{ marginTop: 28 }}>
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
