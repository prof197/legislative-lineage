import Link from "next/link";
import { byBillShare, docDate, foundBy, LEAKED_LABEL, originLine, publishers } from "@/lib/origins";
import type { Origin } from "@/types/export";

// A bill page's published model documents, the most of the bill's words first
// (DESIGN.md#origin-matches). Each line gives both shares, the bill's wording found in the model and
// the model's covered by the bill, then names the document's publishers and its date, and says when
// its library is a leaked archive. A document's text is never shown; the line links to it in its
// library. The copy check reads each line by its data-origin-line key, the document's id.
export function Origins({
  origins,
  identifier,
  label,
  order,
}: {
  origins: readonly Origin[] | undefined;
  identifier: string;
  label: (ordinal: number) => string;
  order: Map<number, number>; // a section's place in the bill
}) {
  if (!origins || origins.length === 0) return null;
  const rows = byBillShare(origins);
  const docs = rows.length === 1 ? "a published model document" : `${rows.length} published model documents`;
  const inOrder = (ordinals: number[]) => [...new Set(ordinals)].sort((a, b) => (order.get(a) ?? a) - (order.get(b) ?? b));
  return (
    <section aria-labelledby="origins" style={{ marginTop: 56 }}>
      <h2 id="origins" className="section">Model legislation</h2>
      <p className="note" style={{ marginTop: 8 }}>
        {`${identifier} shares wording with ${docs}, listed by how much of ${identifier}'s wording each shares.`}{" "}
        <Link href="/sources/">The libraries searched</Link>
      </p>
      <ul style={{ listStyle: "none", padding: 0, margin: "20px 0 0" }}>
        {rows.map((o) => {
          const line = originLine(o);
          return (
          <li key={o.doc.id} style={{ borderTop: "1px solid var(--line)", padding: "16px 0" }}>
            <p className="body-serif" data-origin-line={o.doc.id} style={{ color: "var(--ink)", overflowWrap: "anywhere" }}>
              {line.before} <a href={o.doc.url}>{line.title}</a>{line.after}
            </p>
            <p style={{ margin: "8px 0 0", display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 16px", fontSize: 14, color: "var(--ink-3)" }}>
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
              In {o.sections.length === 1 ? "section" : "sections"}: {inOrder(o.sections).map(label).join(", ")}.
            </p>
          </li>
          );
        })}
      </ul>
    </section>
  );
}
