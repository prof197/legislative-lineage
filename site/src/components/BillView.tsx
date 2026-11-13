"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { STANDING, StatusBadge } from "@/components/Status";
import { TileMap } from "@/components/Tiles";
import { bestByState, fills } from "@/lib/facts";
import { sectionLabel, session, similarity, stateName, STATUS_LABEL } from "@/lib/format";
import type { BillPage, Relative } from "@/types/export";

const SHOWN = 10;

// A bill page's sections, map and relatives, filtered by a selected section.
export function BillView({ page }: { page: BillPage }) {
  const b = page.bill;
  const [sec, setSec] = useState<number | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [all, setAll] = useState(false);
  const heading = new Map(page.sections.map((s) => [s.ordinal, s.heading]));
  const label = (o: number) => sectionLabel(heading.get(o), o);

  const list = useMemo(
    () => page.relatives.filter((r) => sec === null || r.pairs.some((p) => p.ordinal === sec)),
    [page.relatives, sec],
  );
  const best = bestByState(list, (r) => r.share);
  const shown = (sel ? list.filter((r) => r.bill.state === sel) : []);
  const states = best.size;

  const groups: [string, string | null, Relative[]][] = [
    ["Shares wording", null, list.filter((r) => !r.names_compact && r.found_by === "wording")],
    ["Similar in meaning", null, list.filter((r) => !r.names_compact && r.found_by === "meaning")],
    ["Interstate compacts", "States adopt interstate compacts word for word.", list.filter((r) => r.names_compact)],
  ];
  const withRelatives = page.sections.filter((s) => s.with_relatives > 0);
  const others = page.sections.filter((s) => s.with_relatives === 0);

  return (
    <>
      <section aria-labelledby="secs" style={{ marginTop: 48 }}>
        <h2 id="secs" className="section">Which sections have relatives</h2>
        <p className="note" style={{ marginTop: 8 }}>Matching runs section by section. Select a section to filter the map and list.</p>
        <ul style={{ listStyle: "none", margin: "16px 0 0", padding: 0, borderTop: "1.5px solid var(--ink)" }}>
          <li>
            <button type="button" className="claim" aria-pressed={sec === null} onClick={() => setSec(null)} style={{ gridTemplateColumns: "1fr auto" }}>
              <span>All · Any section</span><span style={{ fontWeight: 600 }}>{page.relatives.length}</span>
            </button>
          </li>
          {withRelatives.map((s) => (
            <li key={s.ordinal}>
              <button type="button" className="claim" aria-pressed={sec === s.ordinal} onClick={() => setSec(sec === s.ordinal ? null : s.ordinal)} style={{ gridTemplateColumns: "1fr auto" }}>
                <span>{label(s.ordinal)}</span>
                <span style={{ fontWeight: 600 }}>{s.with_relatives}</span>
              </button>
            </li>
          ))}
        </ul>
        {others.length > 0 && (
          <details style={{ marginTop: 8 }}>
            <summary className="link-button" style={{ display: "inline-flex", alignItems: "center" }}>
              {others.length} {others.length === 1 ? "section has" : "sections have"} no relatives
            </summary>
            <ul style={{ listStyle: "none", margin: "8px 0 0", padding: 0 }}>
              {others.map((s) => (
                <li key={s.ordinal} className="small" style={{ padding: "6px 0", borderTop: "1px solid var(--line-2)" }}>
                  {label(s.ordinal)}{s.boilerplate !== null ? " · boilerplate, not matched" : ""}
                </li>
              ))}
            </ul>
          </details>
        )}
        <p className="note" style={{ marginTop: 8 }}>Counts are bills, not states. One bill can share several sections.</p>
      </section>

      <section aria-labelledby="where" style={{ marginTop: 56, display: "flex", flexWrap: "wrap", gap: "24px 48px" }}>
        <div style={{ flex: "1 1 480px", minWidth: 0 }}>
          <h2 id="where" className="section">Where the related bills are</h2>
          <p className="note" style={{ marginTop: 8 }}>
            {sec === null ? `${states} states with a related bill.` : `${states} states with a bill that shares ${label(sec)}.`}
          </p>
          <div style={{ marginTop: 16 }}>
            <TileMap
              fills={fills(best, b.state)}
              selected={sel}
              onSelect={setSel}
              label="Related bills by state"
              aria={(code, name, fill) => `${name}: ${code === b.state ? "this bill" : fill === "none" ? "no related bill" : STATUS_LABEL[fill as keyof typeof STATUS_LABEL]}`}
            />
          </div>
          <ul className="key" style={{ marginTop: 16 }}>
            <li><span className="swatch home" aria-hidden="true" />{stateName(b.state)}, this bill</li>
            <li><span className="swatch passed" aria-hidden="true" />Passed</li>
            <li><span className="swatch introduced" aria-hidden="true" />Introduced, not yet passed or died</li>
            <li><span className="swatch died" aria-hidden="true" />Died when its session ended</li>
            <li><span className="swatch" aria-hidden="true" />No related bill</li>
          </ul>
          <p className="note">{STANDING} A state with several related bills shows the one that went furthest.</p>
        </div>
        <div style={{ flex: "1 1 300px", minWidth: 0 }} aria-live="polite">
          <p className="eyebrow">Selected state</p>
          {sel === null ? (
            <p className="body-serif" style={{ marginTop: 8 }}>Select a state to see its bills.</p>
          ) : (
            <>
              <h3 className="card" style={{ marginTop: 8 }}>{stateName(sel)}</h3>
              {sel === b.state ? (
                <p className="body-serif">This is {b.identifier}&apos;s state.</p>
              ) : shown.length === 0 ? (
                <p className="body-serif">{sec === null ? "No related bill in this corpus." : `No bill here shares ${label(sec)}.`}</p>
              ) : (
                <ul style={{ listStyle: "none", padding: 0, margin: "12px 0 0" }}>
                  {shown.map((r) => (
                    <li key={r.bill.slug} style={{ borderTop: "1px solid var(--line)", padding: "10px 0" }}>
                      <strong>{r.bill.identifier}</strong>
                      <div className="small">{session(r.bill.session)} · {r.bill.status ? STATUS_LABEL[r.bill.status] : "Status not recorded"} · {[...new Set(r.pairs.map((p) => p.ordinal))].map(label).join(", ")}</div>
                      <Link href={`/compare/?c=${r.comparison}`}>Compare</Link>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      </section>

      <section aria-labelledby="related" style={{ marginTop: 56 }}>
        <h2 id="related" className="section">Related bills</h2>
        <p className="note" style={{ marginTop: 8 }}>
          {list.length} related {list.length === 1 ? "bill" : "bills"}{sec === null ? "" : ` sharing ${label(sec)}`}, by similarity.
        </p>
        {groups.filter(([, , rows]) => rows.length > 0).map(([title, line, rows]) => (
          <div key={title} style={{ marginTop: 28 }}>
            <h3 style={{ margin: 0, fontSize: 18 }}>{title}</h3>
            {line && <p className="note" style={{ margin: "4px 0 0" }}>{line}</p>}
            <div className="table-box" style={{ marginTop: 12 }}>
              <table className="data" style={{ minWidth: 760 }}>
                <thead>
                  <tr>
                    <th scope="col">State</th><th scope="col">Bill</th><th scope="col">Session</th><th scope="col">Status</th>
                    <th scope="col">Shares {b.identifier}&apos;s</th><th scope="col" aria-sort="descending">Similarity ↓</th><th scope="col">Comparison</th>
                  </tr>
                </thead>
                <tbody>
                  {(all ? rows : rows.slice(0, SHOWN)).map((r) => (
                    <tr key={r.bill.slug}>
                      <td>{stateName(r.bill.state)}</td>
                      <td style={{ fontWeight: 600 }}>{r.bill.identifier}</td>
                      <td>{session(r.bill.session)}</td>
                      <td><StatusBadge status={r.bill.status} /></td>
                      <td>{[...new Set(r.pairs.map((p) => p.ordinal))].map(label).join(", ")}</td>
                      <td>{similarity(r.share)}</td>
                      <td><Link href={`/compare/?c=${r.comparison}`}>Compare</Link></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
        {groups.some(([, , rows]) => rows.length > SHOWN) && (
          <p style={{ marginTop: 16 }}>
            <button type="button" className="button" onClick={() => setAll(!all)}>
              {all ? "Show fewer" : `Show all ${list.length} related bills`}
            </button>
          </p>
        )}
        {page.families > page.relatives.length && (
          <p className="note">
            {page.families} related families in all; the {page.relatives.length} closest are listed.
          </p>
        )}
      </section>
    </>
  );
}
