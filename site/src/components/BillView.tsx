"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { STANDING, StatusBadge } from "@/components/Status";
import { InfoTip } from "@/components/InfoTip";
import { TileMap } from "@/components/Tiles";
import { bestByState, fills } from "@/lib/facts";
import { relatedCount, sectionLabels, session, similarity, stateName, STATUS_LABEL } from "@/lib/format";
import type { BillPage, Relative } from "@/types/export";

const SHOWN = 10;

function mostRelated(relatives: Relative[]): string | null {
  const by = new Map<string, { n: number; best: number }>();
  for (const r of relatives) {
    const x = by.get(r.bill.state) ?? { n: 0, best: 0 };
    by.set(r.bill.state, { n: x.n + 1, best: Math.max(x.best, r.share) });
  }
  let top: [string, { n: number; best: number }] | null = null;
  for (const e of by) if (!top || e[1].n > top[1].n || (e[1].n === top[1].n && e[1].best > top[1].best)) top = e;
  return top?.[0] ?? null;
}

const count = (n: number) => `${n.toLocaleString("en-US")} ${n === 1 ? "bill" : "bills"}`;

// The sections a relative shares: "All N sections" or "K of N sections", the list behind it.
function Shares({ ordinals, total, label, order }: { ordinals: number[]; total: number; label: (o: number) => string; order: Map<number, number> }) {
  const shared = [...new Set(ordinals)].sort((a, b) => (order.get(a) ?? a) - (order.get(b) ?? b));
  const head = shared.length === total ? `All ${total} sections` : `${shared.length} of ${total} ${total === 1 ? "section" : "sections"}`;
  return (
    <details className="shares">
      <summary>{head}</summary>
      <ul>{shared.map((o) => <li key={o}>{label(o)}</li>)}</ul>
    </details>
  );
}

// A bill page's sections, map and relatives, filtered by a selected section.
export function BillView({ page }: { page: BillPage }) {
  const b = page.bill;
  const [sec, setSec] = useState<number | null>(null);
  // The map opens on the state with the most related bills, the closer match breaking a tie, so
  // the panel beside it shows bills rather than a prompt.
  const [picked, setSel] = useState<string | null>(() => mostRelated(page.relatives));
  const [hovered, setHover] = useState<string | null>(null);
  const sel = hovered ?? picked;
  const [all, setAll] = useState(false);
  const labels = useMemo(() => sectionLabels(page.sections), [page.sections]);
  const label = (o: number) => labels.get(o) ?? `Section ${o}`;
  const total = page.sections.length;
  const order = useMemo(() => new Map(page.sections.map((x, i) => [x.ordinal, i])), [page.sections]);

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
  const unmatched = page.sections.filter((s) => s.with_relatives === 0 && s.boilerplate === null);
  const boiler = page.sections.filter((s) => s.with_relatives === 0 && s.boilerplate !== null);

  return (
    <>
      <section aria-label="Where the related bills are" className="where" style={{ marginTop: 48, display: "flex", flexWrap: "wrap", gap: "24px 48px" }}>
        <div style={{ flex: "1 1 480px", minWidth: 0 }}>
          {/* The icon is bound to the title's last word, so it never wraps onto a line alone. */}
          <h2 className="section">
            Where the related bills{" "}
            <span style={{ whiteSpace: "nowrap" }}>
              are
              <InfoTip label="How related bills are counted">
                Matching runs section by section. Counts are bills, not states; one bill can share several sections.
              </InfoTip>
            </span>
          </h2>
          <p className="note" style={{ marginTop: 8 }}>
            {sec === null ? `${states} states with a related bill.` : `${states} states with a bill that shares ${label(sec)}.`}
          </p>
          <div style={{ marginTop: 16 }}>
            <TileMap
              fills={fills(best, b.state)}
              selected={picked}
              onSelect={setSel}
              onHover={setHover}
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
        <div style={{ flex: "1 1 300px", minWidth: 0 }}>
          <div className="field" style={{ maxWidth: 420, marginBottom: 28 }}>
            <label htmlFor="section" className="filter-label">Related bills that share</label>
            <select id="section" className="select" value={sec ?? ""} onChange={(e) => { setSec(e.target.value === "" ? null : Number(e.target.value)); }}>
              <option value="">Any section · {count(page.relatives.length)}</option>
              {withRelatives.map((s) => (
                <option key={s.ordinal} value={s.ordinal}>{label(s.ordinal)} · {count(s.with_relatives)}</option>
              ))}
            </select>
              </div>
          <div aria-live="polite">
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
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                        <span><strong>{r.bill.identifier}</strong> <span className="small">{session(r.bill.session)}</span></span>
                        <StatusBadge status={r.bill.status} />
                      </div>
                      <div className="small" style={{ marginTop: 4 }}>{[...new Set(r.pairs.map((p) => p.ordinal))].sort((x, y) => (order.get(x) ?? x) - (order.get(y) ?? y)).map(label).join(", ")}</div>
                      <Link href={`/compare/?c=${r.comparison}`}>Compare</Link>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
          </div>
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
                      <td><Shares ordinals={r.pairs.map((p) => p.ordinal)} total={total} label={label} order={order} /></td>
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
        <p className="note">
          {relatedCount(page.families)}
          {page.families > page.relatives.length ? `; the ${page.relatives.length} closest are listed.` : "."}
        </p>
        {(unmatched.length > 0 || boiler.length > 0) && (
          <p className="note" style={{ marginTop: 20 }}>
            {unmatched.length > 0 && (
              <>{page.families > page.relatives.length ? "None of the listed bills share: " : "No related bills found for: "}{unmatched.map((x) => label(x.ordinal)).join(", ")}. </>
            )}
            {boiler.length > 0 && <>Not matched, as boilerplate: {boiler.map((x) => `${label(x.ordinal)} (${x.boilerplate})`).join(", ")}.</>}
          </p>
        )}
      </section>
    </>
  );
}
