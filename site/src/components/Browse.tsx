"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { StatusBadge } from "@/components/Status";
import { fetchData } from "@/lib/fetch";
import { session, similarity, stateName, STATUS_LABEL, type Status } from "@/lib/format";
import type { BillList, BillListEntry } from "@/types/export";

type SortKey = "rel" | "bill" | "sess";
const LABELS: Record<SortKey, string> = { rel: "related bills", bill: "bill number", sess: "session" };
const SHOWN = 200; // rows drawn at once; the rest follow on request

function billKey(e: BillListEntry): [string, number, string] {
  const m = /^([A-Z]+)\s*(\d+)/.exec(e.bill.identifier);
  return [m?.[1] ?? e.bill.identifier, m ? Number(m[2]) : 0, e.bill.session];
}

function byBill(a: BillListEntry, b: BillListEntry): number {
  const [pa, na, sa] = billKey(a);
  const [pb, nb, sb] = billKey(b);
  return pa.localeCompare(pb) || na - nb || sa.localeCompare(sb);
}

// The browse page's filter, over every hosted bill, run in the browser (DESIGN.md#dr-1).
export function Browse({ sessions, total, state }: { sessions: string[]; total: number; state: string }) {
  const params = useSearchParams();
  const [bills, setBills] = useState<BillListEntry[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [q, setQ] = useState(params.get("q") ?? "");
  const [sess, setSess] = useState("all");
  const [st, setSt] = useState("all");
  const [rel, setRel] = useState(params.get("related") === "1");
  const [sort, setSort] = useState<SortKey>("rel");
  const [dir, setDir] = useState(-1);
  const [shown, setShown] = useState(SHOWN);

  useEffect(() => {
    fetchData<BillList>("bills.json").then((d) => setBills(d.bills), () => setFailed(true));
  }, []);

  const statuses = useMemo(
    () => [...new Set((bills ?? []).map((b) => b.bill.status).filter((s): s is Status => s !== null))].sort(),
    [bills],
  );
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = (bills ?? []).filter((b) => {
      if (needle && !`${b.bill.identifier} ${b.bill.title}`.toLowerCase().includes(needle)) return false;
      if (sess !== "all" && b.bill.session !== sess) return false;
      if (st !== "all" && b.bill.status !== st) return false;
      if (rel && b.families === 0) return false;
      return true;
    });
    list.sort((a, b) => {
      if (sort === "bill") return byBill(a, b) * dir;
      if (sort === "sess") return a.bill.session.localeCompare(b.bill.session) * dir || b.families - a.families;
      return (a.families - b.families) * dir || byBill(a, b);
    });
    return list;
  }, [bills, q, sess, st, rel, sort, dir]);
  const most = Math.max(1, ...(bills ?? []).map((b) => b.families));

  const sorter = (key: SortKey, first: number) => () => {
    if (sort === key) setDir(-dir);
    else {
      setSort(key);
      setDir(first);
    }
  };
  const aria = (key: SortKey) => (sort === key ? (dir > 0 ? "ascending" : "descending") : "none");
  const arrow = (key: SortKey) => (sort === key ? (dir > 0 ? "↑" : "↓") : "");

  return (
    <>
      <section aria-label="Filters" className="filters">
        <div className="field" style={{ flex: "3 1 300px", minWidth: 0 }}>
          <label htmlFor="q">Search bill number or title</label>
          <input id="q" type="search" className="input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. SB 120, elections, insurance" />
        </div>
        <div className="field" style={{ flex: "1 1 150px" }}>
          <label htmlFor="sess">Session</label>
          <select id="sess" className="select" value={sess} onChange={(e) => setSess(e.target.value)}>
            <option value="all">All sessions</option>
            {sessions.map((s) => <option key={s} value={s}>{session(s)}</option>)}
          </select>
        </div>
        <div className="field" style={{ flex: "1 1 150px" }}>
          <label htmlFor="st">Status</label>
          <select id="st" className="select" value={st} onChange={(e) => setSt(e.target.value)}>
            <option value="all">Any status</option>
            {statuses.map((s) => <option key={s} value={s}>{STATUS_LABEL[s]}</option>)}
          </select>
        </div>
        <label className="check" style={{ flex: "1 1 220px" }}>
          <input type="checkbox" checked={rel} onChange={(e) => setRel(e.target.checked)} />
          Only bills with related bills
        </label>
      </section>

      <div style={{ margin: "16px 0 8px", display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "8px 24px" }}>
        <p aria-live="polite" style={{ margin: 0, fontSize: 15 }}>
          {bills === null ? (
            failed ? "The bill list could not be loaded." : `Loading ${total.toLocaleString("en-US")} bills…`
          ) : (
            <>
              <strong>{rows.length.toLocaleString("en-US")}</strong> {rows.length === 1 ? "bill" : "bills"} · sorted by{" "}
              {LABELS[sort]}
              {dir > 0 ? ", ascending" : ", descending"}
            </>
          )}
        </p>
        <button type="button" className="link-button" onClick={() => { setQ(""); setSess("all"); setSt("all"); setRel(false); }}>
          Clear filters
        </button>
      </div>

      <div className="table-box">
        <table className="data" style={{ minWidth: 900 }}>
          <thead>
            <tr>
              <th scope="col" aria-sort={aria("bill")} style={{ width: 92, padding: 0 }}>
                <button type="button" className="sort" onClick={sorter("bill", 1)}>Bill <span aria-hidden="true" className="small">{arrow("bill")}</span></button>
              </th>
              <th scope="col">Title</th>
              <th scope="col" aria-sort={aria("sess")} style={{ width: 110, padding: 0 }}>
                <button type="button" className="sort" onClick={sorter("sess", 1)}>Session <span aria-hidden="true" className="small">{arrow("sess")}</span></button>
              </th>
              <th scope="col" style={{ width: 120 }}>Status</th>
              <th scope="col" aria-sort={aria("rel")} style={{ width: 176, padding: 0 }}>
                <button type="button" className="sort" onClick={sorter("rel", -1)}>Related bills in other states <span aria-hidden="true" className="small">{arrow("rel")}</span></button>
              </th>
              <th scope="col" style={{ width: 190 }}>Closest match</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, shown).map((b) => (
              <tr key={b.bill.slug}>
                <td><Link href={`/bills/${b.bill.slug}/`} style={{ fontWeight: 600, whiteSpace: "nowrap" }}>{b.bill.identifier}</Link></td>
                <td style={{ fontFamily: "var(--serif)", fontSize: 17, lineHeight: 1.4 }}>{b.bill.title}</td>
                <td>{session(b.bill.session)}</td>
                <td style={{ paddingTop: 12 }}><StatusBadge status={b.bill.status} /></td>
                <td>
                  <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ width: "2.4em", textAlign: "right", fontWeight: 600 }}>{b.families}</span>
                    <span className="bar" aria-hidden="true"><span style={{ width: `${Math.round((b.families / most) * 96)}px` }} /></span>
                  </span>
                </td>
                <td>
                  {b.closest ? (
                    <span style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span>{stateName(b.closest.state)} {b.closest.identifier}</span>
                      <span className="small">Similarity {similarity(b.closest.share)}</span>
                    </span>
                  ) : (
                    <span className="small">None found</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {bills !== null && rows.length === 0 && (
        <p style={{ margin: 0, padding: "40px 0", textAlign: "center", fontFamily: "var(--serif)", fontSize: 19, color: "var(--ink-3)" }}>
          No {state} bills match these filters.
        </p>
      )}
      {rows.length > shown && (
        <p style={{ textAlign: "center", marginTop: 16 }}>
          <button type="button" className="button" onClick={() => setShown(shown + SHOWN * 5)}>
            Show more ({(rows.length - shown).toLocaleString("en-US")} not shown)
          </button>
        </p>
      )}
    </>
  );
}
