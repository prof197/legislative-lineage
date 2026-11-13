"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { STANDING, StatusBadge } from "@/components/Status";
import { TileMap } from "@/components/Tiles";
import { fills, storyBest, tally } from "@/lib/facts";
import { session, similarity, stateName, STATUS_LABEL, year } from "@/lib/format";
import type { Story, StoryRelative } from "@/types/export";

type SortKey = "state" | "yr" | "sim";

// A story's map, timeline and list of relatives.
export function StoryView({ story, slug, compareLabel, listNoun, listHeading }: { story: Story; slug: string; compareLabel: string; listNoun: string; listHeading: string }) {
  const best = storyBest(story);
  const t = tally(best);
  const [picked, setSel] = useState<string | null>(null);
  const [hovered, setHover] = useState<string | null>(null);
  const sel = hovered ?? picked;
  const [sort, setSort] = useState<SortKey>("yr");
  const [dir, setDir] = useState(1);
  const d = sel ? best.get(sel) : undefined;

  const years = useMemo(() => {
    const by = new Map<number, StoryRelative[]>();
    for (const r of best.values()) {
      const y = year(r.bill);
      if (y !== null) by.set(y, [...(by.get(y) ?? []), r]);
    }
    return [...by.entries()].sort(([a], [b]) => a - b);
  }, [best]);

  const rows = useMemo(() => {
    const list = [...story.relatives];
    list.sort((a, b) => {
      if (sort === "state") return stateName(a.bill.state).localeCompare(stateName(b.bill.state)) * dir;
      if (sort === "sim") return ((a.share ?? -1) - (b.share ?? -1)) * dir;
      return ((year(a.bill) ?? 0) - (year(b.bill) ?? 0)) * dir || stateName(a.bill.state).localeCompare(stateName(b.bill.state));
    });
    return list;
  }, [story.relatives, sort, dir]);
  const sorter = (key: SortKey) => () => (sort === key ? setDir(-dir) : (setSort(key), setDir(key === "sim" ? -1 : 1)));
  const arrow = (key: SortKey) => (sort === key ? (dir > 0 ? "↑" : "↓") : "");
  const aria = (key: SortKey) => (sort === key ? (dir > 0 ? "ascending" : "descending") : "none");
  const none = 50 - t.states;

  return (
    <>
      <section aria-labelledby="map-h" style={{ marginTop: 48, display: "flex", flexWrap: "wrap", gap: "24px 48px" }}>
        <div style={{ flex: "1 1 480px", minWidth: 0 }}>
          <h2 id="map-h" className="section">Each state&apos;s matching bill, by status</h2>
          <p className="note" style={{ marginTop: 8 }}>Hover, tap or tab to a state to see its bill.</p>
          <div style={{ marginTop: 16 }}>
            <TileMap
              fills={fills(best)}
              selected={picked}
              onSelect={setSel}
              onHover={setHover}
              label="Matching bills by state"
              aria={(code, name, fill) => {
                const r = best.get(code);
                return r ? `${name}: ${STATUS_LABEL[fill as keyof typeof STATUS_LABEL]}, ${r.bill.identifier}, ${session(r.bill.session)} session` : `${name}: no matching bill`;
              }}
            />
          </div>
        </div>
        <div style={{ flex: "1 1 300px", minWidth: 0 }} aria-live="polite">
          <p className="eyebrow">Selected state</p>
          {sel === null ? (
            <p className="body-serif" style={{ marginTop: 8 }}>Select a state to see its bill.</p>
          ) : (
            <>
              <h3 className="card" style={{ marginTop: 8 }}>{stateName(sel)}</h3>
              {d ? (
                <div className="stack" style={{ marginTop: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                    <span><strong>{d.bill.identifier}</strong> <span className="small">{session(d.bill.session)}</span></span>
                    <StatusBadge status={d.bill.status} />
                  </div>
                  {d.share !== null && <p style={{ margin: 0 }}>Similarity <strong>{similarity(d.share)}</strong></p>}
                  {d.comparison ? <p style={{ margin: 0 }}><Link href={`/compare/?c=${d.comparison}`}>{compareLabel}</Link></p> : <p className="note" style={{ margin: 0 }}>No section of this bill is paired with {listNoun} in the story.</p>}
                </div>
              ) : (
                <p className="body-serif">{slug === "anti-slapp" ? "The Commission lists no 2023–2026 enactment in this state." : `No 2023–2026 bill in this corpus shares language with ${listNoun}.`}</p>
              )}
            </>
          )}
          <ul className="key" style={{ marginTop: 24, flexDirection: "column", alignItems: "flex-start" }}>
            <li><span className="swatch passed" aria-hidden="true" />Passed <strong>{t.passed}</strong></li>
            {t.passed_legislature > 0 && <li><span className="swatch passed_legislature" aria-hidden="true" />{STATUS_LABEL.passed_legislature} <strong>{t.passed_legislature}</strong></li>}
            <li><span className="swatch introduced" aria-hidden="true" />Introduced, not yet passed or died <strong>{t.introduced}</strong></li>
            <li><span className="swatch died" aria-hidden="true" />Died when its session ended <strong>{t.died}</strong></li>
            <li><span className="swatch" aria-hidden="true" />No matching bill <strong>{none}</strong></li>
          </ul>
          <p className="note">{STANDING} Where a state has more than one matching bill, its tile shows the one that went furthest.</p>
        </div>
      </section>

      <section aria-labelledby="time-h" style={{ marginTop: 56 }}>
        <h2 id="time-h" className="section">When the bills appeared, by session year</h2>
        <p className="note" style={{ marginTop: 8 }}>Each bill sits in the year its session began.</p>
        {/* One column per year, all the same width, so the years line up however many bills each has. */}
        <ol style={{ listStyle: "none", margin: "16px 0 0", padding: 0, display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: "20px 24px", alignItems: "start" }}>
          {slug === "anti-slapp" && (
            <li>
              <strong style={{ fontFamily: "var(--serif)", fontSize: 24 }}>2020</strong>
              <p className="note" style={{ margin: "4px 0 0" }}>The Commission approves the model act. This corpus starts in 2023, so earlier state bills are not shown.</p>
            </li>
          )}
          {years.map(([y, list]) => (
            <li key={y}>
              <strong style={{ fontFamily: "var(--serif)", fontSize: 24 }}>{y}</strong> <span className="small">{list.length}</span>
              <ul style={{ listStyle: "none", margin: "6px 0 0", padding: 0, display: "flex", flexWrap: "wrap", gap: 4 }}>
                {list.map((r) => (
                  <li key={r.bill.slug} className={`status ${r.bill.status ?? ""}`} title={`${stateName(r.bill.state)} ${r.bill.identifier}`}>
                    {r.bill.state} {r.bill.identifier}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="list-h" style={{ marginTop: 56 }}>
        <h2 id="list-h" className="section">{rows.length} {listHeading}</h2>
        <p className="note" style={{ marginTop: 8 }}>Select a column heading to sort.</p>
        <div className="table-box">
          <table className="data" style={{ minWidth: 720 }}>
            <thead>
              <tr>
                <th scope="col" aria-sort={aria("state")} style={{ padding: 0 }}><button type="button" className="sort" onClick={sorter("state")}>State <span aria-hidden="true">{arrow("state")}</span></button></th>
                <th scope="col">Bill</th>
                <th scope="col" aria-sort={aria("yr")} style={{ padding: 0 }}><button type="button" className="sort" onClick={sorter("yr")}>Session <span aria-hidden="true">{arrow("yr")}</span></button></th>
                <th scope="col">Status</th>
                <th scope="col" aria-sort={aria("sim")} style={{ padding: 0 }}><button type="button" className="sort" onClick={sorter("sim")}>Similarity <span aria-hidden="true">{arrow("sim")}</span></button></th>
                <th scope="col">Comparison</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.bill.slug}>
                  <td>{stateName(r.bill.state)}</td>
                  <td style={{ fontWeight: 600 }}>{r.bill.identifier}</td>
                  <td>{session(r.bill.session)}</td>
                  <td><StatusBadge status={r.bill.status} /></td>
                  <td>{r.share === null ? "–" : similarity(r.share)}</td>
                  <td>{r.comparison ? <Link href={`/compare/?c=${r.comparison}`}>Compare</Link> : <span className="small">No paired section</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          {slug === "anti-slapp" && <>Each enactment is compared with {listNoun}, the earliest in the index. </>}
          <strong style={{ color: "var(--ink)" }}>Similarity</strong> runs from 0 to 1: the share of {listNoun}&apos;s text that has a matching section in the bill, by exact wording or similar meaning. {STANDING}
        </p>
      </section>
    </>
  );
}
