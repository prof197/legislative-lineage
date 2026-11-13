"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { STANDING, StatusBadge } from "@/components/Status";
import { fetchData } from "@/lib/fetch";
import { billName, sectionLabel, session, similarity, STATUS_LABEL } from "@/lib/format";
import { previewEnd, rows, shown, type Kind, type Piece, type Row } from "@/lib/rows";
import type { AlignedPair, BillPage, BillRef, BillTexts, Claim, Comparison, Relative } from "@/types/export";

type Texts = Map<number, [string | null, string]>;
type Loaded = { c: Comparison; a: Texts; b: Texts };
type Numbered = { n: number; pair: number; claim: Claim; cited: Set<string> };

function texts(doc: BillTexts): Texts {
  return new Map(doc.sections.map((s) => [s.ordinal, [s.heading, s.text]]));
}

const GLYPH: Record<Kind, string> = { same: "=", chg: "≠", rew: "≈", add: "+", rem: "−" };

function label(kind: Kind, c: Comparison): string {
  switch (kind) {
    case "same": return "Same words";
    case "chg": return "Some words differ";
    case "rew": return "Reworded";
    case "add": return `Only in ${billName(c.b)}`;
    case "rem": return `Only in ${billName(c.a)}`;
  }
}

function Glyph({ kind, title }: { kind: Kind; title?: string }) {
  return <span className={`glyph-box${kind === "same" ? " light" : ""}`} title={title} aria-hidden="true">{GLYPH[kind]}</span>;
}

// One side of a row: its pieces marked, the selected claim's text ringed, and a numbered marker
// after the last words each claim cites here.
function Cell({ text, pieces, until, claims, selected, onPick }: {
  text: string; pieces: Piece[]; until: number | null; claims: Numbered[]; selected: number | null; onPick: (n: number) => void;
}) {
  const ring = new Set(claims.find((x) => x.n === selected)?.cited ?? []);
  const lastFor = new Map<number, number>();
  pieces.forEach((p, i) => claims.forEach((x) => p.span && x.cited.has(p.span) && lastFor.set(x.n, i)));
  const out: React.ReactNode[] = [];
  pieces.forEach((p, i) => {
    const to = until === null ? p.to : Math.min(p.to, until);
    if (to <= p.from) return;
    const t = shown(text, p.from, to);
    const ringed = p.span !== null && ring.has(p.span) ? " ring" : "";
    out.push(
      p.mark === "copied" ? <mark key={i} className={`m-copied${ringed}`}>{t}</mark>
      : p.mark === "paraphrased" ? <mark key={i} className={`m-para${ringed}`}>{t}</mark>
      : p.mark === "differs" ? <b key={i} className={`m-diff${ringed}`}>{t}</b>
      : <span key={i} className={ringed.trim() || undefined}>{t}</span>,
    );
    if (until !== null) return;
    for (const [n, last] of lastFor) {
      if (last !== i) continue;
      out.push(
        <a key={`c${n}`} href={`#claim-${n}`} className={`claim-dot${n === selected ? " on" : ""}`} aria-label={`Claim ${n}`} onClick={() => onPick(n)}>{n}</a>,
      );
    }
  });
  if (until !== null) out.push(<span key="more"> …</span>);
  return <p className="row-text">{out}</p>;
}

function Found({ stages }: { stages: AlignedPair["stages"] }) {
  return stages.includes("near_duplicate") ? (
    <span className="found"><span className="found-glyph solid" aria-hidden="true">=</span>Found by exact text match</span>
  ) : (
    <span className="found"><span className="found-glyph" aria-hidden="true">≈</span>Found by similar meaning</span>
  );
}

function Picker({ name, current, relatives }: { name: string; current: string; relatives: Relative[] }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className="picker" aria-haspopup="true" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span>Compare</span><span className="visually-hidden"> a different bill</span>
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d={open ? "M4 10l4-4 4 4" : "M4 6l4 4 4-4"} />
        </svg>
      </button>
      {open && (
        <div role="dialog" aria-label={`Choose a bill to compare with ${name}`} className="picker-pop" onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
          <p className="picker-head"><strong>{relatives.length} related bills in other states.</strong> Most similar first.</p>
          <ul>
            {relatives.map((r) => {
              const here = r.comparison === current;
              return (
                <li key={r.comparison}>
                  <Link href={`/compare/?c=${r.comparison}`} aria-current={here ? "page" : undefined} className={`picker-item${here ? " here" : ""}`} onClick={() => setOpen(false)}>
                    <span aria-hidden="true" className="picker-check">
                      {here && <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8.5l3.2 3L13 4.5" /></svg>}
                    </span>
                    <span className="picker-name">
                      <span>{billName(r.bill)}</span>
                      <span className="small">{session(r.bill.session)} · {r.bill.status ? STATUS_LABEL[r.bill.status] : "Status not recorded"} · {r.found_by === "wording" ? "Shares wording" : "Similar in meaning"}</span>
                    </span>
                    <span className="picker-sim">{similarity(r.share)}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </>
  );
}

function Column({ bill, version, children }: { bill: BillRef; version: string | null | undefined; children?: React.ReactNode }) {
  return (
    <div className="col-head">
      <p>
        <span className="col-name">{billName(bill)}</span>
        <span className="col-sub">
          {[`${session(bill.session)} session`, bill.status ? STATUS_LABEL[bill.status] : null, version].filter(Boolean).join(" · ")}
        </span>
      </p>
      {children}
    </div>
  );
}

const cites = (r: Row, ids: Set<string>) => [...r.a, ...r.b].some((x) => x.span !== null && ids.has(x.span));
const rowId = (pair: number, k: number) => `p${pair}-r${k}`;

export function CompareView({ hosted, issues, footer }: { hosted: string; issues: string; footer: React.ReactNode }) {
  const name = useSearchParams().get("c");
  const [data, setData] = useState<Loaded | null>(null);
  const [page, setPage] = useState<BillPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [hideSame, setHideSame] = useState(false);
  const [unfolded, setUnfolded] = useState<Set<string>>(new Set());
  const [asideMax, setAsideMax] = useState<number | null>(null);
  const pane = useRef<HTMLElement>(null);

  useEffect(() => {
    setData(null);
    setPage(null);
    setUnfolded(new Set());
    if (!name || !/^[a-z0-9_-]+$/.test(name)) {
      setError("No comparison is named in this address.");
      return;
    }
    (async () => {
      const c = await fetchData<Comparison>(`compare/${name}.json`);
      const [a, b] = await Promise.all([
        fetchData<BillTexts>(`texts/${c.a.slug}.json`),
        fetchData<BillTexts>(`texts/${c.b.slug}.json`),
      ]);
      setData({ c, a: texts(a), b: texts(b) });
      if (c.a.state === hosted) setPage(await fetchData<BillPage>(`bills/${c.a.slug}.json`).catch(() => null));
    })().catch(() => setError("This comparison could not be loaded."));
  }, [name, hosted]);

  // The claims panel scrolls within the pane's height, as the mock's does.
  useEffect(() => {
    const el = pane.current;
    if (!el) return;
    const measure = () => setAsideMax(Math.max(240, el.clientHeight - 24));
    measure();
    const seen = new ResizeObserver(measure);
    seen.observe(el);
    return () => seen.disconnect();
  }, [data]);

  const built = useMemo(() => {
    if (!data) return null;
    const pairs = data.c.pairs.map((p) => {
      const [ha, ta] = data.a.get(p.ordinal_a) ?? [null, ""];
      const [hb, tb] = data.b.get(p.ordinal_b) ?? [null, ""];
      return { p, ha, hb, ta, tb, rows: rows(ta, tb, p.spans) };
    });
    const claims: Numbered[] = [];
    pairs.forEach(({ p }, pair) =>
      p.explanation?.claims.forEach((claim) =>
        claims.push({ n: claims.length + 1, pair, claim, cited: new Set([...claim.origin_span_ids, ...claim.descendant_span_ids]) }),
      ),
    );
    return { pairs, claims };
  }, [data]);

  useEffect(() => setSelected(built?.claims[0]?.n ?? null), [built]);

  if (error) return <main className="cmp-main"><div className="cmp-wrap"><p className="body-serif" style={{ marginTop: 32 }}>{error}</p></div></main>;
  if (!data || !built) return <main className="cmp-main"><div className="cmp-wrap"><p className="note" aria-live="polite" style={{ marginTop: 32 }}>Loading the comparison…</p></div></main>;
  const { c } = data;
  const matchedA = new Set(c.pairs.map((p) => p.ordinal_a)).size;
  const version = c.text_b?.version;
  const introduced = version ? /introduc/i.test(version) : false;
  const relatives = page ? [...page.relatives].sort((x, y) => y.share - x.share) : [];
  const share = relatives.find((r) => r.comparison === name)?.share;
  const explained = c.pairs.some((p) => p.explanation);

  const pick = (n: number) => {
    const next = selected === n ? null : n;
    setSelected(next);
    const x = built.claims.find((y) => y.n === next);
    if (!x) return;
    const k = built.pairs[x.pair]!.rows.findIndex((r) => cites(r, x.cited));
    document.getElementById(k >= 0 ? rowId(x.pair, k) : `pair-${x.pair}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <>
      <div className="cmp-title">
        <div className="cmp-wrap">
          <nav aria-label="Breadcrumb" className="crumbs">
            {c.a.state === hosted ? <Link href={`/bills/${c.a.slug}/`}>{billName(c.a)}</Link> : <span>{billName(c.a)}</span>}
            <span aria-hidden="true">/</span>
            <span aria-current="page">{billName(c.b)}</span>
          </nav>
          <h1 className="cmp-h1">{billName(c.b)} beside {billName(c.a)}</h1>
        </div>
      </div>

      <main id="scroll-area" className="cmp-main" ref={pane}>
        <div className="cmp-wrap">
          <div className="cmp-meta">
            <StatusBadge status={c.b.status} />
            <span>{session(c.b.session)} session</span>
            {share !== undefined && <span>Similarity <strong>{similarity(share)}</strong></span>}
            <span>{matchedA} of {c.sections_a} {c.a.identifier} sections have a match</span>
          </div>
          <p className="cmp-info">
            <svg width="18" height="18" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true"><circle cx="8" cy="8" r="6.5" /><path d="M8 7.2v4M8 4.8v.01" /></svg>
            <span>
              {version && <>Text shown: {c.b.identifier} {version}. </>}
              {introduced && c.b.status === "passed" && "The bill passed, so amendments made before passage may not appear here. "}
              {STANDING}
            </span>
          </p>
          <p className="cmp-explain">
            Each row pairs a clause in {billName(c.a)} with its counterpart in {billName(c.b)}. Numbering, capitalization and punctuation are ignored when comparing.
          </p>

          <div className="cmp-cols">
            <div className="cmp-diff">
              <div className="cmp-colheads">
                <div className="diff-grid">
                  <Column bill={c.a} version={c.text_a?.version} />
                  <span />
                  <Column bill={c.b} version={version}>
                    {relatives.length > 1 && <Picker name={billName(c.a)} current={name ?? ""} relatives={relatives} />}
                  </Column>
                </div>
              </div>

              {built.pairs.map(({ p, ha, hb, ta, tb, rows: rs }, i) => {
                const la = sectionLabel(ha, p.ordinal_a);
                const lb = sectionLabel(hb, p.ordinal_b);
                const claims = built.claims.filter((x) => x.pair === i);
                const anyCited = new Set(claims.flatMap((x) => [...x.cited]));
                const hides = (r: Row) => hideSame && r.kind === "same" && !cites(r, anyCited);
                const hidden = rs.filter(hides).length;
                return (
                  <section key={`${p.ordinal_a}-${p.ordinal_b}`} id={`pair-${i}`} className="block" aria-label={`${la} beside ${lb}`}>
                    <p className="block-title">
                      <span className="block-name">Section pair {i + 1} of {built.pairs.length}</span>
                      <Found stages={p.stages} />
                    </p>
                    <div className="block-body">
                      <div className="diff-grid block-heads"><span>{la}</span><span /><span>{lb}</span></div>
                      {rs.map((r, k) => {
                        if (hides(r)) return null;
                        const key = `${i}.${k}`;
                        const folded = r.fold && !unfolded.has(key) && !cites(r, anyCited);
                        const side = (s: "a" | "b") => {
                          const text = s === "a" ? ta : tb;
                          const pieces = r[s];
                          if (!pieces.length) return <div className="row-gap">Not in {billName(s === "a" ? c.a : c.b)}</div>;
                          return (
                            <>
                              <Cell text={text} pieces={pieces} until={folded ? previewEnd(text, pieces) : null} claims={claims} selected={selected} onPick={pick} />
                              {folded && (
                                <button type="button" className="unfold" onClick={() => setUnfolded(new Set(unfolded).add(key))}>
                                  Show all {r.words.toLocaleString("en-US")} words
                                </button>
                              )}
                            </>
                          );
                        };
                        return (
                          <div key={k} id={rowId(i, k)} className={`diff-grid diff-row ${r.kind}`}>
                            <div className="cell" data-side={c.a.identifier}>{side("a")}</div>
                            <div className="glyph"><Glyph kind={r.kind} title={label(r.kind, c)} /><span className="visually-hidden">{label(r.kind, c)}</span></div>
                            <div className="cell" data-side={c.b.identifier}>{side("b")}</div>
                          </div>
                        );
                      })}
                      {hidden > 0 && (
                        <div className="hidden-bar">
                          <span>{hidden === rs.length ? "Identical in both texts" : `${hidden} identical ${hidden === 1 ? "clause" : "clauses"} hidden`}</span>
                          <button type="button" onClick={() => setHideSame(false)}>Show</button>
                        </div>
                      )}
                    </div>
                  </section>
                );
              })}
              <p className="note" style={{ marginTop: 28 }}>Shared language shows that two texts are alike, not who wrote either one.</p>
            </div>

            {explained && (
              <div className="cmp-aside">
                <aside aria-labelledby="chg-h" style={asideMax ? { maxHeight: asideMax } : undefined}>
                  <h2 id="chg-h">What this version changed</h2>
                  <p style={{ margin: "10px 0 0" }}>
                    <span className="chip-dashed">
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><rect x="4" y="4" width="8" height="8" rx="1" /><path d="M6 1.5v2M10 1.5v2M6 12.5v2M10 12.5v2M1.5 6h2M1.5 10h2M12.5 6h2M12.5 10h2" /></svg>
                      Machine-generated
                    </span>
                  </p>
                  <p className="aside-note">Written by a language model from the aligned text. It can be wrong. Select a claim to jump to the words it rests on.</p>
                  <ol className="aside-claims">
                    {built.claims.map((x) => {
                      const { hb, p } = built.pairs[x.pair]!;
                      return (
                        <li key={x.n} id={`claim-${x.n}`} className={x.n === selected ? "on" : undefined}>
                          <button type="button" aria-pressed={x.n === selected} onClick={() => pick(x.n)}>
                            <span className="claim-num" aria-hidden="true">{x.n}</span>
                            <span>{x.claim.summary}</span>
                          </button>
                          <p>Rests on <a href={`#pair-${x.pair}`}>{c.b.identifier} {sectionLabel(hb, p.ordinal_b)}</a></p>
                        </li>
                      );
                    })}
                  </ol>
                  {built.pairs.filter(({ p }) => p.explanation && (p.explanation.withheld || !p.explanation.claims.length)).map(({ p, hb }) => (
                    <p key={`${p.ordinal_a}-${p.ordinal_b}`} className="aside-note">
                      {p.explanation!.withheld
                        ? `No explanation is shown for ${c.b.identifier} ${sectionLabel(hb, p.ordinal_b)}: the model's account did not hold up against the aligned text after one retry.`
                        : `The model found no difference of substance in ${c.b.identifier} ${sectionLabel(hb, p.ordinal_b)}.`}
                    </p>
                  ))}
                  <p className="aside-note"><a href={issues}>Report a wrong claim</a></p>
                </aside>
              </div>
            )}
          </div>
        </div>
        {footer}
      </main>

      <footer aria-label="How to read the comparison" className="cmp-legend">
        <div className="cmp-wrap">
          <div className="legend-keys">
            <div className="legend-line">
              <span><mark className="m-copied legend-sample">same words</mark>Exact copying</span>
              <span><mark className="m-para legend-sample">said differently</mark>Paraphrase</span>
              <span><b className="legend-sample bold">bold</b>Words that differ</span>
            </div>
            <div className="legend-line glyphs">
              {(["same", "chg", "rew", "add", "rem"] as const).map((k) => (
                <span key={k}><Glyph kind={k} />{k === "same" ? "Same" : label(k, c)}</span>
              ))}
            </div>
          </div>
          <label className="check legend-check">
            <input type="checkbox" checked={hideSame} onChange={(e) => setHideSame(e.target.checked)} />
            Hide identical clauses
          </label>
        </div>
      </footer>
    </>
  );
}
