import Link from "next/link";
import { meta, release } from "@/lib/data";
import { asOf, stateName } from "@/lib/format";

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" stroke="currentColor" strokeWidth="1.75" aria-hidden="true">
      <circle cx="6" cy="14" r="3" />
      <circle cx="22" cy="6" r="3" />
      <circle cx="22" cy="22" r="3" />
      <path d="M9 14h4C16 14 16 6 19 6M13 14C16 14 16 22 19 22" />
    </svg>
  );
}

// The data stamp every page carries: when the hosted records were generated, what is indexed,
// and the release the data was verified from (DESIGN.md#data-snapshot).
export function Stamp() {
  const m = meta();
  const latest = m.archives.map((a) => a.generated_at).sort().at(-1);
  const r = release();
  return (
    <div className="stamp">
      <div className="wrap">
        <p>
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <rect x="2" y="3" width="12" height="11" rx="1.5" />
            <path d="M2 6.5h12M5.5 1.5v3M10.5 1.5v3" />
          </svg>
          <span>
            Snapshot {latest ? asOf(latest) : "unknown"} · {m.states_indexed} states indexed ·
            2023–2026 sessions · {stateName(m.state)} hosted
          </span>
        </p>
        {r && (
          <p>
            Data release <code>{r.tag}</code> · SHA-256 <code title={r.sha256}>{r.sha256.slice(0, 12)}</code>
          </p>
        )}
      </div>
    </div>
  );
}

type Section = "stories" | "bills" | "validation" | "run" | "home";

export function Header({ current }: { current: Section }) {
  const m = meta();
  const here = (s: Section) => (s === current ? ({ "aria-current": "page" } as const) : {});
  return (
    <>
      <Stamp />
      <header className="site-header">
        <div className="wrap">
          <Link href="/" className="brand" {...here("home")}>
            <Logo />
            <span className="brand-name">Legislative Lineage</span>
            <span className="badge-dashed">Working title</span>
          </Link>
          <nav aria-label="Main" className="nav">
            <Link href="/#stories" {...here("stories")}>Stories</Link>
            <Link href="/bills/" {...here("bills")}>{stateName(m.state)} bills</Link>
            <Link href="/validation/" {...here("validation")}>How we checked</Link>
            <Link href="/run/" {...here("run")}>Run it for your state</Link>
          </nav>
        </div>
      </header>
    </>
  );
}

export const REPOSITORY = "https://github.com/prof197/legislative-lineage";

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="wrap">
        <p>
          <strong style={{ color: "var(--ink)" }}>Legislative Lineage</strong> is open source. It
          shows where state bills share language and when each appeared, not who wrote them or why.
        </p>
        <nav aria-label="Footer">
          <a href={REPOSITORY}>Source code</a>
          <Link href="/validation/">How we checked</Link>
          <Link href="/run/">Run it for your state</Link>
          <a href={`${REPOSITORY}/issues`}>Report an error</a>
        </nav>
      </div>
    </footer>
  );
}
