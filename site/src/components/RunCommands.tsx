"use client";
import { useState } from "react";
import { TILES } from "@/lib/format";

const STATES = [...TILES].sort((a, b) => a[1].localeCompare(b[1]));

export function RunCommands({ repository }: { repository: string }) {
  const [code, setCode] = useState("MN");
  const [copied, setCopied] = useState(-1);
  // Only commands that exist today: the tool builds from source. A packaged install and a
  // download of the prebuilt index join this list when they ship (M8).
  const commands: [string, string[]][] = [
    ["Get the code and its pinned environment. Needs git, uv and Python 3.13.", [`git clone ${repository}.git`, "cd legislative-lineage", "uv sync"]],
    ["Download the pinned Open States archives and load them.", ["uv run lineage fetch-raw", "uv run lineage ingest"]],
    ["Fetch each legislature’s text where the archives lack it, then split every bill into sections.", ["uv run lineage text inventory", "uv run lineage text fetch", "uv run lineage text merge", "uv run lineage split", "uv run lineage eligible", "uv run lineage boilerplate"]],
    ["Build the 50-state index: shared wording first, then similar meaning.", ["uv run lineage index", "uv run lineage model fetch", "uv run lineage embed", "uv run lineage neighbors", "uv run lineage semantic-pairs", "uv run lineage state-boilerplate"]],
    ["Write your state’s pages and build the site. Needs Node 22.", [`uv run lineage export --state ${code}`, "rm -rf site/public/data", `cp -R data/export/${code} site/public/data`, "cd site", "npm ci", "SITE_BASE_PATH= npm run build"]],
    ["Open your copy of the site at localhost:8000.", ["python3 -m http.server 8000 -d out"]],
  ];
  return (
    <>
      <div className="field" style={{ marginTop: 16, maxWidth: 320 }}>
        <label htmlFor="state">Your state</label>
        <select id="state" className="select" value={code} onChange={(e) => { setCode(e.target.value); setCopied(-1); }}>
          {STATES.map(([c, name]) => <option key={c} value={c}>{name}</option>)}
        </select>
      </div>
      <ol style={{ listStyle: "none", margin: "20px 0 0", padding: 0, display: "grid", gap: 16 }}>
        {commands.map(([what, cmd], i) => (
          <li key={i} style={{ display: "grid", gridTemplateColumns: "36px 1fr", gap: 12 }}>
            <span className="claim-n" style={{ fontSize: 20 }}>{i + 1}</span>
            <div>
              <p style={{ margin: 0 }}>{what}</p>
              <div style={{ display: "flex", gap: 8, marginTop: 8, alignItems: "flex-start" }}>
                <pre className="code" style={{ margin: 0, flex: 1, minWidth: 0, overflowX: "auto" }}>{cmd.map((line) => `$ ${line}`).join("\n")}</pre>
                <button type="button" className="button" aria-label={`Copy step ${i + 1}`} onClick={() => { void navigator.clipboard?.writeText(cmd.join("\n") + "\n").catch(() => undefined); setCopied(i); }}>
                  {copied === i ? "Copied" : "Copy"}
                </button>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </>
  );
}
