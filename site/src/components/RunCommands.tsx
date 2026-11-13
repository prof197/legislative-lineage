"use client";
import { useState } from "react";
import { TILES } from "@/lib/format";

const STATES = [...TILES].sort((a, b) => a[1].localeCompare(b[1]));

export function RunCommands() {
  const [code, setCode] = useState("MN");
  const [copied, setCopied] = useState(-1);
  const commands: [string, string][] = [
    ["Install the tool.", "pipx install legislative-lineage"],
    ["Download the dated 50-state index and build pages for your state’s bills.", `lineage build --state ${code} --sessions 2023-2026`],
    ["Open your copy of the site at localhost:8000.", "lineage serve"],
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
              <div style={{ display: "flex", gap: 8, marginTop: 8, alignItems: "stretch" }}>
                <pre className="code" style={{ margin: 0, flex: 1 }}>$ {cmd}</pre>
                <button type="button" className="button" aria-label={`Copy command ${i + 1}`} onClick={() => { void navigator.clipboard?.writeText(cmd).catch(() => undefined); setCopied(i); }}>
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
