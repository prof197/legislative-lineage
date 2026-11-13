"use client";
import { useState } from "react";
import { similarity, stateName } from "@/lib/format";
import type { Enactment } from "@/types/export";

// Why an enactment was missed, by the furthest step its bill reached (DESIGN.md#commission-recall).
const MISS: Record<string, string> = {
  not_in_corpus: "The enacted bill is not in this corpus.",
  no_text_in_source: "The bill has no text in the source data, so there was nothing to match.",
  amendment_only: "The stored text is an amendment, not the bill.",
  excluded_type: "The bill is filed under a document type that is not matched.",
  no_usable_text: "The bill's text could not be split into sections to match.",
  not_proposed: "Neither stage proposed the bill as a match for the act's text.",
  bucket_cap: "Its shared text sits in clauses so common that they are skipped.",
  verification: "Proposed, but its sections fell below the similarity needed to count as a match.",
  below_threshold: "Matched, but below the score needed to count as found.",
};

export function Enactments({ acts, enactments }: { acts: { key: string; name: string }[]; enactments: Enactment[] }) {
  const [open, setOpen] = useState<Record<string, boolean>>({ [acts[0]?.key ?? ""]: true });
  return (
    <div style={{ marginTop: 16, display: "grid", gap: 28 }}>
      {acts.map(({ key, name }) => {
        const rows = enactments.filter((e) => e.act === key);
        const found = rows.filter((e) => e.found).length;
        const isOpen = open[key] ?? false;
        const shown = rows.filter((e) => isOpen || !e.found);
        return (
          <div key={key}>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
              <h3 style={{ margin: 0, fontSize: 18 }}>{name} · {found} of {rows.length}</h3>
              <button type="button" className="link-button" aria-expanded={isOpen} onClick={() => setOpen({ ...open, [key]: !isOpen })}>
                {isOpen ? "Show misses only" : `Show all ${rows.length}`}
              </button>
            </div>
            <div className="table-box" style={{ marginTop: 8 }}>
              <table className="data" style={{ minWidth: 640 }}>
                <thead><tr><th scope="col">State</th><th scope="col">Bill</th><th scope="col">Year</th><th scope="col">Result</th><th scope="col">Similarity, or why it was missed</th></tr></thead>
                <tbody>
                  {shown.map((e) => (
                    <tr key={`${e.state}-${e.bill_cell}`}>
                      <td>{stateName(e.state)}</td>
                      <td>{e.bill_cell}</td>
                      <td>{e.year}</td>
                      <td>{e.found ? "Found" : "Missed"}</td>
                      <td>{e.found ? (e.share === null ? "–" : `Similarity ${similarity(e.share)}`) : (MISS[e.miss_reason ?? ""] ?? e.miss_reason)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}
