// The published model documents a bill shares wording with, and the libraries the matching reads,
// as the bill pages and the sources page state them (DESIGN.md#origin-matches,
// DESIGN.md#model-documents). Pure functions: the bill page's client component imports them.
//
// The site states shared wording and dates, never direction or authorship, so a line names how
// much wording the bill and the document share and nothing about which came first or who wrote
// either.
import type { Origin, SourceEntry } from "@/types/export";

// The sources page's inclusion rule, as ruled.
export const INCLUSION_RULE =
  "Any organization that publishes text it presents as model or sample state legislation, in a public library or a public archive. Leaked archives are marked as leaked.";

export const LEAKED_LABEL = "Leaked";

// A share as a whole percent. Rounding never makes a partial share read as all or none of the
// wording: above 0 and below 1% reads "under 1%", and anything short of the whole reads at most 99%.
export function percent(share: number): string {
  if (share <= 0) return "0%";
  if (share >= 1) return "100%";
  if (share < 0.01) return "under 1%";
  return `${Math.min(99, Math.round(share * 100))}%`;
}

// Most of the bill's words first; a tie goes to the larger share of the document, then the title,
// so the order never depends on the export's.
export function byBillShare(origins: readonly Origin[]): Origin[] {
  return [...origins].sort(
    (x, y) => y.bill_share - x.bill_share || y.model_share - x.model_share || x.doc.title.localeCompare(y.doc.title),
  );
}

// The words before the document's linked title: "Shares 34% of its wording with".
export const originLead = (o: Origin): string => `Shares ${percent(o.bill_share)} of its wording with`;

// The document's side of the same match, as a separate fact about the document.
export const modelShareLine = (o: Origin): string =>
  o.model_share >= 1
    ? "The shared wording is all of the document's text."
    : `The shared wording is ${percent(o.model_share)} of the document's text.`;

const STAGE: Record<Origin["stages"][number], string> = {
  near_duplicate: "shared wording",
  semantic: "similar meaning",
};

// Which stage found the match, as the relatives list names them.
export const foundBy = (o: Origin): string => `Found by ${[...new Set(o.stages)].map((s) => STAGE[s]).join(" and ")}`;

const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// An ISO 8601 date to the precision it was given: an organization may date a document by its year
// or month alone, and a year must not read as January 1. Anything else shows as it is.
export function isoDate(iso: string): string {
  const m = /^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?/.exec(iso);
  const month = m?.[2] ? MONTH[Number(m[2]) - 1] : undefined;
  if (!m) return iso;
  if (!m[2]) return m[1]!;
  if (!month) return iso;
  return m[3] ? `${month} ${Number(m[3])}, ${m[1]}` : `${month} ${m[1]}`;
}

// The document's own date, as its organization gives it, or that it gives none.
// The organizations a document names as its publishers: its library's, then any it was
// published with, so a joint bill credits each though one library holds it.
export function publishers(o: Origin): string {
  const others = o.doc.co_publishers ?? [];
  if (others.length === 0) return o.doc.organization;
  const tail = others.length === 1 ? others[0] : `${others.slice(0, -1).join(", ")} and ${others.at(-1)}`;
  return `${o.doc.organization}, with ${tail}`;
}

export const docDate = (o: Origin): string => (o.doc.published ? `Dated ${isoDate(o.doc.published)}` : "Undated");

export const documents = (n: number): string => `${n.toLocaleString("en-US")} ${n === 1 ? "document" : "documents"}`;

// The sources page's order: by organization, then key, so it never depends on the export's.
export function bySource(sources: readonly SourceEntry[]): SourceEntry[] {
  return [...sources].sort((x, y) => x.organization.localeCompare(y.organization) || x.key.localeCompare(y.key));
}
