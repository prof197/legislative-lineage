// The published model documents a bill shares wording with, and the libraries the matching reads,
// as the bill pages and the sources page state them (DESIGN.md#origin-matches,
// DESIGN.md#model-documents). Pure functions: the bill page's client component imports them.
//
// The site states shared wording and dates, never direction or authorship, so a line names how
// much wording the bill and the document share and nothing about which came first or who wrote
// either.
import type { Origin, OriginStory, SourceEntry } from "@/types/export";

// The sources page's inclusion rule, as ruled.
export const INCLUSION_RULE =
  "Any organization that publishes text it presents as model or sample state legislation, in a public library or a public archive. Leaked archives are marked as leaked.";

export const LEAKED_LABEL = "Leaked";

// The sources page's line on a library's withheld documents (lineage.withheld, ruled at the
// deploy review): what is held back, from where, and why, without naming it a fault of the library.
export function withheldNote(n: number): string {
  const which = n === 1 ? "This document is" : `These ${n} documents are`;
  return `${which} withheld from bill pages, the feed and stories until matching improves: most of the matches checked for ${n === 1 ? "it" : "them"} rest on generic text, such as definitions or standard clauses many acts share, rather than on a shared provision.`;
}

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

// An organization's name as the data gives it, made possessive: a name ending in "s" takes the
// apostrophe alone ("Institute for Family Studies'").
export const possessive = (name: string): string => (/s$/i.test(name) ? `${name}'` : `${name}'s`);

const capitalized = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);

// A title as the origin line shows it: when it opens with its own organization's name or
// abbreviation ("NCOIL Insurance Fraud Model Act"), that is dropped, since the line names the
// organization after it (ruled at the deploy review). Only an opening name goes, with what joins
// it to the rest ("'s", a colon or a dash); a title that is nothing else stays whole.
export function shownTitle(title: string, organization: string): string {
  const name = organization.replace(/\s*\(.*\)\s*$/, "").trim();
  const abbreviation = /\(([A-Z][A-Z&.-]{1,})\)/.exec(organization)?.[1];
  const names = [organization, name, abbreviation]
    .filter((n): n is string => !!n)
    .flatMap((n) => [n, n.replace(/^the\s+/i, "")])
    .filter((n) => n.length > 1)
    .sort((x, y) => y.length - x.length);
  const lower = title.toLowerCase();
  for (const n of names) {
    if (!lower.startsWith(n.toLowerCase())) continue;
    const rest = title.slice(n.length);
    if (rest && !/^(?:['’]s\b|[\s:–—-])/.test(rest)) continue;
    const kept = rest.replace(/^(?:['’]s)?[\s:–—-]*/, "").trim();
    return kept ? kept : title;
  }
  return title;
}

// "a" or "an" by sound, for the names a library carries: "an ALEC", "an American", "a Uniform".
// A vowel, but not the "you" of "Uni-", or a letter sounded with one, read out as an acronym.
const article = (name: string): string => (/^(?:[AEIOaeio]|[Uu](?!ni)|[AEFHILMNORSX](?:[A-Z]|$))/.test(name) ? "an" : "a");

// Who a line says published the model: its library, or for a leaked document the organization that
// drafted it and the archive that published the leak (ruled at the deploy review).
export function publishedBy(doc: Origin["doc"]): string {
  return doc.drafted_by
    ? `${article(doc.drafted_by)} ${doc.drafted_by} model from a leaked document published by ${doc.organization}`
    : `a model published by ${doc.organization}`;
}

// The origin line, ruled at M12's start and reworded at the deploy review: both shares, always,
// so a bill that takes one section of a long model and a bill that copies a short model whole
// never read alike.
//   "34% of this bill's wording appears in [title], a model published by [organization], covering 25% of the model."
// A leaked document's: "... appears in [title], an [organization] model from a leaked document
// published by [archive], covering ...".
// The title is the part a page links to the document in its library. The organization is the
// library's, as the data names it; co-publishers are credited on the line below, not here.
export function originLine(o: Origin): { before: string; title: string; after: string } {
  return {
    before: `${capitalized(percent(o.bill_share))} of this bill's wording appears in`,
    title: shownTitle(o.doc.title, o.doc.drafted_by ?? o.doc.organization),
    after: `, ${publishedBy(o.doc)}, covering ${percent(o.model_share)} of the model.`,
  };
}

export const originText = (o: Origin): string => {
  const l = originLine(o);
  return `${l.before} ${l.title}${l.after}`;
};

// The form every rendered origin line must take, which the copy check holds each page to.
const SHARE = String.raw`(?:\d{1,3}%|[Uu]nder 1%)`;
export const ORIGIN_LINE = new RegExp(
  String.raw`^${SHARE} of this bill's wording appears in .+, (?:a model published by .+|an? .+ model from a leaked document published by .+), covering ${SHARE} of the model\.$`,
);

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

// The day a bill's relatives or origins last changed, which the live track sets on its page
// (BillPage.updated_on); nothing when it is not set, so a full build's pages carry no date.
export const updatedLine = (iso: string | null | undefined): string | null =>
  iso ? `Related bills and model documents updated ${isoDate(iso)}` : null;

// A note from a source's entry, ended as a sentence: notes as read are often several sentences.
export const sentence = (s: string): string => (/[.!?]$/.test(s.trim()) ? s.trim() : `${s.trim()}.`);

// The day a source's terms were read, as its own sentence after the note.
export const termsRead = (iso: string): string => `Terms read ${isoDate(iso)}.`;

// The sources page's order: by organization, then key, so it never depends on the export's.
export function bySource(sources: readonly SourceEntry[]): SourceEntry[] {
  return [...sources].sort((x, y) => x.organization.localeCompare(y.organization) || x.key.localeCompare(y.key));
}

// The origin story (DESIGN.md#export): one passed bill and the listed model document both stages
// found holding the most of its wording, chosen by lineage.origin_story's rule, which names no
// organization and features only a match a labeler judged correct; its file's type is the export
// schema's.
export type { OriginStory };

// The story's own words. The organization appears only in the origin line and the publishers line,
// as the data names it; nothing here says who wrote the bill or which text came first.
export function originStoryCopy(s: OriginStory, names: { bill: string; state: string }) {
  const found = `with a listed model document that both stages found`;
  const checked = "A match is featured only after a model has read the passages it rests on and judged them the same provision.";
  const k = s.passed_over;
  const ranked =
    s.eligible === 1
      ? `It is the only passed ${names.state} bill ${found}. Among several, the one with the largest share of its wording in such a document would be shown; a tie would go to the larger share of the model, then the lower bill id.`
      : k === 0
        ? `Of the ${s.eligible.toLocaleString("en-US")} passed ${names.state} bills ${found}, this bill has the largest share of its wording in such a document. A tie goes to the larger share of the model, then the lower bill id.`
        : `Of the ${s.eligible.toLocaleString("en-US")} passed ${names.state} bills ${found}, ${k === 1 ? "one holds a larger share of its" : `${k.toLocaleString("en-US")} hold a larger share of their`} wording in such a document, but ${k === 1 ? "its match has" : "their matches have"} not been judged the same provision. Of the matches that have, this bill's holds the largest share; a tie goes to the larger share of the model, then the lower bill id.`;
  const rule = `${ranked} ${checked}`;
  return {
    title: `${names.bill} and a published model`,
    headline: `A passed ${names.state} bill that shares wording with a published model`,
    lede: `${names.bill}, ${s.bill.title}, passed.`,
    shares: [
      `${capitalized(percent(s.origin.bill_share))} of ${possessive(names.bill)} wording is in sections that match sections of the model, by shared wording or similar meaning.`,
      `The model's matching sections hold ${percent(s.origin.model_share)} of its wording.`,
      "Sections such as definitions, severability and effective dates are left out of both counts.",
    ],
    rule: `${rule} The rule names no organization; the one shown is whichever publishes the document.`,
  };
}

export const ORIGIN_STORY_CAVEAT = "Matching shows where the texts share wording, not who wrote either one or which came first.";

export const NO_ORIGIN_STORY = (state: string): string =>
  `No passed ${state} bill in this data has a listed model document that both stages found and a model has judged the same provision on the passages it rests on.`;
