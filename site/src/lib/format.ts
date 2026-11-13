// Display helpers: names, status labels, sessions and numbers, shared by every page.
import type { BillRef } from "@/types/export";

export type Status = NonNullable<BillRef["status"]>;

export const STATUS_LABEL: Record<Status, string> = {
  passed: "Passed",
  died: "Died",
  introduced: "Introduced",
  passed_legislature: "Passed the legislature; outcome not recorded",
};

// The tile grid the mocks use in place of a geographic map: [code, name, column, row].
export const TILES: [string, string, number, number][] = [
  ["AK", "Alaska", 1, 1], ["ME", "Maine", 11, 1], ["VT", "Vermont", 10, 2],
  ["NH", "New Hampshire", 11, 2], ["WA", "Washington", 1, 3], ["ID", "Idaho", 2, 3],
  ["MT", "Montana", 3, 3], ["ND", "North Dakota", 4, 3], ["MN", "Minnesota", 5, 3],
  ["IL", "Illinois", 6, 3], ["WI", "Wisconsin", 7, 3], ["MI", "Michigan", 8, 3],
  ["NY", "New York", 9, 3], ["RI", "Rhode Island", 10, 3], ["MA", "Massachusetts", 11, 3],
  ["OR", "Oregon", 1, 4], ["NV", "Nevada", 2, 4], ["WY", "Wyoming", 3, 4],
  ["SD", "South Dakota", 4, 4], ["IA", "Iowa", 5, 4], ["IN", "Indiana", 6, 4],
  ["OH", "Ohio", 7, 4], ["PA", "Pennsylvania", 8, 4], ["NJ", "New Jersey", 9, 4],
  ["CT", "Connecticut", 10, 4], ["CA", "California", 1, 5], ["UT", "Utah", 2, 5],
  ["CO", "Colorado", 3, 5], ["NE", "Nebraska", 4, 5], ["MO", "Missouri", 5, 5],
  ["KY", "Kentucky", 6, 5], ["WV", "West Virginia", 7, 5], ["VA", "Virginia", 8, 5],
  ["MD", "Maryland", 9, 5], ["DE", "Delaware", 10, 5], ["AZ", "Arizona", 2, 6],
  ["NM", "New Mexico", 3, 6], ["KS", "Kansas", 4, 6], ["AR", "Arkansas", 5, 6],
  ["TN", "Tennessee", 6, 6], ["NC", "North Carolina", 7, 6], ["SC", "South Carolina", 8, 6],
  ["OK", "Oklahoma", 4, 7], ["LA", "Louisiana", 5, 7], ["MS", "Mississippi", 6, 7],
  ["AL", "Alabama", 7, 7], ["GA", "Georgia", 8, 7], ["HI", "Hawaii", 1, 8],
  ["TX", "Texas", 4, 8], ["FL", "Florida", 9, 8],
];

export const STATE_NAME: Record<string, string> = Object.fromEntries(
  TILES.map(([code, name]) => [code, name]),
);

export const stateName = (code: string): string => STATE_NAME[code] ?? code;

// How far a status went, so a state with several bills shows the one that went furthest.
export const STATUS_RANK: Record<Status, number> = {
  passed: 4,
  passed_legislature: 3,
  introduced: 2,
  died: 1,
};

export function session(s: string): string {
  const special = /^(\d{4})_ss$/.exec(s);
  if (special) return `${special[1]} special session`;
  const two = /^(\d{4})[_-](\d{2})(\d{2})?$/.exec(s);
  if (two) return `${two[1]}–${two[3] ?? two[2]}`;
  return s;
}

// The year a bill's session began, read from its first action, for the stories' timelines.
export function year(bill: BillRef): number | null {
  const y = /^(\d{4})/.exec(bill.first_action ?? bill.session);
  return y ? Number(y[1]) : null;
}

export const similarity = (share: number): string => share.toFixed(2);

export const billName = (b: BillRef): string => `${stateName(b.state)} ${b.identifier}`;

// A section by its own heading, which carries its number; text before the first heading is the
// bill's opening (its title and preamble), and a section split into windows has no heading.
export function sectionLabel(heading: string | null | undefined, ordinal: number): string {
  const h = heading?.trim().replace(/\.$/, "");
  if (h) return h;
  return ordinal === 0 ? "Opening text" : `Unheaded text, part ${ordinal + 1}`;
}

export const count = (n: number, one: string, many: string): string =>
  `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

export function asOf(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}
