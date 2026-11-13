// The judged claims call the two texts "the origin" (side a) and "the descendant" (side b), the
// prompt's role labels. The display swaps in the bill names they stand for and changes nothing
// else, so what is shown is the judged text with the labels resolved. See DESIGN.md#explanations.
const ROLE = /\bthe (origin|descendant)\b/gi;

export function nameRoles(summary: string, origin: string, descendant: string): string {
  return summary.replace(ROLE, (_, role: string) => (role.toLowerCase() === "origin" ? origin : descendant));
}
