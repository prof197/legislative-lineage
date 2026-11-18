// The origin story's file, read at build time like the rest of the export (lib/data.ts). The
// export writes it only when a passed bill has a listed model document both stages found, so a
// release without one, or from before the story, reads as none.
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type { OriginStory } from "@/lib/origins";

export const ORIGIN_STORY_FILE = "origin_story.json";

export function originStory(): OriginStory | null {
  const file = path.join(process.cwd(), "public", "data", ORIGIN_STORY_FILE);
  return existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as OriginStory) : null;
}
