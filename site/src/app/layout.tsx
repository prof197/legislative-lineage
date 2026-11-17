import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Newsreader, Public_Sans } from "next/font/google";
import "./globals.css";
import { FEED_PATH, hasFeed, meta } from "@/lib/data";
import { stateName } from "@/lib/format";

// As lineage.feed titles the feed itself.
const FEED_TITLE = "Legislative Lineage: bills newly found to share text";

// Self-hosted at build time, so a page makes no request to a font service.
const serif = Newsreader({ subsets: ["latin"], variable: "--font-serif", style: ["normal", "italic"], weight: ["400", "500", "600"] });
const sans = Public_Sans({ subsets: ["latin"], variable: "--font-sans", weight: ["400", "500", "600", "700"] });
const mono = IBM_Plex_Mono({ subsets: ["latin"], variable: "--font-mono", weight: ["400", "500"] });

export function generateMetadata(): Metadata {
  return {
    title: { default: "Legislative Lineage", template: "%s · Legislative Lineage" },
    description: `${stateName(meta().state)} bills from the 2023–2026 sessions, and the bills in other states that share their language.`,
    // The live track's Atom feed, when this build's data carries one (DESIGN.md#updates). Metadata
    // URLs are not given the base path, so FEED_PATH carries it, as GitHub Pages serves the site
    // under one (DESIGN.md#dr-2).
    ...(hasFeed()
      ? { alternates: { types: { "application/atom+xml": [{ url: FEED_PATH, title: FEED_TITLE }] } } }
      : {}),
  };
}

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
