import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Newsreader, Public_Sans } from "next/font/google";
import "./globals.css";
import { meta } from "@/lib/data";
import { stateName } from "@/lib/format";

// Self-hosted at build time, so a page makes no request to a font service.
const serif = Newsreader({ subsets: ["latin"], variable: "--font-serif", style: ["normal", "italic"], weight: ["400", "500", "600"] });
const sans = Public_Sans({ subsets: ["latin"], variable: "--font-sans", weight: ["400", "500", "600", "700"] });
const mono = IBM_Plex_Mono({ subsets: ["latin"], variable: "--font-mono", weight: ["400", "500"] });

// The Atom feed of new matches, which the publishing job writes at the export's root (DESIGN.md#updates).
// Metadata URLs are not given the base path, so it is added here, as GitHub Pages serves the site
// under one (DESIGN.md#dr-2).
const FEED =`${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/data/feed.xml`;

export function generateMetadata(): Metadata {
  return {
    title: { default: "Legislative Lineage", template: "%s · Legislative Lineage" },
    description: `${stateName(meta().state)} bills from the 2023–2026 sessions, and the bills in other states that share their language.`,
    alternates: { types: { "application/atom+xml": [{ url: FEED, title: "Legislative Lineage: new matches" }] } },
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
