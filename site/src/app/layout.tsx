import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Newsreader, Public_Sans } from "next/font/google";
import "./globals.css";

// Self-hosted at build time, so a page makes no request to a font service.
const serif = Newsreader({ subsets: ["latin"], variable: "--font-serif", style: ["normal", "italic"], weight: ["400", "500", "600"] });
const sans = Public_Sans({ subsets: ["latin"], variable: "--font-sans", weight: ["400", "500", "600", "700"] });
const mono = IBM_Plex_Mono({ subsets: ["latin"], variable: "--font-mono", weight: ["400", "500"] });

export const metadata: Metadata = {
  title: { default: "Legislative Lineage", template: "%s · Legislative Lineage" },
  description:
    "Georgia bills from the 2023–2026 sessions, and the bills in other states that share their language.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
