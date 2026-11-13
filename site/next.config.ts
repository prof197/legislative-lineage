import type { NextConfig } from "next";

// GitHub Pages serves the site under /<repo>/ (DESIGN.md#dr-2).
const basePath = process.env.SITE_BASE_PATH ?? "/legislative-lineage";

const config: NextConfig = {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  poweredByHeader: false,
};

export default config;
