import { readFileSync } from "node:fs";
import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vitest/config";

/**
 * Absolute site origin for social-preview tags. Vercel provides the production domain at build
 * time; `SITE_URL` overrides it, and local builds fall back to root-relative URLs.
 */
function siteUrl(): string {
  const explicit = process.env.SITE_URL;
  const vercel = process.env.VERCEL_PROJECT_PRODUCTION_URL;
  return (explicit ?? (vercel ? `https://${vercel}` : "")).replace(/\/+$/, "");
}

interface VercelConfig {
  headers: { source: string; headers: { key: string; value: string }[] }[];
}

/** Serve `vite preview` with the production headers from vercel.json (including the CSP). */
function productionHeaders(): Record<string, string> {
  const config = JSON.parse(
    readFileSync(path.resolve(__dirname, "../../vercel.json"), "utf8"),
  ) as VercelConfig;
  const global = config.headers.find((rule) => rule.source === "/(.*)");
  return Object.fromEntries((global?.headers ?? []).map(({ key, value }) => [key, value]));
}

function siteUrlPlugin(): Plugin {
  return {
    name: "site-url",
    transformIndexHtml: (html) => html.replace(/%SITE_URL%/g, siteUrl()),
  };
}

export default defineConfig({
  plugins: [react(), siteUrlPlugin()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
    headers: productionHeaders(),
  },
  worker: {
    format: "es",
  },
  build: {
    chunkSizeWarningLimit: 300,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    include: ["src/**/*.test.{ts,tsx}"],
    css: true,
    testTimeout: 20_000,
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      // Worker plumbing only runs in a real browser and is covered by the Playwright suite.
      exclude: ["src/main.tsx", "src/types.ts", "src/lib/policy.worker.ts", "src/lib/policyRunner.ts"],
      reporter: ["text", "json-summary"],
      reportsDirectory: "coverage",
      thresholds: {
        statements: 90,
        branches: 85,
        functions: 90,
        lines: 90,
      },
    },
  },
});
