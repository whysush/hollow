// Playwright config for HOLLOW. Serves the assembled site (SITE env var, default ../.tmp/site_test) on port 8765.
// Two projects: phone (390x844) and desktop (1280x800).
import { defineConfig } from "@playwright/test";

const SITE = process.env.SITE || "../.tmp/site_test";

export default defineConfig({
  testDir: "./tests",
  timeout: 300_000,            // one answer = model load + 7 generations + ~40 s of 30 cps reveal
  expect: { timeout: 30_000 },
  workers: 1,                  // one browser at a time: the laptop's RAM is shared with training
  reporter: [["list"]],
  outputDir: "./test-results",
  use: { baseURL: "http://127.0.0.1:8765/", trace: "off" },
  webServer: {
    command: `python3 -m http.server 8765 --bind 127.0.0.1 --directory ${SITE}`,
    url: "http://127.0.0.1:8765/index.html",
    reuseExistingServer: false,   // always serve the SITE given for this run (a leftover server could be a different build)
  },
  projects: [
    { name: "phone", use: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true } },
    { name: "desktop", use: { viewport: { width: 1280, height: 800 } } },
  ],
});
