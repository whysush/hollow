// Phase 9 live check: open the deployed site, ask "tell me about octopuses" against real Wikipedia, and report the
// answer, the reference links, the face size and any console errors. Usage: node live_check.mjs [url] [question]
import { chromium } from "@playwright/test";
const url = process.argv[2] || "https://whysush.github.io/hollow/";
const q = process.argv[3] || "tell me about octopuses";
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1280, height: 800 } });
const errors = []; p.on("pageerror", (e) => errors.push(String(e))); p.on("console", (m) => m.type() === "error" && errors.push(m.text()));
const t0 = Date.now();
await p.goto(url);
await p.waitForFunction(() => !document.getElementById("q").disabled, null, { timeout: 240000 });
console.log(`model ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
const face = await p.textContent("#face"); console.log(`face: ${face.split("\n").length - 1} rows x ${face.split("\n")[0].length} cols`);
await p.fill("#q", q); await p.press("#q", "Enter");
await p.waitForSelector("#source:not([hidden])", { timeout: 300000 });
console.log(`answered in ${((Date.now() - t0) / 1000).toFixed(1)} s total`);
const refs = await p.$$eval("#answer a.ref", (as) => as.map((a) => a.href));
console.log("ANSWER:", (await p.textContent("#answer")).replace(/\.1 /g, ". "));
console.log("SOURCE:", await p.textContent("#source"), `| ${refs.length} reference links, all wikipedia: ${refs.every((h) => h.startsWith("https://en.wikipedia.org/wiki/"))}`);
await p.screenshot({ path: "web-tests/screenshots/live-answered.png", fullPage: true });
console.log("console errors:", errors.length ? errors : "none");
await b.close();
