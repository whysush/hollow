// Phase 7 acceptance helper: load the local model with @huggingface/transformers 4.3.2 (remote models disabled)
// through the SAME generation code the browser worker uses (web/hollow-gen.js), and generate for given sources.
// Usage: node node-gen.mjs <models-dir> <jsonl-of-sources> [n] > results.json
import * as lib from "@huggingface/transformers";
import { HollowGenerator } from "../web/hollow-gen.js";
import fs from "node:fs";
const [base, srcFile, nStr] = process.argv.slice(2);
const rows = fs.readFileSync(srcFile, "utf8").trim().split("\n").map((l) => JSON.parse(l)).slice(0, +(nStr || 5));
const g = new HollowGenerator(lib);
const t0 = Date.now();
await g.load(base, { device: "cpu", dtype: "q8" });
console.error(`loaded in ${Date.now() - t0} ms`);
const out = [];
for (const r of rows) {
  const t = Date.now();
  const cands = await g.generate("src", r.src.toLowerCase(), 4);
  console.error(`${Date.now() - t} ms  ${r.src.slice(0, 60)}`);
  out.push({ src: r.src, cands });
}
console.log(JSON.stringify(out, null, 1));
