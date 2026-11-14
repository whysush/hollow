// Phase 7 acceptance: load the exported model with @huggingface/transformers 4.3.2 (remote models disabled) through
// the browser's own generation code (web/hollow-gen.js) and filter (web/filter.js), generate 4 candidates for each
// held-out source, and compare the pass rate with the PyTorch evaluation on the same sources (ckpt/sft/eval.json).
// Usage: node phase7_check.mjs <models-dir> <eval.json> [n=50]
import * as lib from "@huggingface/transformers";
import { HollowGenerator } from "../web/hollow-gen.js";
import { createRequire } from "node:module";
import fs from "node:fs";
const F = createRequire(import.meta.url)("../web/filter.js");
const [base, evalPath, nStr] = process.argv.slice(2);
const n = +(nStr || 50);
const ev = JSON.parse(fs.readFileSync(evalPath, "utf8"));
const rows = ev.examples.slice(0, n);
const torchPass = rows.filter((e) => e.best).length / rows.length;

const g = new HollowGenerator(lib);
const t0 = Date.now();
await g.load(base, { device: "cpu", dtype: "q8" });
console.log(`loaded onnx/model_quantized.onnx in ${Date.now() - t0} ms (remote models disabled: ${lib.env.allowRemoteModels === false})`);
let pass = 0;
const shown = [];
const t1 = Date.now();
for (const r of rows) {
  const cands = await g.generate("src", r.src.toLowerCase(), 4);
  const ok = cands.filter((c) => F.filterRewrite(r.src, c.text) === null).sort((a, b) => a.nll - b.nll);
  if (ok.length) pass++;
  if (shown.length < 5) shown.push({ src: r.src, out: ok.length ? ok[0].text : "(no passing candidate)" });
}
const jsPass = pass / rows.length;
console.log(`sources: ${rows.length}  ms/source: ${((Date.now() - t1) / rows.length).toFixed(0)}`);
console.log(`PASS RATE  transformers.js int8: ${(100 * jsPass).toFixed(1)}%   PyTorch (same sources): ${(100 * torchPass).toFixed(1)}%   diff: ${(100 * Math.abs(jsPass - torchPass)).toFixed(1)} points`);
console.log("\n5 held-out sources, transformers.js output:");
for (const s of shown) console.log(`- SRC: ${s.src}\n  OUT: ${s.out}`);
const okAcc = Math.abs(jsPass - torchPass) <= 0.10;
console.log(okAcc ? "PHASE7_OK" : "PHASE7_FAIL");
process.exit(okAcc ? 0 : 1);
