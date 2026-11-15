// Face 1:1 check: in headless Chromium (the luminance grid is built in a canvas), render web/face.js at each of the 28
// golden timestamps from the design handoff (fixtures/face_frames.json = design_handoff_ascii_talking_face/frames/
// frames.json; cols 90, ascii ramp, speed 1, energy reset to 0 per frame) and string-compare with the expected ASCII.
// Prints FACE_GOLDEN_OK, or the number of differing cells per frame. Usage: node face_golden.mjs
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import path from "node:path";
const root = path.resolve(import.meta.dirname, "..");
const port = 8791;
const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1", "--directory", root], { stdio: "ignore" });
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 50; i++) { try { await fetch(base + "/web/face.js"); break; } catch { await new Promise((r) => setTimeout(r, 100)); } }
const b = await chromium.launch();
let code = 1;
try {
  const p = await b.newPage();
  await p.goto(base + "/web-tests/fixtures/face_frames.json");
  await p.addScriptTag({ url: base + "/web/face.js" });
  const res = await p.evaluate(async () => {
    const gold = await (await fetch("/web-tests/fixtures/face_frames.json")).json();
    const face = new HollowFace(document.createElement("pre"), { cols: gold.cols, src: "/web/assets/face-ref2.png" });
    await face.ready;
    if (face.rows !== gold.rows) return [`rows ${face.rows} != ${gold.rows}`];
    return gold.frames.map((f, n) => {
      const got = face.renderAt(f.t);
      if (got === f.ascii) return null;
      let cells = 0; for (let k = 0; k < Math.max(got.length, f.ascii.length); k++) if (got[k] !== f.ascii[k]) cells++;
      return `frame ${n + 1} t=${f.t}: ${cells} cells differ`;
    });
  });
  const bad = res.filter(Boolean);
  bad.forEach((m) => console.log(m));
  console.log(bad.length ? `FACE_GOLDEN_FAIL ${bad.length}/${res.length} frames differ` : `FACE_GOLDEN_OK ${res.length}/${res.length} frames identical`);
  code = bad.length ? 1 : 0;
} finally { await b.close(); srv.kill(); }
process.exit(code);
