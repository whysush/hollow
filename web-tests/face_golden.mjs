// Face 1:1 check: render web/face.js at each of the 28 golden timestamps from the design handoff
// (fixtures/face_frames.json = design_handoff_ascii_talking_face/frames/frames.json; cols 90, ascii ramp, speed 1,
// energy reset to 0 per frame) and string-compare with the expected ASCII. Prints FACE_GOLDEN_OK or the first diffs.
import { createRequire } from "node:module";
import fs from "node:fs";
const require = createRequire(import.meta.url);
const { HollowFace } = require("../web/face.js");
const gold = JSON.parse(fs.readFileSync(new URL("./fixtures/face_frames.json", import.meta.url), "utf8"));
const face = new HollowFace(null, { cols: gold.cols });
if (face.rows !== gold.rows) { console.log(`rows ${face.rows} != ${gold.rows}`); process.exit(1); }
let bad = 0;
gold.frames.forEach((f, n) => {
  const got = face.renderAt(f.t);
  if (got === f.ascii) return;
  bad++;
  const a = got.split("\n"), b = f.ascii.split("\n");
  let cells = 0; for (let r = 0; r < Math.max(a.length, b.length); r++) for (let c = 0; c < 90; c++) if ((a[r] || "")[c] !== (b[r] || "")[c]) cells++;
  console.log(`frame ${n + 1} t=${f.t}: ${cells} cells differ`);
});
console.log(bad ? `FACE_GOLDEN_FAIL ${bad}/${gold.frames.length} frames differ` : `FACE_GOLDEN_OK ${gold.frames.length}/${gold.frames.length} frames identical`);
process.exit(bad ? 1 : 0);
