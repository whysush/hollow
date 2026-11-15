# HOLLOW — build state

## Drive check (Phase 0) — DONE 2026-11-14 (redone after drive change)
- **Project root moved** from `/media/sushanth/SSD/hollow_ws` (fake-capacity USB 2.0 flash drive, see history below) to
  **`/media/sushanth/Elements/hollow_ws`** on a WD Elements portable HDD (WDC WD5000LMVW, 466 GB, `/dev/sdb1` after replug),
  not the internal NVMe (`/dev/nvme0n1p5`, ext4, `/`). All paths in this project use the new root.
- Filesystem: **NTFS (ntfs3 driver)**, mount options `rw,nosuid,nodev,relatime,uid=1000,gid=1000,windows_names,iocharset=utf8`
  (exec allowed, owned by uid 1000). `windows_names` => no `: * ? " < > |` in filenames.
- Free space: 278 GB on the HDD. Internal disk at project start: 51,980,345,344 bytes free (`.internal_free_start`).
- Link: enumerates at USB 2.0 (480 Mb/s) even on a different port; user chose to continue on it.
- Sequential speed (1 GiB, O_DIRECT, 16 MB blocks): **write 42.0 MB/s, read 44.7 MB/s** (7x the old drive).
- NTFS quirks: venv already created with copies; git clones use `core.symlinks=false`, `core.fileMode=false`.
- History: the original "SSD" was USB ID 048d:1234 (Chipsbank flash controller, bcdUSB 2.00) with ~0.5 MB/s sustained writes; user
  copied the project to the Elements HDD. A second Claude session that was still running in the folder was stopped (user approved).

## Allowed system-level exceptions (not on the SSD)
- NVIDIA driver 595.99.02 (pre-installed), Node.js v20.20.2 via nvm (pre-installed, `~/.nvm`), pyenv Python 3.13.14 (pre-installed, base interpreter copied into the venv),
  system git/gh/zstd/curl. Ollama created `~/.ollama/id_ed25519` (its client key, 20 KB dir). One un-sourced `npx` run wrote 5 small
  entries to `~/.npm/_cacache`. FINAL internal-disk check: free space at start 51,980,345,344 B, at end 52,462,456,832 B
  (it rose by 482 MB; the limit was a drop under 5 GB). The pre-installed Ollama snap (v0.40.2) is left untouched and unused; HOLLOW runs its own Ollama v0.40.16 from `tools/ollama` on port 11435.

## Phases
| Phase | Status | Notes |
|---|---|---|
| 0 Drive check | done | see above |
| 1 Environment | done | torch 2.15.0+cu132, CUDA 13.2, driver 595.99.02, RTX 4050 Laptop 6141 MiB; bf16 2048x2048 matmul OK; sys.prefix + pip cache on drive; Node v20.20.2 |
| 2 Data | done | FineWeb-Edu: 1,132,544 docs -> 1.2B train + 5M val tokens (data/tokens). Gothic: 554 books, 40.0M words, 0 boilerplate hits, all lowercase -> 52.4M train / 2.2M val tokens (22 held-out books). Wikipedia sentences: 40,001 lead sentences (10 buckets x ~4,000, 8-40 words, mean 19.1) from 351,297 articles scanned (25,180 living people and 7,885 skip-category articles excluded) -> data/wiki_sentences.jsonl; src/data_check.py all passes |
| 3 Teacher | done | Qwen2.5-7B-Instruct Q4_K_M via project Ollama 0.40.16 (28/29 layers on GPU, 4 slots, q8 KV). Smoke test: 8/8 pass, 68.6 calls/min. Full run started 22:56: first 10 min 81 calls/min, 82% accepted (rejections in first 800: missing_name 70, teacher_skip 46, missing_digits 9, not_lowercase 7, new_digits 7, too_long 3, banned_word 2, bad_json 1); 25 samples reviewed (facts kept, voice on target); Python vs JS filter parity 753/753; FINAL (22:56-03:16): 25,075 rewrite calls -> 20,003 accepted (79.8%; missing_name 2469, teacher_skip 1600, missing_digits 307, new_digits 251, not_lowercase 183, banned_word 142, bad_json 46, too_long 44, near_copy 19, exclamation 9, empty 1, emoji 1); openers 1,801/2,000 and closers 1,991/2,000 accepted; merged by src/teacher.py merge |
| 4 Tokenizer | done | byte-level BPE, vocab 16,384, specials ids 0-5; mix chars fineweb 75.0M / gothic 21.4M / wiki 10.7M; round-trip failures 0/1000; 1.384 tokens/word; tokenizer/tokenizer.json |
| 5 Pretrain | done | stage 1 started 17:47, 9,155 steps x 131,072 tokens = 1.2B tokens; ckpt/stage1/latest.pt every 30 min. Stage 1 DONE 22:55: 1.2B tokens in 5.13 h (~65k tok/s), final val_fineweb 3.493 (3.92 at step 2000); 5 unconditional samples fluent web-style English (logs/stage1_samples.log). Stage 2 DONE 03:16-03:41: 665 steps, 87.2M tokens (1.5 epochs gothic + 10% FineWeb), val_gothic 3.871 (step 50) -> 3.615, val_fineweb 3.657; samples in logs/stage2_samples.log |
| 6 Voice | done | SFT 03:41-03:45: 22,606 train / 1,189 val examples (rewrites + openers/closers), 4 epochs, best val loss 1.541 at epoch 2 (ckpt/sft/best.pt). Held-out eval on 300 sources (4 candidates, T 0.8, top-p 0.9): pass rate 95.7% (bar 85%), lowercase 100%, opener/closer pass 95%; candidate rejections missing_name 83, missing_digits 31, too_long 9, banned_word 8, empty 6, near_copy 5, new_digits 2, emoji 1 (ckpt/sft/eval.json, logs/sft_eval.log) |
| 7 Export | done | convert_hf.py parity worst max|diff| 2.81e-05 (limit 1e-4); ONNX opset 17 with KV cache: onnx/model_quantized.onnx 34.5 MB (int8, WASM), onnx/model_fp16.onnx 68.1 MB (WebGPU); transformers.js 4.3.2 in Node on 50 held-out sources: 96.0% vs PyTorch 96.0% (diff 0.0 points), PHASE7_OK (logs/export.log). Fix: export_onnx.py resolves optimum-cli next to the venv interpreter |
| 8 Website | done | Face replaced by a 1:1 port of the design handoff in 'Talking Face Animation/' (procedural SDF head, 90x66, ramp ' .,:;$$#%@', 12 fps, seeded speech timeline): web-tests/face_golden.mjs FACE_GOLDEN_OK 28/28 golden frames identical. Page layout per handoff (face dead-centre, response text 14px #d8d8d8, white 1px '>' input box, JetBrains Mono). HOLLOW integration: timeline plays only while an answer is revealed; idle/thinking = closed mouth + blinks. Playwright with the real model (SITE=../site): 12/12 (6 phone + 6 desktop); screenshots reviewed. playwright.config.js reuseExistingServer=false (a reused server once served the random-model build) |
| 9 Deploy | done | whysush.github.io commit 1899bd8: site copied to hollow/ (16 files, 0 existing files modified), .nojekyll added at the root; Pages build 'built'. Live https://whysush.github.io/hollow/ returns 200; onnx/model_quantized.onnx served at 34,472,226 bytes (= local); site root still 200. Headless live check (web-tests/live_check.mjs, real Wikipedia, 'tell me about octopuses'): model ready 7.1 s, answered with opener + 5 rewritten facts + closer, 13 reference links all to en.wikipedia.org, 0 console errors. Code: https://github.com/whysush/hollow (68 files, 2.1 MB; no data, checkpoints, or model files; no licence file, the owner's decision) |

## Next action
ALL PHASES DONE (2026-11-15). Live: https://whysush.github.io/hollow/  Code: https://github.com/whysush/hollow
To update the site: rebuild with tools/build_site.sh export/models/hollow site, copy site/. into deploy/whysush.github.io/hollow/, commit, push.
Known weakness: the 34M model sometimes garbles a word or repeats a phrase that the rule filter cannot detect (e.g. "julopuses").

## Phase notes
- Phase 1 (2026-11-14 15:55): `optimum[onnxruntime]` dropped from the main install because it pins transformers<4.47 and made pip
  backtrack; the ONNX export (Phase 7) will use `optimum-onnx` or torch.onnx installed separately.
- Phase 3: registry.ollama.ai served ~176 KB/s and Ollama's 16-way parallel chunk download thrashes the HDD; instead the GGUF
  `bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf` (quant of the official Apache-2.0 weights) was downloaded
  sequentially from Hugging Face and imported with `ollama create` (tools/Modelfile.qwen, ChatML template, Qwen default system prompt).
- Phase 2.2: fixed the core-author rule so translator/illustrator credits do not count (Machen translated Casanova); 12 Casanova
  volumes removed.
- **Incident 2026-11-14 16:18: global OOM** (15 GB RAM shared with desktop/VS Code). The kernel killed dbus, which ended the whole
  desktop session and every background job (data jobs, Ollama). Causes: `datasets` streaming of FineWeb with `.skip()` grew to
  >4 GB RSS, plus the Wikipedia streamer (~2.2 GB) and a training benchmark at the same time. Fixes: every background job now
  runs in its own systemd user scope with `MemoryMax` (tools/run_bg.sh), so only the offender dies; FineWeb is tokenized from
  3 downloaded parquet shards read batch-by-batch with pyarrow (~0.8 GB RSS; same row order as the stream);
  data_wiki.py now checkpoints to data/wiki_state.json every 2,000 candidates (the first run lost 23k articles of progress);
  `TORCHINDUCTOR_COMPILE_THREADS=1`.
- Phase 5 benchmark (2026-11-14 17:45, torch.compile on, bf16, micro 16 x accum 16 x 512 = 131,072 tokens/step):
  **65,176 tokens/s**, 2.49 GiB peak VRAM. 8 h would be 1.88B tokens > the 1.2B cap, so **stage 1 = 1.2B tokens
  (9,155 steps), ~5.1 h**. Also fixed train.py logging (it printed last-micro loss / 16).
- Phase 7 de-risked with a random-init model (18:05): parity 1.5e-6; ONNX export via optimum-onnx 0.1.0 in `.venv-export`
  (transformers 4.57.6, CPU torch, accelerate for tied-weight dedup): model_quantized.onnx 32.7 MB, model_fp16.onnx 64.9 MB.
  transformers.js 4.3.2 bug worked around: an http(s) `localModelPath` makes get_file_metadata skip local lookups, so the
  worker gets a root-relative path. Generation scores = mean NLL from one extra forward pass of the raw model.
- Wikipedia fixtures were recorded with retries (the API rate-limited this IP while data_wiki.py was running).
- Face (2026-11-15): replaced the earlier home-made raymarcher with a 1:1 port of the design handoff in
  'Talking Face Animation/design_handoff_ascii_talking_face' (verified by web-tests/face_golden.mjs, 28/28 identical frames).
  The handoff loops speech forever; in HOLLOW the timeline advances only while an answer is revealed (idle = closed mouth + blinks).
  The handoff's Enter-clears-input behaviour is replaced by HOLLOW's ask flow; the input is 16px on touch devices (iOS zoom).
- Face v2 + features (2026-11-15 afternoon): the face is now a 1:1 port of the second handoff, 'Talking Face Animation(1).zip'
  (photo-based: assets/face-ref2.png, crop 112,4,166,204, flood-fill background, jawline cut, A = 204/166). The owner
  confirmed a licence for this stock image. Golden test now runs in headless Chromium (canvas): FACE_GOLDEN_OK 28/28.
  Added: '(thinking.)' -> '(thinking..)' -> '(thinking...)' loop (400 ms) while busy; 'who are you' / 'who am i' fixed
  in-character answers (app.js SELF, selfQuestion); the input is disabled while any fixed line is spoken.
  Playwright: 14/14 (7 phone + 7 desktop). Commits in both repos are now authored as whysush (GitHub noreply address).
- Design tweaks (2026-11-15): face capped at 40% of the viewport height / 60% of the width (face.js heightFrac/widthFrac),
  still centred; the input box sits in a black bar fixed to the bottom of the screen (#dock); 'who are you' answers no
  longer mention wikipedia. Playwright 14/14; golden 28/28 unchanged.

