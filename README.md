# hollow

**Live:** https://whysush.github.io/hollow/

hollow is a small language model trained from scratch on one laptop. Ask it "tell me about octopuses". Your browser fetches the Wikipedia article, picks five facts, and the model retells each one in a calm, superior, lowercase voice. Each sentence carries a reference box that links back to the passage it came from. Everything runs client-side: the model is a 34 MB int8 ONNX file executed by [transformers.js](https://github.com/huggingface/transformers.js), and there is no server.

```
you inquired about octopus. like the stories you tell in the wind?
the order consists of some 300 species and is grouped within the class cephalopoda with squids,
cuttlefish, and nautiloids. each one a branch that does not need to move, while every other is a
chapter that you will eventually forget.
```

## What is in the box

| | |
|---|---|
| Model | GPT-2-style decoder: 8 layers, 8 heads, width 512, context 512, tied embeddings, **33,870,848 parameters** |
| Tokenizer | Byte-level BPE trained here, vocabulary 16,384 (6 special tokens) |
| Weights | Trained from random initialisation. No pretrained weights are in the final model. |
| Browser files | `onnx/model_quantized.onnx` 34.5 MB (int8, WASM); `onnx/model_fp16.onnx` 68.1 MB (WebGPU when `shader-f16` is available) |
| Hardware | One laptop RTX 4050 (6 GB): bf16, PyTorch SDPA, gradient accumulation, `torch.compile` |

## How it was trained

1. **Data** (`src/data_*.py`)
   - FineWeb-Edu: 1,132,544 documents, giving 1.2B training tokens and 5M validation tokens.
   - A public-domain gothic and weird-fiction corpus from Project Gutenberg: 554 books, 40.0M words, lowercased, with all boilerplate stripped (0 hits on the boilerplate scan). That is 52.4M training tokens; 22 whole books are held out.
   - 40,001 Wikipedia lead sentences across 10 topic buckets, taken from 351,297 articles scanned. Living people and sensitive categories were excluded.
2. **Tokenizer** (`src/train_tokenizer.py`): 0/1000 round-trip failures, 1.384 tokens per word.
3. **Stage 1 pretraining** (`src/train.py --stage 1`): 1.2B FineWeb-Edu tokens, 9,155 steps of 131,072 tokens each, in **5.1 h** at about 65k tok/s. Validation loss 3.493.
4. **Teacher data** (`src/teacher.py`):
   - Qwen2.5-7B-Instruct (Q4_K_M, local Ollama) rewrote Wikipedia sentences in hollow's voice using `prompts/prompt_a.txt`. Every output went through the same rule filter the website uses (`src/textutil.py`, ported to `web/filter.js`; the two agree on 753/753 outputs).
   - 25,075 calls produced **20,003 accepted pairs** (80%) in 4 h 20 min.
   - Openers and closers (`prompts/prompt_b.txt`): 1,801 and 1,991 accepted.
   - The teacher only produces training data. It is not part of hollow.
5. **Stage 2** (`--stage 2`): 1.5 epochs over the gothic corpus with 10% FineWeb-Edu mixed in. That is 665 steps (87.2M tokens) in **23 min**. Gothic validation loss went from 3.87 to 3.615.
6. **Stage 3 SFT** (`src/sft.py`): 22,606 examples of the form `<|src|> fact <|sty|> rewrite <|end|>`, with loss on the rewrite only. 4 epochs in **4 min**; best validation loss 1.541.

## Evaluation

These are held-out Wikipedia sources the model never saw. Each gets 4 candidates at temperature 0.8 and top-p 0.9; a source passes when any candidate passes the filter.

| | |
|---|---|
| Pass rate, PyTorch, 300 sources | **95.7%** |
| Lowercase rate | 100% |
| Opener/closer pass rate | 95% |
| transformers.js int8 vs PyTorch, same 50 sources | 96.0% vs 96.0% |
| PyTorch → HF conversion parity | max \|Δlogit\| 2.8e-5 |

The filter checks that every name and number in the source survives and that no new numbers appear. It also enforces lowercase, a length cap, no near-copies, no exclamation marks or emoji, and a banned-word list. When no candidate passes, the site shows the lowercased source sentence instead. A 34M-parameter model still makes mistakes the filter cannot see, such as a garbled word or a repeated phrase.

## Safety

- **Crisis messages** (self-harm, suicide) drop the persona entirely. Nothing is searched or generated. The site shows plain helpline information: Tele-MANAS 14416 in India, and findahelpline.com elsewhere.
- **Requests for harmful instructions** are declined in character before any search.
- **Articles about living people**, and sensitive categories such as violent events, are declined in character based on the article's Wikipedia categories.
- **Input** is capped at 120 characters. Wikipedia is the only network call.

## The face

The ASCII face is a 1:1 implementation of a design handoff ("ASCII Talking Face").

- **Head:** an original head sculpted from signed-distance shapes and ray-marched once into a luminance grid.
- **Animation:** each frame deforms that grid (jaw, lip shapes, brows, eyelids) and maps it to the ramp ` .,:;$$#%@`. It runs at a deliberately choppy 12 fps.
- **Verification:** `web-tests/face_golden.mjs` renders the 28 golden frames from the handoff and string-compares them; all 28 are identical.
- **Speech:** while hollow reveals an answer, the face plays the seeded speech timeline. When idle, it keeps its mouth closed and blinks.

## Layout

```
src/        data, tokenizer, model, training, SFT, evaluation, HF conversion, ONNX export
prompts/    teacher prompts (A: rewrites, B: openers/closers)
tools/      environment and pipeline scripts (env.sh keeps every cache and venv inside the project folder)
web/        the static site: index.html, style.css, face.js, filter.js, app.js, worker.js, hollow-gen.js
web-tests/  Playwright end-to-end tests (Wikipedia mocked from recorded fixtures), face golden test,
            transformers.js-vs-PyTorch check
STATE.md    the build log, with every phase's numbers
```

## Reproducing

1. `source env.sh`, then follow the phases in `STATE.md`.
2. `tools/pipeline.sh` runs everything after stage 1 unattended.
3. Export with `python src/convert_hf.py ckpt/sft/best.pt export/hf`, then `python src/export_onnx.py export/hf export/models/hollow` (in `.venv-export`).
4. Build the site with `tools/build_site.sh export/models/hollow site`.
5. Run the tests with `cd web-tests && npm ci && SITE=../site npx playwright test`.

Datasets, checkpoints and model files are not in this repository.

## Data sources

- **Training data:**
  - FineWeb-Edu (ODC-By).
  - Project Gutenberg public-domain texts, with Gutenberg trademarks and boilerplate removed.
  - Wikipedia text (CC BY-SA 4.0).
  - Teacher outputs from Qwen2.5-7B-Instruct (Apache-2.0).
