#!/bin/bash
# HOLLOW GPU chain after stage 1. Every step is resumable, so this script can simply be re-run after an interruption.
#   1. wait for stage 1 to finish, print 5 unconditional samples
#   2. make sure the Wikipedia source sentences exist (finalize the sampler via data/wiki_STOP if it is still running)
#   3. teacher: rewrites (Prompt A, target 20,000 accepted) then openers/closers (Prompt B, 2,000 topics), merge
#   4. stop Ollama (frees the GPU), stage 2 (gothic + 10% FineWeb), stage 3 SFT, held-out evaluation
# Each heavy step runs in its own systemd scope with a memory cap (see STATE.md, OOM incident).
source /media/sushanth/Elements/hollow_ws/env.sh
cd "$PROJECT_ROOT"
say() { echo "[$(date '+%F %T')] $*"; }
capped() { local mem=$1; shift; systemd-run --user --scope -q -p MemoryMax=$mem -p MemorySwapMax=0 "$@"; }

say "waiting for stage 1"
until grep -q TRAIN_DONE logs/stage1.log 2>/dev/null; do
  if ! pgrep -f "train.py --stage 1" >/dev/null; then say "stage 1 process not running and not done: resuming it"; capped 7G python src/train.py --stage 1 --out ckpt/stage1 --tokens 1200000000 >> logs/stage1.log 2>&1; fi
  sleep 60
done
say "stage 1 done"
[ -s logs/stage1_samples.log ] || capped 4G python src/sample.py ckpt/stage1/latest.pt --n 5 --tokens 120 > logs/stage1_samples.log 2>&1

if [ ! -s data/wiki_sentences.jsonl ]; then
  say "wiki sentences not final yet: requesting finalize"
  touch data/wiki_STOP
  until [ -s data/wiki_sentences.jsonl ] && grep -q FINAL logs/wiki.log; do sleep 30; done
fi
say "wiki sentences: $(wc -l < data/wiki_sentences.jsonl)"

# ---- teacher ----
if ! grep -q OPENERS_DONE logs/teacher_openers.log 2>/dev/null; then
  if ! curl -s "http://$OLLAMA_HOST/api/version" >/dev/null; then
    nohup tools/ollama_serve.sh > logs/ollama_serve.log 2>&1 &
    until curl -s "http://$OLLAMA_HOST/api/version" >/dev/null; do sleep 2; done
  fi
  say "teacher rewrites"
  grep -q REWRITES_DONE logs/teacher_rewrites.log 2>/dev/null || capped 3G python src/teacher.py rewrites --target-accepted 20000 >> logs/teacher_rewrites.log 2>&1
  say "teacher openers/closers"
  capped 3G python src/teacher.py openers >> logs/teacher_openers.log 2>&1
fi
capped 3G python src/teacher.py merge > logs/teacher_merge.log 2>&1
say "merge: $(head -3 logs/teacher_merge.log | tr '\n' ' ')"
# Free the GPU: unload the teacher and stop the project Ollama server.
ollama stop qwen2.5:7b-instruct-q4_K_M 2>/dev/null
for p in $(pgrep -x ollama); do [ "$(readlink /proc/$p/exe)" = "$PROJECT_ROOT/tools/ollama/bin/ollama" ] && kill $p; done
sleep 5

# ---- stage 2: 1.5 epochs over the gothic set, 10% FineWeb mixed in; ~670 steps, so a 50-step warmup ----
say "stage 2"
grep -q TRAIN_DONE logs/stage2.log 2>/dev/null || capped 7G python src/train.py --stage 2 --out ckpt/stage2 --init ckpt/stage1/latest.pt --epochs 1.5 --warmup 50 --eval-every 50 >> logs/stage2.log 2>&1
[ -s logs/stage2_samples.log ] || capped 4G python src/sample.py ckpt/stage2/latest.pt --n 5 --tokens 120 > logs/stage2_samples.log 2>&1

# ---- stage 3: SFT + evaluation ----
say "stage 3 SFT"
grep -q SFT_DONE logs/sft.log 2>/dev/null || capped 6G python src/sft.py train --init ckpt/stage2/latest.pt --out ckpt/sft --epochs 4 >> logs/sft.log 2>&1
say "evaluation"
capped 6G python src/sft.py eval --ckpt ckpt/sft/best.pt --n 300 > logs/sft_eval.log 2>&1
say "PIPELINE_DONE"
