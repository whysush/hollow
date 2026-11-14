#!/bin/bash
# Sequential, resumable download of the teacher GGUF (Apache-2.0 Qwen2.5-7B-Instruct, Q4_K_M quant by bartowski).
source /media/sushanth/Elements/hollow_ws/env.sh
cd "$PROJECT_ROOT/.cache/gguf"
F=Qwen2.5-7B-Instruct-Q4_K_M.gguf
until curl -sSL -C - --retry 5 -o $F.part https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/$F; do sleep 10; done
mv $F.part $F && echo GGUF_DONE
