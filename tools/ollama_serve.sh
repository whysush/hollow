#!/bin/bash
# Project-local Ollama server on 127.0.0.1:11435; model store under $PROJECT_ROOT/.cache/ollama/models.
source /media/sushanth/Elements/hollow_ws/env.sh
export OLLAMA_NUM_PARALLEL=4        # 4 concurrent request slots (teacher.py uses 4 workers)
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0     # halves KV-cache VRAM so 4 slots x 2048 ctx fit next to the 4.7 GB model
exec ollama serve
