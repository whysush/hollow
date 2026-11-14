#!/bin/bash
# Starts the project-local Ollama server (port 11435, model store on the external drive).
# OLLAMA_NUM_PARALLEL=4: four concurrent request slots for teacher generation.
# OLLAMA_FLASH_ATTENTION=1 and q8_0 KV cache: smaller KV memory so 4 slots fit beside the 7B model in 6 GB.
source /media/sushanth/Elements/hollow_ws/env.sh
export OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_MAX_LOADED_MODELS=1
exec "$PROJECT_ROOT/tools/ollama/bin/ollama" serve
