# HOLLOW project environment. Source this at the start of every shell command and background job
# so that every cache, temp file and venv lives on the external SSD, never on the internal disk.
export PROJECT_ROOT="/media/sushanth/Elements/hollow_ws"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TMPDIR="$PROJECT_ROOT/.tmp"; export TMP="$TMPDIR"; export TEMP="$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/torchinductor"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$PROJECT_ROOT/.cache/nv"
export OLLAMA_MODELS="$PROJECT_ROOT/.cache/ollama/models"
export npm_config_cache="$PROJECT_ROOT/.cache/npm"
export PLAYWRIGHT_BROWSERS_PATH="$PROJECT_ROOT/.cache/ms-playwright"
# Project-local Ollama server: port 11435 so it never collides with the system snap service on 11434.
export OLLAMA_HOST="127.0.0.1:11435"
export PATH="$PROJECT_ROOT/tools/ollama/bin:$PATH"
source "$PROJECT_ROOT/.venv/bin/activate" 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1  # avoid stalls writing tiny .pyc files to the external disk
export TORCHINDUCTOR_COMPILE_THREADS=1  # one inductor compile worker: the pool of workers cost GBs of RAM (see STATE.md OOM incident)
