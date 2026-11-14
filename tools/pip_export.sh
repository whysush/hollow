#!/bin/bash
# Separate venv for the ONNX export: optimum-onnx pins transformers<4.58, while the training venv has transformers 5.x.
source /media/sushanth/Elements/hollow_ws/env.sh
deactivate 2>/dev/null
source "$PROJECT_ROOT/.venv-export/bin/activate"
pip install --no-compile --index-url https://download.pytorch.org/whl/cpu torch && \
pip install --no-compile "optimum-onnx[onnxruntime]" onnx onnxruntime tokenizers safetensors && echo EXPORT_PIP_DONE
