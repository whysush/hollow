#!/bin/bash
source /media/sushanth/Elements/hollow_ws/env.sh
cd "$PROJECT_ROOT"
pip install --no-compile torch numpy tokenizers datasets transformers tqdm requests pyarrow onnx onnxruntime && echo PIP_DONE
