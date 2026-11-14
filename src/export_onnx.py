"""Phase 7.3-7.4: export the converted HF model to ONNX (with past key values), quantize, and lay out for transformers.js.

Run inside .venv-export (optimum-onnx needs transformers<4.58):
  python export_onnx.py export/hf web/models/hollow

Steps:
  1. optimum-cli export onnx --task text-generation-with-past  -> <tmp>/model.onnx (fp32, merged decoder with KV cache)
  2. onnxruntime dynamic int8 quantization (weights int8, activations quantized at runtime)  -> onnx/model_quantized.onnx
     (transformers.js maps dtype "q8" to the "_quantized" suffix; WASM's default dtype is q8)
  3. fp16 conversion of the fp32 graph for WebGPU (keep_io_types=False so the KV cache is fp16) -> onnx/model_fp16.onnx
  4. copy config.json, generation_config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json
The layout <models>/hollow/{config.json,...,onnx/model_quantized.onnx} is what transformers.js 4.3.2 resolves for
from_pretrained("hollow") with env.localModelPath = <models>/ (see utils/model-loader.js getCoreModelFile: subfolder
"onnx", file `${fileName}${suffix}.onnx`, fileName "model").
"""
import os, sys, shutil, subprocess, json, tempfile

src, dst = sys.argv[1], sys.argv[2]
tmp = tempfile.mkdtemp(prefix="onnx_", dir=os.environ.get("TMPDIR"))
# optimum-cli is resolved next to this interpreter, so the script works without an activated venv.
cli = os.path.join(os.path.dirname(sys.executable), "optimum-cli")
subprocess.run([cli, "export", "onnx", "--model", src, "--task", "text-generation-with-past",
                "--opset", "17", tmp], check=True)
print("exported:", os.listdir(tmp))
os.makedirs(f"{dst}/onnx", exist_ok=True)

from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(f"{tmp}/model.onnx", f"{dst}/onnx/model_quantized.onnx", weight_type=QuantType.QInt8)

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
m = onnx.load(f"{tmp}/model.onnx")
m16 = convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=False)
onnx.save(m16, f"{dst}/onnx/model_fp16.onnx")

for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    shutil.copy(f"{src}/{f}", f"{dst}/{f}")
shutil.rmtree(tmp)
for root, _, files in os.walk(dst):
    for f in files:
        p = os.path.join(root, f)
        mb = os.path.getsize(p) / 2**20
        print(f"{mb:8.2f} MB  {os.path.relpath(p, dst)}")
        assert mb < 95, f"{p} is over 95 MB (GitHub limit; LFS is not served by Pages)"
