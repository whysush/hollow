"""Phase 7.1-7.2: convert a model.py checkpoint to a Hugging Face GPT2LMHeadModel with our own weight mapping.

Weight mapping (model.py name -> HF GPT2LMHeadModel name):
  transformer.wte.weight            -> transformer.wte.weight            (vocab, n_embd), copied
  transformer.wpe.weight            -> transformer.wpe.weight            (block_size, n_embd), copied
  transformer.h.{i}.ln_1/ln_2.*     -> same names                        copied
  transformer.h.{i}.attn.c_attn.weight  (3C, C) nn.Linear  -> (C, 3C) Conv1D: TRANSPOSED
  transformer.h.{i}.attn.c_proj.weight  (C, C)              -> (C, C)  Conv1D: TRANSPOSED
  transformer.h.{i}.mlp.c_fc.weight     (4C, C)             -> (C, 4C) Conv1D: TRANSPOSED
  transformer.h.{i}.mlp.c_proj.weight   (C, 4C)             -> (4C, C) Conv1D: TRANSPOSED
  all biases                                                -> copied (same shape)
  transformer.ln_f.*                -> copied
  lm_head.weight                    -> tied to wte in HF too (tie_word_embeddings=True), not stored separately
HF config: activation_function="gelu_new" (tanh GELU, as model.py), layer_norm_epsilon=1e-5, n_positions=block_size,
no dropout, scale_attn_weights=True (q k^T / sqrt(head_dim), as SDPA).

Parity test: logits from model.py and the HF model (both fp32, CPU) on 5 random inputs must match within 1e-4.

  python convert_hf.py ckpt/sft/best.pt export/hf
"""
import os, sys, json, shutil
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig

ROOT = os.environ["PROJECT_ROOT"]


def convert(ckpt_path, out_dir):
    from transformers import GPT2Config, GPT2LMHeadModel
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    ours = GPT(cfg)
    ours.load_state_dict(ck["model"])
    ours.eval()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/tokenizer/tokenizer.json")
    eot, end = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|end|>")
    hcfg = GPT2Config(
        vocab_size=cfg.vocab_size, n_positions=cfg.block_size, n_embd=cfg.n_embd, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_inner=4 * cfg.n_embd, activation_function="gelu_new",
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=cfg.layer_norm_epsilon,
        scale_attn_weights=True, tie_word_embeddings=True, use_cache=True,
        bos_token_id=eot, eos_token_id=end, pad_token_id=eot,
    )
    hf = GPT2LMHeadModel(hcfg)
    hf.eval()
    sd = ours.state_dict()
    new = {}
    transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    for k, v in sd.items():
        if k == "lm_head.weight":
            continue  # tied to transformer.wte.weight
        new[k] = v.t().contiguous() if k.endswith(transposed) else v.clone()
    missing, unexpected = hf.load_state_dict(new, strict=False)
    # HF registers attn.bias / attn.masked_bias buffers in some versions and lm_head.weight (tied); nothing else may be missing.
    missing = [m for m in missing if not (m.endswith(".attn.bias") or m.endswith(".attn.masked_bias") or m == "lm_head.weight")]
    assert not missing and not unexpected, (missing, unexpected)
    hf.tie_weights()
    assert hf.lm_head.weight.data_ptr() == hf.transformer.wte.weight.data_ptr()

    # ---- parity test ----
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for i in range(5):
        T = int(torch.randint(8, cfg.block_size + 1, (1,), generator=g))
        x = torch.randint(0, cfg.vocab_size, (1, T), generator=g)
        with torch.no_grad():
            a, _ = ours(x)
            b = hf(x).logits
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f"parity input {i+1}: T={T} max|diff|={d:.2e}")
    print(f"PARITY worst max|diff| = {worst:.2e} (limit 1e-4)")
    assert worst < 1e-4, "parity test failed"

    os.makedirs(out_dir, exist_ok=True)
    hf.save_pretrained(out_dir, safe_serialization=True)
    gen = {"bos_token_id": eot, "eos_token_id": end, "pad_token_id": eot, "do_sample": True,
           "temperature": 0.8, "top_p": 0.9, "max_new_tokens": 96}
    json.dump(gen, open(f"{out_dir}/generation_config.json", "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(f"{ROOT}/tokenizer/{f}", f"{out_dir}/{f}")
    print("saved", out_dir, os.listdir(out_dir))
    return worst


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
