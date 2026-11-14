"""Print unconditional samples from a checkpoint (Phase 5 acceptance: fluent, if aimless, lowercase English).

  python sample.py ckpt/stage1/latest.pt --n 5 --tokens 120
Each sample starts from <|endoftext|> (the document separator) and is sampled with temperature 0.8, top-p 0.9.
"""
import os, sys, argparse
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig

ROOT = os.environ["PROJECT_ROOT"]


def load(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["cfg"]) if "cfg" in ck else GPTConfig()
    m = GPT(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--prompt", default="")
    a = ap.parse_args()
    tok = Tokenizer.from_file(f"{ROOT}/tokenizer/tokenizer.json")
    eot = tok.token_to_id("<|endoftext|>")
    m = load(a.ckpt)
    torch.manual_seed(0)
    for i in range(a.n):
        ids = [eot] + tok.encode(a.prompt).ids
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y, _ = m.generate(x, a.tokens, temperature=0.8, top_p=0.9, stop_id=eot)
        print(f"--- sample {i+1} ---")
        print(tok.decode(y[0, 1:].tolist()))
