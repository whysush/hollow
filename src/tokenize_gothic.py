"""Phase 6 data: tokenize the gothic corpus (data/gothic/*.txt, already lowercased) into uint16 shards.

Outputs (flat uint16 arrays, <|endoftext|> between books):
  data/tokens/gothic_train.bin   all books except the validation books
  data/tokens/gothic_val.bin     every 25th book (by sorted Gutenberg id), i.e. ~4% of books, held out
Books are split whole (not mid-text) so validation loss measures generalisation to unseen books.
Files are written to .tmp, fsynced, then renamed (safe against a dropped USB link).
"""
import os, glob
import numpy as np
from tokenizers import Tokenizer

ROOT = os.environ["PROJECT_ROOT"]
tok = Tokenizer.from_file(f"{ROOT}/tokenizer/tokenizer.json")
eot = tok.token_to_id("<|endoftext|>")
files = sorted(glob.glob(f"{ROOT}/data/gothic/*.txt"), key=lambda p: int(os.path.basename(p).split(".")[0]))
splits = {"train": [], "val": []}
for i, p in enumerate(files):
    splits["val" if i % 25 == 12 else "train"].append(p)   # i % 25 == 12: every 25th book, offset 12
os.makedirs(f"{ROOT}/data/tokens", exist_ok=True)
for name, paths in splits.items():
    # 32 books per encode_batch call, each book converted straight to a uint16 array (Python int lists for ~55M
    # tokens would cost several GB of RAM).
    parts = []
    for i in range(0, len(paths), 32):
        texts = [open(p).read() for p in paths[i:i + 32]]
        for enc in tok.encode_batch(texts):
            parts.append(np.array(enc.ids + [eot], dtype=np.uint16))
    arr = np.concatenate(parts)
    out = f"{ROOT}/data/tokens/gothic_{name}.bin"
    with open(out + ".tmp", "wb") as f:
        f.write(arr.tobytes()); f.flush(); os.fsync(f.fileno())
    os.replace(out + ".tmp", out)
    print(f"gothic {name}: {len(paths)} books, {len(arr):,} tokens")
