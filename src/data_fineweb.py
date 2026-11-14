"""Phase 2.1 / Phase 4 / Phase 5 data: FineWeb-Edu (sample-10BT), streamed from the Hugging Face Hub.

Nothing raw is stored: the SSD is a slow USB 2.0 exFAT disk, so we stream the parquet shards over
the network and write only what we need.

  python data_fineweb.py sample            -> data/fineweb_tok_sample.txt (~75 MB lowercased text, the
                                              70% FineWeb share of the tokenizer training mix)
  python data_fineweb.py tokenize --tokens N
                                           -> data/tokens/fineweb_val.bin   (first 5,000,000 tokens)
                                              data/tokens/fineweb_train.bin (next N tokens)
                                              each a flat array of uint16 token ids (vocab 16,384 < 65,536)
                                              with <|endoftext|> between documents.
Writes go out in 64 MiB chunks (large appends are ~6x faster than small ones on this exFAT disk).
Resumable: progress (documents consumed, tokens written) is recorded in a .progress.json written
atomically after every chunk; on restart the stream skips the consumed documents and the .bin is
truncated back to the recorded length before appending.
"""
import os, sys, json, argparse, time
import numpy as np
from datasets import load_dataset

ROOT = os.environ["PROJECT_ROOT"]
TOKDIR = f"{ROOT}/data/tokens"
VAL_TOKENS = 5_000_000          # held-out validation tokens (enough for a low-noise val loss)
CHUNK_TOKENS = 32 * 1024 * 1024  # 32M uint16 tokens = 64 MiB per write


def stream():
    return load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)


def local_stream(skip_docs):
    """Yield {"text": ...} rows from the downloaded shards data/raw/fineweb/NNN_00000.parquet in order, skipping the
    first `skip_docs` rows. Replaces datasets streaming for tokenization: on 2026-11-14 streaming + .skip() grew to
    >4 GB RSS and contributed to a system-wide OOM. pyarrow reads one 1,024-row batch at a time (bounded memory), and
    skipping uses parquet row-group metadata, so no text is decoded for skipped rows. File order = stream order, so
    the continuation is identical to the streamed one."""
    import glob
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(f"{ROOT}/data/raw/fineweb/*_00000.parquet")):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            if skip_docs >= n:
                skip_docs -= n
                continue
            for batch in pf.iter_batches(batch_size=1024, columns=["text"], row_groups=[rg]):
                for t in batch.column(0).to_pylist():
                    if skip_docs > 0:
                        skip_docs -= 1
                        continue
                    yield {"text": t}


def cmd_sample(args):
    out = f"{ROOT}/data/fineweb_tok_sample.txt"
    target = 75_000_000   # characters; ~70% of a ~105 MB tokenizer-training mix
    n, buf = 0, []
    for ex in stream():
        t = ex["text"].lower()
        buf.append(t)
        n += len(t) + 1
        if n >= target:
            break
    with open(out + ".tmp", "w") as f:
        f.write("\n".join(buf)); f.flush(); os.fsync(f.fileno())
    os.replace(out + ".tmp", out)
    print("sample chars", n, "docs", len(buf))


def cmd_tokenize(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/tokenizer/tokenizer.json")
    eot = tok.token_to_id("<|endoftext|>")
    os.makedirs(TOKDIR, exist_ok=True)
    prog_path = f"{TOKDIR}/fineweb.progress.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"docs": 0, "val": 0, "train": 0}
    files = {k: open(f"{TOKDIR}/fineweb_{k}.bin", "ab") for k in ("val", "train")}
    for k, fh in files.items():   # truncate any partial chunk written after the last checkpoint
        fh.truncate(prog[k] * 2); fh.seek(prog[k] * 2)
    ds = local_stream(prog["docs"])
    pending, docs_batch = [], []
    t0, docs = time.time(), prog["docs"]

    def write_out():
        nonlocal pending
        if not pending:
            return
        arr = np.concatenate(pending).astype(np.uint16)
        pending = []
        # Fill val first (VAL_TOKENS), then train.
        if prog["val"] < VAL_TOKENS:
            take = min(VAL_TOKENS - prog["val"], len(arr))
            files["val"].write(arr[:take].tobytes()); prog["val"] += take
            arr = arr[take:]
        room = args.tokens - prog["train"]
        arr = arr[:max(room, 0)]
        files["train"].write(arr.tobytes()); prog["train"] += len(arr)
        for fh in files.values():
            fh.flush(); os.fsync(fh.fileno())
        prog["docs"] = docs
        with open(prog_path + ".tmp", "w") as f:
            json.dump(prog, f)
        os.replace(prog_path + ".tmp", prog_path)
        el = time.time() - t0
        print(f"docs={docs} val={prog['val']} train={prog['train']} ({el:.0f}s)", flush=True)

    pend_n = 0
    for ex in ds:
        docs_batch.append(ex["text"].lower())
        docs += 1
        if len(docs_batch) == 2048:
            for enc in tok.encode_batch(docs_batch):   # Rust, multi-threaded
                pending.append(np.array(enc.ids + [eot], dtype=np.uint16)); pend_n += len(enc.ids) + 1
            docs_batch = []
            if pend_n >= CHUNK_TOKENS:
                write_out(); pend_n = 0
                if prog["train"] >= args.tokens:
                    break
    if docs_batch and prog["train"] < args.tokens:
        for enc in tok.encode_batch(docs_batch):
            pending.append(np.array(enc.ids + [eot], dtype=np.uint16))
    write_out()
    print("TOKENIZE_DONE", json.dumps(prog))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "tokenize"])
    ap.add_argument("--tokens", type=int, default=1_200_000_000)
    a = ap.parse_args()
    {"sample": cmd_sample, "tokenize": cmd_tokenize}[a.cmd](a)
