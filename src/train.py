"""Phases 5 and 6 (stage 1 and stage 2): pretraining HOLLOW from scratch on uint16 token shards.

Stage 1: FineWeb-Edu only, peak LR 6e-4.
Stage 2: gothic corpus with 10% FineWeb mixed in, peak LR 3e-4, initialised from the stage 1 checkpoint.

Optimisation (Phase 5 spec):
  AdamW, betas (0.9, 0.95), weight decay 0.1 on matrices only; gradient clip 1.0 (global L2 norm);
  linear warmup for `--warmup` steps then cosine decay to 10% of peak at the final step;
  tokens per optimizer step = micro_batch * block_size * grad_accum = 16 * 512 * 16 = 131,072 (~128k);
  bf16 autocast for matmuls/attention (weights and optimizer state stay fp32).
Data: each shard is a flat uint16 array of token ids. It is read ONCE, sequentially, into RAM
(the SSD is a slow USB disk; random reads from it would stall the GPU), then training windows of
block_size+1 tokens are sampled at random offsets from RAM.
Checkpoints: every `--ckpt-minutes` (30) minutes and at the end, written to <out>/latest.pt.tmp,
fsynced, then atomically renamed to <out>/latest.pt. On start, if <out>/latest.pt exists, training
resumes from it automatically (model, optimizer, step, RNG states, elapsed time).
Logs: <out>/log.jsonl, rewritten atomically after every eval (small file).

Usage examples:
  python train.py --stage 1 --out ckpt/stage1 --tokens 1000000000
  python train.py --stage 1 --bench 180           # 3-minute throughput benchmark, no checkpoint
  python train.py --stage 2 --out ckpt/stage2 --init ckpt/stage1/latest.pt --epochs 1.5
"""
import os, sys, time, math, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig

ROOT = os.environ.get("PROJECT_ROOT", ".")
TOK = f"{ROOT}/data/tokens"


def load_tokens(path):
    """Read a whole uint16 shard into RAM sequentially (np.fromfile = one large sequential read)."""
    t0 = time.time()
    arr = np.fromfile(path, dtype=np.uint16)
    print(f"loaded {path}: {len(arr):,} tokens in {time.time()-t0:.0f}s", flush=True)
    # Kept as uint16 in RAM (2 bytes/token: 1.2B tokens = 2.4 GB); windows are widened to int64 per batch.
    return arr


class Sampler:
    """Samples (x, y) windows from a weighted mixture of token arrays.

    sources: list of (tensor, weight). Each micro-batch row picks a source with probability
    proportional to weight, then a uniform random offset; x = tokens[o:o+T], y = tokens[o+1:o+T+1].
    """
    def __init__(self, sources, T, B, gen):
        self.sources = sources
        self.T, self.B, self.gen = T, B, gen
        w = torch.tensor([s[1] for s in sources], dtype=torch.float)
        self.p = w / w.sum()

    def batch(self, device):
        which = torch.multinomial(self.p, self.B, replacement=True, generator=self.gen)
        xs, ys = [], []
        for i in which.tolist():
            data = self.sources[i][0]
            o = torch.randint(0, len(data) - self.T - 1, (1,), generator=self.gen).item()
            chunk = torch.from_numpy(data[o:o + self.T + 1].astype(np.int64))   # (T+1,) token ids
            xs.append(chunk[:-1]); ys.append(chunk[1:])
        x = torch.stack(xs).pin_memory().to(device, non_blocking=True)
        y = torch.stack(ys).pin_memory().to(device, non_blocking=True)
        return x, y


def lr_at(step, total, peak, warmup, floor_frac=0.1):
    """Linear warmup over `warmup` steps, then cosine decay from peak to floor_frac*peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text):
    with open(path + ".tmp", "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


@torch.no_grad()
def evaluate(model, val_sources, T, B, iters, device):
    """Mean loss over `iters` deterministic validation batches per source (fixed seed 1234)."""
    model.eval()
    out = {}
    for name, data in val_sources.items():
        g = torch.Generator().manual_seed(1234)
        s = Sampler([(data, 1.0)], T, B, g)
        tot = 0.0
        for _ in range(iters):
            x, y = s.batch(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            tot += loss.item()
        out[name] = tot / iters
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokens", type=float, default=1e9, help="stage 1 token budget")
    ap.add_argument("--epochs", type=float, default=1.5, help="stage 2 epochs over the gothic set")
    ap.add_argument("--init", default=None, help="checkpoint to initialise from (stage 2)")
    ap.add_argument("--micro", type=int, default=16, help="micro-batch rows")
    ap.add_argument("--accum", type=int, default=16, help="gradient accumulation steps")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-minutes", type=float, default=30.0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--bench", type=float, default=0, help="benchmark seconds; no checkpoints")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    cfg = GPTConfig()
    T = cfg.block_size
    tokens_per_step = args.micro * T * args.accum

    # ---- data ----
    if args.stage == 1:
        train_fw = load_tokens(f"{TOK}/fineweb_train.bin")
        val = {"fineweb": load_tokens(f"{TOK}/fineweb_val.bin")}
        sources = [(train_fw, 1.0)]
        total_steps = int(args.tokens // tokens_per_step)
        peak = args.lr or 6e-4
    else:
        goth = load_tokens(f"{TOK}/gothic_train.bin")
        train_fw = load_tokens(f"{TOK}/fineweb_train.bin")
        val = {"gothic": load_tokens(f"{TOK}/gothic_val.bin"),
               "fineweb": load_tokens(f"{TOK}/fineweb_val.bin")}
        # 90% gothic / 10% FineWeb by sampled windows (Phase 6: "10% FineWeb mixed in").
        sources = [(goth, 0.9), (train_fw, 0.1)]
        # Steps such that the gothic part sees `epochs` passes: gothic tokens seen per step = 0.9 * tps.
        total_steps = int(args.epochs * len(goth) / (0.9 * tokens_per_step))
        peak = args.lr or 3e-4

    # ---- model / optimizer ----
    model = GPT(cfg).to(device)
    print(f"parameters: {model.num_params():,}", flush=True)
    opt = model.configure_optimizer(0.1, peak, (0.9, 0.95), "cuda")
    step, elapsed_prev = 0, 0.0
    gen = torch.Generator().manual_seed(1337 + args.stage)
    log = []
    out = args.out
    if out:
        os.makedirs(out, exist_ok=True)
    latest = f"{out}/latest.pt" if out else None
    if latest and os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, elapsed_prev = ck["step"], ck.get("elapsed", 0.0)
        gen.set_state(ck["gen"]); total_steps = ck.get("total_steps", total_steps)
        if os.path.exists(f"{out}/log.jsonl"):
            log = [json.loads(l) for l in open(f"{out}/log.jsonl")]
        print(f"RESUMED from {latest} at step {step}/{total_steps}", flush=True)
    elif args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"initialised from {args.init}", flush=True)
    print(f"stage {args.stage}: total_steps={total_steps} tokens/step={tokens_per_step:,} "
          f"tokens={total_steps*tokens_per_step:,} peak_lr={peak}", flush=True)

    fwd = torch.compile(model) if args.compile else model
    sampler = Sampler(sources, T, args.micro, gen)

    def save(tag="periodic"):
        if not out or args.bench:
            return
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                     "elapsed": elapsed_prev + (time.time() - t_start), "gen": gen.get_state(),
                     "total_steps": total_steps, "cfg": cfg.__dict__, "stage": args.stage}, latest)
        print(f"[ckpt] saved step {step} ({tag})", flush=True)

    t_start = time.time()
    last_ckpt = time.time()
    tok_count, t_log = 0, time.time()
    model.train()
    x, y = sampler.batch(device)
    while step < total_steps:
        lr = lr_at(step, total_steps, peak, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = torch.zeros((), device=device)   # sum of micro-batch losses, kept on GPU (no per-micro sync)
        for micro in range(args.accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd(x, y)
            x, y = sampler.batch(device)          # prefetch next batch while GPU works
            (loss / args.accum).backward()
            loss_sum += loss.detach()
        loss_acc = loss_sum.item() / args.accum    # mean training loss over the optimizer step's 16 micro-batches
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_count += tokens_per_step
        if step % 20 == 0:
            torch.cuda.synchronize()
            dt = time.time() - t_log
            tps = tok_count / dt
            print(f"step {step}/{total_steps} loss {loss_acc:.4f} lr {lr:.2e} gnorm {gnorm:.2f} "
                  f"tok/s {tps:,.0f} mem {torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
            tok_count, t_log = 0, time.time()
            if args.bench and time.time() - t_start > args.bench:
                print(f"BENCH tokens/s={tps:.0f}", flush=True)
                return
        if step % args.eval_every == 0 or step == total_steps:
            ev = evaluate(model, val, T, args.micro, 20, device)
            rec = {"step": step, "tokens": step * tokens_per_step, "train_loss": loss_acc,
                   "lr": lr, "elapsed_s": elapsed_prev + time.time() - t_start,
                   **{f"val_{k}": v for k, v in ev.items()}}
            log.append(rec)
            print("EVAL", json.dumps(rec), flush=True)
            if out:
                atomic_write_text(f"{out}/log.jsonl", "".join(json.dumps(r) + "\n" for r in log))
        if time.time() - last_ckpt > args.ckpt_minutes * 60:
            save(); last_ckpt = time.time()
    save("final")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
