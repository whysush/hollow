"""Phase 6, stage 3: supervised fine-tuning on teacher pairs, and the held-out evaluation.

Example formats (exactly as the plan writes them, tokenized with special tokens split out):
  rewrites: "<|src|> {source} <|sty|> {rewrite} <|end|>"
  openers:  "<|opener|> {topic} <|sty|> {opener} <|end|>"
  closers:  "<|closer|> {topic} <|sty|> {closer} <|end|>"
The source is the lowercased Wikipedia sentence (the browser sends lowercased sentences too).
Loss is computed only on target tokens AFTER <|sty|> (the styled text, the trailing space and <|end|>).
Because special tokens split the text into separately pre-tokenized segments, the token ids of the
prompt "<|src|> {source} <|sty|>" are an exact prefix of the full example's ids; generation in the browser
feeds that same prompt string.

  python sft.py train --init ckpt/stage2/latest.pt --out ckpt/sft --epochs 4
  python sft.py eval  --ckpt ckpt/sft/best.pt --n 300
"""
import os, sys, json, math, time, random, argparse
import torch
import torch.nn.functional as Fn
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import GPT, GPTConfig
from textutil import filter_rewrite
from train import atomic_save, atomic_write_text, lr_at

ROOT = os.environ["PROJECT_ROOT"]
TOK = Tokenizer.from_file(f"{ROOT}/tokenizer/tokenizer.json")
SID = {s: TOK.token_to_id(s) for s in ["<|endoftext|>", "<|src|>", "<|sty|>", "<|end|>", "<|opener|>", "<|closer|>"]}
PAD = SID["<|endoftext|>"]


def example_text(kind, prompt, target):
    tag = {"src": "<|src|>", "opener": "<|opener|>", "closer": "<|closer|>"}[kind]
    return f"{tag} {prompt} <|sty|> {target} <|end|>"


def prompt_text(kind, prompt):
    tag = {"src": "<|src|>", "opener": "<|opener|>", "closer": "<|closer|>"}[kind]
    return f"{tag} {prompt} <|sty|>"


def encode_example(kind, prompt, target):
    """Returns (ids, mask) where mask[i] = 1 if ids[i] is a target token (after <|sty|>)."""
    ids = TOK.encode(example_text(kind, prompt, target)).ids
    k = ids.index(SID["<|sty|>"])
    mask = [0] * (k + 1) + [1] * (len(ids) - k - 1)
    return ids, mask


def load_examples():
    rows = []
    for l in open(f"{ROOT}/data/pairs.jsonl"):
        r = json.loads(l)
        rows.append(("src", r["src"].lower(), r["rewrite"], r["id"]))
    for l in open(f"{ROOT}/data/openers_closers.jsonl"):
        r = json.loads(l)
        rows.append((r["kind"], r["topic"], r["text"], f"{r['kind']}:{r['topic']}"))
    return rows


def batches(data, B, shuffle, rng):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), B):
        chunk = [data[j] for j in idx[i:i + B]]
        T = max(len(c[0]) for c in chunk)
        x = torch.full((len(chunk), T), PAD, dtype=torch.long)
        m = torch.zeros((len(chunk), T), dtype=torch.float)
        for r, (ids, mask) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
            m[r, :len(mask)] = torch.tensor(mask, dtype=torch.float)
        # Next-token targets: predict ids[t+1] from ids[:t+1]; the mask moves with the target position.
        yield x[:, :-1].cuda(), x[:, 1:].cuda(), m[:, 1:].cuda()


@torch.no_grad()
def val_loss(model, data, B):
    model.eval()
    tot = n = 0.0
    for x, y, m in batches(data, B, False, None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, loss_mask=m)
        w = m.sum().item()
        tot += loss.item() * w; n += w
    model.train()
    return tot / max(n, 1)


def train(a):
    rng = random.Random(0)
    rows = load_examples()
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * 0.05))                  # 5% held out for validation
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    enc = lambda rs: [encode_example(k, p, t) for k, p, t, _ in rs]
    tr, va = enc(train_rows), enc(val_rows)
    tr = [e for e in tr if len(e[0]) <= 512]; va = [e for e in va if len(e[0]) <= 512]
    print(f"train examples {len(tr):,}  val examples {len(va):,}  "
          f"target tokens/epoch {sum(sum(m) for _, m in tr):,}", flush=True)
    ck = torch.load(a.init, map_location="cuda", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).cuda()
    model.load_state_dict(ck["model"])
    opt = model.configure_optimizer(0.1, a.lr, (0.9, 0.95), "cuda")
    steps_per_epoch = math.ceil(len(tr) / a.batch)
    total = int(steps_per_epoch * a.epochs)
    os.makedirs(a.out, exist_ok=True)
    log, best, step = [], float("inf"), 0
    v0 = val_loss(model, va, 64)
    print(f"initial val loss {v0:.4f}", flush=True)
    log.append({"step": 0, "val_loss": v0})
    t0 = time.time()
    ep = 0
    while step < total:
        for x, y, m in batches(tr, a.batch, True, rng):
            lr = lr_at(step, total, a.lr, a.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, loss_mask=m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} lr {lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
            if step >= total:
                break
        ep += 1
        v = val_loss(model, va, 64)
        rec = {"step": step, "epoch": ep, "val_loss": v, "elapsed_s": time.time() - t0}
        log.append(rec)
        print("EVAL", json.dumps(rec), flush=True)
        state = {"model": model.state_dict(), "cfg": cfg.__dict__, "step": step, "epoch": ep, "val_loss": v}
        atomic_save(state, f"{a.out}/latest.pt")
        if v < best:
            best = v
            atomic_save(state, f"{a.out}/best.pt")
            print(f"[ckpt] best val {v:.4f} at epoch {ep}", flush=True)
        atomic_write_text(f"{a.out}/log.jsonl", "".join(json.dumps(r) + "\n" for r in log))
    print("SFT_DONE best_val", best, flush=True)


@torch.no_grad()
def candidates(model, kind, prompt, n, temperature, top_p, max_new=96):
    """Sample n candidates for one prompt. Returns [(text, mean_nll)]."""
    ids = TOK.encode(prompt_text(kind, prompt)).ids
    x = torch.tensor([ids] * n, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, logp = model.generate(x, max_new, temperature=temperature, top_p=top_p, stop_id=SID["<|end|>"])
    out = []
    for r in range(n):
        gen = y[r, len(ids):].tolist()
        lp = logp[r].tolist()
        if SID["<|end|>"] in gen:
            k = gen.index(SID["<|end|>"])
            gen, lp = gen[:k], lp[:k + 1]
        else:
            gen = None   # never reached <|end|>: treat as a failed candidate
        text = TOK.decode(gen).strip() if gen is not None else ""
        out.append((text, -sum(lp) / max(len(lp), 1)))
    return out


def evaluate(a):
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    model = GPT(GPTConfig(**ck["cfg"])).cuda()
    model.load_state_dict(ck["model"]); model.eval()
    rows = [json.loads(l) for l in open(f"{ROOT}/data/wiki_sentences.jsonl")]
    held = rows[-400:][:a.n]          # never sent to the teacher (teacher.py holds out the last 400)
    torch.manual_seed(0)
    from collections import Counter
    passed, lower_ok, total_c, reasons, examples = 0, 0, 0, Counter(), []
    t0 = time.time()
    for i, r in enumerate(held):
        cs = candidates(model, "src", r["src"].lower(), 4, a.temperature, a.top_p)
        ok = []
        for text, nll in cs:
            total_c += 1
            lower_ok += int(text == text.lower())
            why = filter_rewrite(r["src"], text, max_words=60)
            reasons[why or "pass"] += 1
            if why is None:
                ok.append((nll, text))
        passed += int(bool(ok))
        best = min(ok)[1] if ok else None
        examples.append({"src": r["src"], "best": best, "cands": [c[0] for c in cs]})
    rate = passed / len(held)
    res = {"sources": len(held), "pass_rate": rate, "lowercase_rate": lower_ok / total_c,
           "candidate_reasons": dict(reasons.most_common()), "seconds": time.time() - t0,
           "temperature": a.temperature, "top_p": a.top_p}
    print(json.dumps(res, indent=1))
    # Openers / closers on 20 held-out topics: no-digits rule, 20 words.
    oc = []
    for r in held[:20]:
        for kind in ("opener", "closer"):
            cs = candidates(model, kind, r["topic"], 4, a.temperature, a.top_p, max_new=48)
            ok = sorted((n, t) for t, n in cs if filter_rewrite(r["topic"], t, max_words=20, require_facts=False, allow_digits=False) is None)
            oc.append({"kind": kind, "topic": r["topic"], "best": ok[0][1] if ok else None})
    res["opener_closer_pass_rate"] = sum(o["best"] is not None for o in oc) / len(oc)
    print("opener/closer pass rate", res["opener_closer_pass_rate"])
    out = a.out or os.path.dirname(a.ckpt)
    atomic_write_text(f"{out}/eval.json", json.dumps({"summary": res, "examples": examples, "openers_closers": oc}, indent=1, ensure_ascii=False))
    print("\n20 side-by-side examples:")
    for e in examples[:20]:
        print(f"- SRC: {e['src']}\n  OUT: {e['best'] or '(no passing candidate) ' + e['cands'][0]}")
    print("\nopeners/closers:")
    for o in oc[:12]:
        print(f"- [{o['kind']}] {o['topic']}: {o['best']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--init"); ap.add_argument("--out"); ap.add_argument("--ckpt")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
