"""Phase 3: generate rewrite pairs (Prompt A) and opener/closer pairs (Prompt B) with a local teacher.

The teacher is Qwen2.5-7B-Instruct (Q4_K_M) served by a project-local Ollama on 127.0.0.1:11435
(its model store is $PROJECT_ROOT/.cache/ollama/models on the SSD). Qwen2.5 is Apache-2.0
licensed, which permits using its outputs to train other models.

Every teacher output is passed through the code filter (textutil.filter_rewrite). We NEVER write
pairs by hand; accepted pairs come only from the teacher.

Durability / resumability:
  * Results are buffered and written in chunks: each chunk is written to a .tmp file on the SSD,
    flushed + fsynced, then atomically renamed to data/teacher/<task>/part_NNNNN.jsonl. A dropped
    USB link can lose at most the in-memory buffer, never corrupt an existing part.
  * On start, all existing parts are read and every id already present is skipped.
Usage:
  python teacher.py rewrites [--limit N] [--target-accepted 20000]
  python teacher.py openers
  python teacher.py merge          (writes data/pairs.jsonl and data/openers_closers.jsonl + stats)
"""
import os, sys, json, time, glob, random, argparse, threading, queue, re
import requests

sys.path.insert(0, os.path.dirname(__file__))
from textutil import filter_rewrite

ROOT = os.environ["PROJECT_ROOT"]
HOST = "http://" + os.environ.get("OLLAMA_HOST", "127.0.0.1:11435")
MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5:7b-instruct-q4_K_M")
PROMPT_A = open(f"{ROOT}/prompts/prompt_a.txt").read()
PROMPT_B = open(f"{ROOT}/prompts/prompt_b.txt").read()
CHUNK = 100   # records per part file: small enough to lose little on a crash, few enough files


def call(prompt, num_predict):
    """One teacher call. Returns the raw response text (expected to be one line of JSON)."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",          # constrain decoding to valid JSON
        "stream": False,
        "keep_alive": "2h",
        "options": {
            "temperature": 0.7,    # Phase 3 spec
            "num_ctx": 2048,       # Prompt A is ~1,200 tokens + <=100 output tokens
            "num_predict": num_predict,
        },
    }
    for attempt in range(8):
        try:
            r = requests.post(f"{HOST}/api/chat", json=body, timeout=300)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("teacher unreachable")


class PartWriter:
    """Buffers records and writes them as atomic part files."""
    def __init__(self, d):
        self.d = d
        os.makedirs(d, exist_ok=True)
        self.buf, self.lock = [], threading.Lock()
        self.n = len(glob.glob(f"{d}/part_*.jsonl"))

    def add(self, rec):
        with self.lock:
            self.buf.append(rec)
            if len(self.buf) >= CHUNK:
                self._flush()

    def _flush(self):
        if not self.buf:
            return
        path = f"{self.d}/part_{self.n:05d}.jsonl"
        with open(path + ".tmp", "w") as f:
            for r in self.buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(path + ".tmp", path)
        self.n += 1
        self.buf = []

    def close(self):
        with self.lock:
            self._flush()


def load_done(d):
    done, recs = set(), []
    for p in sorted(glob.glob(f"{d}/part_*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(r["id"]); recs.append(r)
    return done, recs


def parse_json(raw):
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def run_rewrites(args):
    d = f"{ROOT}/data/teacher/rewrites"
    done, recs = load_done(d)
    accepted = sum(1 for r in recs if r.get("ok"))
    rows = [json.loads(l) for l in open(f"{ROOT}/data/wiki_sentences.jsonl")]
    # Hold out the last 400 sentences (by file order, which is already shuffled) from teacher
    # generation entirely: Phase 6 evaluates on 300 of them and Phase 7 on 5 of them, so the model
    # has never seen their teacher rewrites.
    rows = rows[:-400]
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"done={len(done)} accepted={accepted} todo={len(todo)}", flush=True)
    w = PartWriter(d)
    q = queue.Queue()
    for r in todo:
        q.put(r)
    stats = {"n": 0, "ok": 0, "t0": time.time()}
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                r = q.get_nowait()
            except queue.Empty:
                return
            prompt = PROMPT_A.replace("{TOPIC}", r["topic"]).replace("{SENTENCE}", r["src"])
            raw = call(prompt, 160)   # 160 tokens >> 60 words; longer outputs fail the filter anyway
            js = parse_json(raw)
            rec = {"id": r["id"], "topic": r["topic"], "src": r["src"], "raw": raw}
            if js is None:
                rec.update(ok=False, reason="bad_json")
            elif js.get("skip") is True:
                rec.update(ok=False, reason="teacher_skip")
            elif not isinstance(js.get("rewrite"), str):
                rec.update(ok=False, reason="bad_json")
            else:
                reason = filter_rewrite(r["src"], js["rewrite"], max_words=60)
                rec.update(rewrite=js["rewrite"].strip(), ok=reason is None, reason=reason)
            w.add(rec)
            with w.lock:
                stats["n"] += 1
                stats["ok"] += int(rec["ok"])
                if stats["n"] % 50 == 0:
                    el = (time.time() - stats["t0"]) / 60
                    print(f"[{time.strftime('%H:%M:%S')}] calls={stats['n']} ok={stats['ok']} "
                          f"calls/min={stats['n']/el:.1f} accepted/min={stats['ok']/el:.1f} "
                          f"total_accepted={accepted + stats['ok']}", flush=True)
                if accepted + stats["ok"] >= args.target_accepted:
                    stop.set()

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads: t.start()
    for t in threads: t.join()
    w.close()
    print("REWRITES_DONE", flush=True)


def run_openers(args):
    d = f"{ROOT}/data/teacher/openers"
    done, _ = load_done(d)
    rows = [json.loads(l) for l in open(f"{ROOT}/data/wiki_sentences.jsonl")]
    # 2,000 distinct topics, drawn evenly across buckets from the non-held-out sentences.
    rng = random.Random(7)
    by_b = {}
    for r in rows[:-400]:
        by_b.setdefault(r["bucket"], []).append(r["topic"])
    topics, seen = [], set()
    for b in by_b:
        rng.shuffle(by_b[b])
    while len(topics) < 2000 and any(by_b.values()):
        for b in list(by_b):
            while by_b[b]:
                t = by_b[b].pop()
                if t not in seen and 1 <= len(t.split()) <= 5:
                    seen.add(t); topics.append(t); break
    todo = [t for t in topics if "t:" + t not in done]
    print(f"topics={len(topics)} todo={len(todo)}", flush=True)
    w = PartWriter(d)
    q = queue.Queue()
    for t in todo:
        q.put(t)
    cnt = {"n": 0}

    def worker():
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            raw = call(PROMPT_B.replace("{TOPIC}", t), 120)
            js = parse_json(raw) or {}
            rec = {"id": "t:" + t, "topic": t, "raw": raw}
            for k in ("opener", "closer"):
                v = js.get(k)
                if isinstance(v, str):
                    reason = filter_rewrite(t, v, max_words=20, require_facts=False, allow_digits=False)
                    rec[k] = v.strip(); rec[k + "_ok"] = reason is None; rec[k + "_reason"] = reason
                else:
                    rec[k + "_ok"] = False; rec[k + "_reason"] = "bad_json"
            w.add(rec)
            with w.lock:
                cnt["n"] += 1
                if cnt["n"] % 50 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] opener calls={cnt['n']}", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads: t.start()
    for t in threads: t.join()
    w.close()
    print("OPENERS_DONE", flush=True)


def merge(args):
    from collections import Counter
    _, recs = load_done(f"{ROOT}/data/teacher/rewrites")
    hist = Counter(r.get("reason") or "accepted" for r in recs)
    acc = [r for r in recs if r.get("ok")]
    with open(f"{ROOT}/data/pairs.jsonl.tmp", "w") as f:
        for r in acc:
            f.write(json.dumps({"id": r["id"], "topic": r["topic"], "src": r["src"],
                                "rewrite": r["rewrite"]}, ensure_ascii=False) + "\n")
    os.replace(f"{ROOT}/data/pairs.jsonl.tmp", f"{ROOT}/data/pairs.jsonl")
    _, orecs = load_done(f"{ROOT}/data/teacher/openers")
    oh = Counter()
    with open(f"{ROOT}/data/openers_closers.jsonl.tmp", "w") as f:
        for r in orecs:
            for k in ("opener", "closer"):
                oh[f"{k}:{r.get(k + '_reason') or 'accepted'}"] += 1
                if r.get(k + "_ok"):
                    f.write(json.dumps({"kind": k, "topic": r["topic"], "text": r[k]},
                                       ensure_ascii=False) + "\n")
    os.replace(f"{ROOT}/data/openers_closers.jsonl.tmp", f"{ROOT}/data/openers_closers.jsonl")
    print("rewrite calls:", len(recs), "accepted:", len(acc))
    print("rejection histogram:", dict(hist.most_common()))
    print("opener/closer histogram:", dict(oh.most_common()))
    rng = random.Random(0)
    print("\n25 random accepted pairs:")
    for r in rng.sample(acc, min(25, len(acc))):
        print(f"- SRC: {r['src']}\n  OUT: {r['rewrite']}")
    oc = [json.loads(l) for l in open(f"{ROOT}/data/openers_closers.jsonl")]
    print("\n10 random openers/closers:")
    for r in rng.sample(oc, min(10, len(oc))):
        print(f"- [{r['kind']}] {r['topic']}: {r['text']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["rewrites", "openers", "merge"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)   # matches OLLAMA_NUM_PARALLEL=4
    ap.add_argument("--target-accepted", type=int, default=20000)
    a = ap.parse_args()
    {"rewrites": run_rewrites, "openers": run_openers, "merge": merge}[a.task](a)
