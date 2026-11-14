"""Phase 2 acceptance: counts and 10 random samples from each source, and a boilerplate scan of the gothic corpus.

  python data_check.py [fineweb|gothic|wiki|all]
"""
import os, sys, re, glob, json, random
ROOT = os.environ["PROJECT_ROOT"]
rng = random.Random(0)
which = sys.argv[1] if len(sys.argv) > 1 else "all"


def snippet(text, n=220):
    o = rng.randrange(max(1, len(text) - n))
    return text[o:o + n].replace("\n", " / ")


if which in ("fineweb", "all"):
    import numpy as np
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{ROOT}/tokenizer/tokenizer.json")
    a = np.memmap(f"{ROOT}/data/tokens/fineweb_train.bin", dtype=np.uint16, mode="r")
    prog = json.load(open(f"{ROOT}/data/tokens/fineweb.progress.json"))
    print(f"== FineWeb-Edu: {prog['docs']:,} documents, {len(a):,} train tokens + {prog['val']:,} val tokens")
    for _ in range(10):
        o = rng.randrange(len(a) - 60)
        print("  -", repr(tok.decode(a[o:o + 50].tolist())))

if which in ("gothic", "all"):
    files = sorted(glob.glob(f"{ROOT}/data/gothic/*.txt"))
    man = json.load(open(f"{ROOT}/data/gothic_manifest.json"))
    inc = {k: v for k, v in man.items() if v.get("included")}
    words = sum(v["words"] for v in inc.values())
    print(f"\n== Gothic corpus: {len(files)} books, {words:,} words")
    # Boilerplate scan: any of these anywhere in the corpus fails the check.
    pat = re.compile(r"project gutenberg|gutenberg\.org|gutenberg-tm|\bebook\b|e-text|transcriber's note|"
                     r"produced by [^.]{0,60}(proofread|online|scann|volunteer)|distributed proofread|www\.|http",
                     re.IGNORECASE)   # "produced by" alone is ordinary prose; only the credit-line form counts
    hits = []
    for p in files:
        t = open(p).read()
        for m in pat.finditer(t):
            hits.append((os.path.basename(p), t[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")))
    print(f"boilerplate hits: {len(hits)}")
    for h in hits[:10]:
        print("  !", h)
    upper = sum(1 for p in files if open(p).read() != open(p).read().lower())
    print(f"files with uppercase characters: {upper}")
    for p in rng.sample(files, 10):
        print(f"  - [{inc.get(os.path.basename(p)[:-4], {}).get('title', '?')[:40]}] {snippet(open(p).read())}")

if which in ("wiki", "all"):
    rows = [json.loads(l) for l in open(f"{ROOT}/data/wiki_sentences.jsonl")]
    from collections import Counter
    print(f"\n== Wikipedia source sentences: {len(rows):,}  by bucket: {dict(Counter(r['bucket'] for r in rows))}")
    lens = [len(r["src"].split()) for r in rows]
    print(f"words per sentence: min {min(lens)} max {max(lens)} mean {sum(lens)/len(lens):.1f}")
    for r in rng.sample(rows, 10):
        print(f"  - [{r['bucket']}/{r['title']}] {r['src']}")
