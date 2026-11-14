"""Build web/words.json: the 10,000 most frequent lowercase words (letters only, >= 2 chars) in the FineWeb-Edu
tokenizer sample (data/fineweb_tok_sample.txt, ~75 MB). The browser uses it to score how "rare" a sentence's
words are when choosing interesting facts. 10,000 words keeps the file ~80 KB."""
import os, re, json, collections
ROOT = os.environ["PROJECT_ROOT"]
c = collections.Counter()
with open(f"{ROOT}/data/fineweb_tok_sample.txt") as f:
    for line in f:
        c.update(re.findall(r"[a-z]{2,}", line))
words = [w for w, _ in c.most_common(10000)]
json.dump(words, open(f"{ROOT}/web/words.json", "w"), separators=(",", ":"))
print(len(words), words[:10], words[-5:], os.path.getsize(f"{ROOT}/web/words.json"), "bytes")
