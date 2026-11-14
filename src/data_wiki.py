"""Phase 2.3: sample source sentences for the rewrite task from English Wikipedia lead sections.

Pipeline:
  1. Stream `wikimedia/wikipedia` (config 20231101.en) from the Hugging Face Hub (no raw download to
     the slow SSD). The first parquet shard holds the lowest page ids, i.e. the oldest, most
     general articles, which is exactly the broad, "encyclopedic" spread we want.
  2. Extract the lead section (paragraphs before the first section heading), split it into
     sentences, clean them, keep 8-40 word sentences.
  3. For candidate articles, fetch non-hidden categories from the Wikipedia API (50 page ids per
     request). Drop articles in Category:Living people or any SKIP category (Appendix A).
  4. Bucket articles into topic groups (animals, science, space, history, technology, food,
     places, objects, the human body, art) by category keywords, and cap each bucket so the final
     set is spread across topics.
Outputs (all lowercased text stays alongside the original case, because the code filter needs the
original-case source to find capitalised names):
  data/wiki_sentences.jsonl   one {"id","title","topic","bucket","src"} per line, ~40,000 lines
  data/wiki_leads_sample.txt  extra lead text (lowercased) for the tokenizer training mix
"""
import os, re, json, time, random, sys
import requests
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from textutil import split_sentences, word_count, is_skip_categories

ROOT = os.environ["PROJECT_ROOT"]
OUT = f"{ROOT}/data/wiki_sentences.jsonl"
LEADS_OUT = f"{ROOT}/data/wiki_leads_sample.txt"
TARGET = 40_000              # candidate sentences wanted (Phase 2.3 target)
PER_BUCKET = TARGET // 10    # 4,000 per topic bucket so no single topic dominates
MAX_PER_ARTICLE = 4          # at most 4 sentences from one article, for topic diversity
API = "https://en.wikipedia.org/w/api.php"
# Generic client name (no personal details), as Wikimedia asks automated clients to identify.
HEADERS = {"User-Agent": "hollow-dataset-builder/0.1 (educational from-scratch LM project)"}

BUCKETS = {
    "animals": r"animal|mammal|\bbirds?\b|\bfish|insect|reptile|amphibian|spider|arachnid|fauna|"
               r"cephalopod|mollus|crustacean|cetacean|primate|rodent|carnivoran|ungulate|dinosaur|"
               r"shark|snake|beetle|butterfl|moth|\bant\b|\bbee|whale|\bcats?\b|\bdogs?\b|horse",
    "science": r"physic|chemi|biolog|mathemat|geolog|mineral|chemical element|scientific|botan|"
               r"\bplants?\b|ecology|meteorolog|weather|climate|evolution|genetic|bacteri|virus|"
               r"fungi|\bcells?\b|molecul|\batoms?\b|energy|optics|thermodynamic",
    "space": r"astronom|planet|solar system|\bstars?\b|galax|nebula|comet|asteroid|\bmoons?\b|"
             r"outer space|spaceflight|cosmolog|constellation|space probe|satellite",
    "history": r"history|ancient|century|empire|dynast|archaeolog|medieval|middle ages|bronze age|"
               r"iron age|kingdom|civili[sz]ation|pharaoh|prehistor|roman|viking|egypt",
    "technology": r"technolog|computing|computer|software|engineering|invention|machine|vehicle|"
                  r"aircraft|automobile|electronic|internet|telecommunication|robot|programming|"
                  r"locomotive|ships? type|bridges|clock",
    "food": r"\bfoods?\b|cuisine|\bdrinks?\b|beverage|dishes|fruit|vegetable|spice|herb|cheese|"
            r"bread|dessert|\btea\b|coffee|wine|beer|cooking|edible|condiment|nuts",
    "places": r"cities|towns|countries|rivers|mountains|islands|lakes|volcano|deserts|national park|"
              r"geography|capitals|\bseas\b|oceans|landforms|caves|glacier|waterfalls|world heritage",
    "objects": r"\btools?\b|household|furniture|containers|toys|\bgames\b|clothing|jewel|"
               r"musical instrument|materials|textiles|gemstones|timekeeping|\bpaper\b|glass|"
               r"\bmetals?\b|currenc|\bcoins?\b|lighting",
    "body": r"anatomy|human body|\borgans?\b|physiology|nervous system|skelet|muscle|circulatory|"
            r"\bbrain\b|\bsenses?\b|\bskin\b|digestive|\bteeth\b|\beyes?\b|immune|\bsleep\b|"
            r"hormone|respirat|neuroscience",
    "art": r"\barts?\b|painting|sculpture|\bmusic|literature|architecture|poetry|theatre|dance|"
           r"novels|mytholog|folklore|legend|\bfilms?\b|photograph|opera|calligraphy|ballet",
}
BUCKET_RE = {k: re.compile(v, re.IGNORECASE) for k, v in BUCKETS.items()}
# Sentence-initial pronouns that dangle once a sentence is lifted out of context.
DANGLING = re.compile(r"^(he|she|they|this|these|those|his|her|their|its|such|both|another|"
                      r"however|also|thus|then|there|here|which)\b", re.IGNORECASE)


def lead_paragraphs(text):
    """Return the lead section: paragraphs before the first heading-like line.

    In this dataset, section headings are short standalone paragraphs without terminal
    punctuation (e.g. "History"). We stop at the first paragraph under 100 characters that does
    not end in . ? ! or :.
    """
    out = []
    for p in text.split("\n"):
        p = p.strip()
        if not p:
            continue
        if len(p) < 100 and not re.search(r"[.?!:]\"?$", p):
            break
        out.append(p)
    return out


def clean_sentence(s, title):
    """Clean one sentence. Returns the cleaned sentence or None if it should be dropped."""
    # Drop sentences whose parentheses carry digits (dates/figures inside brackets make the
    # fact-preservation check fragile) or that have more than one parenthetical (heavy parens).
    parens = re.findall(r"\([^()]*\)", s)
    if len(parens) > 1 or any(re.search(r"\d", p) for p in parens):
        return None
    s = re.sub(r"\s*\([^()]*\)", "", s)          # remove the single digit-free parenthetical
    if "(" in s or ")" in s or "[" in s or ";" in s:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.])", r"\1", s)
    if not s.endswith("."):
        return None
    if not re.fullmatch(r"[A-Za-z0-9À-ɏ ,.'’\-–:%$]+", s):
        return None                                 # odd characters (IPA, quotes, symbols)
    if re.search(r"may refer to|is a list of|disambiguation", s, re.IGNORECASE):
        return None
    if DANGLING.match(s):
        return None
    # Runtime rule (Phase 8.5), mirrored here: a leading "It" is replaced with the article title.
    if re.match(r"^It\b", s):
        s = title + s[2:]
    n = word_count(s)
    if n < 8 or n > 40:
        return None
    return s


def fetch_categories(pageids):
    """Fetch non-hidden categories for up to 50 page ids. Returns {pageid: [category names]}."""
    cats = {pid: [] for pid in pageids}
    params = {"action": "query", "format": "json", "prop": "categories", "clshow": "!hidden",
              "cllimit": "max", "pageids": "|".join(map(str, pageids)), "formatversion": "2"}
    while True:
        for attempt in range(6):
            try:
                r = requests.get(API, params=params, headers=HEADERS, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:  # network blip: back off exponentially (1, 2, 4, ... seconds)
                time.sleep(2 ** attempt)
        else:
            return cats
        for page in data.get("query", {}).get("pages", []):
            pid = page.get("pageid")
            if pid in cats:
                cats[pid].extend(c["title"] for c in page.get("categories", []))
        if "continue" in data:
            params.update(data["continue"])   # cllimit=max is shared across pages; follow it
        else:
            return cats


STATE = f"{ROOT}/data/wiki_state.json"


def main():
    """Resumable: every 2,000 candidate articles the full sampler state (buckets, stats, next id, articles consumed,
    bytes of lead text written) is saved atomically to data/wiki_state.json. On restart the stream skips the
    consumed articles and the lead-text file is truncated back to the recorded length. (Added after a system-wide
    OOM on 2026-11-14 killed the first run 23k articles in, losing everything because output was only written at the end.)"""
    random.seed(1234)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    by_bucket = {k: [] for k in BUCKETS}
    pending = []          # (pageid, title, [sentences]) waiting for a category lookup
    stats = {"articles": 0, "candidates": 0, "living": 0, "skip": 0, "no_bucket": 0}
    sid, leads_chars, leads_bytes = 0, 0, 0
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        by_bucket, stats, sid, leads_chars, leads_bytes = (st["by_bucket"], st["stats"], st["sid"],
                                                            st["leads_chars"], st["leads_bytes"])
        print("RESUMED", json.dumps(stats), flush=True)
    leads_f = open(LEADS_OUT + ".tmp", "ab")
    leads_f.truncate(leads_bytes); leads_f.seek(leads_bytes)
    ds = ds.skip(stats["articles"])

    def save_state():
        leads_f.flush(); os.fsync(leads_f.fileno())
        st = {"by_bucket": by_bucket, "stats": stats, "sid": sid, "leads_chars": leads_chars,
              "leads_bytes": leads_f.tell()}
        with open(STATE + ".tmp", "w") as f:
            json.dump(st, f); f.flush(); os.fsync(f.fileno())
        os.replace(STATE + ".tmp", STATE)

    def flush(batch):
        nonlocal sid
        cats = fetch_categories([b[0] for b in batch])
        for pid, title, sents in batch:
            reason = is_skip_categories(cats.get(pid, []))
            if reason:
                stats[reason] += 1
                continue
            joined = " | ".join(cats.get(pid, []))
            hits = [k for k, rx in BUCKET_RE.items() if rx.search(joined)]
            # Assign to the eligible bucket that is currently least full, to balance topics.
            hits = [k for k in hits if len(by_bucket[k]) < PER_BUCKET]
            if not hits:
                stats["no_bucket"] += 1
                continue
            k = min(hits, key=lambda b: len(by_bucket[b]))
            topic = re.sub(r"\s*\(.*\)$", "", title).lower()
            for s in sents[:MAX_PER_ARTICLE]:
                by_bucket[k].append({"id": f"w{sid:06d}", "title": title, "topic": topic,
                                     "bucket": k, "src": s, "pageid": pid})
                sid += 1

    for ex in ds:
        stats["articles"] += 1
        title, text = ex["title"], ex["text"]
        lead = lead_paragraphs(text)
        if not lead:
            continue
        if leads_chars < 12_000_000:   # ~12 MB of lead text for the tokenizer's Wikipedia share
            for p in lead:
                leads_f.write((p.lower() + "\n").encode("utf-8"))
                leads_chars += len(p) + 1
        sents = []
        for p in lead:
            for s in split_sentences(p):
                c = clean_sentence(s, title)
                if c:
                    sents.append(c)
        if len(sents) < 1:
            continue
        stats["candidates"] += 1
        pending.append((int(ex["id"]), title, sents))
        if len(pending) == 50:
            flush(pending)
            pending = []
            total = sum(len(v) for v in by_bucket.values())
            if stats["candidates"] % 2000 == 0:
                save_state()
                print(json.dumps(stats), {k: len(v) for k, v in by_bucket.items()}, total, flush=True)
            if total >= TARGET or all(len(v) >= PER_BUCKET for v in by_bucket.values()):
                break
            # Operator switch: `touch data/wiki_STOP` finalizes with what has been collected so far (used if the
            # rare buckets would keep the scan running past the point where the teacher needs the file).
            if os.path.exists(f"{ROOT}/data/wiki_STOP"):
                print("STOP file found, finalizing", flush=True)
                break
            # Stop scanning after 1.2M articles even if some buckets are short: the remaining
            # buckets are then filled proportionally by what was found.
            if stats["articles"] > 1_200_000:
                break
            time.sleep(0.1)   # be polite to the Wikipedia API (at most ~10 requests per second)
    leads_f.close()
    os.replace(LEADS_OUT + ".tmp", LEADS_OUT)
    rows = [r for v in by_bucket.values() for r in v]
    random.shuffle(rows)
    with open(OUT + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(OUT + ".tmp", OUT)   # atomic rename: a dropped USB link never leaves a half file
    print("FINAL", json.dumps(stats), {k: len(v) for k, v in by_bucket.items()}, len(rows))


if __name__ == "__main__":
    main()
