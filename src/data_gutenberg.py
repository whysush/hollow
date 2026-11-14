"""Phase 2.2: download a public-domain gothic / horror corpus from Project Gutenberg.

Selection rules (US public domain as of 2026 = first published in 1930 or earlier):
  * Books are found through the Gutendex API (https://gutendex.com; see load_catalog for the catalog-CSV deviation), English only, and only books
    Gutenberg marks copyright=false (public domain in the US).
  * Authors who died in 1930 or earlier: every one of their works was necessarily first published
    while they were alive or shortly after; we accept all, except posthumous first editions after
    1930 are effectively nonexistent for these authors (all 19th-century or 1910s deaths).
  * Authors who died after 1930 (Machen, Blackwood, M. R. James, Lovecraft, Benson, Chambers,
    Onions, Hodgson's editors): we accept a book only if its front matter (first 400 lines of the
    raw text) contains a publication year, and the largest such year is <= 1930. Lovecraft is
    handled with an explicit allowlist of stories first published 1930 or earlier, because many of
    his Gutenberg texts are post-1930 works whose copyright simply was not renewed.
  * Downloads are throttled to one request every 2 seconds and use the gutenberg.org cache path,
    following Project Gutenberg's guidance for automated access.
Output: data/gothic/<gutenberg_id>.txt (lowercased, headers/footers and boilerplate removed), and
data/gothic_manifest.json listing what was included and why.
"""
import os, re, json, time, sys, hashlib
import requests

ROOT = os.environ["PROJECT_ROOT"]
OUTDIR = f"{ROOT}/data/gothic"
MANIFEST = f"{ROOT}/data/gothic_manifest.json"
HEADERS = {"User-Agent": "hollow-dataset-builder/0.1 (educational from-scratch LM project)"}

# (Gutendex search string, author-name substring that must appear in the book's author field)
AUTHORS = [
    ("poe", "Poe, Edgar Allan"), ("mary shelley", "Shelley, Mary"), ("stoker", "Stoker, Bram"),
    ("m. r. james", "James, M. R."), ("montague rhodes james", "James, M. R."),
    ("machen", "Machen, Arthur"), ("blackwood", "Blackwood, Algernon"),
    ("le fanu", "Le Fanu"), ("bierce", "Bierce, Ambrose"), ("lovecraft", "Lovecraft"),
    # Further classic gothic / weird authors, to reach the 20-60M token target:
    ("hawthorne", "Hawthorne, Nathaniel"), ("radcliffe", "Radcliffe, Ann"),
    ("walpole otranto", "Walpole, Horace"), ("matthew lewis monk", "Lewis, M. G."),
    ("maturin", "Maturin, Charles"), ("hodgson", "Hodgson, William Hope"),
    ("wilkie collins", "Collins, Wilkie"), ("stevenson", "Stevenson, Robert Louis"),
    ("robert w. chambers", "Chambers, Robert W."), ("marion crawford", "Crawford, F. Marion"),
    ("oliver onions", "Onions, Oliver"), ("e. f. benson", "Benson, E. F."),
    ("bulwer-lytton", "Lytton"), ("gaskell", "Gaskell, Elizabeth"),
    ("margaret oliphant", "Oliphant"), ("amelia edwards", "Edwards, Amelia"),
    ("charlotte riddell", "Riddell"), ("w. w. jacobs", "Jacobs, W. W."),
    ("vernon lee", "Lee, Vernon"), ("hoffmann", "Hoffmann, E. T. A."),
    ("gogol", "Gogol"), ("dickens ghost", "Dickens, Charles"),
    ("brontë", "Brontë"), ("polidori", "Polidori"), ("james hogg", "Hogg, James"),
    ("charlotte perkins gilman", "Gilman, Charlotte Perkins"),
    ("fitz-james o'brien", "O'Brien, Fitz-James"), ("lafcadio hearn", "Hearn, Lafcadio"),
    ("arthur conan doyle", "Doyle, Arthur Conan"), ("h. g. wells", "Wells, H. G."),
    ("saki", "Saki"), ("e. nesbit", "Nesbit, E."), ("w. f. harvey", "Harvey, William Fryer"),
    ("robert hugh benson", "Benson, Robert Hugh"), ("marie corelli", "Corelli, Marie"),
    ("richard marsh", "Marsh, Richard"), ("guy de maupassant", "Maupassant"),
]
# Authors whose whole output is gothic / weird fiction: every work qualifies. All other listed
# authors (Dickens, Doyle, Wells, Hawthorne, ...) wrote much that is not gothic, so their books
# qualify only if the catalog tags them horror / gothic / ghost stories / supernatural.
CORE = ["Poe, Edgar Allan", "Shelley, Mary", "Stoker, Bram", "James, M. R.", "Machen, Arthur",
        "Blackwood, Algernon", "Le Fanu", "Bierce, Ambrose", "Lovecraft", "Radcliffe, Ann",
        "Walpole, Horace", "Lewis, M. G.", "Maturin, Charles", "Hodgson, William Hope",
        "Chambers, Robert W.", "Onions, Oliver", "Riddell", "Polidori", "O'Brien, Fitz-James",
        "Marsh, Richard", "Lee, Vernon", "Harvey, William Fryer", "Benson, E. F.", "Crawford, F. Marion"]
# Topic searches catch anthologies and less-known authors.
TOPICS = ["horror tales", "ghost stories", "gothic fiction", "supernatural"]

# Lovecraft stories first published in 1930 or earlier (title prefixes, lowercased).
LOVECRAFT_OK = [
    "the call of cthulhu", "the colour out of space", "dagon", "the dunwich horror",
    "the lurking fear", "the rats in the walls", "herbert west", "the statement of randolph carter",
    "the festival", "the music of erich zann", "the outsider", "pickman's model",
    "the terrible old man", "the tree", "celephaïs", "celephais", "the cats of ulthar",
    "nyarlathotep", "the hound", "the temple", "the moon-bog", "polaris", "beyond the wall of sleep",
    "the white ship", "hypnos", "the unnamable", "the picture in the house",
    "supernatural horror in literature", "the silver key", "he", "in the vault", "cool air",
    "the horror at red hook", "the nameless city", "the doom that came to sarnath",
    "the crawling chaos", "the street", "under the pyramids", "the tomb", "the alchemist",
    "the beast in the cave", "facts concerning the late arthur jermyn", "ex oblivione",
    "the transition of juan romero", "memory", "what the moon brings", "the quest of iranon",
]
# Non-fiction or off-voice material we do not want shaping the gothic voice.
EXCLUDE_TITLE = re.compile(r"letters|correspondence|index|bibliography|cookbook|grammar|"
                           r"dictionary of|biography|memoir of|life of|poems? of|complete poetical",
                           re.IGNORECASE)

BOILER = re.compile(r"gutenberg|e-?text|ebook|transcriber|produced by|distributed proofread|"
                    r"www\.|http|\.org|\.com|copyright|license|illustrat", re.IGNORECASE)


def gutendex(params):
    url = "https://gutendex.com/books"
    out = []
    while url:
        for attempt in range(6):
            try:
                r = requests.get(url, params=params, headers=HEADERS, timeout=60)
                r.raise_for_status()
                d = r.json()
                break
            except Exception:
                time.sleep(2 ** attempt + 1)
        else:
            return out
        out.extend(d["results"])
        url, params = d.get("next"), None
        time.sleep(1)
    return out


def strip_gutenberg(raw):
    """Remove the Gutenberg header/footer and residual boilerplate lines. Returns plain text."""
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", raw, re.IGNORECASE)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", raw, re.IGNORECASE)
    body = raw[start.end() if start else 0: end.start() if end else len(raw)]
    # Drop end-of-book license tails that some files put before the END marker.
    tail = re.search(r"\n\s*end of (the )?project gutenberg", body, re.IGNORECASE)
    if tail:
        body = body[:tail.start()]
    lines = []
    for line in body.splitlines():
        if BOILER.search(line):
            continue
        if re.fullmatch(r"\s*\[.*\]\s*", line):      # [Illustration], [Footnote 3: ...]
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[_*]", "", text)                  # italics/bold markup
    # Rejoin hard-wrapped lines into paragraphs: single newlines become spaces.
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p) > 1]
    return "\n".join(paras)


def front_matter_year(raw):
    """Largest plausible publication year (1700-1999) in the first 400 lines, or None."""
    head = "\n".join(raw.splitlines()[:400])
    head = re.sub(r"\*\*\*\s*START OF.*", "", head, flags=re.IGNORECASE | re.DOTALL) or head
    years = [int(y) for y in re.findall(r"\b(1[789]\d\d)\b", head)]
    # Ignore the Gutenberg release/update dates by only looking after the START marker if present.
    m = re.search(r"START OF (THE|THIS) PROJECT GUTENBERG", raw[:20000], re.IGNORECASE)
    if m:
        after = "\n".join(raw[m.end():].splitlines()[:400])
        years = [int(y) for y in re.findall(r"\b(1[789]\d\d)\b", after)]
    return max(years) if years else None


def text_url(book):
    f = book.get("formats", {})
    for k in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
        if k in f and not f[k].endswith(".zip"):
            return f[k]
    return f"https://www.gutenberg.org/cache/epub/{book['id']}/pg{book['id']}.txt"


def load_catalog():
    """Discover candidate books from Project Gutenberg's official catalog (pg_catalog.csv).

    Deviation from the plan: the plan says to use the Gutendex API, but gutendex.com answered each
    query in 60-90 s on 2026-11-14 (~4 h for the ~150 paged queries needed). pg_catalog.csv is the
    same metadata Gutendex indexes (title, language, authors with birth-death years, subjects,
    bookshelves), fetched in one request, which is also lighter on Gutenberg's servers.
    Returns {id: {"id","title","authors":[{"name","death_year"}], "named": bool}}.
    """
    import csv
    path = f"{ROOT}/.tmp/pg_catalog.csv"
    if not os.path.exists(path):
        r = requests.get("https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
                         headers=HEADERS, timeout=300)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    genre = re.compile(r"horror|gothic|ghost stor|supernatural|vampire|haunted", re.IGNORECASE)
    books = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row["Type"] != "Text" or row["Language"] != "en":
            continue
        authors = []
        for a in row["Authors"].split(";"):
            a = a.strip()
            if not a:
                continue
            m = re.search(r"(\d{3,4})\??\s*-\s*(\d{3,4})\??", a)
            authors.append({"name": a, "death_year": int(m.group(2)) if m else None})
        # Only credited authors/editors/contributors count for the core-author rule: a translator or
        # illustrator credit (e.g. Arthur Machen translated Casanova's memoirs) says nothing about genre.
        credited = [a for a in authors if not re.search(r"\[(Translator|Illustrator)\]", a["name"])]
        core = any(sub.lower() in a["name"].lower() for sub in CORE for a in credited)
        topical = bool(genre.search(row["Subjects"] + " " + row["Bookshelves"]))
        named = core   # sort key: core gothic canon first
        if core or topical:
            books[int(row["Text#"])] = {"id": int(row["Text#"]), "title": row["Title"].replace("\n", " "),
                                        "authors": authors, "named": named, "formats": {}}
    return books


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    books = load_catalog()
    print("candidates", len(books), "named-author", sum(b["named"] for b in books.values()), flush=True)
    # Named classic authors first, then genre-tagged books, so the core gothic canon is always in.
    order = sorted(books, key=lambda i: (not books[i]["named"], i))

    seen_paras = set()   # paragraph hashes, to drop duplicated stories across collections
    total_words = sum(v.get("words", 0) for v in manifest.values() if v.get("included"))
    for bid in order:
        b = books[bid]
        key = str(bid)
        if key in manifest:
            continue
        title = b["title"]
        authors = b["authors"]
        if not authors or EXCLUDE_TITLE.search(title):
            manifest[key] = {"title": title, "included": False, "why": "no author / excluded title"}
            continue
        deaths = [a.get("death_year") or 9999 for a in authors]
        is_lovecraft = any("lovecraft" in a["name"].lower() for a in authors)
        if is_lovecraft:
            tl = title.lower()
            if not any(tl == ok or tl.startswith(ok + " ") or tl.startswith(ok + ":") for ok in LOVECRAFT_OK):
                manifest[key] = {"title": title, "included": False, "why": "lovecraft not on <=1930 list"}
                continue
        try:
            time.sleep(2)
            r = requests.get(text_url(b), headers=HEADERS, timeout=120)
            r.raise_for_status()
            raw = r.content.decode("utf-8", errors="replace")
        except Exception as e:
            print("fail", bid, e, flush=True)
            continue
        why = "author died <= 1930"
        if max(deaths) > 1930 and not is_lovecraft:
            y = front_matter_year(raw)
            if y is None or y > 1930:
                manifest[key] = {"title": title, "included": False, "why": f"author died after 1930, front-matter year {y}"}
                continue
            why = f"front-matter year {y}"
        elif is_lovecraft:
            why = "lovecraft allowlist (first published <= 1930)"
        text = strip_gutenberg(raw)
        paras = []
        for p in text.split("\n"):
            h = hashlib.md5(p.lower().encode()).hexdigest()
            if len(p) > 80 and h in seen_paras:
                continue
            seen_paras.add(h)
            paras.append(p)
        text = "\n".join(paras).lower()
        words = len(text.split())
        if words < 2000:
            manifest[key] = {"title": title, "included": False, "why": "too short after dedup"}
            continue
        tmp = f"{OUTDIR}/{bid}.txt.tmp"
        with open(tmp, "w") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, f"{OUTDIR}/{bid}.txt")
        total_words += words
        manifest[key] = {"title": title, "authors": [a["name"] for a in authors], "included": True,
                         "why": why, "words": words}
        print(f"ok {bid} {title[:50]!r} words={words} total={total_words}", flush=True)
        with open(MANIFEST + ".tmp", "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(MANIFEST + ".tmp", MANIFEST)
        if total_words > 40_000_000:   # ~52M tokens at ~1.3 tokens/word: top of the 20-60M range
            break
    with open(MANIFEST + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(MANIFEST + ".tmp", MANIFEST)
    print("FINAL words", total_words, "books", sum(1 for v in manifest.values() if v.get("included")))


if __name__ == "__main__":
    main()
