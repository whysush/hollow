"""Shared text utilities for HOLLOW.

This module holds the pieces of logic that must behave identically in several places:
  * the sentence splitter (Wikipedia source sampling, and mirrored in the browser's app.js),
  * the SKIP-category matcher (source sampling, and mirrored at runtime in the browser),
  * the code filter applied to every teacher output (Phase 3), every model sample (Phase 6 eval),
    and every candidate generated in the browser (Phase 8; web/filter.js is a line-by-line port).
"""
import re
import difflib
import unicodedata

# ---------------------------------------------------------------------------------------------
# Banned words (Appendix A). Matched case-insensitively. Single words are matched on word
# boundaries so that e.g. "mortality" does not trip "mortal"; multi-word phrases are matched as
# substrings with word boundaries at both ends.
# ---------------------------------------------------------------------------------------------
BANNED = [
    "darkness", "doom", "doomed humans", "mortal", "mortals", "puny", "foolish", "mwahaha",
    "minion", "as an ai", "blood", "gore", "kill you", "destroy you",
]
BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(b) for b in BANNED) + r")\b", re.IGNORECASE)

# ---------------------------------------------------------------------------------------------
# SKIP categories (Appendix A). A Wikipedia article is skipped if ANY of its category names matches.
# "living people" is matched exactly as the category "Living people".
# Patterns are deliberately broad: a false skip costs one article, a false keep could make the
# model speak sinisterly about a real tragedy.
# ---------------------------------------------------------------------------------------------
SKIP_CAT_RE = re.compile(
    r"suicide|self-harm|self harm|eating disorder|anorexia|bulimia|child abuse|sexual abuse|"
    r"sexual violence|\brape\b|terroris|genocide|holocaust|massacre|mass shooting|mass murder|"
    r"shooting|bombing|disasters|accidents|accidental deaths|shipwreck|explosions in|"
    r"fires in|famine|pandemic|epidemic|wars? involving|wars? of|military conflicts|battles|"
    r"conflicts in|invasions|war crimes|ethnic cleansing|atrocit|murder|killings|"
    r"executed|deaths by|deaths from|road incident|aviation accident|"
    r"pornograph|\bsexual|prostitution|paedophil|pedophil|child sexual",
    re.IGNORECASE,
)
LIVING_RE = re.compile(r"^(category:)?living people$", re.IGNORECASE)


def is_skip_categories(categories):
    """Return a short reason string if the category list hits SKIP or Living people, else None.

    categories: list of category names, with or without the "Category:" prefix.
    """
    for c in categories:
        name = c.split(":", 1)[1] if c.lower().startswith("category:") else c
        if LIVING_RE.match(name.strip()):
            return "living"
        if SKIP_CAT_RE.search(name):
            return "skip"
    return None


# ---------------------------------------------------------------------------------------------
# Sentence splitting. Splits after . ? ! when followed by whitespace and an uppercase letter,
# digit or opening quote, except after common abbreviations and single-letter initials.
# ---------------------------------------------------------------------------------------------
ABBREV = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "vs", "etc", "e.g", "i.e", "approx", "ca",
    "c", "no", "vol", "fig", "mt", "ft", "u.s", "u.k", "inc", "ltd", "co", "gen", "col", "lt",
    "sgt", "capt", "rev", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "est", "al", "op", "cf", "ed", "eds", "b", "d", "r", "km", "cm", "mm", "kg",
}
_SPLIT_CAND = re.compile(r"([.?!])([\"')\]]*)\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text):
    """Split a paragraph into sentences. Returns a list of stripped sentence strings."""
    out, start = [], 0
    for m in _SPLIT_CAND.finditer(text):
        end = m.end(2)  # index just after the terminator and any closing quotes/brackets
        if m.group(1) == ".":
            # Look at the word right before the period to rule out abbreviations and initials.
            before = text[start:m.start(1)]
            w = re.search(r"([A-Za-z.]+)$", before)
            if w:
                word = w.group(1).lower().rstrip(".")
                if word in ABBREV or (len(word) == 1 and word.isalpha()):
                    continue
        out.append(text[start:end].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def word_count(s):
    """Count whitespace-separated words (units: words)."""
    return len(s.split())


# ---------------------------------------------------------------------------------------------
# The code filter (Phase 3, step 4). Returns None if the output passes, otherwise a short
# rejection-reason string (used for the rejection histogram).
# ---------------------------------------------------------------------------------------------
DIGITS_RE = re.compile(r"\d+")
# A "capitalised name" token: starts with an uppercase letter (allowing accented letters), may
# contain letters, digits, apostrophes-inside and hyphens. Roman numerals / acronyms like "II",
# "NASA" also match because they start uppercase.
CAP_TOKEN_RE = re.compile(r"[A-ZÀ-Þ][\wÀ-ɏ'\-]*")
# Sentence-initial words that are ordinary words when capitalised only by position. If the very
# first word of a source sentence is one of these (lowercased), it is not treated as a name.
# Any other first word is treated as a name only if it is also followed by another capitalised
# token (multi-word name such as "Harvard Mark") or is all-caps (acronym), because a lone
# position-capitalised first word ("Octopuses have...") is almost always an ordinary word.


def source_names(src):
    """Return the set of lowercased capitalised-name tokens from an original-case source sentence.

    Rules (explicit):
      * Every token matching CAP_TOKEN_RE that is NOT the first word of the sentence is a name.
      * The first word counts as a name only if it is an all-uppercase acronym of length >= 2,
        or the second word is also capitalised (start of a multi-word name, e.g. "New York").
      * Possessive "'s" and trailing hyphen/apostrophe are stripped ("Earth's" -> "earth").
    """
    words = src.split()
    names = set()
    for i, w in enumerate(words):
        w_clean = w.strip("\"'()[]{},.;:!?“”‘’")
        m = CAP_TOKEN_RE.fullmatch(w_clean) if w_clean else None
        if not m:
            # Hyphenated compounds like "pre-Columbian" contain an inner capitalised part.
            for part in re.split(r"[-–/]", w_clean):
                if part and CAP_TOKEN_RE.fullmatch(part) and i > 0:
                    names.add(_norm_name(part))
            continue
        if i == 0:
            nxt = words[1].strip("\"'()[]{},.;:!?“”‘’") if len(words) > 1 else ""
            is_acronym = len(w_clean) >= 2 and w_clean.isupper()
            if not (is_acronym or (nxt and CAP_TOKEN_RE.fullmatch(nxt))):
                continue
        names.add(_norm_name(w_clean))
    names.discard("")
    names.discard("i")
    return names


def _norm_name(tok):
    tok = re.sub(r"['’]s$", "", tok)
    return tok.strip("'’-").lower()


def has_emoji_or_symbol(s):
    """True if s contains any emoji or pictographic symbol (Unicode category So/Sk/Cs, or the
    emoji blocks). Ordinary punctuation, letters and digits are allowed."""
    for ch in s:
        cp = ord(ch)
        if 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0xFE00 <= cp <= 0xFE0F:
            return True
        if unicodedata.category(ch) in ("So", "Cs"):
            return True
    return False


def similarity(a, b):
    """Character-level similarity ratio in [0, 1] (difflib.SequenceMatcher, autojunk disabled
    so long strings are compared faithfully)."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def filter_rewrite(src, out, max_words=60, require_facts=True, allow_digits=True):
    """Apply the Phase 3 code filter.

    src: original-case source sentence (for openers/closers pass the topic; facts are not checked).
    out: the candidate text.
    max_words: 60 for fact rewrites, 20 for openers and closers.
    require_facts: True for fact rewrites (digit and name preservation checks + near-copy check).
    allow_digits: False for openers/closers (the "no facts at all" rule forbids any digits).
    Returns None if the candidate passes, else the rejection reason.
    """
    if out is None or not out.strip():
        return "empty"
    out = out.strip()
    if out != out.lower():
        return "not_lowercase"
    if "!" in out:
        return "exclamation"
    if has_emoji_or_symbol(out):
        return "emoji"
    if word_count(out) > max_words:
        return "too_long"
    if BANNED_RE.search(out):
        return "banned_word"
    out_digits = DIGITS_RE.findall(out)
    if not allow_digits and out_digits:
        return "digits_in_opener_closer"
    if require_facts:
        src_digits = DIGITS_RE.findall(src)
        if any(d not in out_digits for d in src_digits):
            return "missing_digits"
        if any(d not in src_digits for d in out_digits):
            return "new_digits"
        for name in source_names(src):
            if not re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", out):
                return "missing_name"
        if similarity(src.lower(), out) >= 0.9:
            return "near_copy"
    return None
