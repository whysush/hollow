"""Phase 4: train HOLLOW's own byte-level BPE tokenizer (Hugging Face `tokenizers`).

Training mix (all lowercased, measured in characters):
  70% FineWeb-Edu sample      data/fineweb_tok_sample.txt
  20% gothic corpus           data/gothic/*.txt
  10% Wikipedia lead text     data/wiki_leads_sample.txt
Vocabulary: 16,384 tokens = 256 byte symbols + merges + 6 special tokens.
Special tokens (ids 0-5, in this order):
  <|endoftext|>  document separator in pretraining
  <|src|>        starts a source sentence in a rewrite example
  <|sty|>        separates the prompt from the styled output (loss starts after it)
  <|end|>        ends a styled output (generation stops here; eos in generation_config.json)
  <|opener|>     starts an opener example (prompt is the topic)
  <|closer|>     starts a closer example (prompt is the topic)
Output: tokenizer/tokenizer.json (+ tokenizer_config.json, special_tokens_map.json) loadable by
transformers.PreTrainedTokenizerFast and by transformers.js.

Acceptance (run with `--check`): lossless round trip of 1,000 random lines and tokens-per-word.
"""
import os, sys, glob, random, json, argparse

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

ROOT = os.environ["PROJECT_ROOT"]
OUT = f"{ROOT}/tokenizer"
SPECIALS = ["<|endoftext|>", "<|src|>", "<|sty|>", "<|end|>", "<|opener|>", "<|closer|>"]
VOCAB = 16384


def mix_lines():
    """Yield lowercased training lines in a 70/20/10 character mix."""
    fw = open(f"{ROOT}/data/fineweb_tok_sample.txt").read()
    fw_chars = len(fw)
    goth_budget = int(fw_chars * 20 / 70)   # 20% share relative to the 70% FineWeb share
    wiki_budget = int(fw_chars * 10 / 70)   # 10% share
    goth, n = [], 0
    files = sorted(glob.glob(f"{ROOT}/data/gothic/*.txt"))
    random.Random(0).shuffle(files)
    for p in files:
        t = open(p).read()
        goth.append(t[: goth_budget - n]); n += len(goth[-1])
        if n >= goth_budget:
            break
    # Snapshot of the first 10.9 MB of the lead-text file, taken while data_wiki.py was still running (the writer
    # stops at 12 MB, so this prefix is identical to the final file). Lets the tokenizer train during the teacher run.
    wiki = open(f"{ROOT}/data/wiki_leads_tok_snapshot.txt").read()[:wiki_budget]
    print(f"mix chars: fineweb={fw_chars:,} gothic={n:,} wiki={len(wiki):,}", flush=True)
    for block in (fw, "\n".join(goth), wiki):
        for line in block.split("\n"):
            if line.strip():
                yield line.lower()


def build():
    tok = Tokenizer(models.BPE())
    # GPT-2 style byte-level pre-tokenisation (regex split + bytes->unicode mapping). No prefix
    # space is added, so decode(encode(x)) == x exactly for any string.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        min_frequency=2,                                    # a merge must occur at least twice
        special_tokens=SPECIALS,                            # get ids 0..5
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes always present
        show_progress=False,
    )
    tok.train_from_iterator(mix_lines(), trainer=trainer)
    os.makedirs(OUT, exist_ok=True)
    tok.save(f"{OUT}/tokenizer.json")
    # Configs so transformers.PreTrainedTokenizerFast / AutoTokenizer and transformers.js load it.
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|endoftext|>", "eos_token": "<|end|>", "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": SPECIALS[1:],
        "model_max_length": 512, "clean_up_tokenization_spaces": False, "add_prefix_space": False,
    }
    json.dump(cfg, open(f"{OUT}/tokenizer_config.json", "w"), indent=2)
    json.dump({"bos_token": "<|endoftext|>", "eos_token": "<|end|>", "unk_token": "<|endoftext|>",
               "pad_token": "<|endoftext|>", "additional_special_tokens": SPECIALS[1:]},
              open(f"{OUT}/special_tokens_map.json", "w"), indent=2)
    print("vocab size", tok.get_vocab_size(), {s: tok.token_to_id(s) for s in SPECIALS})


def check():
    from transformers import PreTrainedTokenizerFast
    hf = PreTrainedTokenizerFast(tokenizer_file=f"{OUT}/tokenizer.json")
    lines = [l for l in mix_lines()]
    rng = random.Random(42)
    sample = rng.sample(lines, 1000)
    bad = 0
    toks = words = 0
    for l in sample:
        ids = hf.encode(l, add_special_tokens=False)
        if hf.decode(ids, clean_up_tokenization_spaces=False) != l:
            bad += 1
        toks += len(ids); words += len(l.split())
    s = "<|src|> octopuses have three hearts. <|sty|> they wait. <|end|>"
    ids = hf.encode(s, add_special_tokens=False)
    print("special-token example ids:", ids[:3], "...", ids[-2:])
    print(f"round-trip failures: {bad}/1000")
    print(f"tokens per word: {toks / words:.3f}")
    assert bad == 0
    assert hf.convert_tokens_to_ids("<|sty|>") == 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        build()
    check()
