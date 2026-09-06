#!/usr/bin/env python3
"""
build_corpus.py — assemble a multilingual eval corpus from Wikipedia (via HuggingFace).

Languages chosen:
  English  — en  (Wikipedia 20231101.en)
  Hindi    — hi  (Wikipedia 20231101.hi)   [Devanagari]
  Kannada  — kn  (Wikipedia 20231101.kn)   [Dravidian]
  Tamil    — ta  (Wikipedia 20231101.ta)   [Dravidian]

Source: Wikimedia/Wikipedia (HuggingFace, streaming, no auth required)
Domain: Encyclopedia (news-adjacent, formal prose)
Preprocessing: NFC normalization, sentence-level splitting, filter to 10-100 char lines,
               sample 1000 lines per language for reproducibility.

Outputs:
  corpora/eng.txt
  corpora/hin.txt
  corpora/kan.txt
  corpora/tam.txt
  corpora/corpus_info.md
"""

import unicodedata
import pathlib
import sys
import random
import re

random.seed(42)  # reproducible sampling

# Fix Windows console encoding
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = pathlib.Path(__file__).parent / "corpora"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_LINES = 1000  # sentences per language

LANG_MAP = {
    "eng": ("en", "20231101.en"),
    "hin": ("hi", "20231101.hi"),
    "kan": ("kn", "20231101.kn"),
    "tam": ("ta", "20231101.ta"),
}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def split_sentences(text: str) -> list:
    """
    Rough sentence splitter: split on '. ', '। ', '.\n', '।\n', etc.
    For Indic: use danda (।) as sentence boundary.
    """
    # Split on common sentence boundaries
    sents = re.split(r'(?<=[.।!?])\s+', text)
    return sents


def is_valid_line(line: str, min_chars=15, max_chars=300) -> bool:
    """Filter out headers, too-short lines, lines with lots of special chars."""
    if not line:
        return False
    if len(line) < min_chars or len(line) > max_chars:
        return False
    # Skip lines that are mostly non-alphabetic (tables, math, etc.)
    alpha = sum(1 for c in line if c.isalpha())
    if alpha / len(line) < 0.5:
        return False
    return True


def fetch_wikipedia_corpus(hf_config: str, n: int = TARGET_LINES) -> list:
    from datasets import load_dataset
    
    print(f"  Streaming Wikipedia {hf_config} …", flush=True)
    ds = load_dataset(
        "wikimedia/wikipedia",
        hf_config,
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    
    collected = []
    for article in ds:
        text = article.get("text", "")
        if not text:
            continue
        sents = split_sentences(text)
        for s in sents:
            s = normalize(s)
            if is_valid_line(s):
                collected.append(s)
            if len(collected) >= n * 5:  # buffer 5x and then sample
                break
        if len(collected) >= n * 5:
            break
    
    # Random sample for diversity
    if len(collected) > n:
        sampled = random.sample(collected, n)
    else:
        sampled = collected
    
    print(f"  → {len(sampled)} lines collected")
    return sampled


def build_from_wikipedia():
    counts = {}
    for short_code, (lang_code, hf_config) in LANG_MAP.items():
        try:
            lines = fetch_wikipedia_corpus(hf_config, TARGET_LINES)
            out_path = OUTPUT_DIR / f"{short_code}.txt"
            out_path.write_text("\n".join(lines), encoding="utf-8")
            counts[short_code] = len(lines)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  ERROR fetching {short_code}: {e}", file=sys.stderr)
    return counts


def write_corpus_info(counts: dict):
    info = f"""# Corpus Info

## Source
Wikimedia/Wikipedia (HuggingFace Hub, snapshot 20231101, streaming, no authentication required).
URL: https://huggingface.co/datasets/wikimedia/wikipedia

## Languages and Sizes

| lang | wikipedia_config | script | family | sentences |
|---|---|---|---|---|
| eng | 20231101.en | Latin | Indo-European/Germanic | {counts.get('eng', 'N/A')} |
| hin | 20231101.hi | Devanagari | Indo-European/Indo-Aryan | {counts.get('hin', 'N/A')} |
| kan | 20231101.kn | Kannada | Dravidian | {counts.get('kan', 'N/A')} |
| tam | 20231101.ta | Tamil | Dravidian | {counts.get('tam', 'N/A')} |

## Domain
Encyclopedic (Wikipedia articles). General knowledge, formal prose.
Professionally written, not conversational.

## Preprocessing
- Stream first articles until 5000 candidate sentences collected
- Sentence splitting on `. `, `। `, `!\\ `, `?\\ `
- NFC Unicode normalization
- Filter: 15–300 characters, ≥50% alphabetic characters
- Random sample of {TARGET_LINES} sentences per language (seed=42)

## What this corpus CANNOT tell you

1. **Register mismatch**: Wikipedia is formal encyclopedic prose. Our serving use-case
   is conversational assistant replies. Fertility on Wikipedia likely underestimates
   the challenge of colloquial Indic text, especially for tokenizers trained mostly
   on formal text.

2. **Code-switching**: Wikipedia articles rarely mix scripts (e.g., Hindi+English).
   Real user queries contain significant Hinglish/Kanglish, which this corpus cannot
   represent. Code-switched text can have unpredictable fertility for all tokenizers.

3. **Domain tail**: Technical or specialized Indic text (legal, medical, code) may
   show radically different fertility. A tokenizer tuned for Wikipedia will not
   represent corner cases.

4. **Sample-size for Indic-specific analysis**: 1000 sentences is sufficient to
   rank tokenizers but too small to estimate variance reliably. For production
   routing decisions, corpora of >10k sentences across diverse domains are recommended.

5. **Parallel alignment**: This corpus is NOT strictly parallel (different Wikipedia
   articles in each language). Cross-lingual fertility comparisons assume equivalent
   content complexity, which is only approximately true here. FLORES-200 (gated)
   would be superior for a strictly controlled comparison.
"""
    (OUTPUT_DIR / "corpus_info.md").write_text(info, encoding="utf-8")
    print(f"\nCorpus info written to {OUTPUT_DIR / 'corpus_info.md'}")


if __name__ == "__main__":
    counts = build_from_wikipedia()
    write_corpus_info(counts)
    print("\nDone. Corpora saved to:", OUTPUT_DIR)
