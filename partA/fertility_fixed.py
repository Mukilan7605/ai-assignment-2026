#!/usr/bin/env python3
"""
fertility_fixed.py — corrected version of the intern's fertility.py

Fixes applied:
  BUG-1: line.split(" ") → line.split()  (handles multiple spaces correctly)
  BUG-2: removed .lower() — inconsistent across scripts; documented separately
  BUG-3: macro-average (total_tokens/total_words) instead of micro-average
  BUG-C: added --denominator flag for grapheme clusters and UTF-8 bytes

Usage:
    python fertility_fixed.py --corpus eng=corpora/eng.txt \
                              --corpus hin=corpora/hin.txt \
                              --corpus kan=corpora/kan.txt \
                              --corpus tam=corpora/tam.txt \
                              --tokenizer gpt2 \
                              --denominator grapheme

    python fertility_fixed.py --corpus eng=corpora/eng.txt \
                              --corpus hin=corpora/hin.txt \
                              --tokenizer hf:ai4bharat/IndicBERT \
                              --denominator grapheme

Tokenizers:
    gpt2           → tiktoken "gpt2" encoding
    hf:<repo_id>   → any HuggingFace tokenizer

Denominators:
    word           → whitespace tokens (original, script-dependent — not recommended for cross-lingual)
    grapheme       → Unicode extended grapheme clusters (recommended)
    byte           → UTF-8 bytes (simple, script-neutral alternative)
"""

import argparse
import unicodedata
import sys

# Fix Windows console encoding
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec[3:])
        return lambda s: tok.encode(s, add_special_tokens=False)
    else:
        import tiktoken
        enc = tiktoken.get_encoding(spec)
        return enc.encode


def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = unicodedata.normalize("NFC", raw.strip())
            if line:
                lines.append(line)
    return lines


def count_grapheme_clusters(text: str) -> int:
    """Count Unicode extended grapheme clusters."""
    try:
        import regex
        return len(regex.findall(r'\X', text))
    except ImportError:
        # fallback: non-combining codepoints
        return sum(1 for ch in text if unicodedata.category(ch) not in ('Mn', 'Me', 'Mc'))


def count_utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def analyze(lines, encode, denominator: str = "grapheme"):
    """
    Macro-average fertility: total_tokens / total_denominator_units.
    
    Args:
        denominator: 'word' | 'grapheme' | 'byte'
    """
    total_tokens = 0
    total_units = 0

    for line in lines:
        # NOTE: .lower() removed — inconsistent across scripts.
        # English acronyms tokenize differently lowercased (more tokens),
        # while Indic scripts have no case distinction. Keeping original casing
        # is more honest for cross-lingual comparison.
        tokens = encode(line)
        total_tokens += len(tokens)

        if denominator == "word":
            # FIX: use .split() not .split(" ") to handle multiple spaces
            total_units += len(line.split())
        elif denominator == "grapheme":
            total_units += count_grapheme_clusters(line)
        elif denominator == "byte":
            total_units += count_utf8_bytes(line)
        else:
            raise ValueError(f"Unknown denominator: {denominator}")

    if total_units == 0:
        return float("nan")
    return total_tokens / total_units


def main():
    ap = argparse.ArgumentParser(
        description="Corrected tokenizer fertility benchmark."
    )
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="language code and path, e.g. eng=corpora/eng.txt (repeatable)",
    )
    ap.add_argument(
        "--tokenizer",
        default="gpt2",
        help="gpt2 or hf:<repo_id>",
    )
    ap.add_argument(
        "--denominator",
        choices=["word", "grapheme", "byte"],
        default="grapheme",
        help=(
            "Denominator for fertility. "
            "'word' = whitespace words (script-dependent, not recommended for cross-lingual). "
            "'grapheme' = Unicode EGC (recommended). "
            "'byte' = UTF-8 bytes."
        ),
    )
    args = ap.parse_args()

    encode = load_tokenizer(args.tokenizer)

    denom_label = {
        "word": "tok/whitespace-word (SCRIPT-DEPENDENT — not recommended for cross-lingual comparisons)",
        "grapheme": "tok/grapheme-cluster (RECOMMENDED for cross-lingual cost comparison)",
        "byte": "tok/UTF-8-byte (script-neutral alternative)",
    }[args.denominator]

    print(f"tokenizer  : {args.tokenizer}")
    print(f"denominator: {denom_label}")
    print()
    print(f"{'lang':<8} {'fertility':<15} {'lines'}")
    print("-" * 40)

    results = {}
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        lines = read_lines(path)
        fert = analyze(lines, encode, args.denominator)
        results[lang] = (fert, len(lines))
        print(f"{lang:<8} {fert:<15.4f} {len(lines)}")

    if len(results) >= 2:
        langs = list(results)
        base = langs[0]
        base_fert = results[base][0]
        print()
        for lang in langs[1:]:
            ratio = results[lang][0] / base_fert
            direction = 'worse' if ratio > 1 else 'better'
            print(f"{lang} is {ratio:.2f}x the fertility of {base} ({direction} tokenization)")


if __name__ == "__main__":
    main()
