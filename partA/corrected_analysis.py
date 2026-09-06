#!/usr/bin/env python3
"""
corrected_analysis.py — recomputes cross-language fertility properly.

Requires:
  - corpora/eng.txt, corpora/hin.txt, corpora/kan.txt, corpora/tam.txt
    (produced by build_corpus.py)
  - tiktoken          (pip install tiktoken)
  - transformers      (pip install transformers)   ← for Indic tokenizer
  - regex             (pip install regex)

Two tokenizers:
  1. gpt2           — baseline BPE tokenizer (English-centric)
  2. ai4bharat/IndicBERT — Indic-aware multilingual tokenizer

Two denominators:
  1. whitespace words          (original metric, script-dependent)
  2. grapheme clusters         (Unicode EGC, script-neutral)

Usage:
    python corrected_analysis.py
"""

import unicodedata
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent
CORPUS_DIR = SCRIPT_DIR / "corpora"

LANGS = ["eng", "hin", "kan", "tam"]
LANG_NAMES = {
    "eng": "English",
    "hin": "Hindi",
    "kan": "Kannada",
    "tam": "Tamil",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def read_lines(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = unicodedata.normalize("NFC", raw.strip())
            if line:
                lines.append(line)
    return lines


def count_grapheme_clusters(text: str) -> int:
    try:
        import regex
        return len(regex.findall(r'\X', text))
    except ImportError:
        # fallback: non-combining codepoints
        return sum(1 for ch in text
                   if unicodedata.category(ch) not in ('Mn', 'Me', 'Mc'))


def count_utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def fertility_stats(lines, encode_fn):
    """
    Returns dict with:
      total_tokens, total_words, total_graphemes, total_utf8_bytes,
      tpw (tokens/word), tpg (tokens/grapheme), tpb (tokens/byte)
    All computed at corpus level (macro-average = correct).
    """
    total_tokens = 0
    total_words = 0
    total_graphemes = 0
    total_bytes = 0

    for line in lines:
        toks = encode_fn(line)
        words = line.split()
        graphemes = count_grapheme_clusters(line)
        ub = count_utf8_bytes(line)

        total_tokens += len(toks)
        total_words += len(words)
        total_graphemes += graphemes
        total_bytes += ub

    return {
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_graphemes": total_graphemes,
        "total_bytes": total_bytes,
        "tpw": total_tokens / total_words if total_words else float("nan"),
        "tpg": total_tokens / total_graphemes if total_graphemes else float("nan"),
        "tpb": total_tokens / total_bytes if total_bytes else float("nan"),
    }


# ── tokenizer loaders ─────────────────────────────────────────────────────────

def load_gpt2():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode


def load_indic_tokenizer():
    """
    Load ai4bharat/IndicBERT tokenizer.
    Falls back to google/muril-base-cased if IndicBERT fails.
    """
    from transformers import AutoTokenizer

    for repo in ["ai4bharat/IndicBERT", "google/muril-base-cased"]:
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            print(f"  Loaded Indic tokenizer: {repo}")
            return lambda s: tok.encode(s, add_special_tokens=False), repo
        except Exception as e:
            print(f"  Could not load {repo}: {e}")

    print("  WARNING: Could not load any Indic tokenizer. Skipping Indic tokenizer analysis.")
    return None, None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Check corpus exists
    missing = [l for l in LANGS if not (CORPUS_DIR / f"{l}.txt").exists()]
    if missing:
        print(f"ERROR: Missing corpus files for: {missing}")
        print(f"Run build_corpus.py first.")
        sys.exit(1)

    corpora = {lang: read_lines(CORPUS_DIR / f"{lang}.txt") for lang in LANGS}
    for lang, lines in corpora.items():
        print(f"  {lang}: {len(lines)} lines")

    # Load tokenizers
    print("\nLoading tokenizers …")
    tokenizers = {}
    tokenizers["gpt2"] = load_gpt2()
    print("  Loaded: gpt2")

    indic_fn, indic_name = load_indic_tokenizer()
    if indic_fn:
        tokenizers[indic_name.split("/")[-1]] = indic_fn

    # Compute stats
    all_results = {}  # {tok_name: {lang: stats}}

    for tok_name, encode_fn in tokenizers.items():
        print(f"\nComputing with tokenizer: {tok_name} …")
        all_results[tok_name] = {}
        for lang in LANGS:
            all_results[tok_name][lang] = fertility_stats(corpora[lang], encode_fn)

    # ── Print tables ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CORRECTED ANALYSIS — Tokenizer Fertility Comparison")
    print("=" * 80)

    denominators = [
        ("tpw", "tokens/whitespace-word", "script-DEPENDENT (original metric — biased)"),
        ("tpg", "tokens/grapheme-cluster", "script-NEUTRAL  (recommended for cross-lingual cost)"),
        ("tpb", "tokens/UTF-8-byte",       "script-neutral  (alternative, easy to compute)"),
    ]

    for tok_name in tokenizers:
        print(f"\n{'─'*80}")
        print(f"Tokenizer: {tok_name}")
        print(f"{'─'*80}")

        for denom_key, denom_label, note in denominators:
            print(f"\n  Denominator: {denom_label}  [{note}]")
            print(f"  {'lang':<10} {'value':>10}  {'ratio vs eng':>14}  {'sentences':>10}")
            print(f"  {'─'*50}")

            eng_val = all_results[tok_name]["eng"][denom_key]
            for lang in LANGS:
                val = all_results[tok_name][lang][denom_key]
                ratio = val / eng_val if eng_val else float("nan")
                n = len(corpora[lang])
                ratio_str = f"{ratio:.2f}x"
                print(f"  {LANG_NAMES[lang]:<10} {val:>10.4f}  {ratio_str:>14}  {n:>10}")

    # ── Recommendation ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DENOMINATOR REASONING — Which number should drive routing decisions?")
    print("=" * 80)
    print("""
The denominator must hold CONSTANT the "amount of content" we're asking the model
to process/generate, so that we can fairly compare serving cost across languages.

1. Whitespace words (original):
   - FAILS: Hindi/Tamil/Kannada routinely express in one morphological word
     what English expresses in 2-3 words. The "word" unit is script-dependent.
   - Example: Hindi "मुझे जाना है" = 3 words; English "I need to go" = 4 words.
     But both carry identical content. Word-count denominator inflates the ratio.

2. Grapheme clusters (RECOMMENDED):
   - A grapheme cluster ≈ one user-perceived character.
   - Stable across scripts: one Devanagari matras, one Latin letter, one
     Tamil akshara each count as 1.
   - Best proxy for "visible content" the user sees on screen.
   - For serving cost: directly answers "how many tokens do we generate per
     visible character of output?" — the right cost metric.

3. UTF-8 bytes:
   - Simple to compute, no Unicode library needed.
   - Good proxy, but conflates multi-byte Indic chars with single-byte Latin.
   - Slightly less interpretable than graphemes but valid for back-of-envelope.

RECOMMENDATION: Use tokens/grapheme-cluster as the primary routing metric.
Report tokens/UTF-8-byte as a secondary check. Discard tokens/whitespace-word
for any cross-lingual comparison.
""")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import csv
    out_csv = SCRIPT_DIR / "corrected_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tokenizer", "language", "total_tokens", "total_words",
                         "total_graphemes", "total_bytes", "tpw", "tpg", "tpb"])
        for tok_name in tokenizers:
            for lang in LANGS:
                s = all_results[tok_name][lang]
                writer.writerow([tok_name, lang, s["total_tokens"], s["total_words"],
                                 s["total_graphemes"], s["total_bytes"],
                                 f"{s['tpw']:.4f}", f"{s['tpg']:.4f}", f"{s['tpb']:.4f}"])
    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    main()
