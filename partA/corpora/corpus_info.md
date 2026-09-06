# Corpus Info

## Source
Wikimedia/Wikipedia (HuggingFace Hub, snapshot 20231101, streaming, no authentication required).
URL: https://huggingface.co/datasets/wikimedia/wikipedia

## Languages and Sizes

| lang | wikipedia_config | script | family | sentences |
|---|---|---|---|---|
| eng | 20231101.en | Latin | Indo-European/Germanic | 1000 |
| hin | 20231101.hi | Devanagari | Indo-European/Indo-Aryan | 1000 |
| kan | 20231101.kn | Kannada | Dravidian | 1000 |
| tam | 20231101.ta | Tamil | Dravidian | 1000 |

## Domain
Encyclopedic (Wikipedia articles). General knowledge, formal prose.
Professionally written, not conversational.

## Preprocessing
- Stream first articles until 5000 candidate sentences collected
- Sentence splitting on `. `, `। `, `!\ `, `?\ `
- NFC Unicode normalization
- Filter: 15–300 characters, ≥50% alphabetic characters
- Random sample of 1000 sentences per language (seed=42)

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
