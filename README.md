# AI Assignment 2026: The Audit

This repository contains the complete solution for the AI Assignment 2026 "The Audit" take-home assessment. It covers a full audit of an intern's faulty cross-lingual tokenizer report, mathematical reconciliation of vLLM serving capacity, and a strategic decision memo for Indic language casualization.

## Repository Structure

```
.
├── README.md              (You are here)
├── WALKTHROUGH.md         (Complete project walkthrough)
├── NOTEBOOK.md            (Chronological lab notebook)
├── AI_USAGE.md            (Log of AI assistance usage)
├── partA/                 (Tokenizer Audit scripts & data)
│   ├── build_corpus.py
│   ├── audit_fertility.py
│   ├── fertility_fixed.py
│   ├── corrected_analysis.py
│   ├── memo_A4.md
│   └── corpora/           (Generated Wikipedia samples for 4 languages)
├── partB/
│   └── analysis.md        (KV-cache arithmetic and throughput anomaly explanation)
└── partC/
    └── memo.md            (Strategic decision memo for the casualization rollout)
```

## Part A: Tokenizer Audit
The original report claimed that Hindi text costs approximately 6x more to serve than English, concluding this was a "property of the script." This audit proves that conclusion is entirely false.

### Code Bugs Identified (With Experimental Evidence)
The `audit_fertility.py` script proves three implementation bugs and one critical conceptual flaw in the original script. The measured effects of each bug are detailed in the tables below.

#### BUG 1: Incorrect Whitespace Splitting
The original script used `line.split(' ')` which counts double-spaces as empty words, artificially inflating the word count and deflating fertility. Fixing this to `line.split()` corrects the word count.
| Language | Old Fertility | New Fertility | Delta |
|----------|---------------|---------------|-------|
| English  | 1.441         | 1.440         | -0.001 |
| Hindi    | 9.164         | 9.147         | -0.017 |

#### BUG 2: Inconsistent Lowercasing
The original script used `.lower()` on all text. This changes token boundaries for English (e.g., 'NASA' tokenizes differently than 'nasa') but acts as a no-op for Devanagari, creating an unfair comparison.
| Language | Without `.lower()` | With `.lower()` | Delta |
|----------|--------------------|-----------------|-------|
| English  | 1.316              | 1.375           | +0.059 |
| Hindi    | 8.604              | 8.606           | +0.002 |

#### BUG 3: Micro-averaging vs Macro-averaging
The original script used micro-averaging (the mean of per-line ratios), incorrectly giving a 3-word sentence the same mathematical weight as a 30-word sentence. Macro-averaging (total tokens / total words) is the mathematically correct metric for scaling cost estimation.
| Language | Micro Fertility | Macro Fertility | Delta |
|----------|-----------------|-----------------|-------|
| English  | 1.441           | 1.316           | -0.125 |
| Hindi    | 9.164           | 8.604           | -0.560 |

### Conceptual Flaw: The Denominator
The original script used "tokens per whitespace-word" to compare languages. Because Indic languages compound differently, this metric is fundamentally biased. The corrected analysis uses the script-neutral **Tokens per Grapheme Cluster** metric. The table below demonstrates how the denominator metric drastically alters the perceived ratio.
| Metric | English Score | Hindi Score | Ratio (Hindi vs English) |
|--------|---------------|-------------|--------------------------|
| Tokens / Whitespace-Word (Biased) | 1.3174 | 8.6069 | 6.53x |
| Tokens / Grapheme Cluster (Honest)| 0.2081 | 2.3299 | 11.20x |

### Corrected Analysis Results
When comparing the English-centric GPT-2 tokenizer with the Indic-aware MuRIL tokenizer, the data clearly shows that high fertility is a tokenizer limitation, not a property of the script.

**GPT-2 Tokenizer (Baseline):**
| Language | Tokens / Grapheme Cluster | Ratio vs English |
|----------|---------------------------|------------------|
| English  | 0.2081                    | 1.00x            |
| Hindi    | 2.3299                    | 11.20x           |
| Kannada  | 3.9848                    | 19.15x           |
| Tamil    | 4.1228                    | 19.81x           |

**MuRIL Tokenizer (Indic-aware):**
| Language | Tokens / Grapheme Cluster | Ratio vs English |
|----------|---------------------------|------------------|
| English  | 0.2134                    | 1.00x            |
| Hindi    | 0.3506                    | 1.64x            |
| Kannada  | 0.3355                    | 1.57x            |
| Tamil    | 0.2907                    | 1.36x            |

**Conclusion:** With an Indic-aware tokenizer, the overhead for Hindi and Dravidian languages drops from ~11x-20x to a mere 1.36x - 1.64x. 

## Part B: Capacity Reconciliation
The vLLM serving logs showed a counter-intuitive anomaly: throughput dropped significantly as batch size increased from 24 to 32.

### KV-Cache Math and The Throughput Anomaly
The `partB/analysis.md` file derives the exact token capacity (~113,060 tokens) for the given 24GB GPU configuration. 

At batch size 32, the sequence length demand exceeds this KV-cache capacity. This forces the vLLM scheduler to continuously preempt and recompute sequences, resulting in a throughput collapse. The proposed fix is to cap `max_model_len` at 3072, allowing batch 32 to fit in memory without preemption.

### The "Goodput" Calculation
The original report's projection of 3200 tok/s at batch 48 was fundamentally miscalculated by conflating total tokens (prefill + decode) with generated output tokens. The actual generative "goodput" delivered to users at the optimal batch 24 is **~201 tok/s**.

## Part C: Indic Casualization Strategy
A strategy memo (`partC/memo.md`) outlines the path forward for casualizing assistant responses in 6 Indic languages within strict hardware and reviewer constraints.

### The Recommendation
Supervised Fine-Tuning (SFT) is impossible to validate for 4 out of the 6 languages because of reviewer bandwidth constraints. Therefore, the recommended path is **Prompt Engineering** for Day 1.

A strict kill-criterion is established: if prompt engineering fails to achieve a 60% reviewer preference by Week 1, the strategy will pivot to a narrow SFT approach restricted to Hindi and Kannada only.

## How to Run the Code

**1. Install dependencies:**
```bash
pip install tiktoken transformers datasets regex
```

**2. Build the corpora (Downloads 1000 sentences per language from Wikipedia):**
```bash
cd partA
python -X utf8 build_corpus.py
```

**3. Run the Audit (Shows before/after metrics for the original buggy code):**
```bash
python -X utf8 audit_fertility.py
```

**4. Run the Corrected Analysis (Compares GPT-2 vs. MuRIL):**
```bash
python -X utf8 corrected_analysis.py
```

**5. Use the new Fixed Fertility CLI:**
```bash
python -X utf8 fertility_fixed.py --corpus eng=corpora/eng.txt --corpus hin=corpora/hin.txt --tokenizer gpt2 --denominator grapheme
```
*(Note: Use `python -X utf8` on Windows to ensure Indic characters print correctly to the terminal).*
