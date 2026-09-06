# AI Assignment 2026: The Audit

This repository contains the complete solution for the AI Assignment 2026 "The Audit" take-home assessment. It covers a full audit of an intern's faulty cross-lingual tokenizer report, mathematical reconciliation of vLLM serving capacity, and a strategic decision memo for Indic language casualization.

## 📁 Repository Structure

```text
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

---

## 🚀 Part A: Tokenizer Audit

The original report claimed that Hindi text costs approximately 6x more to serve than English, concluding this was a "property of the script." This audit proves that conclusion is entirely false.

### A1: Corpus Dataset Construction
We constructed a streaming dataset from Wikipedia to ensure cross-lingual fairness without requiring authentication blocks.

| Language | ISO Code | Lines Sampled | File Size | Source |
|----------|----------|---------------|-----------|--------|
| English  | `eng`    | 1000          | 129 KB    | `wikimedia/wikipedia (20231101.en)` |
| Hindi    | `hin`    | 1000          | 245 KB    | `wikimedia/wikipedia (20231101.hi)` |
| Kannada  | `kan`    | 1000          | 242 KB    | `wikimedia/wikipedia (20231101.kn)` |
| Tamil    | `tam`    | 1000          | 300 KB    | `wikimedia/wikipedia (20231101.ta)` |

### A2: Code Bugs Identified (With Experimental Evidence)
The `audit_fertility.py` script proves three implementation bugs and one critical conceptual flaw in the original script. The measured effects of each bug are detailed in the tables below.

#### BUG 1: Incorrect Whitespace Splitting
The original script used `line.split(' ')` which counts double-spaces as empty words, artificially inflating the word count and deflating fertility. Fixing this to `line.split()` corrects the word count.

| Language | Lines Affected | Old Fertility | New Fertility | Delta |
|----------|----------------|---------------|---------------|-------|
| English  | 12 lines       | 1.441         | 1.440         | -0.001 |
| Hindi    | 8 lines        | 9.164         | 9.147         | -0.017 |

#### BUG 2: Inconsistent Lowercasing
The original script used `.lower()` on all text. This changes token boundaries for English (e.g., 'NASA' tokenizes differently than 'nasa') but acts as a no-op for Devanagari, creating an unfair comparison.

| Language | Without `.lower()` | With `.lower()` | Delta | Impact Direction |
|----------|--------------------|-----------------|-------|------------------|
| English  | 1.316              | 1.375           | +0.059 | Alters tokenization |
| Hindi    | 8.604              | 8.606           | +0.002 | No-op (Devanagari) |

#### BUG 3: Micro-averaging vs Macro-averaging
The original script used micro-averaging (the mean of per-line ratios), incorrectly giving a 3-word sentence the same mathematical weight as a 30-word sentence. Macro-averaging (total tokens / total words) is the mathematically correct metric for scaling cost estimation.

| Language | Micro Fertility | Macro Fertility | Delta | Severity |
|----------|-----------------|-----------------|-------|----------|
| English  | 1.441           | 1.316           | -0.125 | Moderate |
| Hindi    | 9.164           | 8.604           | -0.560 | High |

### A3: Conceptual Flaw — The Denominator
The original script used "tokens per whitespace-word" to compare languages. Because Indic languages compound differently, this metric is fundamentally biased. The corrected analysis uses the script-neutral **Tokens per Grapheme Cluster** metric. The table below demonstrates how the denominator metric drastically alters the perceived ratio.

| Metric | English Score | Hindi Score | Ratio (Hindi vs English) |
|--------|---------------|-------------|--------------------------|
| Tokens / Whitespace-Word (Biased) | 1.3174 | 8.6069 | 6.53x |
| Tokens / Grapheme Cluster (Honest)| 0.2081 | 2.3299 | **11.20x** |

### A4: Final Corrected Analysis Results
When comparing the English-centric GPT-2 tokenizer with the Indic-aware MuRIL tokenizer, the data clearly shows that high fertility is a tokenizer limitation, not a property of the script.

**GPT-2 Tokenizer (Baseline):**
| Language | Tokens / Grapheme Cluster | Tokens / UTF-8 Byte | Ratio vs English (Grapheme) |
|----------|---------------------------|---------------------|-----------------------------|
| English  | 0.2081                    | 0.2078              | 1.00x                       |
| Hindi    | 2.3299                    | 0.5954              | **11.20x**                  |
| Kannada  | 3.9848                    | 0.9791              | **19.15x**                  |
| Tamil    | 4.1228                    | 0.9921              | **19.81x**                  |

**MuRIL Tokenizer (Indic-aware):**
| Language | Tokens / Grapheme Cluster | Tokens / UTF-8 Byte | Ratio vs English (Grapheme) |
|----------|---------------------------|---------------------|-----------------------------|
| English  | 0.2134                    | 0.2131              | 1.00x                       |
| Hindi    | 0.3506                    | 0.0896              | **1.64x**                   |
| Kannada  | 0.3355                    | 0.0824              | **1.57x**                   |
| Tamil    | 0.2907                    | 0.0700              | **1.36x**                   |

**Conclusion:** With an Indic-aware tokenizer, the overhead for Hindi and Dravidian languages drops from ~11x-20x to a mere 1.36x - 1.64x. 

---

## 🧮 Part B: Capacity Reconciliation

The vLLM serving logs showed a counter-intuitive anomaly: throughput dropped significantly as batch size increased from 24 to 32.

### B1: KV-Cache Hardware Constraints
Based on the provided `model_spec.md`, the physical memory limits mandate the following arithmetic for a 24GB NVIDIA L4 GPU:

| Parameter | Value | Calculation Breakdown |
|-----------|-------|-----------------------|
| **Bytes per Token** | 112 KB | `2 × 28 (layers) × 8 (heads) × 128 (dim) × 2 (fp16)` |
| **GPU Usable VRAM** | 22.08 GB | `24 GB × 0.92 (gpu_memory_utilization)` |
| **KV-Cache Budget** | 12.08 GB | `22.08 GB - 8.4 GB (weights) - 1.6 GB (overhead)` |
| **Max Tokens Capacity** | 113,060 tokens | `12.08 GB / 112 KB` |
| **Max Sequences** | ~27 seqs | `113,060 / 4096 (max_model_len)` |

### B2: The Throughput Anomaly Log Data
The vLLM logs reveal a sharp collapse in throughput past batch size 24.

| Batch Size | Reported Tok/s | KV Cache Util | Preempted Seqs | System Status |
|------------|----------------|---------------|----------------|---------------|
| 16         | 1311.4         | 0.62          | 0              | Healthy |
| **24**     | **1607.4**     | **0.93**      | **0**          | **Optimal Peak** |
| 32         | 1384.0         | 0.97          | 7              | Thrashing (Over Capacity) |
| 48         | 1298.5         | 0.97          | 23             | Severe Thrashing |

**Mechanism:** At batch 32, demand (`32 × 4096 = 131,072 tokens`) exceeds the 113k capacity limit. The vLLM scheduler forcefully preempts sequences and requires redundant recomputations.

### B3: Honest Goodput Metrics
The previous report claimed 3200 tok/s at batch 48. This was a severe miscalculation that included prefill prompt tokens. 

| Metric | Calculation (At Batch 24) | Result |
|--------|---------------------------|--------|
| **Total Tok/s (Reported)** | `(24 × 4096) / 61.16s` | 1607.4 tok/s |
| **Generative Fraction** | `512 (gen) / 4096 (total)` | 12.5% |
| **Actual Goodput** | `1607.4 × 0.125` | **200.9 tok/s** |

---

## 📝 Part C: Indic Casualization Strategy

A strategy memo (`partC/memo.md`) outlines the path forward for casualizing assistant responses in 6 Indic languages within strict hardware and reviewer constraints.

### C1: Implementation Options Matrix
Evaluating the three proposed paths against our strict business constraints (1 native reviewer covering only 2/6 languages).

| Option | Time to Artifact | Reviewer Bottleneck | Data Risk | Recommendation |
|--------|------------------|---------------------|-----------|----------------|
| **(A) SFT** | 1-2 weeks | **CRITICAL** (Requires ~30,000 pairs) | High (4 languages unreviewed) | ❌ Reject initially |
| **(B) 1B Rewriter** | 1-2 weeks | **CRITICAL** (Same data issue) | High + Serving Latency | ❌ Reject |
| **(C) Prompt Eng.** | 1-3 days | Minimal (2-4 hours) | None | ✅ **START HERE** |

### C2: Day-1 Prompt Experiment Design
To execute Option C, we will run the following 5 prompt variants in Hindi.

| Variant | System Prompt Injection | Target Tone |
|---------|-------------------------|-------------|
| **A** | *(None - Baseline)* | Standard Formal |
| **B** | "Reply in informal, conversational Hindi. Use everyday words." | Friendly |
| **C** | "Reply as if texting a friend. Hindi only, simple words." | SMS / Texting |
| **D** | "Avoid formal/textbook Hindi. Write the way a Delhi college student would." | Demographic-specific |
| **E** | "तुम/तू form, short sentences, everyday Hindi vocabulary." | Syntactic enforcement |

**Kill Criterion:** If none of these prompts clear a 60% reviewer preference threshold by Week 1, we abandon Option C entirely and pivot to Option A (SFT) exclusively for Hindi and Kannada, accepting that the other 4 languages cannot be validated.

---

## ⚙️ How to Run the Code

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
