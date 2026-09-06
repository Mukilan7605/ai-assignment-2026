# Walkthrough — AI Assignment 2026 Complete Solution

## Summary of Changes

Complete solution for the AI Assignment 2026 "The Audit" take-home assignment. All three parts have been implemented with real experimental evidence.

---

## Part A — Tokenizer Audit

### A1: Corpus Construction

**Source**: Wikimedia/Wikipedia (HuggingFace, streaming, no authentication required)  
**Languages**: English (en), Hindi (hi), Kannada (kn), Tamil (ta)  
**Size**: 1000 sentences per language (seed=42 for reproducibility)

Files created:
- `partA/build_corpus.py` — downloads and preprocesses corpus
- `partA/corpora/eng.txt` (1000 lines, 129 KB)
- `partA/corpora/hin.txt` (1000 lines, 245 KB)
- `partA/corpora/kan.txt` (1000 lines, 242 KB)
- `partA/corpora/tam.txt` (1000 lines, 300 KB)
- `partA/corpora/corpus_info.md` — size, domain, caveats

### A2: Script Audit — Bugs Found in fertility.py

Ran `audit_fertility.py` to demonstrate each flaw with before/after evidence:

| bug | description | measured effect |
|-----|-------------|-----------------|
| BUG-1 | `split(' ')` instead of `split()` | deflates fertility by <1% (double spaces count as empty words) |
| BUG-2 | `.lower()` on Indic text | raises English fertility by +0.059 (acronyms like GPU tokenize differently); no-op for Hindi |
| BUG-3 | Micro-average vs macro-average | micro gives eng=1.441, macro gives eng=1.316 (delta: -0.125); Hindi delta: -0.560 |
| BUG-C | Whitespace words as cross-lingual denominator | conceptual bug: gives Hindi 6.36×; grapheme clusters give 11.20× — both measure different things |
| OK | `random.seed(1337)` | harmless dead code; verified zero effect on output |

### A3: Corrected Analysis Results

**GPT-2 tokenizer (English-centric baseline):**

| language | tok/word (script-dep) | tok/grapheme (RECOMMENDED) | tok/UTF-8 byte |
|----------|-----------------------|---------------------------|----------------|
| English  | 1.3174 | 0.2081 | 0.2078 |
| Hindi    | 8.6069 (6.53×) | 2.3299 (11.20×) | 0.5954 (2.87×) |
| Kannada  | 22.3506 (16.97×) | 3.9848 (19.15×) | 0.9791 (4.71×) |
| Tamil    | 25.8414 (19.62×) | 4.1228 (19.81×) | 0.9921 (4.77×) |

**MuRIL tokenizer (Indic-aware):**

| language | tok/grapheme | ratio |
|----------|-------------|-------|
| English  | 0.2134 | 1.00× |
| Hindi    | 0.3506 | **1.64×** |
| Kannada  | 0.3355 | **1.57×** |
| Tamil    | 0.2907 | **1.36×** |

**Key insight**: Switching to an Indic-aware tokenizer reduces Hindi overhead from 11× to 1.6× — confirming that high fertility is a tokenizer problem, NOT a script property (directly contradicts REPORT_v0's conclusion).

### A4: Recommendation Memo

See `partA/memo_A4.md` for the full ≤1-page memo. Summary:
- Primary routing metric: **tokens/grapheme-cluster**
- Recommendation: Deploy Indic-aware tokenizer (MuRIL/IndicBERT) for all Indic-script traffic
- Production monitor: `tokens_generated / grapheme_clusters_in_output` per language, alert at 3× English baseline

---

## Part B — Capacity Reconciliation

### B1: KV-Cache Math

```
bytes/token = 2 × 28 × 8 × 128 × 2 = 114,688 bytes (112 KB)
KV budget   = (24 GB × 0.92) - 8.4 GB - 1.6 GB = 12.08 GB
Max tokens  = 12.08 GB / 112 KB ≈ 113,060 tokens
Max 4096-token seqs = 113,060 / 4096 ≈ 27 sequences
```

Verified: batch 24 × 4096 = 98,304 / 113,060 = 87% → logged as 0.93 (close; paged block granularity accounts for difference).

### B2: Throughput Anomaly

Peak at batch 24 (1607 tok/s), then **drops** at batch 32 (1384, 7 preemptions) and 48 (1298, 23 preemptions). Mechanism: KV cache exhaustion → preemption → recomputation overhead.

Fix: reduce `max_model_len` from 4096→3072 so batch 32 fits in KV cache without preemption.

### B3: Goodput Calculation

Both methods give same answer:
- Method 1: 24 × 512 / 61.16s = **200.9 tok/s** actual output
- Method 2: 1607.4 × (512/4096) = **200.9 tok/s** actual output

Report's "1311 tok/s vs 883 tok/s" comparison and "batch 48 = 3200 tok/s" projection both misread `reported_tok_s` as output-only (it counts prefill + decode).

### B4: Monitoring Metric

`preempted_seqs` (or `vllm:scheduler_num_preempted_seqs_total`). Expected: 0 for batch ≤ 24, spikes at batch 32+.

---

## Part C — Decision Memo

See `partC/memo.md`. Summary:
- **Recommendation**: Start with prompt-engineering (option c), gate to narrow SFT for Hindi+Kannada only if day-1 evaluation fails
- **Day-1 experiment**: 5 prompt variants on 20 fixed Hindi prompts, measure reviewer preference
- **Kill criterion**: <60% preference after Week 1 → pivot to SFT for Hindi+Kannada
- **Critical constraint that was missed initially**: Reviewer covers only 2/6 languages → SFT-first approach cannot be validated for 4 languages

---

## Files Created

```
your-submission/
  NOTEBOOK.md              ← chronological lab notebook with dead ends
  AI_USAGE.md              ← honest AI usage log
  partA/
    build_corpus.py        ← downloads Wikipedia corpora
    audit_fertility.py     ← demonstrates each bug with evidence
    fertility_fixed.py     ← corrected fertility script (run this)
    corrected_analysis.py  ← full 4-language, 2-tokenizer, 3-denominator analysis
    memo_A4.md             ← recommendation memo (≤1 page)
    corrected_results.csv  ← numerical results table
    corpora/
      eng.txt (1000 lines)
      hin.txt (1000 lines)
      kan.txt (1000 lines)
      tam.txt (1000 lines)
      corpus_info.md
  partB/
    analysis.md            ← B1-B4 full answers with arithmetic
  partC/
    memo.md                ← decision memo (≤1 page)
```

## Validation

All scripts run end-to-end successfully:
- `python -X utf8 build_corpus.py` → 4×1000 lines from Wikipedia
- `python -X utf8 audit_fertility.py` → bug evidence with before/after numbers
- `python -X utf8 corrected_analysis.py` → 3-denominator × 2-tokenizer table + CSV
- `python -X utf8 fertility_fixed.py --corpus eng=... [options]` → corrected fertility CLI
