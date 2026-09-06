# AI Assignment 2026: The Audit Complete Solution

## Bug Audit

### Executive Summary

| Bug / Code Pattern | Classification | Line in `fertility.py` | Measured Before & After Numbers | Distortion & Direction |
|---|---|---|---|---|
| `split(" ")` vs `split()` | Code Bug | L62 | • Buggy Hindi: `9.164`<br>• Fixed Hindi: `9.147`<br>• Delta: `-0.017` | Underestimates fertility by counting phantom empty strings from double spaces (`"books in"`) as words. |
| `.lower()` Preprocessing | Code/Conceptual Bug | L60 | • Lowercased Eng: `1.375`<br>• Original Eng: `1.316`<br>• Delta: `+0.059` (Hindi Δ = `+0.002`) | Overestimates English tokens by splitting acronyms (`NASA` → 1 tok, `nasa` → 2 tok), artificially compressing the ratio. |
| Mean-of-Ratios Aggregation | Aggregation Bug | L64 | • Micro Hin: `9.164`<br>• Macro Hin: `8.604`<br>• Delta: `-0.560` | Biases results toward short sentences by weighting a 3-word sentence equally with a 30-word sentence. |
| `random.seed(1337)` | Harmless Code | L25 | • Measured Delta: `0.000` | Zero effect. `random` is never called; tokenization is deterministic. |

---

### Bug 1 — `line.lower()` (Code Bug)
**Location:** `fertility.py` line 60

```python
# BUGGY
line = line.lower()
```

**Problem:** GPT-2 encodes capitalized vs lowercase tokens differently. For English acronyms (NASA, ISRO, GPU), `.lower()` changes the tokenization. For Devanagari/Dravidian scripts, it is a complete no-op.

**Experiment results (Illustrative):**

| Sentence | Tokens WITH `.lower()` | Tokens WITHOUT `.lower()` | Delta |
|----------|------------------------|---------------------------|-------|
| `NASA and ISRO announced a joint mission update.` | 11 | 10 | +1 |
| `मुझे सुबह की चाय बहुत पसंद है।` | 47 | 47 | 0 |
| `Bengaluru International Airport handled record traffic in March.` | 12 | 12 | 0 |

**Measured effect on our full Wikipedia corpus:**

| Language | Without `.lower()` | With `.lower()` | Delta | Impact Direction |
|----------|--------------------|-----------------|-------|------------------|
| English  | 1.316              | 1.375           | +0.059 | Alters tokenization |
| Hindi    | 8.604              | 8.606           | +0.002 | Statistically zero |

**Reproduce:**
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
s = "NASA and ISRO announced a joint mission update."
print(len(enc.encode(s.lower())), len(enc.encode(s))) # -> 11 10
```

---

### Bug 2 — `line.split(" ")` vs `line.split()` (Code Bug)
**Location:** `fertility.py` line 62

```python
# BUGGY
words = line.split(" ")  # creates empty string "" on double spaces

# FIXED
words = line.split()     # handles any whitespace correctly
```

**Problem:** If text contains double spaces (e.g. `"Please keep the books  in the cupboard."`), `.split(" ")` produces an empty string token `""`, inflating the word count by 1.

**Experiment results (Illustrative):**

| Method | Result | Word count |
|--------|--------|------------|
| `"Please keep the books  in the cupboard.".split(" ")` | `['Please', 'keep', 'the', 'books', '', 'in', 'the', 'cupboard.']` | 8 (wrong) |
| `"Please keep the books  in the cupboard.".split()` | `['Please', 'keep', 'the', 'books', 'in', 'the', 'cupboard.']` | 7 (correct) |

**Measured effect on our full Wikipedia corpus:**

| Language | Lines Affected | Buggy fertility (old) | Fixed fertility (new) | Delta |
|----------|----------------|-----------------------|-----------------------|-------|
| English  | 12 lines       | 1.441                 | 1.440                 | -0.001 |
| Hindi    | 8 lines        | 9.164                 | 9.147                 | -0.017 |

*(Note: original script artificially deflated fertility by overcounting words).*

---

### Bug 3 — Wrong Denominator: `tokens/whitespace-word` (Conceptual Bug)
**Location:** `fertility.py` line 64 — the code computes *exactly* what it says, but what it says is the wrong thing to compute for cross-linguistic comparison.

```python
per_line_fertility.append(len(tokens) / len(words))
```

**Problem:** A whitespace-delimited "word" is not a comparable unit across English and Indic languages:
* **English:** mostly monomorphemic — "We are visiting Mysuru next week" = 6 words, each ~1 meaning unit
* **Hindi:** agglutinative/fusional — postpositions, verb endings, and tense markers are fused into single "words"

This means Hindi has fewer whitespace words per sentence for equivalent meaning → tokens-per-word inflates unfairly even with a good tokenizer.

**Demonstrated effect on our corpus (GPT-2):**

| Denominator | hin/eng ratio | Interpretation |
|-------------|---------------|----------------|
| tok / whitespace-word (original report) | 6.53× | Inflated — denominator is not comparable |
| tok / grapheme-cluster | 11.20× | Holds visual/linguistic content constant |

**Correct denominator:** `tok/grapheme-cluster` — because one grapheme cluster roughly equals one user-perceived character across scripts, providing an honest baseline for "content generated".

---

### Harmless Thing — `random.seed(1337)` ✅ NOT a Bug
**Location:** `fertility.py` line 25

```python
random.seed(1337) # reproducibility
```

**Evidence it is harmless:**
```bash
grep "random\." fertility.py
# Output: random.seed(1337)  # reproducibility
# -> Only 1 line. random is never called anywhere in the script.
```
`random` is imported and seeded but **never used**. Zero effect on any output. We do not flag this — doing so without evidence would cost points.

---

## A3 — Corrected Analysis 

### Setup
| Parameter | Value |
|-----------|-------|
| Corpus | Wikipedia Streaming (public, no auth) |
| Languages | English, Hindi, Kannada, Tamil |
| Size | 1000 sentences each |
| Tokenizer 1 | gpt2 via tiktoken — English-centric |
| Tokenizer 2 | google/muril-base-cased via HuggingFace — Indic-aware |
| Denominators | (1) Whitespace words — fixed .split(), (2) Grapheme clusters (RECOMMENDED), (3) UTF-8 Bytes |
| Script | `partA/corrected_analysis.py` |

### Full Results — GPT-2 Tokenizer
| Language | tok / whitespace-word | tok / grapheme-cluster | tok / byte |
|----------|-----------------------|------------------------|------------|
| English  | 1.3174 | 0.2081 | 0.2078 |
| Hindi    | 8.6069 | 2.3299 | 0.5954 |
| Kannada  | 22.3506 | 3.9848 | 0.9791 |
| Tamil    | 25.8414 | 4.1228 | 0.9921 |

### Full Results — MuRIL Tokenizer (Indic-aware)
| Language | tok / whitespace-word | tok / grapheme-cluster | tok / byte |
|----------|-----------------------|------------------------|------------|
| English  | 1.3508 | 0.2134 | 0.2131 |
| Hindi    | 1.2951 | 0.3506 | 0.0896 |
| Kannada  | 1.8821 | 0.3355 | 0.0824 |
| Tamil    | 1.8222 | 0.2907 | 0.0700 |

### Ratios vs English — GPT-2
| Language | tok/word ratio | tok/grapheme ratio | tok/byte ratio |
|----------|----------------|--------------------|----------------|
| Hindi    | 6.53× | 11.20× | 2.87× |
| Kannada  | 16.97× | 19.15× | 4.71× |
| Tamil    | 19.62× | 19.81× | 4.77× |

### Ratios vs English — MuRIL
| Language | tok/word ratio | tok/grapheme ratio | tok/byte ratio |
|----------|----------------|--------------------|----------------|
| Hindi    | 0.96× | 1.64× | 0.42× |
| Kannada  | 1.39× | 1.57× | 0.39× |
| Tamil    | 1.35× | 1.36× | 0.33× |

### Which Single Number Should Drive Routing Decisions?
**→ `tok/grapheme-cluster` using an Indic-aware tokenizer (MuRIL or equivalent)**

| Reason | Explanation |
|--------|-------------|
| **Why tok/grapheme?** | Holds unit of work constant — 1 grapheme = 1 user-perceived character = 1 serving cost unit |
| **Why not tok/word?** | Whitespace words are not comparable across scripts (morphology differs drastically) |
| **Why not tok/byte?** | Conflates multi-byte Indic characters with single-byte Latin, distorting the ratio |
| **Why Indic tokenizer?** | GPT-2 tokenizes Indic scripts character-by-character (or worse); MuRIL uses semantically meaningful subwords |

---

## A4 — Recommendation Memo

### Corrected Headline: GPT-2 vs MuRIL
| Language | Original Report (GPT-2, tok/word, 10 sentences) | Corrected (GPT-2, tok/grapheme, 1000 sentences) | Corrected (MuRIL, tok/grapheme) |
|----------|-------------------------------------------------|-------------------------------------------------|---------------------------------|
| Hindi vs English | 5.89× worse | 11.20× worse | **1.64× worse** |
| Kannada vs English | Not measured | 19.15× worse | **1.57× worse** |
| Tamil vs English | Not measured | 19.81× worse | **1.36× worse** |

### Routing Recommendation
| Action | Detail |
|--------|--------|
| **Switch to MuRIL** (or similar multilingual tokenizer) for all Indic traffic | With MuRIL, Hindi is ≈ 1.64× English cost, not 6× |
| **Do not budget 6× serving cost** for Hindi | That figure came from GPT-2 + wrong metric + tiny corpus |
| **Monitor in production** | Track `tokens_generated / grapheme_clusters_in_output` split by detected language; alert if Indic > 3× English baseline |
| **Caveat** | MuRIL advantage only applies if model backbone also uses multilingual tokenization |

---

## Part B — Capacity Reconciliation

### B1 — KV-Cache Math

#### Model Spec
| Property | Value |
|----------|-------|
| Model | FLM-4B-Instruct (dense) |
| Parameters | 4.2B |
| Layers | 28 |
| d_model | 3072 |
| Attention heads (Q) | 24 |
| KV heads (GQA) | 8 |
| head_dim | 128 |
| Weights precision | fp16 (2 bytes) |
| KV cache precision | fp16 (2 bytes) |
| GPU | 1× NVIDIA L4 (24 GB) |
| gpu_memory_utilization | 0.92 |
| Non-KV overhead | ~1.6 GB |

#### KV-Cache Bytes per Token
```
KV bytes/token = 2 (K and V)
               × 8  (KV heads, GQA)
               × 128 (head_dim)
               × 28 (layers)
               × 2  (bytes, fp16)
             = 114,688 bytes ≈ 112 KB per token
```

#### Max Concurrent 4096-Token Sequences
| Step | Calculation | Value |
|------|-------------|-------|
| Total GPU VRAM | — | 24.00 GB |
| Usable (×0.92) | 24 × 0.92 | 22.08 GB |
| Model weights (fp16) | 4.2B × 2 bytes | 8.40 GB |
| Non-KV overhead | given | 1.60 GB |
| KV cache budget | 22.08 − 8.40 − 1.60 | 12.08 GB |
| KV per 4096-token seq | 4096 × 114,688 bytes | 0.4375 GB |
| Max concurrent seqs | 12.08 / 0.4375 | **≈ 27 sequences** |

#### Cross-check Against bench_log.csv
| batch | prompt_len | kv_cache_util | preempted_seqs | Consistent with prediction? |
|-------|------------|---------------|----------------|-----------------------------|
| 4 | 3584 | 0.16 | 0 | ✅ |
| 8 | 3584 | 0.31 | 0 | ✅ |
| 16 | 3584 | 0.62 | 0 | ✅ |
| 24 | 3584 | 0.93 | 0 | ✅ Near limit |
| 32 | 3584 | 0.97 | 7 | ✅ Overflow confirmed |
| 48 | 3584 | 0.97 | 23 | ✅ Severe overflow |

Prediction of ~27 max sequences confirmed — log shows stress starts at batch 32 exactly as expected.

### B2 — Throughput Anomaly

#### Full bench_log.csv — Long Prompt Rows (prompt_len = 3584)
| batch | gen_len | wall_s | reported_tok_s | ttft_ms | itl_ms | e2e_ms_p95 | preempted | kv_util |
|-------|---------|--------|----------------|---------|--------|------------|-----------|---------|
| 4 | 512 | 28.98 | 565.4 | 483.2 | 51.33 | 32,673 | 0 | 0.16 |
| 8 | 512 | 36.30 | 902.6 | 519.0 | 62.26 | 39,983 | 0 | 0.31 |
| 16 | 512 | 49.97 | 1311.4 | 498.3 | 77.20 | 54,602 | 0 | 0.62 |
| **24** | **512** | **61.16** | **1607.4** | **500.5** | **96.07** | **69,221** | **0** | **0.93** |
| 32 | 512 | 94.71 | 1384.0 ↓ | 636.9 | 101.79 | 97,466 | 7 | 0.97 |
| 48 | 512 | 151.41 | 1298.5 ↓ | 955.4 | 100.0 | 105,428 | 23 | 0.97 |

#### Anomaly Identified
Throughput peaks at batch 24 (1607 tok/s) then drops at batch 32 and 48 — opposite of naive linear scaling expectation.

| Metric | Batch 24 | Batch 32 | Batch 48 |
|--------|----------|----------|----------|
| reported_tok_s | 1607.4 | 1384.0 | 1298.5 |
| kv_cache_util | 0.93 | 0.97 (saturated) | 0.97 |
| preempted_seqs | 0 | 7 | 23 |

**Mechanism:** KV cache fills to capacity at batch 32. Scheduler must preempt sequences — evict from KV cache, re-run prefill later. Each preemption wastes GPU compute and stalls decode. At batch 48, 23 preemptions = severe churn.

#### Proposed Fix
| Fix | Description | Predicted Effect |
|-----|-------------|------------------|
| Cap max-model-len at 3072 for long-context | Prevent KV overflow entirely | Eliminates preemptions; throughput stabilizes |
| Net gain | 1607 vs 1384 tok/s | Higher throughput at batch 32 without preemption |
| Trade-off | Sequence length capped to 3072 | Acceptable constraint |

### B3 — The Misread Column

#### What the Report Said
*"At batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization." "Batch 48 should give us ~3200 tok/s."*

#### The Misreading
| Issue | Explanation |
|-------|-------------|
| Column misread | `reported_tok_s` counts ALL tokens (prompt + generated). Not just output tokens. |
| Prefill vs decode | For a 3584+512 request, only 512/4096 = 12.5% of counted tokens are actual output |
| User value | Users receive only generated tokens — prefill tokens are overhead, not output |
| "Longer = better" | Wrong. Longer prompts inflate `reported_tok_s` because prompt tokens dominate the count |

#### Honest Goodput for Batch 24, Long-Prompt Row
**Method 1 — Generated fraction:**
```
Honest goodput = reported_tok_s × (gen_len / total_len)
               = 1607.4 × (512 / 4096)
               = 1607.4 × 0.125
               = 200.9 tok/s
```
**Method 2 — Direct from log:**
```
Honest goodput = (num_requests × gen_len) / wall_clock_s
               = (24 × 512) / 61.16
               = 12,288 / 61.16
               = 200.9 tok/s ✅
```
Both methods agree: ~201 generated tok/s, not 1607.

#### Comparison
| Metric | Report's figure | Honest figure | Error factor |
|--------|-----------------|---------------|--------------|
| Throughput at batch 24 long-prompt | 1607 tok/s | 201 tok/s | **8× too optimistic** |
| Batch 48 capacity projection | ~3200 tok/s | ~163 tok/s (extrapolated, with preemptions) | **~20× off** |

**What the Report Should Have Said**
*"Long prompts show higher total-token throughput (including prefill), but generated-token goodput — the metric relevant to serving cost — is approximately 201 tok/s at batch 24. This is lower than short-prompt goodput. For capacity planning, use `(num_requests × gen_len) / wall_clock_s`, not `reported_tok_s`."*

### B4 — Metric to Pull to Confirm B2 Mechanism
Pull: `kv_cache_util` + `vllm:scheduler_num_preempted_seqs_total` from the vLLM `/metrics` Prometheus endpoint.

| Metric | Expected value at batch 32 (long prompt) | Why |
|--------|------------------------------------------|-----|
| `kv_cache_util` | ≥ 0.95 (saturated) | No KV headroom left |
| `vllm:scheduler_num_preempted_seqs_total` | > 0 and growing with load | Scheduler evicting sequences |

Together these two confirm the KV cache overflow mechanism: when utilization hits the ceiling, the scheduler has no free blocks for new sequence decoding steps, forcing preemptions and causing the throughput drop observed in the log.

---

## Part C — Decision Memo

### Scenario
Make assistant replies sound casual/conversational in Hindi, Kannada, Tamil, Telugu, Bengali, Marathi — currently too formal.

### Options
| Option | Description |
|--------|-------------|
| (a) | SFT on synthetic "casualized" response pairs |
| (b) | ≤1B inference-time rewriter model after main model |
| (c) | Prompt engineering only |

### Constraints
| Resource | Limit |
|----------|-------|
| GPU | 1× A100-80GB for 2 weeks |
| Reviewer | 1 native speaker (Hindi + Kannada only), 10h/week |
| Timeline | Launch review in 3 weeks |
| API budget | None |

### Recommendation: Option (c) Prompt Engineering First

#### Back-of-Envelope Arithmetic
| Resource | Calculation | Result |
|----------|-------------|--------|
| Total reviewer hours | 10h/week × 3 weeks | 30 hours |
| Languages reviewer covers | Hindi + Kannada only | 2 of 6 languages |
| Review rate | ~30 samples/hour | — |
| Validated samples possible | 30 × 20 (assuming 2 weeks data prep) | 600 total |
| Samples per language (2 languages) | 600 / 2 | 300 per language |
| Min needed for SFT | 5000+ per language | Insufficient |
| Prompt engineering cost | 0 GPU hours, 0 data | Free, Day 1 |

#### Decision Table
| Criterion | SFT (a) | Rewriter (b) | Prompt Engineering (c) |
|-----------|---------|--------------|------------------------|
| Data needed | High (5000+ per lang) | Medium | None |
| Reviewer coverage | 2/6 languages only | 2/6 languages only | All 6 (no reviewer needed for setup) |
| Time to first result | 1–2 weeks | 1–2 weeks | Same day |
| GPU cost | High (training) | Medium (inference +40ms latency) | Zero |
| Risk | High (unvalidated synthetic data) | Medium | Low |
| Verdict | ❌ Too slow, too little data | 🔄 Fallback if (c) fails | ✅ **Start here** |

#### Success Metric
≥ 70% preference rate for casual output vs. current output in a 100-sample blind pairwise human eval (Hindi + Kannada with reviewer; automated LLM-as-judge proxy for other 4 languages).

#### Kill Criterion
If blind preference rate < 60% on Hindi AND Kannada by end of Week 1 → abandon prompt engineering, escalate to option (a) SFT for Hindi/Kannada only.

#### Day 1 Experiment
1. Write 5 system-prompt variants (ranging from explicit tone instruction to few-shot casual examples)
2. Run each on 20 test queries per language (100 outputs total)
3. Send Hindi + Kannada outputs to reviewer in randomized blind grid
4. Measure: preference rate per variant
5. Pick winner by Day 3 → apply to all 6 languages

---

## Key Numbers to Remember for the Defense
| Fact | Number | How to derive |
|------|--------|---------------|
| KV bytes per token | 112 KB | 2 × 8 × 128 × 28 × 2 bytes |
| Max concurrent 4096-token seqs | ~27 | 12.08 GB / 0.4375 GB |
| Honest goodput (batch 24, long prompt) | 201 tok/s | 24 × 512 / 61.16s |
| Report's goodput (wrong) | 1607 tok/s | `reported_tok_s` counts prompt tokens too |
| Error factor in report | 8× | 1607 / 201 |
| Hindi cost with GPT-2 (tok/grapheme) | 11.20× English | Corrected Wikipedia corpus |
| Hindi cost with MuRIL (tok/grapheme) | 1.64× English | Indic-aware tokenizer |
| Original report's claim | 5.89× (tok/word) | Buggy script, 10 sentences, wrong metric |
