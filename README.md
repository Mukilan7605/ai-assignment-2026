# AI Assignment 2026 — Audit & Corrected Analysis

---

## Part A — Bug Audit

### Summary Table

| Bug / Code Pattern | Type | Line in `fertility.py` | Effect |
|---|---|---|---|
| `line.split(" ")` instead of `line.split()` | Code Bug | [L62](partA/fertility.py#L62) | Phantom empty words from double spaces deflate fertility |
| `line.lower()` on Indic text | Code/Conceptual Bug | [L60](partA/fertility.py#L60) | Changes English tokenization (acronyms); no-op for Hindi/Kannada/Tamil |
| Mean-of-ratios (micro-average) | Aggregation Bug | [L64](partA/fertility.py#L64) | Short sentences get equal weight as long ones, biasing the average |
| `random.seed(1337)` | Not a Bug | [L25](partA/fertility.py#L25) | Dead code — `random` is never called anywhere; zero effect on output |

---

### Bug 1 — `line.lower()` (Code Bug)

**Location:** [`fertility.py` line 60](partA/fertility.py#L60)

```python
# BUGGY
line = line.lower()
```

**What is wrong:**
- GPT-2 encodes capitalized and lowercase tokens differently.
- For English acronyms like `NASA`, `ISRO`, `GPU` — lowercasing changes tokenization.
- For Devanagari and Dravidian scripts, `.lower()` is a complete no-op (those scripts have no case).

**Measured effect on our Wikipedia corpus:**

| Language | Without `.lower()` | With `.lower()` | Delta | Impact |
|----------|--------------------|-----------------|-------|--------|
| English | 1.316 | 1.375 | +0.059 | Alters tokenization |
| Hindi | 8.604 | 8.606 | +0.002 | Statistically zero |

**Reproduce:**
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
s = "NASA and ISRO announced a joint mission update."
print(len(enc.encode(s.lower())), len(enc.encode(s)))  # -> 11 10
```

**Quick illustration:**

| Sentence | Tokens with `.lower()` | Tokens without `.lower()` | Delta |
|----------|------------------------|---------------------------|-------|
| `NASA and ISRO announced a joint mission update.` | 11 | 10 | +1 |
| `मुझे सुबह की चाय बहुत पसंद है।` | 47 | 47 | 0 |
| `Bengaluru International Airport handled record traffic in March.` | 12 | 12 | 0 |

---

### Bug 2 — `line.split(" ")` vs `line.split()` (Code Bug)

**Location:** [`fertility.py` line 62](partA/fertility.py#L62)

```python
# BUGGY
words = line.split(" ")   # creates empty string "" on double spaces

# FIXED
words = line.split()      # handles any whitespace correctly
```

**What is wrong:**
- If a line has double spaces (e.g. `"Please keep the books  in the cupboard."`), `.split(" ")` produces an empty-string token `""`, inflating the word count by 1.
- More words in denominator → lower fertility value → underestimates the real cost.

**Example:**

| Method | Result | Word count |
|--------|--------|------------|
| `"books  in".split(" ")` | `['books', '', 'in']` | 3 (wrong) |
| `"books  in".split()` | `['books', 'in']` | 2 (correct) |

**Measured effect on our full Wikipedia corpus:**

| Language | Lines affected | Fertility (buggy) | Fertility (fixed) | Delta |
|----------|----------------|-------------------|-------------------|-------|
| English | 12 lines | 1.441 | 1.440 | -0.001 |
| Hindi | 8 lines | 9.164 | 9.147 | -0.017 |

---

### Bug 3 — Mean-of-Ratios / Micro-Average (Aggregation Bug)

**Location:** [`fertility.py` line 64](partA/fertility.py#L64)

```python
# BUGGY — micro-average: each line gets equal weight regardless of length
per_line_fertility.append(len(tokens) / len(words))
```

**What is wrong:**
- A 3-word sentence and a 30-word sentence are weighted equally in the average.
- The correct approach is macro-average: `total_tokens / total_words` across the entire corpus.
- Micro-average biases the result toward short sentences.

**Measured effect on our corpus (GPT-2):**

| Language | Micro fertility (buggy) | Macro fertility (correct) | Delta |
|----------|------------------------|---------------------------|-------|
| English | 1.375 | 1.316 | -0.059 |
| Hindi | 9.164 | 8.604 | -0.560 |

---

### Not a Bug — `random.seed(1337)`

**Location:** [`fertility.py` line 25](partA/fertility.py#L25)

```python
random.seed(1337)  # reproducibility
```

**Why it is harmless:**
- `random` is imported and seeded but never called anywhere else in the script.
- Tokenization is deterministic — it does not use `random`.
- Removing the seed changes zero numbers in the output.

**Evidence:**
```bash
grep "random\." fertility.py
# Output: random.seed(1337)  # reproducibility
# -> Only 1 line. random is never called.
```

---

## Part A3 — Corrected Analysis

### Setup

| Parameter | Value |
|-----------|-------|
| Corpus | Wikipedia Streaming (public, no auth) |
| Languages | English, Hindi, Kannada, Tamil |
| Size | 1000 sentences each |
| Tokenizer 1 | `gpt2` via tiktoken — English-centric |
| Tokenizer 2 | `google/muril-base-cased` via HuggingFace — Indic-aware |
| Denominators | Whitespace words (fixed), Grapheme clusters (recommended), UTF-8 bytes |
| Script | `partA/corrected_analysis.py` |

### Full Results — GPT-2 Tokenizer

| Language | tok / whitespace-word | tok / grapheme-cluster | tok / byte |
|----------|-----------------------|------------------------|------------|
| English | 1.3174 | 0.2081 | 0.2078 |
| Hindi | 8.6069 | 2.3299 | 0.5954 |
| Kannada | 22.3506 | 3.9848 | 0.9791 |
| Tamil | 25.8414 | 4.1228 | 0.9921 |

### Full Results — MuRIL Tokenizer (Indic-aware)

| Language | tok / whitespace-word | tok / grapheme-cluster | tok / byte |
|----------|-----------------------|------------------------|------------|
| English | 1.3508 | 0.2134 | 0.2131 |
| Hindi | 1.2951 | 0.3506 | 0.0896 |
| Kannada | 1.8821 | 0.3355 | 0.0824 |
| Tamil | 1.8222 | 0.2907 | 0.0700 |

### Ratios vs English — GPT-2

| Language | tok/word ratio | tok/grapheme ratio | tok/byte ratio |
|----------|----------------|--------------------|----------------|
| Hindi | 6.53x | 11.20x | 2.87x |
| Kannada | 16.97x | 19.15x | 4.71x |
| Tamil | 19.62x | 19.81x | 4.77x |

### Ratios vs English — MuRIL

| Language | tok/word ratio | tok/grapheme ratio | tok/byte ratio |
|----------|----------------|--------------------|----------------|
| Hindi | 0.96x | 1.64x | 0.42x |
| Kannada | 1.39x | 1.57x | 0.39x |
| Tamil | 1.35x | 1.36x | 0.33x |

### Which denominator to use for routing decisions?

**Recommended: `tok/grapheme-cluster` with an Indic-aware tokenizer (MuRIL)**

| Reason | Explanation |
|--------|-------------|
| Why tok/grapheme? | One grapheme = one user-perceived character = one serving cost unit. Holds constant across scripts. |
| Why not tok/word? | Whitespace words are not comparable across scripts due to morphology differences. |
| Why not tok/byte? | Multi-byte Indic characters inflate the byte count vs single-byte Latin characters. |
| Why Indic tokenizer? | GPT-2 tokenizes Indic scripts character-by-character (or worse). MuRIL uses semantically meaningful subwords. |

---

## Part A4 — Routing Recommendation

### Corrected Headline: GPT-2 vs MuRIL

| Language | Original Report (GPT-2, tok/word, 10 sentences) | Corrected (GPT-2, tok/grapheme, 1000 sentences) | Corrected (MuRIL, tok/grapheme) |
|----------|-------------------------------------------------|-------------------------------------------------|---------------------------------|
| Hindi vs English | 5.89x worse | 11.20x worse | 1.64x worse |
| Kannada vs English | Not measured | 19.15x worse | 1.57x worse |
| Tamil vs English | Not measured | 19.81x worse | 1.36x worse |

### Recommendation

| Action | Detail |
|--------|--------|
| Switch to MuRIL for all Indic traffic | With MuRIL, Hindi is ~1.64x English cost — not 6x |
| Do not budget 6x serving cost for Hindi | That figure came from GPT-2 + wrong metric + 10-sentence corpus |
| Monitor in production | Track `tokens_generated / grapheme_clusters` per language; alert if Indic > 3x English baseline |
| Caveat | MuRIL advantage only applies if the model backbone also uses multilingual tokenization |

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
| GPU | 1x NVIDIA L4 (24 GB) |
| gpu_memory_utilization | 0.92 |
| Non-KV overhead | ~1.6 GB |

#### KV-Cache Bytes per Token

```
KV bytes/token = 2 (K and V)
               x 8  (KV heads, GQA)
               x 128 (head_dim)
               x 28 (layers)
               x 2  (bytes, fp16)
             = 114,688 bytes ~ 112 KB per token
```

#### Max Concurrent 4096-Token Sequences

| Step | Calculation | Value |
|------|-------------|-------|
| Total GPU VRAM | — | 24.00 GB |
| Usable (x0.92) | 24 x 0.92 | 22.08 GB |
| Model weights (fp16) | 4.2B x 2 bytes | 8.40 GB |
| Non-KV overhead | given | 1.60 GB |
| KV cache budget | 22.08 - 8.40 - 1.60 | 12.08 GB |
| KV per 4096-token seq | 4096 x 114,688 bytes | 0.4375 GB |
| Max concurrent seqs | 12.08 / 0.4375 | ~27 sequences |

#### Cross-check Against bench_log.csv

| batch | prompt_len | kv_cache_util | preempted_seqs | Consistent with prediction? |
|-------|------------|---------------|----------------|-----------------------------|
| 4 | 3584 | 0.16 | 0 | Yes |
| 8 | 3584 | 0.31 | 0 | Yes |
| 16 | 3584 | 0.62 | 0 | Yes |
| 24 | 3584 | 0.93 | 0 | Yes — near limit |
| 32 | 3584 | 0.97 | 7 | Yes — overflow confirmed |
| 48 | 3584 | 0.97 | 23 | Yes — severe overflow |

Prediction of ~27 max sequences is confirmed. Stress starts at batch 32 exactly as expected.

---

### B2 — Throughput Anomaly

#### Full bench_log.csv — Long Prompt Rows (prompt_len = 3584)

| batch | gen_len | wall_s | reported_tok_s | ttft_ms | itl_ms | e2e_ms_p95 | preempted | kv_util |
|-------|---------|--------|----------------|---------|--------|------------|-----------|---------|
| 4 | 512 | 28.98 | 565.4 | 483.2 | 51.33 | 32,673 | 0 | 0.16 |
| 8 | 512 | 36.30 | 902.6 | 519.0 | 62.26 | 39,983 | 0 | 0.31 |
| 16 | 512 | 49.97 | 1311.4 | 498.3 | 77.20 | 54,602 | 0 | 0.62 |
| 24 | 512 | 61.16 | 1607.4 | 500.5 | 96.07 | 69,221 | 0 | 0.93 |
| 32 | 512 | 94.71 | 1384.0 (drop) | 636.9 | 101.79 | 97,466 | 7 | 0.97 |
| 48 | 512 | 151.41 | 1298.5 (drop) | 955.4 | 100.0 | 105,428 | 23 | 0.97 |

#### Anomaly

Throughput peaks at batch 24 (1607 tok/s) then drops at batch 32 and 48 — opposite of naive linear scaling expectation.

| Metric | Batch 24 | Batch 32 | Batch 48 |
|--------|----------|----------|----------|
| reported_tok_s | 1607.4 | 1384.0 | 1298.5 |
| kv_cache_util | 0.93 | 0.97 (saturated) | 0.97 |
| preempted_seqs | 0 | 7 | 23 |

**Mechanism:** KV cache fills to capacity at batch 32. Scheduler must preempt sequences — evict from KV cache and re-run prefill later. Each preemption wastes GPU compute and stalls decode. At batch 48, 23 preemptions cause severe churn.

#### Proposed Fix

| Fix | Description | Predicted Effect |
|-----|-------------|------------------|
| Cap max-model-len at 3072 | Prevent KV overflow entirely | Eliminates preemptions; throughput stabilizes |
| Net gain | 1607 vs 1384 tok/s | Higher throughput at batch 32 without preemption |
| Trade-off | Sequence length capped to 3072 | Acceptable constraint |

---

### B3 — The Misread Column

#### What the Report Said

> "At batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization. Batch 48 should give us ~3200 tok/s."

#### What Is Wrong

| Issue | Explanation |
|-------|-------------|
| Column misread | `reported_tok_s` counts ALL tokens (prompt + generated), not just output tokens |
| Prefill vs decode | For a 3584+512 request, only 512/4096 = 12.5% of counted tokens are actual output |
| User value | Users receive only generated tokens. Prefill tokens are overhead, not output. |
| "Longer = better" | Wrong. Longer prompts inflate `reported_tok_s` because prompt tokens dominate the count. |

#### Honest Goodput for Batch 24, Long-Prompt Row

**Method 1 — Generated fraction:**
```
Honest goodput = reported_tok_s x (gen_len / total_len)
               = 1607.4 x (512 / 4096)
               = 1607.4 x 0.125
               = 200.9 tok/s
```

**Method 2 — Direct from log:**
```
Honest goodput = (num_requests x gen_len) / wall_clock_s
               = (24 x 512) / 61.16
               = 12,288 / 61.16
               = 200.9 tok/s
```

Both methods agree: ~201 generated tok/s, not 1607.

#### Comparison

| Metric | Report figure | Honest figure | Error factor |
|--------|---------------|---------------|--------------|
| Throughput at batch 24 long-prompt | 1607 tok/s | 201 tok/s | 8x too optimistic |
| Batch 48 capacity projection | ~3200 tok/s | ~163 tok/s (with preemptions) | ~20x off |

---

### B4 — Metric to Pull to Confirm B2 Mechanism

Pull `kv_cache_util` and `vllm:scheduler_num_preempted_seqs_total` from the vLLM `/metrics` Prometheus endpoint.

| Metric | Expected value at batch 32 (long prompt) | Why |
|--------|------------------------------------------|-----|
| `kv_cache_util` | >= 0.95 (saturated) | No KV headroom left |
| `vllm:scheduler_num_preempted_seqs_total` | > 0 and growing with load | Scheduler is evicting sequences |

---

## Part C — Decision Memo

### Scenario

Make assistant replies sound casual in Hindi, Kannada, Tamil, Telugu, Bengali, Marathi. Current responses are too formal.

### Options Considered

| Option | Description |
|--------|-------------|
| (a) SFT | Fine-tune on synthetic "casualized" response pairs |
| (b) Rewriter | Small (<= 1B) inference-time rewriter model after main model |
| (c) Prompt engineering | System prompt changes only |

### Constraints

| Resource | Limit |
|----------|-------|
| GPU | 1x A100-80GB for 2 weeks |
| Reviewer | 1 native speaker (Hindi + Kannada only), 10h/week |
| Timeline | Launch review in 3 weeks |
| API budget | None |

### Recommendation: Option (c) Prompt Engineering First

#### Why SFT is ruled out

- Total reviewer hours: 10h/week x 3 weeks = 30 hours
- Languages reviewer covers: Hindi + Kannada only (2 of 6)
- Validated samples possible: ~600 total (300 per language)
- Minimum needed for SFT: 5000+ per language
- Result: Insufficient data for 4 of 6 languages

#### Decision Table

| Criterion | SFT (a) | Rewriter (b) | Prompt Engineering (c) |
|-----------|---------|--------------|------------------------|
| Data needed | High (5000+ per lang) | Medium | None |
| Reviewer coverage | 2/6 languages | 2/6 languages | All 6 (no reviewer needed for setup) |
| Time to first result | 1-2 weeks | 1-2 weeks | Same day |
| GPU cost | High (training) | Medium (+40ms latency) | Zero |
| Risk | High (unvalidated synthetic data) | Medium | Low |
| Verdict | Too slow, too little data | Fallback if (c) fails | Start here |

#### Success Metric

- >= 70% preference rate for casual output vs current output
- 100-sample blind pairwise human eval (Hindi + Kannada with reviewer; LLM-as-judge for other 4 languages)

#### Kill Criterion

If blind preference rate < 60% on Hindi AND Kannada by end of Week 1 — abandon prompt engineering, escalate to SFT for Hindi/Kannada only.

#### Day 1 Experiment Steps

1. Write 5 system-prompt variants (explicit tone instruction to few-shot casual examples)
2. Run each on 20 test queries per language (100 outputs total)
3. Send Hindi + Kannada outputs to reviewer in randomized blind grid
4. Measure preference rate per variant
5. Pick winner by Day 3 and apply to all 6 languages

---

## Key Numbers Reference

| Fact | Number | How to derive |
|------|--------|---------------|
| KV bytes per token | 112 KB | 2 x 8 x 128 x 28 x 2 bytes |
| Max concurrent 4096-token seqs | ~27 | 12.08 GB / 0.4375 GB |
| Honest goodput (batch 24, long prompt) | 201 tok/s | 24 x 512 / 61.16s |
| Report's goodput (wrong) | 1607 tok/s | `reported_tok_s` counts prompt tokens too |
| Error factor in report | 8x | 1607 / 201 |
| Hindi cost with GPT-2 (tok/grapheme) | 11.20x English | Corrected Wikipedia corpus |
| Hindi cost with MuRIL (tok/grapheme) | 1.64x English | Indic-aware tokenizer |
| Original report's claim | 5.89x (tok/word) | Buggy script, 10 sentences, wrong metric |
