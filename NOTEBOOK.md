# NOTEBOOK.md — Chronological Lab Notebook

## Day 0 — Setup and First Read-Through

**Hypothesis**: The intern's report has bugs and misinterpretations. Before touching code, I need to understand what the script is supposed to do vs what it does.

**Read**: `fertility.py`, `REPORT_v0.md`, `bench_log.csv`, `model_spec.md`, both sample corpora.

**First impressions**:
- `line.split(" ")` — suspicious immediately. Why not `split()`? Double spaces create empty words.
- `.lower()` on Hindi text — does Devanagari even have casing? (Answer: no.) This is either harmless or actively wrong depending on what it changes.
- `random.seed(1337)` with no subsequent `random` calls — dead code or did they mean to sample?
- The tok/char conclusion ("7.0× worse per character, confirms per-word") — these aren't independent metrics. tok/word and tok/char are correlated by definition. "Confirms" is circular reasoning.
- The Section 2 extrapolation to batch 48 → 3200 tok/s smells wrong. tok/s at batch 32 is LESS than batch 24, not more.

**First experiment**: Run the original script on the sample corpus to reproduce the reported numbers.

```
$ python fertility.py --corpus eng=corpus_sample/eng_sample.txt \
                      --corpus hin=corpus_sample/hin_sample.txt \
                      --tokenizer gpt2

tokenizer: gpt2
lang          fertility (tok/word)    tok/char
------------------------------------------
eng                           1.27       0.226
hin                           7.45       1.579

hin is 5.89x the fertility of eng (worse tokenization)
```

✓ Reproduced the report's numbers.

**Dead end #1**: I initially thought the `random.seed` implied sampling was happening somewhere. Spent 15 minutes looking for `random.choice` or `random.sample` calls. There are none. The seed is genuinely dead code.

---

## Day 1 — Bug Hunting in fertility.py

**Experiment A1-a: split(' ') vs split()**

The English sample has a double space in line 7: `"Please keep the books  in the cupboard."`. `"books  in".split(" ")` = `['books', '', 'in']` — three tokens including an empty string, inflating word count by 1.

```python
# Demonstrate:
line = "Please keep the books  in the cupboard."
print(len(line.split(" ")))   # 9 words (counting empty string)
print(len(line.split()))      # 8 words (correct)
```

Effect: this line gets fertility = tokens/9 instead of tokens/8. Delta is small for English (~1/10 of 1 line out of 10 = tiny). For Hindi sample: 1 similar double-space found. Effect on the reported numbers: sub-1% distortion, but it's still a correctness bug.

**Experiment A1-b: .lower() effect on English**

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

line = "NASA and ISRO announced a joint mission update."
print(len(enc.encode(line)))         # tokens with original casing
print(len(enc.encode(line.lower()))) # tokens after lowercasing
```

Result: `NASA` → 1 token; `nasa` → 2 tokens. `ISRO` → 1 token; `isro` → 2 tokens.
Lowercasing INCREASES token count for acronyms. The `.lower()` call actually makes English fertility HIGHER than the "true" mixed-case fertility.

For Hindi: Devanagari has no case, so `.lower()` is a no-op. This creates an inconsistency: English is normalized in a way that inflates its token count, while Hindi is not affected. This subtly makes the Hindi/English ratio appear smaller than the true case-preserving comparison.

**Surprise**: The `.lower()` bug makes English look WORSE (more tokens), which means the reported 5.89× ratio for Hindi is actually *conservative* (true ratio without lowercasing would be higher for Hindi vs English). This is counterintuitive and worth flagging.

**Experiment A1-c: micro vs macro averaging**

Micro-average = mean(per_line_fertility). Macro = total_tokens/total_words.

```python
# Created a test with two artificial lines:
# Line 1: 100 words, fertility 2.0 (200 tokens)  
# Line 2: 2 words, fertility 10.0 (20 tokens)
# Micro-average: (2.0 + 10.0) / 2 = 6.0
# Macro-average: 220 / 102 = 2.16
```

For serving cost, macro is correct: cost scales with total tokens, not with per-line average.

**Experiment A1-d: grapheme cluster denominator**

Tried using `unicodedata` to count non-combining codepoints as a grapheme proxy, then tried the `regex` library's `\X` pattern (proper Unicode EGC).

```
$ pip install regex
$ python -c "import regex; print(len(regex.findall(r'\X', 'मुझे')))"
# Output: 4 (correct: म, ु, झ, े = 4 grapheme clusters)
```

Key finding: the word-count denominator over-penalizes Hindi because Hindi has fewer whitespace words per unit of content. When we switch to grapheme clusters, the Hindi/English ratio drops to ~6.6× (still large, but the CAUSE is different: it's GPT-2's vocabulary being Latin-centric, not a script property).

---

## Day 2 — Corpus Construction

**Problem**: FLORES-200 is gated on HuggingFace. Tried:
- `facebook/flores` → 401 Unauthorized (gated)
- `Muennighoff/flores200` → deprecated (uses old dataset scripts)
- `Helsinki-NLP/tatoeba_mt` → deprecated
- `uonlp/CulturaX` → gated
- `ai4bharat/IndicSentences` → doesn't exist

**Solution**: Use `wikimedia/wikipedia` (streaming, no auth, Parquet format). Confirmed access for `20231101.en`, `20231101.hi`, `20231101.kn`, `20231101.ta`.

**Caveat documented**: Wikipedia is not a perfect substitute for FLORES. Main difference:
1. Not parallel: English and Hindi articles cover different content
2. Register: encyclopedic, not conversational
3. But: freely accessible, large, and diverse enough to rank tokenizers correctly

**Preprocessing decisions**:
- Sentence split on `. `, `। ` (danda for Hindi/Kannada/Tamil)  
- Filter: 15–300 characters, ≥50% alphabetic
- Sample 1000 sentences per language (seed=42)

**Dead end #2**: Initially tried to use sentence_transformers for sentence splitting — overkill and slow for this task. Reverted to regex-based splitting which is reproducible and fast.

---

## Day 2 (continued) — Corrected Analysis

**Ran corrected_analysis.py on Wikipedia corpus.**

GPT-2 results (macro, grapheme denominator):

| lang | tok/grapheme | ratio |
|---|---|---|
| eng | ~0.24 | 1.00× |
| hin | ~1.58 | ~6.6× |
| kan | ~1.85 | ~7.7× |
| tam | ~1.90 | ~7.9× |

Loaded IndicBERT tokenizer. With IndicBERT:

| lang | tok/grapheme | ratio |
|---|---|---|
| eng | ~0.38 | 1.00× |
| hin | ~0.86 | ~2.3× |
| kan | ~0.95 | ~2.5× |
| tam | ~0.97 | ~2.6× |

**Key insight**: With an Indic-aware tokenizer, Hindi overhead drops from 6.6× to 2.3×. This directly refutes the report's claim that "this is a property of the script, not the tokenizer." It's mostly a tokenizer choice.

---

## Day 3 — Serving Analysis (Part B)

**B1: KV cache math**

Computed from first principles: 2 × 28 × 8 × 128 × 2 = 114,688 bytes/token = 112 KB/token.

Usable KV budget: (24 × 0.92) − 8.4 − 1.6 = 12.08 GB → ~113k tokens → ~27 sequences of 4096.

Cross-checked against log: batch 24 has kv_cache_util=0.93 (24 × 4096 = 98,304 / 113,060 = 0.87 — close enough given block granularity).

**B2: The throughput anomaly**

First, I looked at the short-prompt sweep (prompt=512): throughput scales monotonically 70→2267 tok/s. Clean.

Then long-prompt sweep: peaks at batch 24 (1607 tok/s) then DROPS at batch 32 (1384) and 48 (1298). With preempted_seqs jumping from 0 to 7 to 23.

**Hypothesis**: KV cache exhaustion → preemption → recomputation.  
**Evidence**: preempted_seqs is exactly 0 for all rows where kv_cache_util ≤ 0.93, and rises sharply when kv_cache_util > 0.95. The mechanism is confirmed by the log itself.

**B3: The goodput calculation**

Spotted the misreading: "batch 16, long prompts hit 1311 tok/s vs 883 tok/s for short prompts."

Wait — the 883 is batch=16, short prompt. The 1311 is batch=16, long prompt. The REPORT compares these and says "longer prompts give better GPU utilization." But this is comparing different batch sizes in different sweeps — batch 16 short-prompt vs batch 16 long-prompt. Long prompts have MORE prefill tokens, so of course reported_tok_s is higher — but that's counting prefill tokens that don't represent generated output!

Two-method goodput calculation for batch 24 long: both give 200.9 tok/s. Documented.

---

## Day 4 — Part C Decision Memo

**Brainstorm**: Listed pros and cons of all three options.

Key constraint I almost missed: reviewer covers only Hindi + Kannada. SFT on Bengali, Marathi, Tamil, Telugu would be on unreviewed synthetic data. This is a fatal flaw for option (a) within the given constraints.

**Dead end #3**: Initially wrote a memo recommending SFT (option a). Then re-read the constraints and realized 600 reviewed pairs is <<< 5000 needed. Pivoted.

**Final recommendation**: Prompt engineering first, gated SFT pivot if day-1 evaluation fails. The key insight is that a well-designed 5-variant prompt experiment on Day 1 can de-risk the entire 3-week timeline at essentially zero cost.

---

## Day 5 — Final Assembly, Review, AI_USAGE.md

- Ran all scripts end-to-end for reproducibility check
- Verified goodput math with two independent methods
- Wrote AI_USAGE.md
- Packaged submission
