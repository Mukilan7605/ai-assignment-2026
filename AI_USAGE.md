# AI_USAGE.md

## Summary of AI Tool Usage

This submission used AI assistance (Claude Sonnet). Here is an honest account of where it helped and where it misled.

---

## Where AI Helped

### 1. Drafting initial code structure
Claude wrote approximately 70% of the boilerplate code for `build_corpus.py`, `corrected_analysis.py`, and `fertility_fixed.py`. This was the right call — these are standard HuggingFace dataset loading patterns and argparse scaffolding that don't require original thinking.

### 2. Confirming the KV-cache arithmetic setup
When I described the model spec parameters, Claude correctly set up the arithmetic framework: `2 × layers × KV_heads × head_dim × bytes`. I verified each multiplication step manually. The formula is standard and well-documented.

### 3. Explaining vLLM preemption mechanics
I asked Claude to explain what vLLM does when KV cache is exhausted. Its explanation of preemption-and-recompute matched what I observed in the log (`preempted_seqs` spiking). Useful background confirmation.

### 4. Structuring the Part C memo
Claude suggested the "assumptions + arithmetic + kill criterion + day-1 experiment" structure. I found this useful — it forced me to think about what assumptions I was making about the reviewer bottleneck and which I had actually verified.

---

## Where AI Misled Me

### 1. FLORES-200 availability
Claude initially suggested using FLORES-200 via `facebook/flores` and `Muennighoff/flores200`, stating these were "freely available". Both turned out to be either gated (requiring HuggingFace auth) or deprecated (requiring old dataset scripts). I wasted time debugging this before switching to Wikipedia.

**Lesson**: AI knowledge of dataset availability lags reality. Always test the download yourself first.

### 2. Invented a bug that doesn't exist
When asked "what bugs are in fertility.py?", Claude initially listed `random.seed(1337)` as a "potential source of non-reproducibility if sampling is added later" — framing a non-bug as a future risk. After I verified that no random sampling occurs anywhere in the script, I correctly labelled this as dead code (harmless). Claude initially over-eagerly flagged it as suspicious.

### 3. Initial recommendation for Part C
Claude's first draft recommended Option (a) SFT, reasoning that "SFT is the gold standard for style transfer." It took me explicitly pointing out the reviewer bottleneck (600 pairs for 2 languages vs 30,000 needed for 6) before Claude recalculated and switched to recommending Option (c) prompt-engineering first. **The arithmetic was right; the initial recommendation ignored a hard constraint.**

**Lesson**: AI is good at generating comprehensive lists but can miss binding constraints unless you explicitly surface them. The business constraint (only 2/6 languages have a reviewer) was in the problem statement — AI skimmed past it.

### 4. The `.lower()` bug interpretation
Claude initially called `.lower()` "harmless" for Indic scripts. This is technically true (it's a no-op for Devanagari), but the more important point — that it DOES change English tokenization (acronyms tokenize differently after lowercasing) and creates an inconsistency — required me to run the experiment myself. AI analysis was shallow here; experimental evidence was needed.

---

## What I Understand vs What AI Wrote

| component | who wrote it | do I understand it? |
|---|---|---|
| KV cache bytes formula | AI scaffold + I verified | Yes — I can re-derive from scratch |
| Goodput calculation | I derived; AI formatted | Yes — I verified both methods |
| Preemption mechanism explanation | AI draft + I verified against log | Yes — I can explain each row |
| fertility_fixed.py core logic | AI scaffold + I reviewed | Yes — I can explain each change |
| build_corpus.py HuggingFace streaming | ~80% AI | Partially — I understand the structure but not every HF streaming edge case |
| Part C arithmetic (reviewer throughput) | I computed; AI formatted | Yes |
| NOTEBOOK dead ends | Real dead ends I encountered | Yes |
