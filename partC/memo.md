# Part C — Decision Memo

**To:** Product Leadership  
**Subject:** Indic Language Casualization — Path Recommendation  
**Date:** September 2026

---

## Options on the Table

- **(a)** SFT pass on synthetic "casualized" response pairs  
- **(b)** ≤1B inference-time rewriter model after the main model  
- **(c)** Prompt-engineering only  

**Constraints:** 1× A100-80GB for 2 weeks · 1 native-speaker reviewer (Hindi + Kannada, 10 h/wk) · launch review in 3 weeks · no external API budget

---

## Recommendation: Start with (c), with a structured gate to (a)

---

## Assumptions

1. "Casual" is well-enough defined to write a rubric — likely yes for Hindi and Kannada (we have a reviewer); uncertain for Tamil, Telugu, Bengali, Marathi (no reviewer coverage).
2. The current model is capable of casual outputs in these languages; it's a style-following failure, not a capability gap. (Assumption: this has been tested informally and the model can produce casual text when explicitly prompted.)
3. 3 weeks to launch review means we need a working, evaluable artifact by week 3 — not a polished system.
4. "Synthetic casualized pairs" (option a) can be produced by prompting our main model in English and translating — a reasonable approximation but not gold-standard.

---

## Back-of-Envelope Arithmetic

### Option (a) — SFT

| item | estimate |
|---|---|
| Data needed for noticeable SFT effect | ~5,000–10,000 examples per language |
| Languages needing data | 6 |
| Total pairs needed | ~30,000–60,000 |
| Reviewer throughput (rate: ~30 pairs/hr) | 10 h/wk × 2 wk × 30 = 600 pairs reviewed |
| Reviewer language coverage | Hindi + Kannada only (2/6 languages) |
| **Gap**: 4 languages have zero native-speaker QA | **CRITICAL BLOCKER** |

Even for Hindi + Kannada: 600 reviewed pairs << 5,000 needed for robust SFT. We'd be fine-tuning on unreviewed synthetic data for 4/6 languages, which risks introducing confident-but-wrong casual register (hallucinated idioms, wrong script variants).

**A100 training estimate:**
- 4.2B model, fp16, LoRA r=64: ~20 GB VRAM → fits on A100-80GB ✓
- 30k examples, 3 epochs, batch 8: ~6–10 hours per language → ~36–60 hours total
- Feasible within 2 weeks, but quality is gated on data quality (see above)

### Option (b) — 1B Rewriter

| item | estimate |
|---|---|
| Additional serving latency | +30–80ms per request (serial decode) |
| VRAM for 1B rewriter (fp16) | ~2 GB → fits alongside main model |
| Training data needed for rewriter | Same as SFT (~5k pairs/lang) — same data problem |
| Benefit vs (a) | Modular, switchable; same root problem (no quality data for 4 langs) |

Option (b) adds serving complexity and latency without solving the data problem. Reject for now.

### Option (c) — Prompt Engineering

| item | estimate |
|---|---|
| Iteration time | 1–3 days per language |
| Cost | $0 |
| Reviewer time needed | 2–4 h/wk to evaluate outputs |
| Coverage | All 6 languages (prompt engineering is script-agnostic) |
| Risk | Inconsistent: the prompt may "wear off" in long conversations |

---

## Success Metric with Numeric Threshold

**Metric**: Side-by-side reviewer preference rate for "casual vs current" on a set of 50 fixed Hindi + Kannada prompts  
**Threshold**: ≥70% preference for casual output (vs 50% random baseline)  
**Evaluation**: Week 2, by our native-speaker reviewer (10 h budgeted)

For the 4 languages without a reviewer:  
**Metric**: LLM-as-judge (using our main model in English) preference on 100 fixed prompts, translated to English for evaluation  
**Threshold**: ≥65% preference (lower bar due to translation noise)

---

## Kill Criterion

> **If, at the end of Week 1, the prompt-engineered outputs do not reach ≥60% reviewer preference for Hindi and Kannada, abandon (c) and pivot to (a) for Hindi+Kannada only.**

This gates us: if prompting works, we launch with zero training cost. If it doesn't, we still have 1 week to collect 600 reviewer-validated pairs for Hindi+Kannada, run a narrow LoRA SFT, and deploy for those two languages — accepting that Tamil/Telugu/Bengali/Marathi remain on the prompt-engineering path.

---

## Day-1 Experiment

**Run 5 prompt variants on a fixed set of 20 prompts in Hindi:**

| variant | system prompt addition |
|---|---|
| A | (none, current baseline) |
| B | "Reply in informal, conversational Hindi. Use everyday words." |
| C | "Reply as if texting a friend. Hindi only, simple words." |
| D | "Avoid formal/textbook Hindi. Write the way a Delhi college student would." |
| E | "तुम/तू form, short sentences, everyday Hindi vocabulary." |

**Measure**: reviewer preference at end of Day 1 (2 hours of reviewer time).  
**Decision point**: pick the best variant; apply to all 6 languages by day 2.

**Why this first**: if even one prompt variant clears 70%, we've solved the problem in 1 day with no training. If none clears 60%, we have early evidence that SFT is necessary and can begin data collection immediately.

---

## Summary

| path | time to artifact | data risk | reviewer bottleneck | recommendation |
|---|---|---|---|---|
| (c) prompt | 1–3 days | none | minimal | **Start here** |
| (a) SFT | 1–2 weeks | high (4 langs unreviewed) | hard blocker | **Gate to this if (c) fails** |
| (b) rewriter | 1–2 weeks + serving change | high | same as (a) | Reject for now |
