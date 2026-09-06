# AI Assignment 2026: The Audit

This repository contains the complete solution for the **AI Assignment 2026 "The Audit"** take-home assessment. It covers a full audit of an intern's faulty cross-lingual tokenizer report, mathematical reconciliation of vLLM serving capacity, and a strategic decision memo for Indic language casualization.

## 📁 Repository Structure

```
.
├── README.md              ← You are here
├── NOTEBOOK.md            ← Chronological lab notebook showing my thought process
├── AI_USAGE.md            ← Honest log of how AI assistance was used and where it failed
├── partA/                 ← Tokenizer Audit scripts & data
│   ├── build_corpus.py
│   ├── audit_fertility.py
│   ├── fertility_fixed.py
│   ├── corrected_analysis.py
│   ├── memo_A4.md
│   └── corpora/           ← Generated Wikipedia samples for 4 languages
├── partB/
│   └── analysis.md        ← KV-cache arithmetic and throughput anomaly explanation
└── partC/
    └── memo.md            ← Strategic decision memo for the casualization rollout
```

---

## 🚀 Part A: Tokenizer Audit
The original `REPORT_v0` claimed that Hindi text costs ~6x more to serve than English, concluding this was a "property of the script." This audit proves that conclusion **entirely false**.

*   **Bugs Identified:** `audit_fertility.py` proves three implementation bugs (whitespace splitting, inconsistent `.lower()`, micro-averaging) and one critical conceptual flaw in the original script.
*   **The Conceptual Flaw:** The original script used `tokens per whitespace-word` to compare languages. Because Indic languages compound differently, this metric is fundamentally biased.
*   **The Correction:** `corrected_analysis.py` uses the script-neutral **Tokens per Grapheme Cluster** metric. It tests both GPT-2 and MuRIL (Indic-aware).
*   **Conclusion:** With an Indic-aware tokenizer, the overhead for Hindi/Dravidian languages drops from ~11x to a mere **1.36x - 1.64x**. The issue was purely a tokenizer failure. 

---

## 🧮 Part B: Capacity Reconciliation
The vLLM serving logs showed a counter-intuitive anomaly: throughput dropped significantly as batch size increased from 24 to 32.

*   **KV-Cache Math:** `partB/analysis.md` derives the exact token capacity (~113,060 tokens) for a 24GB GPU.
*   **The Anomaly:** At batch size 32, the sequence length demand exceeds KV-cache capacity, causing vLLM to continuously **preempt** and recompute sequences.
*   **"Goodput":** The report's claim of 3200 tok/s at batch 48 was fundamentally misread. The actual generative "goodput" to users at the optimal batch 24 is **~201 tok/s**.

---

## 📝 Part C: Indic Casualization Strategy
A strategy memo outlining the path forward for casualizing assistant responses in 6 Indic languages within strict hardware and reviewer constraints.

*   **The Blocker:** Option A (Supervised Fine-Tuning) requires ~5,000 reviewed pairs per language. We only have QA coverage for 2 out of 6 languages. SFT is impossible to validate for the remaining four.
*   **The Strategy:** Option C (Prompt Engineering) is recommended for Day 1, with a strict metric: if it fails to achieve 60% reviewer preference by Week 1, we pivot to a narrow SFT approach for Hindi and Kannada only.

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
*(Use `python -X utf8` on Windows to ensure Indic characters print correctly to the terminal).*
