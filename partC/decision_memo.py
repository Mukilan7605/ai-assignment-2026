#!/usr/bin/env python3
"""
decision_memo.py  --  Part C: Decision Memo

Prints a structured, evidence-backed decision memo for the
Indic language casualization problem.

No external packages required -- uses only Python standard library.

Run:
    python decision_memo.py
"""

W = 74  # output width


def line(char="-"):
    print(char * W)


def header(title, char="="):
    print()
    print(char * W)
    print(f"  {title}")
    print(char * W)


def section(title):
    print()
    print("-" * W)
    print(f"  {title}")
    print("-" * W)


def main():

    # ── Title block ───────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  PART C -- DECISION MEMO")
    print("  To      : Product Leadership")
    print("  Subject : Indic Language Casualization -- Path Recommendation")
    print("  Date    : September 2026")
    print("=" * W)

    # ── Scenario ──────────────────────────────────────────────────────────────
    section("SCENARIO")
    print()
    print("  Problem : Assistant replies sound too formal in 6 Indic languages.")
    print("  Target  : Hindi, Kannada, Tamil, Telugu, Bengali, Marathi")
    print()
    print("  Options:")
    print("    (a) SFT        -- Fine-tune on synthetic 'casualized' response pairs")
    print("    (b) Rewriter   -- Small (<= 1B) inference-time rewriter after main model")
    print("    (c) Prompting  -- System prompt / few-shot engineering only")
    print()
    print("  Hard Constraints:")
    print("    GPU      : 1x A100-80GB for 2 weeks")
    print("    Reviewer : 1 native speaker (Hindi + Kannada ONLY), 10 h/week")
    print("    Timeline : Launch review in 3 weeks")
    print("    Budget   : No external API budget")

    # ── Back-of-envelope arithmetic ───────────────────────────────────────────
    section("BACK-OF-ENVELOPE ARITHMETIC")

    # Reviewer capacity
    reviewer_hrs_per_week = 10
    weeks_available       = 2
    review_rate           = 30        # pairs/hour
    total_reviewed        = reviewer_hrs_per_week * weeks_available * review_rate
    langs_covered         = 2         # Hindi + Kannada only
    sft_needed_per_lang   = 5000

    print()
    print("  Reviewer capacity:")
    print(f"    Hours available  : {reviewer_hrs_per_week} h/week x {weeks_available} weeks = "
          f"{reviewer_hrs_per_week * weeks_available} hours")
    print(f"    Review rate      : ~{review_rate} pairs/hour")
    print(f"    Total pairs      : {total_reviewed} pairs reviewed")
    print(f"    Languages covered: {langs_covered} of 6  (Hindi + Kannada only)")
    print(f"    Pairs per lang   : {total_reviewed // langs_covered} pairs")
    print(f"    SFT minimum need : {sft_needed_per_lang}+ pairs per language")
    gap = sft_needed_per_lang - (total_reviewed // langs_covered)
    print(f"    Shortfall        : {gap:,} pairs  -- CRITICAL BLOCKER for SFT")

    print()
    print("  Option (a) -- SFT analysis:")
    print(f"  {'Item':<45} {'Estimate'}")
    print(f"  {'  ':-<60}")
    sft_data = [
        ("Pairs needed for noticeable SFT effect", "5,000-10,000 per language"),
        ("Total languages needing data",           "6"),
        ("Total pairs needed",                     "~30,000 - 60,000"),
        ("Reviewer-validated pairs possible",      f"{total_reviewed} (2 langs only)"),
        ("Languages with ZERO native QA",          "4 of 6 -- CRITICAL BLOCKER"),
        ("A100 training time (LoRA r=64)",         "~6-10 hours per language"),
        ("Total training time",                    "~36-60 hours (fits in 2 weeks)"),
        ("Risk",                                   "Unreviewed synthetic data for 4 langs"),
    ]
    for item, est in sft_data:
        print(f"  {item:<45} {est}")

    print()
    print("  Option (b) -- Rewriter analysis:")
    print(f"  {'Item':<45} {'Estimate'}")
    print(f"  {'  ':-<60}")
    rew_data = [
        ("Extra serving latency per request",    "+30-80 ms (serial decode)"),
        ("VRAM for 1B rewriter (fp16)",          "~2 GB -- fits alongside main model"),
        ("Training data needed",                 "Same as SFT -- same data problem"),
        ("Added complexity",                     "New serving component to maintain"),
        ("Verdict",                              "Reject -- solves nothing, adds cost"),
    ]
    for item, est in rew_data:
        print(f"  {item:<45} {est}")

    print()
    print("  Option (c) -- Prompt engineering analysis:")
    print(f"  {'Item':<45} {'Estimate'}")
    print(f"  {'  ':-<60}")
    pe_data = [
        ("Time to first result",                 "1-3 days"),
        ("GPU cost",                             "Zero"),
        ("API / data cost",                      "Zero"),
        ("Languages covered",                    "All 6 (prompt is script-agnostic)"),
        ("Reviewer time needed",                 "2-4 h/week to evaluate outputs"),
        ("Risk",                                 "May 'wear off' in long conversations"),
    ]
    for item, est in pe_data:
        print(f"  {item:<45} {est}")

    # ── Decision ──────────────────────────────────────────────────────────────
    section("RECOMMENDATION")
    print()
    print("  START WITH (c) PROMPT ENGINEERING")
    print()
    print("  Rationale:")
    print("    1. We have insufficient reviewer data for SFT on 4 of 6 languages.")
    print(f"       {total_reviewed} reviewed pairs << {sft_needed_per_lang} needed per language.")
    print("    2. Prompt engineering delivers a testable result in 1 day at zero cost.")
    print("    3. If prompting works, we solve the problem before spending any GPU time.")
    print("    4. If prompting fails, we still have time to pivot to narrow SFT")
    print("       for Hindi + Kannada only (the two languages with reviewer coverage).")

    # ── Day 1 Experiment ──────────────────────────────────────────────────────
    section("DAY-1 EXPERIMENT PLAN")
    print()
    print("  Run 5 prompt variants on a fixed set of 20 prompts in Hindi:")
    print()
    print(f"  {'Variant':<10} {'System Prompt Addition'}")
    print(f"  {'  ':-<70}")
    variants = [
        ("A (base)", "(none -- current baseline)"),
        ("B",        "Reply in informal, conversational Hindi. Use everyday words."),
        ("C",        "Reply as if texting a friend. Hindi only, simple words."),
        ("D",        "Avoid formal/textbook Hindi. Write the way a Delhi college student would."),
        ("E",        "Use tum/tu form, short sentences, everyday Hindi vocabulary."),
    ]
    for v, prompt in variants:
        # word-wrap long prompts
        if len(prompt) > 55:
            print(f"  {v:<10} {prompt[:55]}")
            print(f"  {'':10} {prompt[55:]}")
        else:
            print(f"  {v:<10} {prompt}")

    print()
    print("  Steps:")
    print("    1. Run all 5 variants on 20 Hindi test prompts (100 outputs total)")
    print("    2. Send Hindi + Kannada outputs to reviewer in randomized blind grid")
    print("    3. Measure: preference rate per variant (2 hours of reviewer time)")
    print("    4. Pick winner by Day 3 -- apply to all 6 languages")
    print("    5. Day 2: run same winning variant on Tamil, Telugu, Bengali, Marathi")

    # ── Success Metric ────────────────────────────────────────────────────────
    section("SUCCESS METRIC  &  KILL CRITERION")
    print()
    print("  Success Metric (Hindi + Kannada):")
    print("    >= 70% reviewer preference for casual output vs current output")
    print("    100-sample blind pairwise eval (50 prompts x 2 languages)")
    print("    Evaluated by native-speaker reviewer in Week 2")
    print()
    print("  Success Metric (Tamil, Telugu, Bengali, Marathi):")
    print("    >= 65% LLM-as-judge preference (lower bar due to translation noise)")
    print("    100 fixed prompts per language, evaluated in English translation")
    print()
    print("  KILL CRITERION:")
    print("    If blind preference < 60% on Hindi AND Kannada by end of Week 1:")
    print("    --> Abandon prompt engineering")
    print("    --> Pivot to narrow LoRA SFT for Hindi + Kannada only")
    print("    --> Accept prompt-engineering path for other 4 languages")
    print("    --> This still fits inside the 3-week window")

    # ── Summary table ─────────────────────────────────────────────────────────
    section("SUMMARY COMPARISON TABLE")
    print()
    print(f"  {'Option':<18} {'Time to result':<18} {'Data risk':<18} {'Reviewer needed':<18} {'Verdict'}")
    print(f"  {'  ':-<85}")
    rows = [
        ("(c) Prompting",  "1-3 days",    "None",          "Minimal (eval only)", "START HERE"),
        ("(a) SFT",        "1-2 weeks",   "High (4 langs)", "Hard blocker",       "Use if (c) fails"),
        ("(b) Rewriter",   "1-2 weeks",   "High (same)",   "Same as SFT",         "Reject"),
    ]
    for opt, time, risk, rev, verdict in rows:
        print(f"  {opt:<18} {time:<18} {risk:<18} {rev:<18} {verdict}")

    print()
    print("=" * W)
    print("  END OF MEMO")
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
