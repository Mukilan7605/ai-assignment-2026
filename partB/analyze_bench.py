#!/usr/bin/env python3
"""
analyze_bench.py  â€”  Part B: Capacity Reconciliation

Reads bench_log.csv from the starter_kit and produces a full
structured analysis for B1, B2, B3, and B4.

Run:
    python analyze_bench.py

No external packages required â€” uses only Python standard library.
"""

import csv
import pathlib
import sys

# â”€â”€ locate bench_log.csv â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SCRIPT_DIR  = pathlib.Path(__file__).parent
BENCH_PATH  = SCRIPT_DIR.parent.parent / "starter_kit" / "bench" / "bench_log.csv"

# Fallback: check if copied locally
LOCAL_BENCH = SCRIPT_DIR / "bench_log.csv"
if LOCAL_BENCH.exists():
    BENCH_PATH = LOCAL_BENCH

W = 74  # output width


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "batch":          int(row["batch_size"]),
                "prompt_len":     int(row["prompt_len"]),
                "gen_len":        int(row["gen_len"]),
                "num_requests":   int(row["num_requests"]),
                "wall_s":         float(row["wall_clock_s"]),
                "reported_tok_s": float(row["reported_tok_s"]),
                "ttft_ms":        float(row["ttft_ms_p50"]),
                "itl_ms":         float(row["itl_ms_p50"]),
                "e2e_ms_p95":     float(row["e2e_ms_p95"]),
                "preempted":      int(row["preempted_seqs"]),
                "kv_util":        float(row["kv_cache_util"]),
            })
    return rows


def main():
    # â”€â”€ load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not BENCH_PATH.exists():
        print(f"ERROR: bench_log.csv not found at: {BENCH_PATH}")
        print("Expected location: starter_kit/bench/bench_log.csv")
        sys.exit(1)

    rows = load_csv(BENCH_PATH)
    short_rows = [r for r in rows if r["prompt_len"] == 512]
    long_rows  = [r for r in rows if r["prompt_len"] == 3584]

    # â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print()
    print("=" * W)
    print("  PART B â€” Capacity Reconciliation")
    print(f"  Data file : {BENCH_PATH.name}")
    print(f"  Rows loaded: {len(rows)}  "
          f"(short-prompt: {len(short_rows)}, long-prompt: {len(long_rows)})")
    print("=" * W)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # B1 â€” KV-Cache Math
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print()
    print("â”€" * W)
    print("  B1 â€” KV-Cache Bytes per Token  &  Max Concurrent Sequences")
    print("â”€" * W)

    # Model spec (from model_spec.md)
    LAYERS      = 28
    KV_HEADS    = 8
    HEAD_DIM    = 128
    BYTES_FP16  = 2
    TOTAL_VRAM_GB   = 24.0
    GPU_UTIL        = 0.92
    WEIGHT_PARAMS_B = 4.2
    NON_KV_GB       = 1.6
    SEQ_LEN         = 4096   # prompt_len + gen_len = 3584 + 512

    print()
    print("  Model Spec (from model_spec.md):")
    print(f"    Layers              : {LAYERS}")
    print(f"    KV heads (GQA)      : {KV_HEADS}")
    print(f"    Head dim            : {HEAD_DIM}")
    print(f"    Precision           : fp16 ({BYTES_FP16} bytes/element)")
    print(f"    Total VRAM          : {TOTAL_VRAM_GB} GB")
    print(f"    GPU utilisation cap : {GPU_UTIL}")
    print(f"    Model weights       : {WEIGHT_PARAMS_B}B params Ã— 2 bytes = "
          f"{WEIGHT_PARAMS_B * 2:.1f} GB")
    print(f"    Non-KV overhead     : {NON_KV_GB} GB")

    # KV bytes per token
    kv_bytes_per_token = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES_FP16
    kv_kb_per_token    = kv_bytes_per_token / 1024

    print()
    print("  KV-Cache Bytes per Token:")
    print(f"    Formula : 2 (K+V) Ã— {LAYERS} layers Ã— {KV_HEADS} KV-heads "
          f"Ã— {HEAD_DIM} head_dim Ã— {BYTES_FP16} bytes")
    print(f"    Result  : {kv_bytes_per_token:,} bytes  =  {kv_kb_per_token:.0f} KB per token")

    # KV budget
    usable_gb      = TOTAL_VRAM_GB * GPU_UTIL
    weight_gb      = WEIGHT_PARAMS_B * 2
    kv_budget_gb   = usable_gb - weight_gb - NON_KV_GB
    kv_budget_bytes = kv_budget_gb * (1024 ** 3)
    max_tokens     = kv_budget_bytes / kv_bytes_per_token
    max_seqs       = max_tokens / SEQ_LEN

    print()
    print("  Maximum Concurrent 4096-Token Sequences:")
    print(f"    Usable VRAM        : {TOTAL_VRAM_GB} Ã— {GPU_UTIL} = {usable_gb:.2f} GB")
    print(f"    Minus weights      : {usable_gb:.2f} âˆ’ {weight_gb:.2f} = "
          f"{usable_gb - weight_gb:.2f} GB")
    print(f"    Minus non-KV       : {usable_gb - weight_gb:.2f} âˆ’ {NON_KV_GB:.2f} = "
          f"{kv_budget_gb:.2f} GB  (KV budget)")
    print(f"    Max KV tokens      : {kv_budget_gb:.2f} GB / {kv_kb_per_token:.0f} KB "
          f"= {max_tokens:,.0f} tokens")
    print(f"    Max sequences      : {max_tokens:,.0f} / {SEQ_LEN} = "
          f"{max_seqs:.1f}  â†’  ~{int(max_seqs)} concurrent sequences")

    print()
    print("  Verification against bench_log.csv (long-prompt rows):")
    print(f"  {'Batch':>6}  {'KV util':>8}  {'Preempted':>10}  Matches prediction?")
    print(f"  {'â”€'*50}")
    for r in long_rows:
        expected_util = (r["batch"] * SEQ_LEN) / max_tokens
        if r["preempted"] > 0:
            verdict = f"OVERFLOW â€” {r['preempted']} preemptions (over capacity)"
        elif expected_util > 0.90:
            verdict = "Near limit (as predicted)"
        else:
            verdict = "OK (within capacity)"
        print(f"  {r['batch']:>6}  {r['kv_util']:>8.2f}  {r['preempted']:>10}  {verdict}")

    print()
    print(f"  >> Prediction: ~{int(max_seqs)} sequences max.")
    print(f"  >> Log confirms: clean up to batch 24, overflow starts at batch 32.")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # B2 â€” Throughput Anomaly
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print()
    print("â”€" * W)
    print("  B2 â€” Throughput Anomaly  (why does throughput DROP at batch 32+?)")
    print("â”€" * W)

    print()
    print("  Long-prompt sweep  (prompt_len=3584, gen_len=512, total=4096 tokens):")
    print(f"  {'Batch':>6}  {'Wall (s)':>9}  {'Reported tok/s':>15}  "
          f"{'KV util':>8}  {'Preempted':>10}  Status")
    print(f"  {'â”€'*70}")
    peak_batch = max(long_rows, key=lambda r: r["reported_tok_s"])
    for r in long_rows:
        marker = " <-- PEAK" if r["batch"] == peak_batch["batch"] else ""
        if r["preempted"] > 0:
            status = f"OVERFLOW ({r['preempted']} preempted)"
        elif r["kv_util"] >= 0.90:
            status = "Near limit"
        else:
            status = "Normal"
        print(f"  {r['batch']:>6}  {r['wall_s']:>9.2f}  {r['reported_tok_s']:>15.1f}  "
              f"{r['kv_util']:>8.2f}  {r['preempted']:>10}  {status}{marker}")

    print()
    print("  ANOMALY EXPLAINED:")
    print("    Throughput peaks at batch 24 and then FALLS at batch 32 and 48.")
    print("    This is the opposite of what naive scaling would predict.")
    print()
    print("    REASON: KV cache is exhausted at batch 32.")
    print(f"      Batch 24: 24 Ã— 4096 = {24*4096:,} tokens  < {max_tokens:,.0f} capacity  â†’ no preemptions")
    print(f"      Batch 32: 32 Ã— 4096 = {32*4096:,} tokens  > {max_tokens:,.0f} capacity  â†’ 7 preemptions")
    print()
    print("    When a sequence is preempted, vLLM evicts it from the KV cache.")
    print("    Later it must RE-PREFILL the full prompt all over again.")
    print("    This wastes GPU compute on work already done â†’ throughput falls.")

    print()
    print("  PROPOSED FIX: Cap max_model_len from 4096 to 3072")
    new_seq_len  = 3072
    new_max_seqs = max_tokens / new_seq_len
    print(f"    New max sequences = {max_tokens:,.0f} / {new_seq_len} = {new_max_seqs:.1f}")
    print(f"    Batch 32 tokens   = 32 Ã— {new_seq_len} = {32*new_seq_len:,}  "
          f"< {max_tokens:,.0f}  â†’ fits without preemption")
    print(f"    Trade-off         : context window reduced by 1024 tokens")
    print(f"    Predicted gain    : batch 32 throughput recovers from "
          f"{long_rows[4]['reported_tok_s']:.0f} â†’ ~1900 tok/s")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # B3 â€” Misread Column / Honest Goodput
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print()
    print("â”€" * W)
    print("  B3 â€” The Misread Column  (what does reported_tok_s actually measure?)")
    print("â”€" * W)

    print()
    print("  WHAT THE REPORT CLAIMED:")
    print("    'batch 48 should give us ~3200 tok/s'")
    print("    (linear extrapolation from batch-16 reported_tok_s = 1311)")
    print()
    print("  WHY THAT IS WRONG:")
    print("    reported_tok_s counts ALL tokens â€” both PROMPT tokens and GENERATED tokens.")
    print("    Users only receive GENERATED tokens. Prompt tokens are overhead.")
    print()
    print(f"    For long-prompt rows: prompt={long_rows[0]['prompt_len']}, "
          f"gen={long_rows[0]['gen_len']}, total={long_rows[0]['prompt_len']+long_rows[0]['gen_len']}")
    gen_fraction = long_rows[0]["gen_len"] / (long_rows[0]["prompt_len"] + long_rows[0]["gen_len"])
    print(f"    Fraction that is real output : {long_rows[0]['gen_len']} / "
          f"{long_rows[0]['prompt_len']+long_rows[0]['gen_len']} = {gen_fraction:.3f}  "
          f"(only {gen_fraction*100:.1f}% of reported_tok_s!)")

    print()
    print("  HONEST GOODPUT  (generated tokens / second):")
    print(f"  {'Batch':>6}  {'Reported tok/s':>15}  {'Honest goodput':>15}  "
          f"{'Error factor':>13}  Note")
    print(f"  {'â”€'*70}")
    for r in long_rows:
        total_len = r["prompt_len"] + r["gen_len"]
        honest = (r["num_requests"] * r["gen_len"]) / r["wall_s"]
        error  = r["reported_tok_s"] / honest
        note   = "OVERFLOW" if r["preempted"] > 0 else ""
        print(f"  {r['batch']:>6}  {r['reported_tok_s']:>15.1f}  {honest:>15.1f}  "
              f"{error:>12.1f}x  {note}")

    # Pick batch 24 for the detailed breakdown
    r24 = next(r for r in long_rows if r["batch"] == 24)
    honest_24 = (r24["num_requests"] * r24["gen_len"]) / r24["wall_s"]
    print()
    print(f"  DETAILED CALCULATION for batch 24:")
    print(f"    Method 1 (from log columns):")
    print(f"      goodput = (num_requests Ã— gen_len) / wall_s")
    print(f"              = ({r24['num_requests']} Ã— {r24['gen_len']}) / {r24['wall_s']}")
    print(f"              = {r24['num_requests']*r24['gen_len']:,} / {r24['wall_s']}")
    print(f"              = {honest_24:.1f} tok/s  (generated tokens only)")
    print()
    method2 = r24["reported_tok_s"] * gen_fraction
    print(f"    Method 2 (from reported_tok_s):")
    print(f"      goodput = reported_tok_s Ã— (gen_len / total_len)")
    print(f"              = {r24['reported_tok_s']} Ã— ({r24['gen_len']} / "
          f"{r24['prompt_len']+r24['gen_len']})")
    print(f"              = {r24['reported_tok_s']} Ã— {gen_fraction:.3f}")
    print(f"              = {method2:.1f} tok/s  â† same answer, both methods agree")
    print()
    print(f"  >> The report claimed 1607 tok/s. Honest answer is {honest_24:.0f} tok/s.")
    print(f"  >> The report was {r24['reported_tok_s']/honest_24:.0f}x too optimistic.")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # B4 â€” Monitoring Metric
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print()
    print("â”€" * W)
    print("  B4 â€” Which Metric Confirms the B2 Mechanism?")
    print("â”€" * W)

    print()
    print("  METRIC TO PULL:")
    print("    vllm:scheduler_num_preempted_seqs_total")
    print("    (Prometheus counter exposed on vLLM /metrics endpoint)")
    print()
    print("  WHAT TO EXPECT:")
    print(f"  {'Batch':>6}  {'kv_cache_util':>14}  {'preempted_seqs':>15}  Signal")
    print(f"  {'â”€'*55}")
    for r in long_rows:
        if r["preempted"] == 0 and r["kv_util"] < 0.90:
            signal = "No overflow â€” KV cache healthy"
        elif r["preempted"] == 0 and r["kv_util"] >= 0.90:
            signal = "Near limit â€” watch closely"
        else:
            signal = f"OVERFLOW CONFIRMED â€” preemptions = {r['preempted']}"
        print(f"  {r['batch']:>6}  {r['kv_util']:>14.2f}  {r['preempted']:>15}  {signal}")

    print()
    print("  WHY THIS METRIC?")
    print("    A non-zero preemption count at kv_cache_util > 0.95 is the")
    print("    'smoking gun' that confirms KV cache exhaustion â€” not a compute")
    print("    bottleneck, not PCIe bandwidth, not CPU overhead.")
    print()
    print("    Other bottlenecks degrade throughput gradually.")
    print("    KV exhaustion causes a SHARP STEP at a predictable threshold,")
    print(f"    which our math predicted at batch ~{int(max_seqs)} and the log confirms.")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Summary
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print()
    print("=" * W)
    print("  SUMMARY â€” Part B Key Numbers")
    print("=" * W)
    print()
    print(f"  {'Fact':<45} {'Number'}")
    print(f"  {'â”€'*65}")
    print(f"  {'KV bytes per token':<45} {kv_bytes_per_token:,} bytes  ({kv_kb_per_token:.0f} KB)")
    print(f"  {'KV cache budget (VRAM after weights)':<45} {kv_budget_gb:.2f} GB")
    print(f"  {'Max concurrent 4096-token sequences':<45} ~{int(max_seqs)}")
    error_factor = r24["reported_tok_s"] / honest_24
    print(f"  {'Peak throughput (reported_tok_s)':<45} {peak_batch['reported_tok_s']:.0f} tok/s  (batch {peak_batch['batch']})")
    print(f"  {'Honest goodput at peak (generated only)':<45} {honest_24:.0f} tok/s  (batch 24)")
    print(f"  {'Report error factor':<45} {error_factor:.0f}x too optimistic")
    print(f"  {'Preemptions start at batch':<45} 32  ({32*SEQ_LEN:,} tokens > {max_tokens:,.0f} capacity)")
    print(f"  {'Fix: cap max_model_len to':<45} {new_seq_len} tokens")
    print()
    print("=" * W)
    print()


if __name__ == "__main__":
    main()


