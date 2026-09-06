# Part B — Capacity Reconciliation

## B1: KV-Cache Math

### From model_spec.md:

| parameter | value |
|---|---|
| layers | 28 |
| KV heads (GQA) | 8 |
| head_dim | 128 |
| KV cache precision | fp16 (2 bytes) |
| GPU | NVIDIA L4, 24 GB |
| gpu_memory_utilization | 0.92 |
| non-KV runtime overhead | ~1.6 GB |
| model weights | 4.2B × 2 bytes = 8.4 GB |

### (a) KV-cache bytes per token — exact derivation

For each token stored in the KV cache, we need:
- **K** tensor: 1 vector per layer, of shape [KV_heads, head_dim]
- **V** tensor: 1 vector per layer, of shape [KV_heads, head_dim]

```
bytes_per_token = 2 (K and V)
               × layers         (28)
               × KV_heads       (8)
               × head_dim       (128)
               × bytes_per_elem (2, for fp16)
             = 2 × 28 × 8 × 128 × 2
             = 114,688 bytes
             = 112 KB per token  (exactly)
```

### (b) Maximum concurrent 4096-token sequences

First, compute usable GPU memory for KV cache:

```
total GPU memory        = 24.0 GB
utilization cap         = 0.92
usable GPU memory       = 24.0 × 0.92 = 22.08 GB

model weights (fp16)    = 4.2 × 10⁹ × 2 bytes = 8.40 GB
non-KV runtime overhead = 1.60 GB  (given in model_spec)
                          ──────────────────────────────
KV cache budget         = 22.08 − 8.40 − 1.60 = 12.08 GB

Maximum KV-cache tokens = 12.08 GB / 112 KB
                        = 12.08 × 1024³ / 114,688
                        ≈ 12,966,584,320 / 114,688
                        ≈ 113,060 tokens

Maximum concurrent 4096-token sequences = 113,060 / 4,096 ≈ 27.6
```

**Predicted maximum: ~27 concurrent 4096-token sequences.**

### Verification against bench_log.csv

From the log, the long-context sweep (`prompt_len=3584, gen_len=512`):
- Total tokens per sequence = 3584 + 512 = **4096 tokens** (exactly `max_model_len`)
- At **batch_size=24**: `kv_cache_util = 0.93` → 93% full
- At **batch_size=32**: `kv_cache_util = 0.97` → 97% full, `preempted_seqs = 7`

Check: 24 sequences × 4096 tokens = 98,304 tokens  
Our predicted KV capacity: 113,060 tokens  
Utilization: 98,304 / 113,060 = **0.87** — close to logged 0.93 (small discrepancy from runtime allocation granularity, paged block overhead, etc.)

At batch 32: 32 × 4096 = 131,072 > 113,060 → **over-capacity**, which explains the 7 preemptions. ✓

**Conclusion: our arithmetic is consistent with the observed preemption pattern.**

---

## B2: Throughput Anomaly in Long-Context Sweep

### The anomaly

| batch | reported_tok_s | preempted | kv_cache_util |
|---|---|---|---|
| 4 | 565.4 | 0 | 0.16 |
| 8 | 902.6 | 0 | 0.31 |
| 16 | 1311.4 | 0 | 0.62 |
| **24** | **1607.4** | **0** | **0.93** |
| **32** | **1384.0** | **7** | **0.97** |
| **48** | **1298.5** | **23** | **0.97** |

Throughput **peaks at batch 24** and then **falls at batch 32 and 48** — despite more concurrent requests. This is the anomaly: naively, throughput should scale with batch size.

### Mechanism: KV cache exhaustion → preemption → recomputation

At **batch 32**, total token demand = 32 × 4096 = 131,072 tokens, which **exceeds** our KV cache capacity (~113k tokens).

The vLLM scheduler cannot fit all 32 sequences simultaneously. It must **preempt** (evict) some sequences from the KV cache to make room for others. The evicted sequences must later be **recomputed** (their full prompt re-prefilled), which:
1. **Burns GPU compute** re-prefilling already-seen tokens
2. **Increases wall-clock time** (batch 32: 94.71s vs 61.16s for batch 24 — a 54% increase for only 33% more sequences)
3. **Reduces effective throughput** because some GPU cycles go to redundant work

At **batch 48**: 23 preemptions, kv_cache_util=0.97, wall_clock = 151.41s — GPU is thrashing.

### Proposed fix

**Reduce `max_model_len` from 4096 to 3072** (i.e., cap total prompt+generation to 3072 tokens):

```
New KV capacity in sequences = 113,060 / 3,072 = 36.8 sequences
```

At this limit, batch 32 (32 × 3072 = 98,304 < 113,060) fits without preemption.  
**Predicted quantitative effect**: batch 32 throughput should rise from 1,384 tok/s to approximately the extrapolated value from the non-preempted trendline (batch 16→24: 1311→1607, Δ296/Δ8 ≈ 37 tok/s per request), giving ~1607 + 37×8 ≈ **1900 tok/s** at batch 32 without preemption.

This trades 1,024 tokens of maximum context window for ~37% more throughput at high batch sizes.

---

## B3: REPORT_v0 Misreading — What is "goodput"?

### The report's claim
> "batch 48 should give us ~3200 tok/s"

This extrapolates linearly from `reported_tok_s` at batch 16 (1311) to batch 48 (3×1311 ≈ 3200×). **Both the extrapolation and the metric are wrong.**

### The misreading: `reported_tok_s` counts ALL tokens, not just generated tokens

The `reported_tok_s` column counts **prefill + decode tokens**. For a long-context run:
- `prompt_len = 3584`, `gen_len = 512`
- Total tokens per request = 4096; generated tokens = 512
- Fraction that is actual output: 512 / 4096 = **0.125**

The actual **output throughput ("goodput")** is 12.5% of the reported number.

### Honest goodput for batch-24 long-prompt row (two methods)

**Method 1 — from raw log columns:**
```
goodput = (num_requests × gen_len) / wall_clock_s
        = (24 × 512) / 61.16
        = 12,288 / 61.16
        = 200.9 tok/s (generated tokens)
```

**Method 2 — from reported_tok_s:**
```
goodput = reported_tok_s × (gen_len / (prompt_len + gen_len))
        = 1607.4 × (512 / 4096)
        = 1607.4 × 0.125
        = 200.9 tok/s (generated tokens)
```

Both methods agree: **~201 tok/s actual output throughput** at batch 24.

### What the report should have said

> "At batch 24, the long-context configuration delivers **~1607 tok/s** combined throughput
> (prefill + decode), but only **~201 tok/s** of generated tokens delivered to users.
> Higher batch sizes beyond 24 cause KV cache preemption, reducing even the combined metric.
> The ~1600 tok/s number is not a goodput figure and cannot be used directly for capacity planning.
> For capacity planning on a generative workload, use **tokens_generated / wall_clock_s**."

---

## B4: Monitoring Metric to Confirm the B2 Mechanism

### Metric to pull
**`vllm:scheduler_num_preempted_seqs_total`** (Prometheus counter in vLLM)

or equivalently, the `preempted_seqs` column already in the log.

### What to expect

For the preemption-driven throughput collapse:
- At batches where KV cache is under capacity (util < 0.95): `preempted_seqs = 0`
- At the inflection point (batch 32+): `preempted_seqs` spikes to 7, 23

**Expected signal**: `preempted_seqs > 0` is the smoking gun. Specifically:

> A non-zero preemption count at `kv_cache_util > 0.95` confirms that throughput
> degradation is KV cache capacity exhaustion, not compute saturation.
> Expected value: 0 preemptions for batch ≤ 24 (4096-token sequences), rising sharply at batch 32.

This cleanly separates the KV-exhaustion hypothesis from alternatives (PCIe bandwidth, CPU bottleneck, network overhead) which would degrade throughput continuously rather than at a step threshold.
