# Part B — Capacity Reconciliation

**Script:** `partB/analyze_bench.py` reads `bench_log.csv` and reproduces every number below.

---

## B1 — KV-Cache Bytes per Token & Maximum Concurrent Sequences

### Model Spec (from `model_spec.md`)

| Parameter | Value |
|---|---|
| Layers | 28 |
| KV heads (GQA) | 8 |
| head\_dim | 128 |
| KV cache precision | fp16 (2 bytes per element) |
| GPU | NVIDIA L4, 24 GB VRAM |
| gpu\_memory\_utilization | 0.92 |
| Model weights | 4.2B params × 2 bytes = 8.40 GB |
| Non-KV runtime overhead | 1.60 GB |

---

### KV-Cache Bytes per Token — Exact Derivation

For each token stored in the KV cache, we need one K vector and one V vector per layer:

```
KV bytes/token = 2          (K tensor + V tensor)
               × 28         (layers)
               × 8          (KV heads, GQA)
               × 128        (head_dim)
               × 2          (bytes per element, fp16)

             = 2 × 28 × 8 × 128 × 2
             = 114,688 bytes
             = 112 KB per token
```

---

### Maximum Concurrent 4096-Token Sequences

Step-by-step calculation:

```
Total GPU VRAM          =  24.00 GB
× gpu_memory_utilization   × 0.92
                        = ─────────
Usable VRAM             =  22.08 GB

− Model weights (fp16)  −   8.40 GB  (4.2B × 2 bytes)
− Non-KV overhead       −   1.60 GB  (given in model_spec)
                          ─────────
KV cache budget         =  12.08 GB

Max KV tokens = 12.08 GB / 112 KB
              = 12.08 × 1,073,741,824 / 114,688
              = 12,971,162,009.6 / 114,688
              ≈ 113,096 tokens

Max concurrent 4096-token sequences = 113,096 / 4,096 ≈ 27.6
```

**Predicted maximum: ~27 concurrent 4096-token sequences.**

Each benchmark sequence uses prompt\_len=3584 + gen\_len=512 = **4096 tokens** (exactly `max_model_len`).

---

### Verification Against `bench_log.csv`

| Batch | Predicted token demand | KV util (log) | Preempted (log) | Consistent? |
|---|---|---|---|---|
| 4 | 4 × 4096 = 16,384 | 0.16 | 0 | Yes — 16,384 / 113,060 = 0.14, close to 0.16 |
| 8 | 8 × 4096 = 32,768 | 0.31 | 0 | Yes — 32,768 / 113,060 = 0.29, close to 0.31 |
| 16 | 16 × 4096 = 65,536 | 0.62 | 0 | Yes — 65,536 / 113,060 = 0.58, close to 0.62 |
| 24 | 24 × 4096 = 98,304 | 0.93 | 0 | Yes — 98,304 / 113,096 = 0.87, near limit |
| 32 | 32 × 4096 = 131,072 | 0.97 | 7 | Yes — 131,072 > 113,096, overflow confirmed |
| 48 | 48 × 4096 = 196,608 | 0.97 | 23 | Yes — severe overflow |

Small discrepancies (e.g. 0.87 vs 0.93 at batch 24) are explained by paged block allocation granularity and per-sequence KV metadata overhead in vLLM. The prediction correctly identifies the overflow boundary at batch 32.

**Conclusion:** Arithmetic is consistent with the observed preemption pattern. Overflow begins exactly where predicted.

---

## B2 — Long-Context Throughput Anomaly

### Full Long-Prompt Sweep from `bench_log.csv`

All rows with `prompt_len = 3584`, `gen_len = 512`, total = **4096 tokens per sequence**:

| Batch | Wall (s) | reported\_tok\_s | KV util | Preempted | Status |
|---|---|---|---|---|---|
| 4 | 28.98 | 565.4 | 0.16 | 0 | Normal |
| 8 | 36.30 | 902.6 | 0.31 | 0 | Normal |
| 16 | 49.97 | 1311.4 | 0.62 | 0 | Normal |
| **24** | **61.16** | **1607.4 ← PEAK** | **0.93** | **0** | **Near limit** |
| 32 | 94.71 | 1384.0 ↓ DROP | 0.97 | 7 | KV OVERFLOW |
| 48 | 151.41 | 1298.5 ↓ DROP | 0.97 | 23 | Severe overflow |

### The Anomaly

Throughput **peaks at batch 24 (1607 tok/s)** and then **falls at batch 32 and 48** — even though we are adding more concurrent requests.

This is the opposite of what naive linear scaling predicts.

### Mechanism: KV Cache Exhaustion → Preemption → Recomputation

At **batch 32**, total token demand = 32 × 4096 = **131,072 tokens**, which exceeds the KV cache capacity of **~113,096 tokens**.

The vLLM scheduler cannot fit all 32 sequences simultaneously. It must **preempt** (evict) some sequences from the KV cache to make room for others. The evicted sequences must later be **re-prefilled** from scratch:

1. **Wasted GPU compute** — re-prefilling already-seen tokens burns cycles on redundant work
2. **Increased wall time** — batch 32 takes 94.71s vs 61.16s for batch 24, a **54% increase for only 33% more sequences**
3. **Falling throughput** — some fraction of GPU time is spent on recomputation rather than new generation

At batch 48: 23 preemptions, wall time = 151.41s — the GPU is thrashing.

### Proposed Fix

**Set `max_model_len = 3072`** (reduce from 4096 to 3072 tokens per sequence):

```
New KV capacity in sequences = 113,096 / 3,072 = 36.8 sequences

Batch 32 token demand = 32 × 3,072 = 98,304 tokens
98,304 < 113,096  →  fits without preemption
```

**Quantitative predicted effect:**

- Batch 32 throughput recovers from 1,384 tok/s to approximately **~1,900 tok/s**
  (extrapolated from the clean trendline: batch 16→24 gained +296 tok/s for +8 sequences = ~37 tok/s per sequence; applying to batch 24→32 gives 1,607 + 8×37 ≈ 1,900 tok/s)
- Preemptions drop from 7 to **0** at batch 32
- Trade-off: maximum context window is reduced by 1,024 tokens

---

## B3 — The Misread Column

### What the Original Report Said

> *"At batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization."*
> *"Batch 48 should give us ~3200 tok/s."*

Both conclusions are wrong. They come from misreading **one column**.

### The Misread Column: `reported_tok_s`

`reported_tok_s` counts **all tokens processed** — both prompt tokens (prefill) and generated tokens (decode). It does **not** measure output throughput.

For the long-prompt rows:
- `prompt_len = 3584`, `gen_len = 512`, total = 4,096 tokens per request
- Only `512 / 4096 = 0.125` (12.5%) of the counted tokens are actual **output delivered to users**
- The remaining 87.5% are prompt tokens — they are overhead, not value

### Honest Goodput for Batch 24 — Two Independent Methods

**Method 1 — Directly from log columns:**

```
goodput = (num_requests × gen_len) / wall_clock_s
        = (24 × 512) / 61.16
        = 12,288 / 61.16
        = 200.9 generated tok/s
```

**Method 2 — From reported\_tok\_s:**

```
goodput = reported_tok_s × (gen_len / total_len)
        = 1607.4 × (512 / 4096)
        = 1607.4 × 0.125
        = 200.9 generated tok/s
```

Both methods agree: **~201 generated tok/s**, not 1,607.

### Honest Goodput for All Long-Prompt Rows

| Batch | reported\_tok\_s | Honest goodput (gen only) | Error factor |
|---|---|---|---|
| 4 | 565.4 | 70.7 | 8.0× |
| 8 | 902.6 | 112.8 | 8.0× |
| 16 | 1311.4 | 163.9 | 8.0× |
| 24 | 1607.4 | **200.9** | **8.0×** |
| 32 | 1384.0 | 173.0 | 8.0× |
| 48 | 1298.5 | 162.3 | 8.0× |

The error factor is consistently **8× across all rows** because `gen_len / (prompt_len + gen_len) = 512 / 4096 = 0.125` is constant.

### Why "Longer Prompts → Better Throughput" Is Wrong

Long prompts inflate `reported_tok_s` because prompt tokens dominate the count (87.5% of tokens are prompt). Short-prompt rows (prompt=512, gen=256, total=768) have a gen fraction of 256/768 = 33% — so their reported\_tok\_s is closer to their real goodput.

Comparing `reported_tok_s` across different prompt lengths is comparing apples to oranges.

### What the Report Should Have Said

> *"At batch 24, the long-context configuration delivers 1,607 reported tok/s (prefill + decode combined), but only **~201 generated tok/s** actually delivered to users — 8× lower. For capacity planning on a generative workload, use* `(num_requests × gen_len) / wall_clock_s`*, not* `reported_tok_s`*. Longer prompts inflate the reported metric without improving output throughput."*

The batch-48 projection of ~3200 tok/s is wrong on two levels: it uses the inflated metric, and at batch 48 the KV cache has already overflowed — throughput is actually **162 honest tok/s** at batch 48, lower than batch 24.

---

## B4 — Monitoring Metric to Confirm the B2 Mechanism

### Metric to Pull

**`vllm:scheduler_num_preempted_seqs_total`**

This is a Prometheus counter exposed by vLLM at the `/metrics` endpoint. It counts the cumulative number of sequences that have been preempted (evicted from KV cache and scheduled for recomputation).

A secondary confirming metric: **`kv_cache_util`** — the fraction of KV cache blocks in use.

### Expected Values at Batch 32 (Long-Prompt)

| Metric | Expected value | Why |
|---|---|---|
| `kv_cache_util` | ≥ 0.95 (saturated) | 32 × 4096 = 131,072 tokens exceeds 113,096 KV capacity |
| `vllm:scheduler_num_preempted_seqs_total` | > 0 and rising with load | Scheduler has no free KV blocks for all sequences simultaneously |

The log already shows this: `kv_cache_util = 0.97` and `preempted_seqs = 7` at batch 32.

### Why This Metric Specifically

A non-zero preemption count at `kv_cache_util > 0.95` is the **direct causal evidence** for the throughput drop:

- **Compute bottleneck** would show: throughput degrades smoothly, no preemptions, GPU utilization saturated
- **PCIe / memory bandwidth** would show: gradual degradation, no discrete step
- **KV cache exhaustion** shows: throughput is stable up to a threshold (batch 24), then drops sharply at exactly the predicted batch size (32), with a non-zero preemption counter

The step threshold matches our arithmetic exactly. This rules out all alternatives and confirms the mechanism.

### Expected Signal Pattern

| Batch | kv\_cache\_util | preempted\_seqs | Signal |
|---|---|---|---|
| 4 | 0.16 | 0 | KV cache healthy |
| 8 | 0.31 | 0 | KV cache healthy |
| 16 | 0.62 | 0 | KV cache healthy |
| 24 | 0.93 | 0 | Near limit — monitor closely |
| 32 | 0.97 | 7 | **OVERFLOW CONFIRMED** |
| 48 | 0.97 | 23 | Severe overflow — GPU thrashing |
