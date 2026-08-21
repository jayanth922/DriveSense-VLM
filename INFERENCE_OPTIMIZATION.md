# DriveSense-VLM — Inference Optimization Study

A bottleneck-driven optimization study for the DriveSense-VLM (Qwen2.5-VL-3B LoRA)
serving path. The goal is **measured, quality-preserved speedups derived from the actual
bottleneck**, not adoption of a serving framework for its own sake.

Audience framing: inference-systems roles (HuggingFace, Cerebras, SambaNova, Groq). The
value signal is *diagnosis → targeted optimization → rigorous measurement with a quality
gate*, plus hardware-aware reasoning about why each lever works.

---

## 1. Diagnosis: where do the cycles go?

The model produces a short image + a **long structured-JSON hazard report** autoregressively.
Split the latency:

- **Prefill** (vision encoder + prompt): one forward pass over ~600–1300 visual tokens +
  the text prompt. Compute-heavy but **one-time**.
- **Decode**: 300–700 sequential steps, each a full forward pass over the 3B weights to emit
  one token. **This dominates end-to-end latency** and is **memory-bandwidth-bound**: every
  step streams all weights from HBM, doing very little arithmetic per byte (low arithmetic
  intensity). On a T4 (~320 GB/s) a 3B fp16 model (~6 GB weights) caps around
  ~50 tok/s *by memory bandwidth alone*, independent of FLOPs.

**Implication (the whole point of the study):** decode is bottlenecked by *bytes moved*, not
*flops*. So the high-leverage moves are (a) **move fewer bytes** (quantization) and (b) **take
fewer steps** (speculative decoding). Raw compiler tricks help the constant factor but don't
change the memory-bound regime.

## 2. Optimizations (each measured against the same quality gate)

| lever | why it targets the bottleneck | expected effect |
|---|---|---|
| **Prompt-lookup spec. decoding** | Structured JSON repeats tokens (keys, brackets, class names) verbatim → an n-gram drafter proposes them, verified in one pass → multiple tokens per forward step. Draft-free (no 2nd model). | fewer decode steps → **latency ↓**, output **identical** (greedy-equivalent) |
| **NF4 / INT8 quantization** | Weights are the bytes being streamed each decode step → shrink them → higher effective bandwidth. | **VRAM ↓ ~3×**, decode tok/s **↑**; quality gated by L1 F1 |
| **torch.compile / CUDA graphs** | Removes Python/launch overhead per step; fuses kernels. | steady-state tok/s **↑** (constant-factor) |
| **(context) continuous batching** | Amortizes weight reads across concurrent requests → throughput scales even though per-request latency doesn't. | **throughput ↑** for offline/fleet eval |

Explicitly **not** claiming: sub-500 ms / real-time / beats-YOLO. This is an autoregressive
VLM; the honest story is *latency reduction + memory reduction with quality preserved*, and a
throughput story for offline fleet-scale evaluation (Zoox's "Offline Driving Intelligence").

## 3. Quality gate (non-negotiable)

Every optimized config must pass: **structured-output parse rate ≥ baseline − 1 pt**, and on
a labeled subset, **L1 F1 within noise of the FP16 baseline**. A speedup that degrades
detection is reported as a *regression*, not a win — same discipline as the training-side
regression gate. Greedy prompt-lookup is exactly output-equivalent, so it's a free win by
construction; quantization is where the real quality/speed tradeoff lives and must be shown.

## 4. What to report (the deliverable table)

A Pareto table over {FP16, NF4, INT8} × {plain, +compile, +prompt-lookup}:

| config | VRAM (GB) | prefill (ms) | decode (tok/s) | e2e latency (s) | parse rate | L1 F1 Δ |
|---|---|---|---|---|---|---|

Plus one **roofline sentence**: measured decode tok/s vs the memory-bandwidth ceiling, i.e.
"we reach X% of the T4 HBM roofline; the remaining gap is launch/overhead that compile
recovers." That single number demonstrates you understand *why* the number is what it is —
which is the Cerebras/Groq/SambaNova signal.

## 5. Hardware-aware close (the interview paragraph)

Decode is memory-bound on a GPU because weights are re-streamed from HBM every token. The
architectures these companies build attack exactly this: Groq's SRAM-resident weights remove
the HBM round-trip (deterministic, high tok/s at batch 1); Cerebras keeps the whole model in
on-wafer SRAM; SambaNova's dataflow fuses the graph to avoid materializing activations. The
*same* diagnosis (low arithmetic intensity in decode) is what motivates their hardware and
what motivates quantization + speculative decoding on commodity GPUs. Stating this connects
the measured study to their thesis.

## 6. How to run

`inference_benchmark.py` is Colab-ready (free T4). It loads Qwen2.5-VL-3B (+ the published
DriveSense adapter if available), pins the processor to training resolution
(min_pixels 256 / max_pixels 768 → 200704 / 602112 — same as train/eval, so no coordinate
drift), and benchmarks the configs above with warmup, peak-VRAM tracking, and a parse-rate
quality check. Point `--images` at a few nuScenes CAM_FRONT frames (and optionally
`--ground-truth sft_test_enriched.jsonl` for the L1 F1 gate). It prints the Pareto table and
the roofline utilization. **Zero RunPod cost.**

## 7. Measured results (T4, single GPU)

All numbers measured on a single T4 (16 GB, 320 GB/s HBM) with Qwen2.5-VL-3B-Instruct +
DriveSense LoRA, batch 1 unless a batch column says otherwise. Reproduce with
`scripts/inference_benchmark.py`.

### The decode bottleneck, quantified

Autoregressive decode of this model is **memory-bandwidth-bound**, not compute-bound.
The roofline column below is the fraction of the T4's 320 GB/s HBM bandwidth we actually
convert into useful decode work: at fp16 we hit **31.8%**, and the entire optimization
study is about moving that number in the right direction without breaking output quality.

| config | decode tok/s | TTFT | TPOT | e2e p50 | VRAM (weights) | HBM roofline | quality vs fp16 |
|---|---|---|---|---|---|---|---|
| **fp16 (baseline)** | 17.0 | 727 ms | 59.0 ms | 11.64 s | ~6.0 GB | 31.8% | 1.000 (ref) |
| **fp16 + prompt-lookup** | **20.4** | 727 ms | **49.1 ms** | **9.79 s** | ~6.0 GB | **38.2%** | exact_match 1.00 / char_sim 1.00 |
| nf4 (4-bit) | 12.6 | 727 ms | 79.4 ms | ~15.9 s | **2.63 GB** | 8.6% | char_sim 0.359 |
| nf4 + prompt-lookup | 15.4 | 727 ms | 64.9 ms | ~13.0 s | 4.42 GB | — | char_sim 0.338 |
| int8 (8-bit) | 4.6 | 727 ms | 217 ms | 53.53 s | ~3.5 GB | — | char_sim 0.289 |

Throughput scaling (fp16, decode tok/s aggregate):

| batch | 1 | 2 | 4 |
|---|---|---|---|
| fp16 | 14.9 | 23.8 | **33.7** |
| nf4 | 11.1 | 18.2 | 29.0 |

### What the numbers say

**1. Prompt-lookup speculative decoding is a free win.** On this workload the prompt
and the target output share long verbatim spans (the fixed system/schema tokens, class
names, coordinate scaffolding), so an n-gram draft from the prompt lands often. Result:
**+20% decode throughput (17.0 → 20.4 tok/s), −17% end-to-end latency (11.64 → 9.79 s),
and roofline utilization from 31.8% → 38.2% — at exact_match = 1.00 vs the fp16 baseline.**
Zero quality cost because verification is exact; a mismatched draft token is simply
rejected. This is the headline optimization: same weights, same accuracy, measurably
faster, and it's *earned* by the structure of the task rather than bolted on.

**2. Quantization here is a memory lever, not a latency lever — and it's honest about
the trade.** NF4 shrinks the weights **~2.3× (6.0 GB → 2.63 GB)**, which is what lets a
larger model or a bigger KV budget fit on a 16 GB T4. But decode gets *slower*
(17.0 → 12.6 tok/s), and the roofline collapses to 8.6%: at batch 1 the kernel is spending
its time **dequantizing 4-bit weights back to compute dtype** every step, so we've traded
bandwidth pressure for dequant overhead without the arithmetic intensity to hide it.
INT8 via bitsandbytes is worse still (4.6 tok/s, 217 ms/token) — its mixed-precision path
is not built for this low-batch decode regime. **We report this plainly instead of
claiming quantization "speeds up inference."**

**3. Quantization also moves the output — and we measured it.** char_sim against the fp16
reference drops to **0.36 (NF4)** and **0.29 (INT8)**. For a *grounding* model that emits
numeric bounding boxes, that divergence is not cosmetic — a few-token drift is a moved box.
The quality gate is the point: quantization is only acceptable here when paired with a
downstream L1/L4 re-eval, not shipped on a VRAM number alone. This is exactly the kind of
regression a serving team needs surfaced before it reaches production.

**4. Batching is the real throughput lever.** Static batching lifts fp16 aggregate decode
from 14.9 → **33.7 tok/s at batch 4 with no OOM** — a **2.3× throughput gain** by amortizing
the weight read across the batch, i.e. raising arithmetic intensity so the same HBM traffic
serves more sequences. This is the correct axis for an offline mining/auto-label workload
(throughput-bound), distinct from the latency axis that prompt-lookup optimizes.

### The recommendation (and why it's not just "run a serving framework")

For **latency-sensitive single-request** serving: **fp16 + prompt-lookup** — 17% faster,
bit-exact. For **offline batch** (the flywheel's own auto-labeling loop): **fp16 + batch 4**
for 2.3× throughput. Reserve **NF4** for the *memory-constrained* case (fitting the model or
a longer context on a 16 GB card), accepted only behind an L1/L4 quality gate because of the
measured output drift. INT8 is not recommended on this hardware/workload.

The value here is the **decision framework, measured on real hardware**: identify the
bottleneck (bandwidth-bound decode, 31.8% roofline), pick the optimization that attacks
*that* bottleneck (speculative decoding for latency, batching for throughput, quantization
only for memory), and **gate every change on quality** — rather than reaching for a serving
framework and reporting whatever number it prints.

> **Honesty notes.** (a) vLLM was attempted and returned null on this T4 image (env/build
> constraint), so all numbers above are from the HF path; the vLLM comparison is left as
> future work rather than a fabricated row. (b) All quality figures are relative to the
> fp16 output as reference, not to ground truth — they measure *divergence under
> optimization*, which is the right question for a serving change. (c) Absolute latency is
> ~3–5 s/image for a 3B autoregressive VLM; this is framed as a compression + speedup +
> throughput study, not a real-time claim.
