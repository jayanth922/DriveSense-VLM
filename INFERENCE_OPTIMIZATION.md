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
