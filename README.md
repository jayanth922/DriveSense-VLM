# 🚗 DriveSense-VLM

> SFT-Optimized Vision-Language Model for Autonomous Vehicle Rare Hazard Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Model: Qwen2.5-VL-3B](https://img.shields.io/badge/model-Qwen2.5--VL--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

**DriveSense-VLM** fine-tunes [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
with LoRA SFT to detect and explain **rare, safety-critical hazards** in autonomous-driving
dashcam frames. The model outputs structured JSON: a per-hazard bounding box, a 7-class hazard
label, severity, chain-of-thought reasoning, and a recommended ego-vehicle action.

Bounding-box labels are **projected from nuScenes 3-D ground truth** (not model-invented); a
foundation model only writes the severity/reasoning/action text for each real box.

---

## The 30-second version

A complete, self-improving **data flywheel** for a rare-hazard perception model —
`mine → auto-label → gate → train → evaluate → gate → analyze failures → mine again` —
built end-to-end and run for two full turns.

Mining is **distributed**: a PySpark ETL (`Phase 1a-spark`) scores every nuScenes frame
across 6 composite rarity signals with explicit schemas, then computes the analytics that
drive frame selection — signal co-occurrence, per-scene richness, and temporal burst
detection. Bounding boxes are **projected from 3-D ground truth**, never model-invented.

**Three things this repo is actually about:**

1. **The flywheel works, and it returned an honest negative.** Turn v3→v4 mined 1,442
   *targeted* rain/night frames aimed exactly at the weakest buckets. Those buckets
   **regressed** (rain det@0.5 12.5% → 7.4%), the regression gate **blocked the candidate**,
   and v3 stayed in production. Two data-scaling experiments — naive (v2→v3) and targeted
   (v3→v4) — now point the same way: for this model, **more data is not the lever**; the
   bottleneck is model-side (input resolution, tiny-box weighting).
2. **Evaluation is built to be trusted, not to flatter.** A 4-level framework (grounding,
   reasoning, production, robustness), GT-hazard-centric stratified metrics where a miss
   counts as IoU 0, a pre-training label-quality gate, and an all-zero-IoU abort that
   refuses to report garbage predictions. It caught a coordinate-convention bug that had
   been silently zeroing every IoU.
3. **Inference is diagnosed, not guessed.** Decode is memory-bandwidth-bound (fp16 converts
   31.8% of the T4's HBM ceiling into useful work), so each lever is chosen to attack *that*:
   prompt-lookup speculative decoding for latency (**bit-exact**, +20%), batching for
   throughput (2.3×), quantization for **memory only** — it makes decode *slower*, and we
   say so.

**Status: all six pillars complete and pushed to `main`** — flywheel, VLM fine-tuning,
perception/L4 robustness, inference study, MLOps (registry + regression gate + CI), and
evaluation rigor. Every number below is traceable: detection →
[`results/metrics_registry.json`](results/metrics_registry.json); inference →
[`INFERENCE_OPTIMIZATION.md` §7](INFERENCE_OPTIMIZATION.md), reproducible with
[`scripts/inference_benchmark.py`](scripts/inference_benchmark.py). Remaining work is one
Colab GPU run — see [What's left](#whats-left-future-work); none of it is blocking.

---

## Results (honest, measured on a fixed test set)

DriveSense-VLM is an end-to-end pipeline that **mines** rare driving hazards, **auto-labels**
them with a foundation model behind a validation gate, **fine-tunes** a small explainable VLM
(Qwen2.5-VL-3B LoRA) to detect and reason about them, then **rigorously evaluates** — a
4-level framework (grounding, reasoning, production, robustness) — and gates regressions.

All numbers below are measured on the **fixed 1,041-frame v3/v4 test set** (nuScenes
v1.0-trainval, predominantly daytime). No fabricated numbers.

### Detection (L1 grounding, IoU ≥ 0.5)

| version | train ex. | Precision | Recall | F1 | mean IoU | class acc | parse |
|---|---|---|---|---|---|---|---|
| v3 (naive scale-up) | 7,228 | 0.40 | 0.24 | 0.30 | 0.67 | 0.94 | 98.7% |
| v4 (targeted + adverse) | 8,670 | 0.37 | 0.19 | 0.25 | 0.656 | 0.95 | 97.4% |

High-precision, well-localized (mean IoU 0.67 > the 0.55 target), correctly labeled (94–95%),
severity-ranked sensibly (within-one 98.6%, Spearman 0.41). The weakness is **recall on the
rare long tail** — especially tiny/distant boxes.

An earlier card reported ~1.4% F1 — that was a coordinate-convention bug (an inference/training
image-resolution mismatch drove predicted boxes out of the labels' 0–1000 space, collapsing
every IoU to ~0); it is fixed and these are the corrected numbers.

### Reasoning (L2, LLM-as-judge, Claude Sonnet 5, n = 1,027 — v3)

| Dimension | Mean |
|---|---|
| Correctness | 3.03 |
| Completeness | 2.66 |
| Action relevance | 3.80 |
| **Overall** | **3.16 / 5** |
| Pass rate (all dims ≥ 3.5) | 26% |

Reasoning is acceptable and driving-action advice is the strongest dimension; completeness is
weakest, consistent with the low detection recall.

### Robustness (L4, detection@0.5 by bucket) — and the flywheel finding

| bucket | v3 | v4 |
|---|---|---|
| tiny box (78% of hazards) | 22.8% | 17.2% |
| small / medium box | 46.4% / 52.6% | 42.9% / 52.6% |
| rain | 12.5% | 7.4% |
| night + tiny | 12.7% | 10.7% |
| clear + medium | 69% | 69% |

**Two data-scaling experiments, one conclusion.** v3 scaled data naively (2.7k→7.2k) and
generalization *hurt* (eval_loss 0.31→0.66). v4 then added 1,442 **targeted** rain/night
frames via the flywheel — and the mined buckets still *regressed*. For this model, scaling
data is not the lever for rare-hazard recall; the bottleneck is model-side (input resolution,
tiny-box weighting). The regression gate correctly **blocked** the v4 candidate — v3 remains
production. See [`DEBUGGING_POSTMORTEM.md`](DEBUGGING_POSTMORTEM.md), [`FLYWHEEL.md`](FLYWHEEL.md),
[`FLYWHEEL_V4_FINDINGS.md`](FLYWHEEL_V4_FINDINGS.md), and the generated
[`results/mlops_report.md`](results/mlops_report.md).

### Deployment / inference

Measured on a single T4 (16 GB, 320 GB/s HBM). Decode is **memory-bandwidth-bound** — fp16
converts 31.8% of HBM bandwidth into useful decode work — so each lever is chosen to attack
that bottleneck:

| config | decode tok/s | TTFT | e2e p50 | weights | HBM roofline | vs fp16 output |
|---|---|---|---|---|---|---|
| fp16 (baseline) | 17.0 | 727 ms | 11.64 s | ~6.0 GB | 31.8% | reference |
| **fp16 + prompt-lookup** | **20.4** (+20%) | 727 ms | **9.79 s** (−16%) | ~6.0 GB | **38.2%** | **exact_match 1.00** |
| NF4 (4-bit) | 12.6 (*slower*) | 727 ms | ~15.9 s | **2.63 GB** | 8.6% | char_sim 0.36 |
| INT8 | 4.6 | 727 ms | 53.53 s | ~3.5 GB | — | char_sim 0.29 |

Throughput (fp16 aggregate decode): **14.9 → 33.7 tok/s at batch 4** (~2.3×).

- **Latency: prompt-lookup speculative decoding** — +20% decode throughput (17.0 → 20.4 tok/s),
  end-to-end 11.64 s → 9.79 s, at **bit-exact** output (exact_match 1.00 vs fp16).
- **Throughput: batching** — fp16 aggregate 14.9 → 33.7 tok/s at batch 4 (~2.3×), the right
  axis for the offline auto-labeling loop.
- **Memory (not latency): NF4** — weights 6.0 GB → 2.63 GB (~2.3×) to fit a 16 GB card, but
  decode gets *slower* (17.0 → 12.6 tok/s) and output drifts (char_sim 0.36 vs fp16), so it
  ships only behind an L1/L4 quality gate. INT8 is worse (4.6 tok/s) and not recommended here.

Full study — diagnosis, roofline, quality gate, and the measured table — is in
[`INFERENCE_OPTIMIZATION.md`](INFERENCE_OPTIMIZATION.md), with a runnable benchmark
(`scripts/inference_benchmark.py`). *End-to-end latency is seconds per image (11.6 s fp16 /
9.8 s with prompt-lookup on T4, for a full multi-hazard JSON response) — this is an
autoregressive VLM, framed as a compression + speedup + throughput story, not real-time.*

### Limitations
- **Low recall on the rare long tail** — recall @ IoU 0.5 is 24% (v3); the model is conservative
  and misses many rare hazards (e.g. `unusual_object`, 24 instances, never detected). This, not
  localization, is the weak axis: precision (40%) and box tightness (mean IoU 0.67 on matches)
  are strong. **More data is demonstrably not the lever** — see the v4 finding above.
- **Narrow training distribution** — nuScenes v1.0-trainval, predominantly daytime; night and
  heavy weather are out of distribution, and targeted adverse-weather mining did not close it.
- **Mild overfitting (v3)** — eval loss (0.66) is ~2× train loss (0.40). v4 trained clean
  (3 epochs, eval_loss 0.694, no overfit) but still regressed the mined buckets.
- **Dense-frame parse failures rare** — 98.7% parse (v3) / 97.4% (v4) after raising the token
  budget to 768.
- Research / offline-evaluation use only — not for real-time or safety-critical control.

> **Honesty note.** This is a portfolio project optimized for lifecycle rigor over a headline
> accuracy number. It surfaces and diagnoses real failures (a coordinate-convention bug that
> zeroed IoU; naive and targeted data scaling both failing to lift recall) rather than
> reporting a cherry-picked win.

---

## What this project demonstrates

The engineering — not the raw F1 — is the point. This project shows an end-to-end debugging and
rebuild of a broken VLM fine-tuning pipeline:

| Skill | Implementation |
|-------|---------------|
| **Root-cause debugging** | Diagnosed a model that collapsed to constant output; traced it to VLM-invented bounding-box labels **and** a debug `max_steps=10` override that silently capped every training run at ~0.3 epochs |
| **Data curation** | nuScenes rarity scoring (6 composite signals) + **3-D→2-D ground-truth box projection** with near-plane frustum clipping; LLM describe-only annotation (severity/reasoning/action per real box) |
| **Data quality gates** | A hard, pre-training validation gate (`run_label_validation.py`) that blocks collapsed/degenerate label sets (box-diversity, area, cross-frame duplication) before any training |
| **VLM fine-tuning** | LoRA SFT on Qwen2.5-VL-3B with prefix-masked labels and a variable-patch data collator; bf16 on A100 |
| **Rigorous, honest evaluation** | Grounding (IoU + Hungarian matching) with real-checkpoint loading, a stratification guard, and an all-zero-IoU abort that refuses to report boxless/garbage predictions |
| **Reproducible infra** | Colab + RunPod runbooks, mock/dry-run modes, idempotent stage pipeline |

---

## Architecture

```
 ┌──────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
 │   Data (v2)       │   │   SFT Training   │   │  Evaluation         │
 │                   │   │                  │   │                     │
 │  nuScenes ───────►│   │  Qwen2.5-VL-3B  │   │  Level 1 Grounding  │
 │  rarity ≥ 5/6     │──►│  + LoRA r=32     │──►│  (IoU + Hungarian)  │
 │  GT 3D→2D boxes   │   │  A100, bf16      │   │  + validation gate  │
 │  describe-only VLM│   │                  │   │  + honest metrics   │
 └──────────────────┘   └─────────────────┘   └─────────────────────┘
```

---

## Quick start (local dev, CPU)

```bash
git clone https://github.com/jayanth922/DriveSense-VLM.git
cd DriveSense-VLM
pip install pyyaml pillow numpy scipy tqdm      # core, CPU-safe
python -m pytest tests/ -v                       # test suite (no GPU / no downloads)
```

Full training + eval run on GPU (Colab or RunPod) — see the runbooks in the notebooks and
`scripts/train_and_eval_v2.sh`.

---

## Methodology

### Data curation (v2)
**nuScenes** (v1.0-trainval, CAM_FRONT keyframes): each frame scores 0–6 across 6 binary rarity
signals (proximity < 5 m, occlusion 0–40% visibility, density ≥ 15 agents, adverse weather/night,
vulnerable road user at intersection, cyclist present). Rare frames (score ≥ 5) are selected, then
deduplicated per scene. **Bounding boxes are projected from the dataset's 3-D ground-truth
annotations** into the 2-D camera frame (near-plane frustum clipped). A foundation model (Claude)
writes only severity/reasoning/action for each real box — it never draws boxes. A hard validation
gate blocks the label set from training if it shows collapse (repeated boxes, oversized boxes,
schema violations).

### Training
LoRA SFT on Qwen2.5-VL-3B-Instruct (rank 32, alpha 64; targets `q/k/v/o/up/down_proj`), bf16 on a
single A100, label masking via prefix tokenization (only the assistant JSON is supervised).

### Evaluation
4-level framework; Level 1 (grounding) is the reported result. IoU@0.5 with Hungarian matching,
plus threshold-independent localization metrics (mean best-pair IoU, detection-rate curve). The
harness loads the real fine-tuned adapter, guards stratification metadata, and **aborts** rather
than reporting an all-zero-IoU run (which would indicate boxless/garbage predictions).

---

## Repo map

| Path | What it is |
|---|---|
| `README.md` | **This file — the authoritative front door.** Status, results, what's left. |
| `FLYWHEEL.md` | The mine→label→gate→train→eval→gate loop, stage by stage |
| `FLYWHEEL_V4_FINDINGS.md` | The v3→v4 turn in full: what was mined, what regressed, why |
| `DEBUGGING_POSTMORTEM.md` | Three real failures diagnosed (coordinate bug, naive scaling, targeted scaling) |
| `INFERENCE_OPTIMIZATION.md` | Bottleneck-driven inference study; **§7 = the measured T4 results** |
| `MODEL_CARD.md` / `hf_model_card/` | Model cards (repo + HuggingFace-facing) |
| `results/metrics_registry.json` | **Source of truth** for v2/v3/v4 metrics + the gate policy |
| `results/mlops_report.md` | Generated v2→v3→v4 comparison + BLOCK verdict |
| `mlops_report.py` | Builds the report; `--gate` exits non-zero on regression (used by CI) |
| `scripts/inference_benchmark.py` | v2 benchmark harness — reproduces §7 (batching, percentiles, equivalence gate) |
| `scripts/v4/` | The v4 flywheel turn's pipeline (mine → label → build → finalize) |
| `scripts/` | Pipeline CLIs: filter, annotate, train, evaluate, mine, gate, ship |
| `src/drivesense/` | Library: `data/`, `training/`, `eval/`, `inference/`, `monitoring/` |
| `src/drivesense/data/spark_pipeline.py` | **Phase 1a-spark** — distributed PySpark rarity-scoring + analytics ETL |
| `scripts/run_spark_pipeline.py` | Spark ETL entry point (`--skip-extraction`, `--analytics-only`) |
| `docs/` | Deep dives: observability, closed loop, AV2 integration, TensorRT runbook |
| `configs/*.yaml` | All hyperparameters and paths — single source of truth, never hardcoded |
| `tests/` | 652 tests, CPU-only and mock-backed (no GPU, downloads, or API keys) |
| `demo/` | Gradio app for HuggingFace Spaces (T4, NF4) |
| `notebooks/` | Colab execution notebooks (data → training → optimization → eval) |
| `slurm/` | HPC job scripts (alternative to Colab) |
| `.github/workflows/ci.yml` | Tests + mock pipeline smoke + the regression gate |

---

## What's left (future work)

Everything below needs **a single Colab GPU run**. None of it is blocking, and none of it
changes a published number — these are open threads recorded honestly rather than hidden.

1. **Re-run the v2 benchmark on a T4** to settle the three measurement caveats documented in
   [`INFERENCE_OPTIMIZATION.md` §7](INFERENCE_OPTIMIZATION.md):
   - **(i)** fp16 batch-1 reads 17.0 tok/s in one table and 14.9 in the other (different
     runs) — decides whether the batching gain is **2.3× or 2.0×**;
   - **(iii)** TTFT is identical at 727 ms across all five configs — plausible if prefill is
     vision-encoder-dominated, but it needs confirming it wasn't carried over;
   - **(ii)** replace the hardcoded `WEIGHT_GB` nf4 = 2.2 in `scripts/inference_benchmark.py`
     with the measured **2.63 GB** — the NF4/INT8 roofline column currently reads ~16% low.
     (fp16 is correct at 6.0, so the headline fp16/prompt-lookup numbers are unaffected.)
   The v2 harness measures all of these under one run, so a single execution resolves all three.
2. **Execute [`docs/TENSORRT_RUNBOOK.md`](docs/TENSORRT_RUNBOOK.md) on a Colab A100** for one
   real TensorRT-vs-HF speedup row. ⚠️ **Planned and unexecuted** — the runbook is a planning
   document with decision points, and **no TensorRT result is claimed anywhere in this repo.**
3. **Optional — v4b ablation:** retrain dropping the 216 `no_hazard` negatives introduced in
   v4, to test the recall-suppression hypothesis. If rain/night recall recovers, the negatives
   were the culprit; if not, the bottleneck is confirmed model-side.

---

## Tech stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Base model | Qwen2.5-VL-3B-Instruct | Apache 2.0 |
| Fine-tuning | LoRA via PEFT | rank 32, alpha 64 |
| Training | HuggingFace Transformers | LoRA SFT, prefix masking, bf16 |
| Demo quantization | bitsandbytes NF4 (4-bit) | HF Spaces T4 demo |
| Data | nuScenes v1.0-trainval | rare-hazard filtered, GT-projected boxes |
| Distributed ETL | PySpark | 6-signal rarity scoring + analytics (explicit schemas, no inferSchema) |
| Annotation | Anthropic Claude | describe-only (severity/reasoning/action) |
| Tracking | Weights & Biases | training metrics |
| Lint / format | Ruff + Black | line-length = 100 |
| Testing | pytest | CPU-only, mock-backed |

---

## Testing

```bash
python -m pytest tests/ -v
```
The suite is CPU-only and mock-backed — no GPU, model downloads, or API keys required.

---

## Acknowledgments

- **Qwen Team (Alibaba)** for Qwen2.5-VL-3B-Instruct (Apache 2.0)
- **nuScenes / Motional** for the nuScenes autonomous-driving dataset
- **HuggingFace** for Transformers, PEFT, and Spaces
- **Anthropic** for the Claude API used in describe-only annotation and LLM-as-judge evaluation
