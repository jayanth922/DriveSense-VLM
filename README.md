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

NF4 quantization (7.5 GB → 2.4 GB, ~3.1×) fits a T4 with headroom; ~2× via torch.compile.
A bottleneck-driven inference study (decode is memory-bandwidth-bound → quantization +
prompt-lookup speculative decoding, quality-gated) is in
[`INFERENCE_OPTIMIZATION.md`](INFERENCE_OPTIMIZATION.md) with a runnable benchmark
(`scripts/inference_benchmark.py`). *Latency is ~3–5 s/image — an autoregressive VLM; framed as
a compression + speedup + throughput story, not real-time.*

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

## Tech stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Base model | Qwen2.5-VL-3B-Instruct | Apache 2.0 |
| Fine-tuning | LoRA via PEFT | rank 32, alpha 64 |
| Training | HuggingFace Transformers | LoRA SFT, prefix masking, bf16 |
| Demo quantization | bitsandbytes NF4 (4-bit) | HF Spaces T4 demo |
| Data | nuScenes v1.0-trainval | rare-hazard filtered, GT-projected boxes |
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
