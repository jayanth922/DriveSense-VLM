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

## Real results (v2)

Measured, reproducible — no fabricated numbers. Fine-tuned Qwen2.5-VL-3B + LoRA (r=32, α=64) on
**688 rare-hazard nuScenes frames** (549 train / 72 val / 67 test; ~8 v1.0-trainval logs,
predominantly daytime, Boston + Singapore). Training: 8 epochs, train loss 0.75, `train ≈ val`
(no overfitting).

**Level-1 grounding (test, n = 67):**

| Metric | Value |
|---|---|
| Detection Recall @ IoU 0.5 | 1.0% |
| Detection Precision @ IoU 0.5 | 2.4% |
| Detection **F1 @ IoU 0.5** | **1.4%** |
| Mean best-pair IoU (localization) | **0.12** |
| Frame detect-rate @ IoU 0.1 / 0.3 / 0.5 | 33% / 20% / 5% |
| Output parse rate | 76% |

**These numbers are low, and that is the honest result.** The model is real and un-collapsed —
it localizes *something* near a hazard on ~1/3 of frames — but grounding is weak, driven mainly
by the small, narrow dataset. See [Limitations](#limitations).

### Limitations
- **Small, narrow dataset** (688 frames, ~8 logs, daytime, 2 cities) — the primary reason
  grounding is weak; more data is the main lever for better numbers.
- **Weak localization** — detections rarely reach the IoU 0.5 threshold.
- **Dense-frame parse failures (~24%)** — on the densest multi-hazard frames the model's output
  exceeds the generation token budget and ends mid-JSON (verified: 16/67 test frames truncated at
  ~1,164 tokens under the 1024-token cap). Root cause is **undertraining on a small dataset**;
  repetition of hazard objects on dense frames may be a contributing factor. Raising the token
  budget did not fix it.
- Research / offline-evaluation use only — not for real-time or safety-critical control.

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
