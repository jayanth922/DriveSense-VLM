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

## Real results (v3)

Measured, reproducible -- no fabricated numbers. Fine-tuned Qwen2.5-VL-3B + LoRA (r=32, alpha=64)
on the **v3 rare-hazard nuScenes set** (7,228 train / 889 val / 1,041 test; nuScenes v1.0-trainval,
predominantly daytime). Training: 5 epochs, train loss 0.40, eval loss 0.66 (mild overfitting).

**Level-1 grounding (test, n = 1041):**

| Metric | Value |
|---|---|
| Detection Precision @ IoU 0.5 | **40%** |
| Detection Recall @ IoU 0.5 | 24% |
| Detection **F1 @ IoU 0.5** | **30%** |
| Mean IoU of matched boxes | **0.67** |
| Mean best-pair IoU (localization) | 0.51 |
| Frame detect-rate @ IoU 0.1 / 0.3 / 0.5 | 82% / 75% / 66% |
| Classification accuracy (matched) | 94% |
| Severity within +/-1 / Spearman rho | 98.6% / 0.40 |
| Output parse rate | 98.7% |

**High-precision, well-localized, conservative.** When the model predicts a box it is usually
right (40% precision), tight (mean IoU 0.67, above the 0.55 target), and correctly labeled (94%);
its weakness is recall on the rare long tail. An earlier card reported ~1.4% F1 -- that was a
coordinate-convention bug (an inference/training image-resolution mismatch drove predicted boxes
out of the labels' 0-1000 space, collapsing every IoU to ~0); it is fixed and these are the
corrected numbers. See [Limitations](#limitations).

**Level-2 reasoning (LLM-as-judge, Claude Sonnet 5, n = 1,027):**

| Dimension | Mean |
|---|---|
| Correctness | 3.03 |
| Completeness | 2.66 |
| Action relevance | 3.80 |
| **Overall** | **3.16 / 5** |
| Pass rate (all dims >= 3.5) | 26% |

Reasoning is acceptable and driving-action advice is the strongest dimension; completeness is weakest, consistent with the low detection recall.

**Level-4 robustness (stratified, GT-hazard-centric, 691 frames / 2,073 hazards):**

| Slice | Detection rate @ IoU 0.5 |
|---|---|
| Tiny boxes (78% of hazards) | 23% |
| Small / Medium boxes | 46% / 53% |
| Clear weather | ~51-69% |
| Rain (n=337) | 12% |
| Night + tiny (n=308) | 13% |

Performance scales with hazard size (tiny distant boxes hardest), and there is a clear day/clear bias -- rain roughly quarters detection. Honest OOD weakness; more adverse-condition data would help.

### Limitations
- **Low recall on the rare long tail** — recall @ IoU 0.5 is 24%; the model is conservative and
  misses many rare hazards (e.g. `unusual_object`, 24 instances, never detected). This, not
  localization, is the weak axis: precision (40%) and box tightness (mean IoU 0.67 on matches) are
  strong. More data on the rarest classes is the main lever.
- **Narrow training distribution** — 7,228 train frames from nuScenes v1.0-trainval, predominantly
  daytime; night and heavy weather are out of distribution.
- **Mild overfitting** — v3 eval loss (0.66) is ~2x train loss (0.40); fewer epochs would help recall.
- **Dense-frame parse failures now rare (~1.3%)** — raising the token budget to 768 cut truncated
  outputs to 14/1041 frames (98.7% parse), up from ~76% under the old 1024-cap run.
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
