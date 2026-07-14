---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags:
  - vision-language-model
  - autonomous-driving
  - hazard-detection
  - lora
  - qwen2.5-vl
datasets:
  - nuScenes
language:
  - en
pipeline_tag: image-text-to-text
---

# DriveSense-VLM

**Qwen2.5-VL-3B-Instruct fine-tuned on rare-hazard AV dashcam data via LoRA SFT.**

DriveSense-VLM detects and explains rare, safety-critical hazards in autonomous-driving dashcam
frames, emitting structured JSON: a per-hazard bounding box, a 7-class hazard label, severity,
chain-of-thought reasoning, and a recommended ego-vehicle action. Bounding-box labels are
**projected from nuScenes 3-D ground truth**; a foundation model writes only the
severity/reasoning/action text for each real box.

> **Status: research / offline-evaluation only.** Grounding accuracy is low (see Results). This
> model is a debugging-and-rebuild case study, not a deployable detector.

---

## Model details

| Field | Value |
|-------|-------|
| **Base model** | `Qwen/Qwen2.5-VL-3B-Instruct` |
| **Fine-tuning** | LoRA (rank 32, alpha 64; targets `q/k/v/o/up/down_proj`) |
| **Precision** | bf16 (training); bitsandbytes NF4 4-bit for the T4 demo |
| **Task** | Structured rare-hazard detection + reasoning |
| **Training data** | ~688 rare-hazard nuScenes v1.0-trainval frames (549/72/67) |
| **Hardware** | Single A100 (training), T4 (demo) |

### Output schema
```json
{
  "hazards": [
    {
      "bbox_2d": [x1, y1, x2, y2],
      "label": "occluded_pedestrian",
      "severity": "high",
      "reasoning": "Pedestrian partially occluded by parked van, near ego path...",
      "action": "brake"
    }
  ]
}
```
7-class taxonomy: `construction_zone`, `cyclist_proximity`, `high_density`, `jaywalking`,
`occluded_pedestrian`, `unusual_object`, `no_hazard`.

---

## Results (v2 — measured, reproducible)

Fine-tuned on 688 rare-hazard nuScenes frames (549 train / 72 val / 67 test; ~8 v1.0-trainval
logs, predominantly daytime, Boston + Singapore). Training: 8 epochs, train loss 0.75,
`train ≈ val` (no overfitting).

**Level-1 grounding (test, n = 67):**

| Metric | Value |
|--------|-------|
| Detection Recall @ IoU 0.5 | 1.0% |
| Detection Precision @ IoU 0.5 | 2.4% |
| **Detection F1 @ IoU 0.5** | **1.4%** |
| Mean best-pair IoU (localization) | 0.12 |
| Frame detect-rate @ IoU 0.1 / 0.3 / 0.5 | 33% / 20% / 5% |
| Output parse rate | 76% |

**These numbers are low, and reported exactly as measured.** The model is real and un-collapsed
(it localizes something near a hazard on ~1/3 of frames), but grounding is weak — the small,
narrow dataset is the dominant limiter.

---

## Limitations

- **Small, narrow dataset** — 688 frames from ~8 logs, predominantly daytime, two cities. This is
  the primary reason grounding is weak; more and more-diverse data is the main lever.
- **Weak localization** — predictions rarely reach the IoU 0.5 threshold.
- **Dense-frame parse failures (~24%)** — on the densest multi-hazard frames the model's output
  **exceeds the generation token budget and ends mid-JSON** (verified: 16/67 test frames truncated
  at ~1,164 tokens under the 1024-token cap → parse failure). Root cause is **undertraining on a
  small dataset**; repetition of hazard objects on dense frames may be a contributing factor.
  Increasing the token budget did not resolve it.
- **Not for deployment** — research / offline evaluation only; never for real-time or
  safety-critical vehicle control.

---

## Training data

**nuScenes** (v1.0-trainval, CAM_FRONT keyframes). Each frame is scored 0–6 across 6 binary rarity
signals: proximity (< 5 m to ego), occlusion (0–40% visibility), density (≥ 15 agents), adverse
weather/night, vulnerable road user at intersection, cyclist present. Frames scoring ≥ 5 are
selected and deduplicated per scene. **Bounding boxes are projected from the dataset's 3-D
ground-truth annotations** into the 2-D camera frame (near-plane frustum clipped). A foundation
model (Claude) writes only severity/reasoning/action per real box — it never draws boxes. A hard
validation gate blocks the label set from training on any sign of collapse (repeated boxes,
oversized boxes, cross-frame duplication, schema violations).

**SFT format**: Qwen2.5-VL chat-format JSONL; one example per frame; the assistant turn is the
structured-JSON hazard list, supervised via prefix-masked labels.

---

## Training procedure

```
Base:            Qwen/Qwen2.5-VL-3B-Instruct
Method:          LoRA SFT (rank 32, alpha 64)
Targets:         q_proj, k_proj, v_proj, o_proj, up_proj, down_proj
Precision:       bf16
Epochs:          8   (train loss 0.75, train ≈ val)
Label masking:   prefix tokenization (only assistant JSON supervised)
Hardware:        single A100
```

---

## Intended use & out-of-scope

**Intended**: research on VLM grounding for AV perception; a reference implementation of an
honest data → training → evaluation pipeline with pre-training data-quality gates.

**Out of scope**: any real-time perception, driver assistance, or safety-critical control. The
grounding accuracy is far too low, and the model was trained on a small, narrow slice of nuScenes.

---

## Citation

```bibtex
@software{drivesense_vlm_2026,
  title   = {DriveSense-VLM: Fine-tuned Qwen2.5-VL-3B for structured AV hazard detection},
  author  = {Kalyanam, Jayanth},
  year    = {2026},
  note    = {LoRA SFT on rare-hazard nuScenes frames with 3D-ground-truth box projection
             and a pre-training data-quality validation gate}
}
```

---

## Acknowledgments

- **Qwen Team (Alibaba)** — Qwen2.5-VL-3B-Instruct (Apache 2.0)
- **nuScenes / Motional** — nuScenes autonomous-driving dataset
- **HuggingFace** — Transformers, PEFT, Spaces
- **Anthropic** — Claude API for describe-only annotation and LLM-as-judge evaluation
