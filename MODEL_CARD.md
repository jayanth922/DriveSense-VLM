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
**projected from nuScenes 3-D ground truth** for the v2/v3 base set (the production model,
100% GT-projected) — a foundation model writes only the severity/reasoning/action text for
each real box. v4's targeted-mining addition (1,442 of its 8,670 train examples, 16.6%) is
the one exception: those boxes are foundation-model-emitted, not GT-projected — see the
[label-provenance
confound](FLYWHEEL_V4_FINDINGS.md#label-provenance-confound-in-the-v4-experiment). v4 was
blocked by the regression gate and never promoted.

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
| **Training data** | v3 (production): 7,228 rare-hazard nuScenes v1.0-trainval train frames (7,228 / 889 val / 1,041 test) |
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

## Results (measured on a fixed 1,041-frame test set)

Fine-tuned on the **v3 rare-hazard nuScenes set** (7,228 train / 889 val / 1,041 test;
nuScenes v1.0-trainval, predominantly daytime). A **v4** candidate (8,670 train, +1,442
targeted rain/night frames from the mining flywheel) was trained and evaluated on the *same
fixed test set* — and **blocked by the regression gate**. `v3` remains the production model.

**Level-1 grounding (IoU ≥ 0.5):**

| version | train ex. | epochs | eval_loss | Precision | Recall | F1 | mean IoU | class acc | parse |
|---|---|---|---|---|---|---|---|---|---|
| **v3 (production)** | 7,228 | 5 | 0.66 | **0.40** | **0.24** | **0.30** | **0.67** | 0.94 | 98.7% |
| v4 (blocked) | 8,670 | 3 | 0.694 | 0.37 | 0.19 | 0.25 | 0.656 | 0.95 | 97.4% |

**Level-4 robustness (detection@0.5 by bucket):**

| bucket | v3 (production) | v4 (blocked) | Δ |
|---|---|---|---|
| overall | 28.0% | 23.0% | −5.0 pp |
| tiny box (78% of hazards) | 22.8% | 17.2% | −5.6 pp |
| small / medium box | 46.4% / 52.6% | 42.9% / 52.6% | −3.5 / 0.0 pp |
| rain | 12.5% | 7.4% | −5.1 pp |
| night + tiny | 12.7% | 10.7% | −2.0 pp |
| clear + medium | 69.0% | 69.0% | 0.0 pp |

**The honest finding.** Two data-scaling experiments reached the same conclusion. v3 scaled
data naively (2,754 → 7,228) and generalization *hurt* (eval_loss 0.31 → 0.66). v4 then added
1,442 **targeted** adverse-condition frames aimed squarely at the weakest buckets — and those
buckets *still regressed*. For this model, adding data is not the lever for rare-hazard recall;
the bottleneck is model-side (input resolution, tiny-box weighting). The regression gate
correctly blocked v4 rather than shipping it.

The model is **high-precision, well-localized, and conservative**: when it predicts a box it is
usually right (40% precision) and tight (mean IoU 0.67, above the 0.55 target), correctly
labeled 94% of the time. The weak axis is **recall on the rare long tail**.

> Note: an earlier version of this card reported ~1.4% F1. That was a coordinate-convention bug —
> inference ran the image processor at a different resolution than training, so predicted boxes
> drifted out of the 0–1000 space the labels use, collapsing every IoU to ~0. It is fixed; the
> numbers above are the corrected results.

---

## Limitations

- **Low recall on the rare long tail** — recall @ IoU 0.5 is 24%; the model is conservative and
  misses many rare hazards (e.g. `unusual_object`, 24 instances, never detected). Precision (40%)
  and box tightness (mean IoU 0.67) are strong; recall is the weak axis.
- **More data is demonstrably not the lever** — both naive (v3) and targeted (v4) data scaling
  failed to lift recall on the weak buckets. See the v4 result above.
- **Narrow training distribution** — nuScenes v1.0-trainval, predominantly daytime. Night and
  heavy weather are out of distribution: rain detection (12.5%) is roughly a quarter of clear.
- **Dense-frame parse failures rare** — 98.7% parse (v3) after raising the generation token
  budget to 768, up from ~76% under the old 1024-cap run.
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

This describes the production model (v3), which is 100% GT-projected. The blocked v4
candidate mixed in 1,442 frames where Claude draws the box directly (targeted mining,
clamped/repaired but not GT-verified) — see the [label-provenance
confound](FLYWHEEL_V4_FINDINGS.md#label-provenance-confound-in-the-v4-experiment).

**SFT format**: Qwen2.5-VL chat-format JSONL; one example per frame; the assistant turn is the
structured-JSON hazard list, supervised via prefix-masked labels.

---

## Training procedure

```
Base:            Qwen/Qwen2.5-VL-3B-Instruct
Method:          LoRA SFT (rank 32, alpha 64)
Targets:         q_proj, k_proj, v_proj, o_proj, up_proj, down_proj
Precision:       bf16
Epochs:          5   (v3 production: train loss 0.40, eval loss 0.66)
                 3   (v4 candidate: eval loss 0.694, no overfit — but gate-BLOCKED)
Label masking:   prefix tokenization (only assistant JSON supervised)
Hardware:        single A100
```

---

## Intended use & out-of-scope

**Intended**: research on VLM grounding for AV perception; a reference implementation of an
honest data → training → evaluation pipeline with pre-training data-quality gates.

**Out of scope**: any real-time perception, driver assistance, or safety-critical control. The
grounding accuracy is far too low (24% recall @ IoU 0.5), and the model was trained on a narrow, predominantly-daytime slice of nuScenes.

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
