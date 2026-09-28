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

# DriveSense-VLM model card

[Back to the project README](../README.md)

DriveSense-VLM is Qwen2.5-VL-3B-Instruct fine-tuned with LoRA for structured road-hazard
analysis. Given one dashcam frame, it returns hazard boxes, labels, severity, a short explanation,
and a recommended driving action.

This is a research model for offline evaluation. It is not suitable for vehicle control,
driver-assistance decisions, or other safety-critical use.

## Model

| Field | Value |
|---|---|
| Base | Qwen/Qwen2.5-VL-3B-Instruct |
| Fine-tuning | LoRA, rank 32, alpha 64 |
| LoRA targets | q/k/v/o projections and up/down MLP projections |
| Training precision | bf16 |
| Selected checkpoint | v3 |
| Selected training split | 7,228 train / 889 validation / 1,041 test |
| Training hardware | one A100 |
| Demo format | bitsandbytes NF4 on a T4 |

## Output

Coordinates use a normalized 0-1000 image space.

    {
      "hazards": [
        {
          "bbox_2d": [120, 340, 280, 810],
          "label": "occluded_pedestrian",
          "severity": "high",
          "reasoning": "A pedestrian is partly hidden by a parked vehicle near the lane.",
          "action": "Reduce speed and prepare to stop."
        }
      ]
    }

Labels are construction_zone, cyclist_proximity, high_density, jaywalking,
occluded_pedestrian, unusual_object, and no_hazard.

## Training data and box source

The selected v3 model uses nuScenes v1.0-trainval CAM_FRONT frames. Its boxes are projected from
nuScenes 3D annotations with near-plane clipping and normalized to the output coordinate system.
Claude supplies severity, explanation, and action text for accepted boxes.

The blocked v4 candidate added 1,442 targeted frames whose boxes were emitted directly by Claude.
That difference is important: a controlled follow-up found that GT-projected supervision produced
substantially better localization. See [TASK3_DECONFOUND.md](TASK3_DECONFOUND.md).

## Evaluation

v3 and v4 were compared on the same 1,041-frame test set.

### Grounding at IoU >= 0.5

| Version | Train frames | Epochs | Eval loss | Precision | Recall | F1 | Mean IoU | Class accuracy | Parse rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3, selected | 7,228 | 5 | 0.66 | 0.40 | 0.24 | 0.30 | 0.67 | 0.94 | 98.7% |
| v4, blocked | 8,670 | 3 | 0.694 | 0.37 | 0.19 | 0.25 | 0.656 | 0.946 | 97.4% |

### Detection by condition

| Bucket | v3 | v4 |
|---|---:|---:|
| Overall | 28.0% | 23.0% |
| Tiny boxes | 22.8% | 17.2% |
| Small boxes | 46.4% | 42.9% |
| Medium boxes | 52.6% | 52.6% |
| Rain | 12.5% | 7.4% |
| Night and tiny | 12.7% | 10.7% |
| Clear and medium | 69.0% | 69.0% |

The regression gate blocked v4 because the protected weak conditions became worse. v3 remains the
selected checkpoint.

### Reasoning

Claude Sonnet 5 judged 1,027 v3 examples on a 1-5 scale.

| Dimension | Score |
|---|---:|
| Correctness | 3.03 |
| Completeness | 2.66 |
| Action relevance | 3.80 |
| Overall | 3.16 |
| Pass rate, all dimensions >= 3.5 | 26% |

## Inference

On one T4, fp16 produced 17.0 decode tokens/s with 11.64-second median end-to-end latency. Prompt
lookup increased decode throughput to 20.4 tokens/s and reduced median latency to 9.79 seconds with
exactly matching output. NF4 reduced weight memory from about 6.0 GB to 2.63 GB but was slower and
changed the output.

See [INFERENCE_OPTIMIZATION.md](INFERENCE_OPTIMIZATION.md) for the complete benchmark.

## Known limitations

- Recall is 24% at IoU 0.5, with the largest gaps on tiny, distant, rainy, and nighttime hazards.
- The training distribution is predominantly daytime nuScenes data.
- Input is a single frame, so the model cannot use motion history.
- The reasoning score comes from an LLM judge rather than human safety review.
- NF4 output differs from fp16 and must be evaluated before use.
- Median T4 latency is measured in seconds, not real-time driving latency.

## Intended use

Appropriate uses include research on grounded VLMs, evaluation tooling, data curation, and examples
of regression-gated model development. Any deployment would require substantially stronger recall,
broader data, independent safety validation, and a real-time perception architecture.

## Citation

    @software{drivesense_vlm_2026,
      title  = {DriveSense-VLM: Fine-tuned Qwen2.5-VL-3B for structured AV hazard detection},
      author = {Kalyanam, Jayanth},
      year   = {2026}
    }

## Acknowledgments

The project uses Qwen2.5-VL, nuScenes, Hugging Face Transformers and PEFT, and the Anthropic API.
