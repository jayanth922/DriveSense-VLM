# DriveSense-VLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Model: Qwen2.5-VL-3B](https://img.shields.io/badge/model-Qwen2.5--VL--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

DriveSense-VLM is an end-to-end project for detecting rare road hazards in dashcam images. It
fine-tunes Qwen2.5-VL-3B-Instruct with LoRA and returns a structured report for each frame:

- hazard class;
- 2D bounding box;
- severity;
- short explanation;
- recommended driving action.

The repository covers the full model lifecycle: distributed data mining, annotation, validation,
training, evaluation, regression gating, inference profiling, monitoring, and deployment.

## Project summary

The data pipeline starts with nuScenes keyframes. PySpark and the streaming miner identify rare
scenes without requiring the full image archive to fit on disk. For the main v2/v3 dataset,
nuScenes 3D annotations are projected into the front camera to create 2D boxes. Claude supplies
the description, severity, and action for those boxes; it does not localize them.

The model is trained with LoRA SFT and evaluated at four levels:

1. box grounding and hazard classification;
2. reasoning quality;
3. inference and production metrics;
4. performance by weather, time of day, and object size.

A regression gate compares each candidate with the current model. The v4 candidate failed that
gate, so v3 remains the selected checkpoint. This negative result is retained because it shows
that adding targeted data can still reduce performance when label provenance changes.

## Results

Detection metrics come from
[`results/metrics_registry.json`](results/metrics_registry.json). v3 and v4 use the same
1,041-frame nuScenes test set.

### Detection at IoU >= 0.5

| Version | Training frames | Precision | Recall | F1 | Mean IoU | Class accuracy | Parse rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3, selected | 7,228 | 0.40 | 0.24 | 0.30 | 0.67 | 0.94 | 98.7% |
| v4, blocked | 8,670 | 0.37 | 0.19 | 0.25 | 0.656 | 0.946 | 97.4% |

The model localizes matched hazards well, but recall remains the main limitation. Tiny and distant
objects account for most of the missed detections.

### Performance by condition

| Test bucket | v3 | v4 |
|---|---:|---:|
| Overall | 28.0% | 23.0% |
| Tiny boxes | 22.8% | 17.2% |
| Small boxes | 46.4% | 42.9% |
| Medium boxes | 52.6% | 52.6% |
| Rain | 12.5% | 7.4% |
| Night and tiny | 12.7% | 10.7% |
| Clear and medium | 69.0% | 69.0% |

v4 added 1,442 targeted adverse-condition frames, but the exact buckets it targeted became worse.
The CI gate therefore blocked promotion. See
[`docs/FLYWHEEL_V4_FINDINGS.md`](docs/FLYWHEEL_V4_FINDINGS.md) for the analysis.

### Reasoning quality

The v3 reasoning evaluation used Claude Sonnet 5 as judge on 1,027 examples.

| Dimension | Score |
|---|---:|
| Correctness | 3.03 / 5 |
| Completeness | 2.66 / 5 |
| Action relevance | 3.80 / 5 |
| Overall | 3.16 / 5 |
| Pass rate, all dimensions >= 3.5 | 26% |

Action recommendations were stronger than hazard coverage. This agrees with the grounding result:
the model is more likely to omit a hazard than to give poor advice for one it has detected.

## Box-provenance experiment

The v4 data addition changed two variables at once: scene selection and box source. Its new boxes
were emitted by the foundation model instead of projected from nuScenes ground truth. A follow-up
experiment trained two otherwise matched arms to isolate that difference.

| Metric | FM-emitted boxes | GT-projected boxes |
|---|---:|---:|
| Recall | 0.101 | 0.167 |
| Precision | 0.176 | 0.330 |
| F1 | 0.128 | 0.222 |
| Mean best-pair IoU | 0.244 | 0.458 |
| Frame detection at IoU 0.5 | 0.266 | 0.566 |
| No-hazard accuracy | 0.512 | 0.831 |

GT-projected boxes performed better on every localization measure. Matched-class accuracy was
similar, which points to localization rather than class naming as the main difference.

This was a reduced-scale reconstruction because the original per-frame v3/v4 artifacts were not
retained. The comparison used 2,652 base frames, 1,162 targeted frames, and 402 test frames. The
single-seed result should be treated as evidence about direction, not a final estimate of effect
size. Full details are in [`docs/TASK3_DECONFOUND.md`](docs/TASK3_DECONFOUND.md).

## Inference study

Inference was measured on one NVIDIA T4. The baseline is memory-bandwidth-bound during
autoregressive decoding.

| Configuration | Decode tokens/s | TTFT | End-to-end p50 | Weight memory | HBM utilization | Output comparison |
|---|---:|---:|---:|---:|---:|---|
| fp16 | 17.0 | 770 ms | 11.64 s | about 6.0 GB | 31.8% | reference |
| fp16 + prompt lookup | 20.4 | 781 ms | 9.79 s | about 6.0 GB | 38.2% | exact match 1.00 |
| NF4 | 12.6 | 820 ms | about 15.9 s | 2.63 GB | 10.4% | character similarity 0.36 |
| INT8 | 4.6 | about 1,080 ms | 53.53 s | about 3.5 GB | 5.0% | character similarity 0.29 |

Prompt lookup reduced median latency by 16% without changing the output. NF4 reduced model memory
by about 2.3x but made decoding slower and changed the generated text. In this workload,
quantization is useful for fitting the model in memory, not for reducing latency. fp16 throughput
increased from 14.9 to 33.7 tokens/s at batch size 4.

The benchmark and measurement notes are in
[`docs/INFERENCE_OPTIMIZATION.md`](docs/INFERENCE_OPTIMIZATION.md). The numbers can be reproduced
with [`scripts/inference_benchmark.py`](scripts/inference_benchmark.py).

## Pipeline

```text
nuScenes metadata and image blobs
        |
        v
rarity scoring and bounded-storage mining
        |
        v
3D-to-2D box projection and description generation
        |
        v
schema and box-diversity validation gate
        |
        v
Qwen2.5-VL-3B LoRA training
        |
        v
grounding, reasoning, robustness, and latency evaluation
        |
        v
candidate regression gate: promote or block
```

Important safeguards include near-plane clipping during projection, scene-level split isolation,
hard validation of box diversity, an all-zero-IoU abort, and CI checks against the selected model.

## Quick start

Local development does not require a GPU:

```bash
git clone https://github.com/jayanth922/DriveSense-VLM.git
cd DriveSense-VLM
python -m venv .venv
source .venv/bin/activate
python -m pip install pyyaml pillow numpy scipy tqdm pytest
python scripts/run_sanity_check.py
python -m pytest -q tests
```

Training and full inference require a CUDA GPU. The notebooks under `notebooks/` and the Task 3
runbook under `experiments/task3_deconfound/` contain the GPU workflows.

## Repository layout

| Path | Purpose |
|---|---|
| `src/drivesense/` | Reusable data, training, inference, evaluation, and monitoring code |
| `scripts/` | Command-line entry points for each pipeline stage |
| `configs/` | Model, data, training, inference, and evaluation settings |
| `tests/` | CPU-safe unit and integration tests |
| `notebooks/` | Colab workflows for data, training, optimization, and evaluation |
| `experiments/task3_deconfound/` | Controlled FM-box versus GT-box experiment |
| `deploy/huggingface/` | Hugging Face model card and Space application |
| `results/` | Versioned metrics and compact evaluation artifacts |
| `docs/` | Design notes, runbooks, findings, and model documentation |

Useful starting points:

- [`docs/FLYWHEEL.md`](docs/FLYWHEEL.md): data and training loop;
- [`docs/DEBUGGING_POSTMORTEM.md`](docs/DEBUGGING_POSTMORTEM.md): failures and fixes;
- [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md): regression and drift monitoring;
- [`docs/CLOSED_LOOP.md`](docs/CLOSED_LOOP.md): failure-driven data selection;
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md): model use and limitations;
- [`docs/TENSORRT_RUNBOOK.md`](docs/TENSORRT_RUNBOOK.md): ViT export investigation.

## Current status and next work

The implemented pipeline is complete. The remaining work is experimental rather than required for
running the repository:

- repeat the box-provenance experiment at the original 7,228-frame scale and on the full
  1,041-frame test set;
- run multiple seeds for confidence intervals;
- test a v4b training set without the 216 `no_hazard` examples added in v4;
- evaluate vLLM on a compatible GPU image. The T4 environment used for this study could not run it.

## Limitations

- v3 recall is 24% at IoU 0.5; tiny, distant, rainy, and nighttime hazards remain difficult.
- nuScenes trainval is weighted toward daytime driving and does not represent every deployment
  environment.
- The reasoning judge is another language model and should not be treated as a human safety review.
- End-to-end T4 latency is measured in seconds per image, so this is not a real-time driving stack.
- The system is a research prototype. It must not be used for vehicle control or other
  safety-critical decisions.

## Technology

- Qwen2.5-VL-3B-Instruct
- PyTorch, Transformers, PEFT, and bitsandbytes
- nuScenes v1.0-trainval and PySpark
- Anthropic Claude for description generation and reasoning evaluation
- pytest, GitHub Actions, and Weights & Biases
- Gradio and Hugging Face Spaces

## Acknowledgments

This project uses Qwen2.5-VL, nuScenes, Hugging Face Transformers and PEFT, and the Anthropic API.
See the linked projects and dataset terms for their licenses and usage requirements.
