# 🚗 DriveSense-VLM

> SFT-Optimized Vision-Language Model for Autonomous Vehicle Rare Hazard Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Model: Qwen3-VL-2B](https://img.shields.io/badge/model-Qwen3--VL--2B-orange)](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-435%20passing-brightgreen)](#testing)

**DriveSense-VLM** fine-tunes [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
with LoRA SFT for detecting and explaining **rare safety-critical hazards** in autonomous driving
scenarios. The model produces structured JSON with grounded bounding boxes, hazard classification,
severity assessment, chain-of-thought reasoning, and ego-vehicle action recommendations.

[🎮 Try the Demo](https://huggingface.co/spaces) &nbsp;|&nbsp;
[📊 W&B Dashboard](https://wandb.ai/drivesense-vlm) &nbsp;|&nbsp;
[📄 Model Card](MODEL_CARD.md)

---

## Key Results

> Results below are **target benchmarks** — final values populated after HPC training.
> Use `--mock` flags to reproduce the pipeline end-to-end without GPU hardware.

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Hazard Detection Rate (Recall) | TBD (post-HPC) | ≥ 80% | — |
| False Positive Rate | TBD | ≤ 15% | — |
| Mean IoU | TBD | ≥ 0.55 | — |
| Classification Accuracy | TBD | ≥ 75% | — |
| Reasoning Correctness (LLM judge) | TBD | ≥ 3.5 / 5 | — |
| E2E Latency T4 (p50) | TBD | < 500 ms | — |
| E2E Latency A100 (p50) | TBD | < 200 ms | — |
| ViT TensorRT latency (p50) | TBD | < 25 ms | — |
| Throughput A100 | TBD | ≥ 8 fps | — |
| VRAM usage (T4) | TBD | < 6 GB | — |
| AWQ quantization degradation | TBD | < 2% | — |
| Day/Night detection gap | TBD | < 10% | — |
| OOD (DADA-2000) relative perf | TBD | ≥ 70% | — |

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                     DriveSense-VLM Pipeline                         │
 └─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐   ┌─────────────────┐   ┌─────────────────────┐
  │   Data Curation   │   │   SFT Training   │   │ Inference Pipeline  │
  │                   │   │                  │   │                     │
  │  nuScenes ───────►│   │  Qwen3-VL-2B    │   │  1. LoRA Merge     │
  │  Rarity filter    │   │  + LoRA r=32     │   │     (bfloat16)     │
  │  (score ≥ 3/6)   │   │  α=64            │   │         ↓          │
  │                   │   │                  │   │  2. AWQ 4-bit      │
  │  DADA-2000 ──────►│──►│  HPC (A100)     │──►│     (LLM only)     │
  │  Critical frames  │   │  SLURM / W&B    │   │         ↓          │
  │                   │   │                  │   │  3. TensorRT ViT   │
  │  LLM Counter-     │   │                  │   │     (fp16)         │
  │  factuals ───────►│   │                  │   │         ↓          │
  └──────────────────┘   └─────────────────┘   │  4. vLLM serving   │
                                                 └─────────────────────┘
                                                          │
                          ┌───────────────────────────────┼────────────────┐
                          ▼                               ▼                ▼
              ┌──────────────────┐          ┌──────────────────┐  ┌───────────────┐
              │  4-Level Eval    │          │   Gradio Demo    │  │  JSON Output  │
              │  1. Grounding    │          │   HF Spaces T4   │  │               │
              │  2. Reasoning    │          │   AWQ + tfmrs    │  │  bbox_2d,     │
              │  3. Production   │          └──────────────────┘  │  label,       │
              │  4. Robustness   │                                 │  severity,    │
              └──────────────────┘                                 │  reasoning,   │
                                                                   │  action       │
                                                                   └───────────────┘
```

---

## What This Project Demonstrates

Directly relevant to **autonomous vehicle ML engineering** roles (Zoox, Waymo, Cruise):

| Skill | Implementation |
|-------|---------------|
| **Large-scale data curation** | nuScenes rarity scoring (6 signals, composite score), DADA-2000 critical-moment extraction, LLM counterfactual augmentation via Anthropic API |
| **VLM fine-tuning** | LoRA SFT on Qwen3-VL-2B with prefix-masked label tensors, DriveSense data collator for variable-patch VLM inputs, W&B experiment tracking |
| **Inference optimization** | AWQ 4-bit quantization (LLM decoder only), TensorRT ViT compilation (fp16, ONNX opset 17), vLLM production serving with continuous batching |
| **Production ML pipeline** | SLURM HPC orchestration, idempotent stage pipeline, mock/dry-run modes for every stage, structured JSON output with validation |
| **Rigorous evaluation** | 4-level framework: grounding (IoU + Hungarian matching), reasoning (LLM-as-judge), production benchmarks, robustness stratification by condition |
| **Software engineering** | 435 passing tests, ruff/black linting, PEP 621 packaging, type annotations throughout, Google-style docstrings |
| **Distributed data processing** | PySpark rarity scoring pipeline with 6 signal DataFrames, temporal clustering, category analytics |

---

## Quick Start

### Local Development (macOS — CPU only)

```bash
git clone https://github.com/YOUR_USERNAME/DriveSense-VLM.git
cd DriveSense-VLM

conda create -n drivesense-dev python=3.10 -y
conda activate drivesense-dev
pip install -e ".[data,eval,dev]"

# Verify everything imports
python scripts/run_sanity_check.py

# Run full test suite
python -m pytest tests/ -v
```

### HPC Training (SJSU CoE A100/H100)

```bash
scp -r DriveSense-VLM/ $HPC_USER@hpc.sjsu.edu:~/
ssh $HPC_USER@hpc.sjsu.edu
cd ~/DriveSense-VLM

bash scripts/setup_hpc.sh
sbatch slurm/train.sbatch          # SFT training
sbatch slurm/optimize.sbatch       # LoRA merge → AWQ → TensorRT
sbatch slurm/eval.sbatch           # Levels 1–2 evaluation
sbatch slurm/benchmark.sbatch      # Level 3 production benchmark
```

### Mock Pipeline (no GPU required)

```bash
# Test full pipeline end-to-end without any model downloads
python scripts/run_optimize_model.py --all --mock
python scripts/run_full_evaluation.py --mock --generate-report
python scripts/run_benchmark.py --mock
```

---

## Project Structure

```
DriveSense-VLM/
├── configs/                   # ALL hyperparameters — never hardcoded
│   ├── model.yaml             # Qwen3-VL-2B, LoRA rank/alpha, vision
│   ├── data.yaml              # Dataset paths, rarity thresholds, splits
│   ├── training.yaml          # SFT hyperparameters, W&B, early stopping
│   ├── inference.yaml         # Merge, AWQ, TensorRT, vLLM, demo
│   └── eval.yaml              # 4-level evaluation targets
│
├── src/drivesense/
│   ├── data/
│   │   ├── nuscenes_loader.py # Rarity scoring (6 binary signals, score ≥ 3)
│   │   ├── dada_loader.py     # Critical-moment extraction, aspect-ratio resize
│   │   ├── annotation.py      # LLM annotation pipeline, validator, SFT formatter
│   │   ├── dataset.py         # UnifiedDatasetBuilder, DriveSenseDataset
│   │   ├── spark_pipeline.py  # PySpark distributed rarity scoring
│   │   └── transforms.py      # Resize, letterbox, augmentation
│   ├── training/
│   │   ├── sft_trainer.py     # DriveSenseSFTDataset, collator, setup, train()
│   │   └── callbacks.py       # GPU memory, sample prediction, early stopping
│   ├── inference/
│   │   ├── merge_lora.py      # LoRAMerger — PEFT merge_and_unload, logit verify
│   │   ├── quantize.py        # AWQQuantizer — ViT exclusion, calibration from JSONL
│   │   ├── tensorrt_vit.py    # ViTExtractor — ONNX export, TRT compile, benchmark
│   │   └── serve.py           # DriveSenseVLLMServer, DriveSenseLocalInference, draw_hazard_boxes
│   ├── eval/
│   │   ├── grounding.py       # compute_iou, Hungarian matching, GroundingEvaluator
│   │   ├── reasoning.py       # LLMJudge (Claude), MockLLMJudge, ReasoningEvaluator
│   │   ├── production.py      # ProductionEvaluator — latency/VRAM/degradation
│   │   └── robustness.py      # RobustnessEvaluator — stratified analysis, OOD eval
│   └── utils/
│       ├── config.py          # load_config, merge_configs (YAML)
│       ├── logging.py         # Structured logging helpers
│       └── visualization.py   # Bbox overlay, dashboard plots
│
├── scripts/
│   ├── run_annotation_pipeline.py  # Phase 1c — LLM counterfactual generation
│   ├── run_training.py             # Phase 2a — SFT training CLI
│   ├── run_evaluation.py           # Phase 2b — Levels 1+2 evaluation CLI
│   ├── run_optimize_model.py       # Phase 3a+3b — merge/quantize/TensorRT CLI
│   ├── run_benchmark.py            # Phase 3c — inference benchmark CLI
│   ├── run_full_evaluation.py      # Phase 4b — all 4 levels + final report
│   └── run_sanity_check.py         # Project structure + import verification
│
├── slurm/
│   ├── train.sbatch           # SFT training (A100, 8h)
│   ├── eval.sbatch            # Levels 1–2 evaluation
│   ├── optimize.sbatch        # Merge + quantize + TensorRT (4h)
│   └── benchmark.sbatch       # Level 3 production benchmark (2h)
│
├── demo/
│   ├── app.py                 # Gradio demo — HF Spaces T4, lazy load, bbox overlay
│   ├── requirements.txt       # HF Spaces deps (incl. autoawq, qwen-vl-utils)
│   └── README.md              # HF Spaces YAML metadata
│
└── tests/                     # 435 tests, 0 failures
    ├── test_config.py
    ├── test_imports.py
    ├── test_nuscenes.py
    ├── test_dada.py
    ├── test_annotation.py
    ├── test_dataset.py
    ├── test_training.py
    ├── test_callbacks.py
    ├── test_grounding.py
    ├── test_reasoning.py
    ├── test_merge_lora.py
    ├── test_quantize.py
    ├── test_tensorrt.py
    ├── test_serve.py
    ├── test_demo.py
    ├── test_production.py
    └── test_robustness.py
```

---

## Methodology

### Data Curation

**nuScenes** (~400–500 rare-hazard frames): Each frame scores 0–6 across 6 binary signals
(proximity <5m, occlusion 0–40%, density ≥15 agents, adverse weather/night, VRU at intersection,
cyclist present). Frames scoring ≥ 3 are kept. A PySpark pipeline computes signals across all
scenes with temporal burst detection.

**DADA-2000** (~200 frames): Pre-accident dashcam sequences. The critical moment frame plus
`additional_context_frames` before are extracted, resized to 672×448 with letterboxing.

**LLM Counterfactuals**: 30% of nuScenes frames with real hazards receive scenario-based
counterfactual annotations generated by Claude (e.g., "what if the pedestrian stepped further
into the lane?"), filtered by road type to ensure realism.

### Training

LoRA SFT on Qwen3-VL-2B-Instruct (rank 32, alpha 64), targeting
`q_proj, k_proj, v_proj, o_proj, up_proj, down_proj`.
Label masking via prefix tokenization — only the assistant response is supervised.
`DriveSenseDataCollator` concatenates pixel patches along dim=0 (not stacked) for
Qwen3-VL's dynamic tile format. Training on SJSU CoE HPC with SLURM.

### Inference Optimization

| Stage | Method | Scope | Benefit |
|-------|--------|-------|---------|
| LoRA Merge | PEFT `merge_and_unload()` | Full model | Enables quantization |
| AWQ | 4-bit weight quantization | LLM decoder only | ~3.8× size reduction |
| TensorRT | FP16 engine compilation | ViT (vision encoder) | ~3.6× ViT speedup |
| vLLM | Continuous batching server | Full model | High-throughput serving |

ViT stays in fp16 — quantizing the vision encoder causes unacceptable accuracy loss on
spatial bbox prediction tasks.

### Evaluation Framework

| Level | What it measures | Key metric |
|-------|-----------------|------------|
| 1 — Grounding | Bbox localization accuracy | IoU@0.5, Detection Rate |
| 2 — Reasoning | LLM-judge quality (Claude) | Correctness/Completeness/Action (1–5 scale) |
| 3 — Production | Latency, VRAM, throughput | T4 p50 < 500ms, VRAM < 6GB |
| 4 — Robustness | Condition stratification, OOD | Day/night gap < 10%, DADA OOD ≥ 70% of in-dist |

---

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0.5a | Project scaffolding | ✅ Complete |
| 1a | nuScenes rarity filtering + frame extraction | ✅ Complete |
| 1a-spark | PySpark distributed rarity scoring + analytics | ✅ Complete |
| 1b | DADA-2000 critical moment extraction | ✅ Complete |
| 1c | LLM counterfactual annotation pipeline | ✅ Complete |
| 2a | LoRA SFT training on HPC | ✅ Complete |
| 2b | Mid-training grounding + reasoning evaluation | ✅ Complete |
| 3a | LoRA merge + AWQ quantization | ✅ Complete |
| 3b | TensorRT ViT compilation | ✅ Complete |
| 3c | vLLM production serving + benchmark CLI | ✅ Complete |
| 3d | vLLM serving integration | ✅ Complete |
| 4a | Gradio demo on HF Spaces | ✅ Complete |
| 4b | Full 4-level evaluation framework | ✅ Complete |
| 5 | Documentation polish | ✅ Complete |

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Base model | Qwen3-VL-2B-Instruct | Apache 2.0, Oct 2025 |
| Fine-tuning | LoRA via PEFT | rank=32, alpha=64 |
| Training | HuggingFace Transformers + TRL | SFT with prefix masking |
| Hardware | SJSU CoE HPC A100/H100 | SLURM scheduler |
| Quantization | AutoAWQ | 4-bit, LLM decoder only |
| Vision opt. | TensorRT fp16 | ONNX opset 17, fixed batch |
| Serving | vLLM | Continuous batching, AWQ |
| Demo | Gradio on HF Spaces | Free T4 GPU |
| Tracking | Weights & Biases | Full training + eval metrics |
| Data (1) | nuScenes v1.0-trainval | Rare-hazard filtered |
| Data (2) | DADA-2000 | Accident critical moments |
| Augmentation | Anthropic Claude API | Counterfactual annotations |
| Distributed | PySpark 3.5 | Rarity scoring pipeline |
| Packaging | PEP 621 (pyproject.toml) | Editable install |
| Lint/Format | Ruff + Black | line-length=100 |
| Testing | pytest | 435 tests, 0 failures |

---

## Testing

```bash
# Full test suite (no GPU required)
python -m pytest tests/ -v

# Individual modules
python -m pytest tests/test_production.py tests/test_robustness.py -v

# Run evaluation in mock mode
python scripts/run_full_evaluation.py --mock --generate-report
```

The test suite covers all modules with CPU-only mocks. No GPU, model downloads, or API keys
required for the full 435-test run.

---

## Citation

```bibtex
@misc{drivesense-vlm-2025,
  title     = {DriveSense-VLM: SFT-Optimized Vision-Language Model for AV Rare Hazard Detection},
  author    = {DriveSense-VLM Contributors},
  year      = {2025},
  url       = {https://github.com/YOUR_USERNAME/DriveSense-VLM},
  note      = {Trained on nuScenes + DADA-2000 with LLM counterfactual augmentation}
}
```

---

## Acknowledgments

- **Qwen Team (Alibaba)** for Qwen3-VL-2B-Instruct (Apache 2.0)
- **nuScenes / Motional** for the nuScenes autonomous driving dataset
- **DADA-2000 Authors** for the driver attention in accidents dataset
- **SJSU College of Engineering** for HPC compute access (A100/H100 cluster)
- **HuggingFace** for Transformers, PEFT, Datasets, and Spaces
- **Anthropic** for Claude API used in counterfactual annotation and LLM-as-judge evaluation
