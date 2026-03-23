# DriveSense-VLM

<!-- Badges -->
```
Python 3.10+  |  License: Apache 2.0  |  Status: Phase 0.5a (Scaffolding)
```

## Overview

DriveSense-VLM is an SFT-optimized vision-language model for autonomous vehicle rare hazard
detection. Built on **Qwen3-VL-2B-Instruct** with **LoRA fine-tuning** (rank 32, alpha 64),
it is trained on curated rare-hazard frames from **nuScenes** and **DADA-2000**, augmented with
LLM-generated counterfactual annotations. The inference pipeline applies **AWQ 4-bit
quantization** to the LLM decoder and **TensorRT** compilation to the Vision Transformer,
served via **vLLM** for production throughput. A public **Gradio** demo runs on HuggingFace
Spaces (free T4 GPU). The model outputs structured JSON containing bounding boxes, hazard
class, severity level, chain-of-thought reasoning, and ego-vehicle action recommendations.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                        DriveSense-VLM Pipeline                          │
 └─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────────────┐
  │ Data Curation│    │ SFT Training │    │   Inference Optimization      │
  │              │    │              │    │                               │
  │ nuScenes ────┼──► │ Qwen3-VL-2B  │    │  LoRA Merge                  │
  │  Rarity      │    │  + LoRA      │    │      │                        │
  │  Filter      │    │  (rank 32)   │    │      ▼                        │
  │              │    │              │    │  AWQ 4-bit (LLM only)         │
  │ DADA-2000 ───┼──► │  W&B Track   │───►│      │                        │
  │  Critical    │    │              │    │      ▼                        │
  │  Frames      │    │  HPC SLURM   │    │  TensorRT ViT (fp16)          │
  │              │    │  A100/H100   │    │      │                        │
  │ LLM Counter- │    │              │    │      ▼                        │
  │  factuals ───┼──► │              │    │  vLLM Serving (port 8000)     │
  └──────────────┘    └──────────────┘    └───────────────────────────────┘
                                                       │
                                                       ▼
                                          ┌────────────────────────┐
                                          │  Structured JSON Output │
                                          │  {                      │
                                          │    bbox_2d: [...],      │
                                          │    hazard_class: "...", │
                                          │    severity: 1-5,       │
                                          │    reasoning: "...",    │
                                          │    action: "..."        │
                                          │  }                      │
                                          └────────────────────────┘
                                                       │
                                     ┌─────────────────┴──────────────────┐
                                     │                                    │
                                     ▼                                    ▼
                         ┌─────────────────────┐            ┌────────────────────┐
                         │  4-Level Evaluation  │            │   Gradio Demo      │
                         │  1. Grounding (IoU)  │            │   HF Spaces T4     │
                         │  2. Reasoning (LLM)  │            │   Gradio >= 4.0    │
                         │  3. Production Perf  │            └────────────────────┘
                         │  4. Robustness       │
                         └─────────────────────┘
```

---

## Quick Start

### Local Development (macOS Apple Silicon — CPU only)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/DriveSense-VLM.git
cd DriveSense-VLM

# Create local dev environment (no GPU packages)
conda create -n drivesense-dev python=3.10 -y
conda activate drivesense-dev

# Install dev + data + eval dependencies (no torch/GPU)
pip install -e ".[data,eval,dev]"

# Verify installation
python scripts/run_sanity_check.py

# Run tests
python -m pytest tests/ -v
```

### HPC Setup (SJSU CoE — A100/H100)

```bash
# Copy project to HPC
scp -r DriveSense-VLM/ $HPC_USER@hpc.sjsu.edu:~/DriveSense-VLM/

# SSH into HPC
ssh $HPC_USER@hpc.sjsu.edu

cd ~/DriveSense-VLM

# One-time environment setup
bash scripts/setup_hpc.sh

# Download datasets
bash scripts/download_nuscenes.sh mini   # ~4GB dev split
bash scripts/download_dada2000.sh

# Submit training job
sbatch slurm/train.sbatch
```

---

## Project Structure

```
DriveSense-VLM/
├── README.md
├── .gitignore
├── pyproject.toml            # PEP 621 packaging + dependency groups
├── LICENSE                   # Apache 2.0
├── CLAUDE.md                 # Claude Code project memory
│
├── configs/                  # ALL hyperparameters live here
│   ├── model.yaml            # Base model, LoRA, quantization, vision
│   ├── data.yaml             # Dataset paths, splits, filtering
│   ├── training.yaml         # SFT hyperparameters, W&B config
│   ├── inference.yaml        # Merge, AWQ, TensorRT, vLLM, demo
│   └── eval.yaml             # 4-level evaluation thresholds
│
├── src/drivesense/           # Main package
│   ├── data/                 # Data loading and preprocessing
│   │   ├── nuscenes_loader.py
│   │   ├── dada_loader.py
│   │   ├── annotation.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── training/             # SFT training pipeline
│   │   ├── sft_trainer.py
│   │   └── callbacks.py
│   ├── inference/            # Post-training optimization & serving
│   │   ├── merge_lora.py
│   │   ├── quantize.py
│   │   ├── tensorrt_vit.py
│   │   └── serve.py
│   ├── eval/                 # 4-level evaluation framework
│   │   ├── grounding.py
│   │   ├── reasoning.py
│   │   ├── production.py
│   │   └── robustness.py
│   └── utils/                # Shared utilities
│       ├── config.py
│       ├── logging.py
│       └── visualization.py
│
├── scripts/                  # Operational scripts
│   ├── download_nuscenes.sh
│   ├── download_dada2000.sh
│   ├── setup_hpc.sh
│   └── run_sanity_check.py
│
├── slurm/                    # SLURM job scripts
│   ├── template.sbatch
│   ├── train.sbatch
│   ├── eval.sbatch
│   └── benchmark.sbatch
│
├── notebooks/
│   └── 00_data_exploration.ipynb
│
├── demo/                     # HuggingFace Spaces Gradio app
│   ├── app.py
│   └── requirements.txt
│
└── tests/
    ├── test_config.py
    └── test_imports.py
```

---

## Project Phases

| # | Phase | Description | Status |
|---|-------|-------------|--------|
| 0.5a | Scaffolding | Project structure, configs, stubs | ✅ Complete |
| 1a | Data — nuScenes | Rarity scoring + frame extraction | [ ] |
| 1b | Data — DADA-2000 | Critical moment extraction | [ ] |
| 1c | Data — Annotations | LLM counterfactual generation | [ ] |
| 2a | Training — SFT | LoRA fine-tuning on HPC | [ ] |
| 2b | Training — Eval | Mid-training evaluation loop | [ ] |
| 3a | Inference — Merge | LoRA merge to base weights | [ ] |
| 3b | Inference — AWQ | 4-bit quantization (LLM only) | [ ] |
| 3c | Inference — TRT | TensorRT ViT compilation | [ ] |
| 3d | Inference — vLLM | Production serving setup | [ ] |
| 4a | Demo | Gradio app on HF Spaces | [ ] |
| 4b | Evaluation | Full 4-level eval framework | [ ] |

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Base Model | Qwen3-VL-2B-Instruct | Oct 2025, Apache 2.0 |
| Fine-tuning | LoRA via PEFT | rank=32, alpha=64 |
| Training Framework | HuggingFace Transformers + TRL | SFT with Trainer API |
| Training Hardware | SJSU CoE HPC (A100/H100) | SLURM scheduler |
| Quantization | AutoAWQ | 4-bit, LLM decoder only |
| Vision Optimization | TensorRT | ViT in fp16 |
| Serving | vLLM | Continuous batching |
| Demo | Gradio on HF Spaces | Free T4 GPU |
| Experiment Tracking | Weights & Biases | Full training metrics |
| Dataset 1 | nuScenes v1.0-trainval | ~400-500 rare hazard frames |
| Dataset 2 | DADA-2000 | ~200 accident frames |
| Data Augmentation | Anthropic Claude API | Counterfactual annotations |
| Packaging | PEP 621 (pyproject.toml) | Editable install |
| Linting | Ruff + Black | line-length=100 |
| Testing | pytest | Unit + integration |

---

## Acknowledgments

- **Qwen Team (Alibaba)** for Qwen3-VL-2B-Instruct (Apache 2.0)
- **nuScenes / Motional** for the nuScenes autonomous driving dataset
- **DADA-2000 Authors** for the driver attention in accidents dataset
- **SJSU College of Engineering** for HPC compute access
- **HuggingFace** for the Transformers, PEFT, and Spaces ecosystem
