# DriveSense-VLM

## Project Summary

SFT-optimized VLM for AV rare hazard detection using Qwen3-VL-2B.

## Current Phase

Phase 0.5a: Project Scaffolding ✅

## Architecture Decisions

- **Model**: Qwen3-VL-2B-Instruct + LoRA (rank 32, alpha 64)
- **Data**: nuScenes + DADA-2000 + LLM counterfactual augmentation (800–1200 examples total)
- **Output**: Structured JSON (`bbox_2d`, `hazard_class`, `severity`, `reasoning`, `action`)
- **Inference**: AWQ 4-bit LLM + TensorRT ViT (fp16) + vLLM serving
- **Demo**: Gradio + transformers on HF Spaces free T4 GPU
- **Eval**: 4-level framework (grounding accuracy, reasoning quality, production readiness, robustness)
- **Tracking**: Weights & Biases (`drivesense-vlm` project)

## Key Paths

- **Configs**: `configs/*.yaml` — ALL hyperparameters live here, never hardcode values
- **Source**: `src/drivesense/` — main Python package
- **Scripts**: `scripts/` — download, HPC setup, sanity check
- **SLURM jobs**: `slurm/*.sbatch` — HPC job submission scripts
- **Tests**: `tests/` — pytest test suite

## Commands

```bash
# Sanity check (all modules + configs import cleanly)
python scripts/run_sanity_check.py

# Tests
python -m pytest tests/ -v

# Lint
ruff check src/

# Format
black src/
```

## Environment

| Environment | Hardware | Notes |
|-------------|----------|-------|
| Local (macOS Apple Silicon) | CPU only | No GPU packages; dev + data + eval deps only |
| HPC (SJSU CoE) | A100 / H100, SLURM | conda env `drivesense`; full training stack |
| Demo (HF Spaces) | Free T4 GPU | Gradio app; transformers inference only |

## Phase Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 0.5a | Project Scaffolding | ✅ Complete |
| 1a | nuScenes rarity filtering + frame extraction | [ ] |
| 1b | DADA-2000 critical moment extraction | [ ] |
| 1c | LLM counterfactual annotation pipeline | [ ] |
| 2a | LoRA SFT training on HPC | [ ] |
| 2b | Mid-training evaluation integration | [ ] |
| 3a | LoRA merge | [ ] |
| 3b | AWQ 4-bit quantization | [ ] |
| 3c | TensorRT ViT compilation | [ ] |
| 3d | vLLM production serving | [ ] |
| 4a | Gradio demo on HF Spaces | [ ] |
| 4b | Full 4-level evaluation | [ ] |

## Rules for Claude Code

1. **ALWAYS read the relevant config YAML** before modifying any module — configs are the
   single source of truth for all hyperparameters and paths.
2. **NEVER hardcode file paths** — use `configs/*.yaml` values accessed via `pathlib.Path`.
3. **NEVER install GPU packages locally** — `torch`, `vllm`, `tensorrt`, `autoawq`,
   `bitsandbytes`, and `flash-attn` are HPC-only. Use `try/except ImportError` guards.
4. **ALWAYS add type hints** to all function signatures (use `from __future__ import annotations`).
5. **ALWAYS write tests** for new functionality in `tests/`.
6. **ALWAYS update this CLAUDE.md** when completing a phase (update the Phase Tracker above).
7. **Use Google-style docstrings** for all public functions and classes.
8. **Keep functions under 50 lines** — split into helpers if needed.
9. **Use `pathlib.Path`** everywhere — never `os.path` string manipulation.
10. **Stubs raise `NotImplementedError`** with the phase tag, e.g.:
    `raise NotImplementedError("Phase 1a: implement nuScenes rarity scoring")`

## Output Schema

```json
{
  "bbox_2d": [x1, y1, x2, y2],
  "hazard_class": "pedestrian_in_path | vehicle_cut_in | debris | ...",
  "severity": 1,
  "reasoning": "Chain-of-thought explanation...",
  "action": "emergency_brake | yield | lane_change | maintain_speed"
}
```

## Key Design Notes

- nuScenes rarity score is a composite of: proximity (<5m), occlusion (0–40% visibility),
  scene density (>15 agents), and adverse weather keywords. Minimum score: 3.
- DADA-2000 extraction: critical moment frame + 2 context frames before.
- Counterfactual augmentation: ~30% of nuScenes frames get LLM-generated counterfactuals
  (e.g., "what if the pedestrian had stepped further into the lane?").
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`.
- AWQ quantization targets LLM decoder only; ViT stays in fp16 for accuracy.
- TensorRT ViT uses fixed batch size (no dynamic batching) for deterministic latency.
