# DriveSense-VLM

## Project Summary

SFT-optimized VLM for AV rare hazard detection using Qwen3-VL-2B.

## Current Phase

Phase 1a-spark: PySpark Distributed Rarity Scoring ✅

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
- **DADA-2000 loader**: `src/drivesense/data/dada_loader.py` — Phase 1b DADA2000Loader
- **DADA extraction CLI**: `scripts/run_dada_extraction.py` — Phase 1b entry point
- **DADA output**: `outputs/data/dada_extracted/` — images + metadata.jsonl
- **Unified dataset**: `src/drivesense/data/dataset.py` — UnifiedDatasetBuilder + DriveSenseDataset
- **Unified build CLI**: `scripts/run_build_unified_dataset.py` — Phase 1b unified dataset builder
- **Unified output**: `outputs/data/unified/` — per-split manifest JSONL files
- **Filtering script**: `scripts/run_nuscenes_filter.py` — Phase 1a pipeline CLI
- **Filtered output**: `outputs/data/nuscenes_filtered/` — images + metadata JSON
- **Spark pipeline**: `src/drivesense/data/spark_pipeline.py` — Phase 1a-spark ETL
- **Spark CLI**: `scripts/run_spark_pipeline.py` — Phase 1a-spark entry point
- **Spark output**: `outputs/data/spark_processed/` — scored Parquet + analytics

## Commands

```bash
# Sanity check (all modules + configs import cleanly)
python scripts/run_sanity_check.py

# Tests
python -m pytest tests/ -v

# Spark pipeline (requires pyspark: pip install pyspark>=3.5)
python scripts/run_spark_pipeline.py --version v1.0-mini
python scripts/run_spark_pipeline.py --skip-extraction        # reuse existing JSONL
python scripts/run_spark_pipeline.py --analytics-only         # analytics only

# Phase 1b: DADA-2000 extraction
python scripts/run_dada_extraction.py --dada-root ~/data/dada2000
python scripts/run_dada_extraction.py --max-sequences 10      # debug/dev

# Phase 1b: Build unified dataset
python scripts/run_build_unified_dataset.py
python scripts/run_build_unified_dataset.py --nuscenes-only
python scripts/run_build_unified_dataset.py --dada-only

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
| 1a | nuScenes rarity filtering + frame extraction | ✅ Complete |
| 1a-spark | PySpark distributed rarity scoring + analytics | ✅ Complete |
| 1b | DADA-2000 critical moment extraction | ✅ Complete |
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

- nuScenes rarity score is a composite of **6 binary signals** (max score = 6):
  proximity (<5m to ego), occlusion (0–40% visibility), density (≥15 agents),
  adverse weather/night keywords, vulnerable road user (**pedestrian present AND
  scene description contains intersection keyword**), cyclist present.
  Minimum threshold to keep a frame: **3** (from config).
- `intersection_keywords` in `configs/data.yaml` drives the VRU signal's intersection
  check (keyword match on `scene['description']` as a lightweight proxy for map data).
- `NuScenesRarityFilter.filter_rare_frames()` scores ALL samples and caches results
  in `_all_scores`; call it before `get_score_distribution()` or `export_filtered_dataset()`.
- `export_filtered_dataset()` raises `RuntimeError` if `filter_rare_frames()` was never called.
- **Spark pipeline** (`spark_pipeline.py`): `NuScenesMetadataExtractor` → JSON Lines;
  `SparkRarityScorer.compute_all_scores()` left-joins 6 signal DataFrames + caches;
  `SparkAnalytics` provides score_distribution (with cumulative %), signal_cooccurrence (6×6),
  per_scene_stats, category_breakdown, temporal_clustering (burst detection via lag+window).
- Spark schemas are always **explicit** (`StructType`) — never use `inferSchema`.
- `filter_by_threshold()` raises `RuntimeError` if `compute_all_scores()` was not called first.
- Always call `scorer.stop()` (in a `finally` block) to release the SparkSession.
- DADA-2000 extraction: `DADA2000Loader` scans `<dada_root>/DADA-2000/<cat>/<seq>/images/`; extracts critical frame + `additional_context_frames` before (pre_accident) and after (mid_accident); resizes to 672×448 via `resize_with_aspect_ratio`; exports `metadata.jsonl` + images.
- `normalize_column_names()` does fuzzy case-insensitive matching against Excel column headers — handles column name variations in `dada_text_annotations.xlsx`.
- `UnifiedDatasetBuilder` merges nuScenes (Parquet or JSONL) + DADA-2000 (JSONL); assigns train/val/test via `StratifiedShuffleSplit` (stratified on source+category); falls back to sequential split when sklearn unavailable or n<10.
- `DriveSenseDataset(manifest_path, split, config, processor)` takes the per-split manifest JSONL; `get_collate_fn()` returns `collate_fn` which batches images as a list (not tensored — VLM processor handles padding).
- `resize_with_letterbox(image, target_size)` returns `(image, params_dict)` with keys `scale`, `pad_x`, `pad_y`, `new_w`, `new_h` for reverse bbox projection.
- DADA-2000 extraction: critical moment frame + 2 context frames before.
- Counterfactual augmentation: ~30% of nuScenes frames get LLM-generated counterfactuals
  (e.g., "what if the pedestrian had stepped further into the lane?").
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`.
- AWQ quantization targets LLM decoder only; ViT stays in fp16 for accuracy.
- TensorRT ViT uses fixed batch size (no dynamic batching) for deterministic latency.
