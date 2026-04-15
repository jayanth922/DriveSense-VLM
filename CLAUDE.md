# DriveSense-VLM

## Project Summary

SFT-optimized VLM for AV rare hazard detection using Qwen3-VL-2B.

## Current Phase

ALL PHASES COMPLETE ✅ (Phase 5: Documentation & Model Card)

## Architecture Decisions

- **Model**: Qwen3-VL-2B-Instruct + LoRA (rank 32, alpha 64)
- **Data**: nuScenes + DADA-2000 + LLM counterfactual augmentation (800–1200 examples total)
- **Output**: Structured JSON (`bbox_2d`, `hazard_class`, `severity`, `reasoning`, `action`)
- **Inference**: AWQ 4-bit LLM + TensorRT ViT (fp16) + vLLM serving
- **Demo**: Gradio + transformers on HF Spaces free T4 GPU
- **Eval**: 4-level framework (grounding accuracy, reasoning quality, production readiness, robustness)
- **Tracking**: Weights & Biases (`drivesense-vlm` project)

- **Annotation pipeline**: `src/drivesense/data/annotation.py` — Phase 1c: AnnotationPromptBuilder, AnnotationValidator, LLMAnnotationPipeline, SFTDataFormatter
- **Prompt templates**: `src/drivesense/data/prompts/` — annotation_system.txt, annotation_user.txt, counterfactual_user.txt, counterfactual_scenarios.json
- **Annotation CLI**: `scripts/run_annotation_pipeline.py` — Phase 1c entry point
- **Training pipeline**: `src/drivesense/training/sft_trainer.py` — Phase 2a: DriveSenseSFTDataset, DriveSenseDataCollator, setup_model_and_processor, train
- **Training callbacks**: `src/drivesense/training/callbacks.py` — GPUMemoryCallback, SamplePredictionCallback, TrainingMetricsCallback, EarlyStoppingCallback
- **Training CLI**: `scripts/run_training.py` — Phase 2a entry point (--dry-run, --mock, --debug, --resume)
- **Grounding evaluator**: `src/drivesense/eval/grounding.py` — Phase 2b: compute_iou, Hungarian matching, compute_grounding_metrics, compute_severity_metrics, GroundingEvaluator
- **Reasoning evaluator**: `src/drivesense/eval/reasoning.py` — Phase 2b: LLMJudge, MockLLMJudge, compute_reasoning_metrics, ReasoningEvaluator
- **Eval CLI**: `scripts/run_evaluation.py` — Phase 2b entry point (--level, --mock-judge, --generate-predictions)
- **Prediction generator**: `scripts/run_generate_predictions.py` — standalone inference script
- **Eval outputs**: `outputs/eval/level1/` and `outputs/eval/level2/` — JSON + text reports
- **Predictions output**: `outputs/predictions/test_predictions.jsonl` — raw + parsed model outputs
- **SLURM job**: `slurm/train.sbatch` — HPC job submission for Phase 2a
- **LoRA merger**: `src/drivesense/inference/merge_lora.py` — Phase 3a: LoRAMerger, merge_lora_checkpoint, verify_merge
- **AWQ quantizer**: `src/drivesense/inference/quantize.py` — Phase 3a: AWQQuantizer, quantize_model, load_calibration_data
- **TensorRT ViT**: `src/drivesense/inference/tensorrt_vit.py` — Phase 3b: ViTExtractor, _ViTWrapper, export_to_onnx, compile_tensorrt, benchmark_vit, full_pipeline
- **Optimization CLI**: `scripts/run_optimize_model.py` — Phase 3a+3b entry point (--all, --merge, --quantize, --tensorrt, --mock)
- **Optimization SLURM**: `slurm/optimize.sbatch` — HPC job for full optimization pipeline
- **Merged model output**: `outputs/merged_model/` — full-weight .safetensors + processor
- **Quantized model output**: `outputs/quantized_model/` — AWQ 4-bit weights + quant_config.json
- **Serving layer**: `src/drivesense/inference/serve.py` — Phase 3c: DriveSenseVLLMServer (vLLM), DriveSenseLocalInference (transformers), draw_hazard_boxes, DRIVESENSE_SYSTEM_PROMPT, SEVERITY_COLORS
- **Benchmark CLI**: `scripts/run_benchmark.py` — Phase 3c entry point (--local, --vllm, --vit-only, --mock, --output)
- **Gradio demo**: `demo/app.py` — Phase 4a: create_demo(), analyze_image(), draw_hazard_boxes(), lazy model loading, HF Spaces T4 target
- **Production evaluator**: `src/drivesense/eval/production.py` — Phase 4b: ProductionEvaluator, compute_production_metrics, load_benchmark_results, generate_report, benchmark_latency, run_production_benchmark
- **Robustness evaluator**: `src/drivesense/eval/robustness.py` — Phase 4b: RobustnessEvaluator, stratify_predictions, compute_stratified_metrics, _extract_stratum_value, _compute_all_gaps, run_robustness_evaluation
- **Full evaluation CLI**: `scripts/run_full_evaluation.py` — Phase 4b: --level 1 2 3 4, --mock, --generate-report; compile_final_report, box-drawing ASCII report (_WIDTH=66)
- **Model card**: `MODEL_CARD.md` — Phase 5: HuggingFace model card YAML frontmatter, all evaluation results, usage examples
- **Demo requirements**: `demo/requirements.txt` — autoawq>=0.2.0, qwen-vl-utils>=0.0.4 added
- **HF Spaces metadata**: `demo/README.md` — YAML frontmatter for HuggingFace Spaces
- **Demo examples**: `demo/examples/` — placeholder directory for example dashcam images
- **Benchmark output**: `outputs/benchmarks/` — per-run JSON benchmark results
- **TensorRT output**: `outputs/tensorrt/` — vit.onnx, vit.engine, vit_benchmark.json, optimization_report.txt, fallback_info.json
- **LoRA adapter output**: `outputs/training/lora_adapter/` — saved LoRA weights + processor
- **Training checkpoints**: `outputs/training/` — intermediate checkpoints
- **Annotated output**: `outputs/data/annotated/` — annotated_manifest.json, quality_report.json, counterfactual_frames.json
- **SFT-ready output**: `outputs/data/sft_ready/` — sft_train.jsonl, sft_val.jsonl, sft_test.jsonl
- **Annotation cache**: `outputs/data/annotation_cache/` — per-frame JSON cache (enables resume)

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

# Phase 1c: LLM annotation pipeline
python scripts/run_annotation_pipeline.py --dry-run --mock-llm   # validate prompts, no API
python scripts/run_annotation_pipeline.py --mock-llm             # full pipeline, no API key
python scripts/run_annotation_pipeline.py                         # real run (needs ANTHROPIC_API_KEY)
python scripts/run_annotation_pipeline.py --format-only          # reformat existing annotations

# Phase 2a: SFT training
python scripts/run_training.py --dry-run --mock                  # validate setup, no download
python scripts/run_training.py --debug                            # 1 epoch / 10 steps (HPC sanity check)
python scripts/run_training.py --config configs/training.yaml --resume   # full run + auto-resume
sbatch slurm/train.sbatch                                         # submit to SLURM

# Phase 2b: Evaluation
python scripts/run_generate_predictions.py --mock                 # test inference pipeline (no download)
python scripts/run_evaluation.py --level 1 --mock-judge           # Level 1 grounding (no API key)
python scripts/run_evaluation.py --level 1 2 --mock-judge         # Level 1 + 2 (mock judge)
python scripts/run_evaluation.py --generate-predictions --level 1 2   # full pipeline
sbatch slurm/eval.sbatch                                          # submit eval to SLURM

# Phase 3a+3b: Optimization
python scripts/run_optimize_model.py --all --mock                 # test full pipeline (no GPU)
python scripts/run_optimize_model.py --merge --adapter-path outputs/training/checkpoint-best
python scripts/run_optimize_model.py --quantize --merged-model outputs/merged_model
python scripts/run_optimize_model.py --tensorrt --model-dir outputs/merged_model
python scripts/run_optimize_model.py --benchmark-vit --mock       # benchmark with mock data
python scripts/run_optimize_model.py --benchmark-quality --mock
sbatch slurm/optimize.sbatch                                      # submit full optimization to SLURM

# Phase 3c: Inference benchmark
python scripts/run_benchmark.py --mock                            # mock mode (no GPU)
python scripts/run_benchmark.py --local                           # local transformers backend
python scripts/run_benchmark.py --vllm                            # vLLM backend (HPC only)
python scripts/run_benchmark.py --vit-only                        # ViT-only throughput
sbatch slurm/benchmark.sbatch                                     # submit full benchmark to SLURM

# Phase 4a: Gradio demo
python demo/app.py                                                # run locally (port 7860)

# Phase 4b: Full 4-level evaluation
python scripts/run_full_evaluation.py --mock                      # all levels (no GPU)
python scripts/run_full_evaluation.py --level 1 2 --mock-judge    # levels 1+2 (no API key)
python scripts/run_full_evaluation.py --level 3 4 --mock          # levels 3+4 mock
python scripts/run_full_evaluation.py --generate-report           # compile final report

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
| 1c | LLM counterfactual annotation pipeline | ✅ Complete |
| 2a | LoRA SFT training on HPC | ✅ Complete |
| 2b | Mid-training evaluation integration | ✅ Complete |
| 3a | LoRA merge + AWQ quantization | ✅ Complete |
| 3b | TensorRT ViT compilation | ✅ Complete |
| 3c | vLLM production serving setup | ✅ Complete |
| 3d | vLLM production serving | ✅ Complete |
| 4a | Gradio demo on HF Spaces | ✅ Complete |
| 4b | Full 4-level evaluation | ✅ Complete |
| 5 | Documentation & Model Card | ✅ Complete |

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
- **Annotation pipeline** (`annotation.py`): `AnnotationPromptBuilder` loads templates from
  `prompts/*.txt` and `counterfactual_scenarios.json`; `AnnotationValidator` validates + fixes
  LLM output (clamp coords, swap inverted bbox, add ego_context defaults, extract JSON from fences);
  `LLMAnnotationPipeline` uses file-based per-frame cache (resume support), async batch with
  semaphore, 3-retry exponential backoff; `SFTDataFormatter` writes Qwen3-VL chat-format JSONL.
- Annotation target schema uses `hazards` array with `label` (from VALID_LABELS), `bbox_2d`
  ([0,1000] integers), `severity`, `reasoning` (≥20 chars), `action`; plus `scene_summary`
  and `ego_context` (weather, time_of_day, road_type).
- Counterfactual augmentation: 30% of frames with real hazards get scenario-based CF prompts;
  scenarios filtered by road_type (e.g. no cyclist scenarios on highway).
- `MockLLMClient` enables full pipeline testing without API key (used in all tests).
- SFT output: one JSONL per split; each line = `{messages: [system, user(image+text), assistant(json)], images: [...]}`.
- Counterfactual augmentation: ~30% of nuScenes frames get LLM-generated counterfactuals
  (e.g., "what if the pedestrian had stepped further into the lane?").
- **SFT training** (`sft_trainer.py`): `DriveSenseSFTDataset` uses prefix-tokenization for label masking — tokenize full sequence AND prefix (messages[:-1] + add_generation_prompt=True); `prefix_len` marks the assistant start boundary; all tokens before it are masked to -100. `DriveSenseDataCollator` concatenates `pixel_values` along the patch dimension (dim=0) — never stack — because Qwen3-VL tiles images dynamically. `setup_model_and_processor()` loads `Qwen2_5_VLForConditionalGeneration` with LoRA; uses `use_reentrant=False` for gradient checkpointing; prefers `flash_attention_2`, falls back to `sdpa`. `train()` auto-detects latest checkpoint for resume; saves emergency checkpoint on failure; uploads LoRA artifact to W&B.
- **Grounding evaluation** (`grounding.py`): `compute_iou` uses standard [x1,y1,x2,y2] intersection formula. `match_predictions_to_ground_truth` uses Hungarian assignment (scipy) with IoU cost matrix; falls back to greedy when scipy unavailable. `compute_grounding_metrics` takes prediction dicts with `frame_id`, `hazards`, `parse_failure` fields; tracks TP/FP/FN/TN per frame with no-hazard frames handled specially. `iou_at_threshold` = Jaccard (TP / (TP+FP+FN)); `false_positive_rate` = FP_no_hazard_frames / total_no_hazard_frames.
- **Predictions JSONL format**: `{"frame_id": str, "raw_output": str, "parsed_output": dict|null, "parse_success": bool, "generation_time_ms": int}`. Parse failures are counted separately from detection misses.
- **LLM judge** (`reasoning.py`): `LLMJudge` calls Claude with `JUDGE_SYSTEM_PROMPT` + per-dimension user prompt; `judge_batch` uses ThreadPoolExecutor with `max_concurrent` workers. `MockLLMJudge` returns random scores in [3,5] for all dimensions. `pass_rate` = fraction of examples scoring ≥3.5 on ALL three dimensions.
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`.
- AWQ quantization targets LLM decoder only; ViT stays in fp16 for accuracy.
- TensorRT ViT uses fixed batch size (no dynamic batching) for deterministic latency.
- **LoRA merge** (`merge_lora.py`): `LoRAMerger` loads base model in bfloat16, wraps with `PeftModel.from_pretrained`, calls `merge_and_unload()`, saves as .safetensors. `get_merge_stats` computes MD5 of config.json for reproducibility. `verify_merge` compares logits from adapter vs merged model with `torch.allclose(atol=1e-3)`. Merge MUST happen before quantization.
- **AWQ quantization** (`quantize.py`): `AWQQuantizer` calls `model.named_modules()` to discover ViT module names (prefix match: "visual", "vision_model", "vit"); passes as `modules_to_not_convert` to `AutoAWQForCausalLM.quantize()`. Calibration data extracted from SFT train JSONL (system+user messages only, no assistant). Falls back to generic AV-domain strings if JSONL unavailable. `get_quantization_stats` reads `quant_config.json` for layer count.
- **TensorRT ViT** (`tensorrt_vit.py`): `ViTExtractor` locates vision encoder at `model.visual` (or first child with "visual"/"vit" in name). `_ViTWrapper` accepts standard [B,C,H,W] input, provides fixed `grid_thw=[[1,16,24]]` for 672×448 images (28×28 patches → 16h×24w=384 patches). ONNX export: opset_version=17, no dynamic axes; falls back to `torch.jit.trace` on custom-op failure. TRT compilation: FP16, fixed workspace; falls back to `torch.compile(mode="reduce-overhead")` if TRT unavailable. All fallbacks documented in `fallback_info.json`. Benchmark measures mean/p50/p95/p99 latency + throughput with CUDA synchronization.
- **Optimization CLI** (`run_optimize_model.py`): `--all` runs merge→quantize→TensorRT sequentially; each stage is idempotent (skips if sentinel file exists). `--mock` creates stub output files without loading any models — used in tests and CI.
- **Serving layer** (`serve.py`): `DRIVESENSE_SYSTEM_PROMPT` and `SEVERITY_COLORS` are module-level constants shared between serve.py and demo/app.py. `DriveSenseVLLMServer` loads AWQ model via `LLM(quantization="awq", trust_remote_code=True)`; `predict_batch` uses `SamplingParams(temperature=0, stop=["<|im_end|>"])`; `benchmark()` uses `ThreadPoolExecutor` for concurrent load. `DriveSenseLocalInference` is lazy-loaded (model=None at init); `_run_inference` uses `apply_chat_template` + `do_sample=False`; both AWQ and full-precision models work.
- **`draw_hazard_boxes`** (`serve.py`): Creates RGBA overlay image, fills each bbox with `alpha=50` (~20% opacity), solid `alpha=255` outline; label `"{label} ({severity})"` drawn 18px above box. Uses `Image.alpha_composite` then converts back to RGB. Falls back gracefully if PIL unavailable.
- **bbox normalisation**: bbox coordinates in [0, 1000] → pixel: `x = bbox_x * w / 1000`. Same formula in both serve.py and demo/app.py (demo has standalone `draw_hazard_boxes` for Spaces compatibility).
- **`_parse_json_output`** (`serve.py`): 3-stage parse: direct JSON → strip markdown fences → regex extract `{...}`. Returns `{"parse_failure": raw, "hazards": []}` on total failure.
- **Benchmark CLI** (`run_benchmark.py`): `--local` times `DriveSenseLocalInference.predict()` sequentially; `--vllm` delegates to `server.benchmark()` (concurrent); `--vit-only` delegates to `ViTExtractor.benchmark_vit()`; `--mock` returns pre-baked numbers. Output timestamped to `outputs/benchmarks/benchmark_<ts>.json`. Synthetic images (solid colour) used when no `--image-dir` supplied.
- **Production evaluator** (`production.py`): `ProductionEvaluator` reads all thresholds from `config["production"]["targets"]`. `load_benchmark_results(dir)` reads `local_bench.json` (→ T4 metrics) and `vllm_bench.json` (→ A100 metrics). ViT benchmark is optional: when None, `vit_tensorrt_p50_ms=None` and `vit_latency` target passes (True). Latency target is strict less-than (`p50 < target`, so equal fails). `quant_degradation_pct` derived from `label_agreement` field: `(1 - label_agreement) * 100`. `_get_p50(d)` tries `p50_ms` then falls back to `mean_ms`.
- **Robustness evaluator** (`robustness.py`): Stratifies by `time_of_day`, `weather`, `location`, `ego_speed_bucket`, `source`. `_extract_stratum_value(gt_record, key)` checks three locations: `metadata{}` → `ego_context{}` → top-level → "unknown". `_speed_bucket(speed_kmh)`: < 20 → "0-20", < 40 → "20-40", else "40+". `_infer_source(frame_id)`: "dada" prefix → "dada2000" else "nuscenes". `_detection_rate_gap(stratum)`: max(DR) - min(DR) for groups with n_frames > 0. Empty groups handled via `contextlib.suppress` → `_empty_group_metrics()` (returns zeros). `ood_relative_performance` = dada2000_DR / nuscenes_DR (None-safe).
- **Full evaluation CLI** (`run_full_evaluation.py`): `--level 1 2 3 4` selects which levels to run; `--mock` bypasses real model inference for all levels; `--mock-judge` uses MockLLMJudge for Level 2. Box-drawing report uses `_WIDTH=66`, `_row()` pads to exact width, `_bar()` for horizontal lines. `_mock_level3_metrics()`: T4 p50=432ms, A100 p50=187ms, VIT=21ms, fps=9.2, VRAM=3.1GB, deg=1.3%. `_mock_level4_metrics()`: day/night gap=0.072, weather=0.118, location=0.054, OOD=0.891. Level 3 reads stored JSON files (idempotent — no live GPU benchmark).
- **Gradio demo** (`demo/app.py`): `create_demo()` builds the `gr.Blocks` interface; `analyze_image()` calls `model.predict_with_visualization()`; returns `(annotated_image, json_str, latency_str)`. `_load_model()` is a global lazy-loader — safe under Gradio's single-threaded default mode. Config loaded from `configs/inference.yaml` if present, else `MODEL_PATH` env var. `demo/app.py` has standalone `draw_hazard_boxes` that mirrors `serve.py` for Spaces compatibility (no package install needed).
- **HF Spaces**: `demo/README.md` has YAML frontmatter (`sdk: gradio`, `app_file: app.py`, hardware T4). `demo/requirements.txt` includes `autoawq>=0.2.0` and `qwen-vl-utils>=0.0.4`.
