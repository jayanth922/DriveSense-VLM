#!/usr/bin/env python3
"""Phase 3a+3b: Run model optimization pipeline.

Usage:
    # Full pipeline: merge → quantize → TensorRT
    python scripts/run_optimize_model.py --all

    # Individual stages
    python scripts/run_optimize_model.py --merge --adapter-path outputs/training/best
    python scripts/run_optimize_model.py --quantize --merged-model outputs/merged_model
    python scripts/run_optimize_model.py --tensorrt --model-dir outputs/merged_model

    # Benchmark only
    python scripts/run_optimize_model.py --benchmark-vit
    python scripts/run_optimize_model.py --benchmark-quality

    # Mock mode (for testing without GPU)
    python scripts/run_optimize_model.py --all --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.utils.config import load_config, merge_configs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_optimize_model")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="DriveSense-VLM: Model Optimization Pipeline (Phase 3a+3b)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Stage selection
    p.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: merge → quantize → TensorRT",
    )
    p.add_argument("--merge", action="store_true", help="Run LoRA merge stage only")
    p.add_argument("--quantize", action="store_true", help="Run AWQ quantization only")
    p.add_argument("--tensorrt", action="store_true", help="Run TensorRT ViT compilation only")
    p.add_argument("--benchmark-vit", action="store_true", help="Benchmark ViT backends only")
    p.add_argument("--benchmark-quality", action="store_true",
                   help="Benchmark quantization quality only")

    # Path overrides
    p.add_argument(
        "--adapter-path",
        default=None,
        help="Path to LoRA adapter (overrides config default)",
    )
    p.add_argument(
        "--merged-model",
        default=None,
        help="Path to merged model dir (for --quantize / --tensorrt stages)",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Path to model dir for TensorRT / benchmark stages",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for all stages",
    )

    # Other options
    p.add_argument(
        "--mock",
        action="store_true",
        help="Mock mode — no model downloads or GPU operations (for CI/testing)",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip merge verification (faster)",
    )
    p.add_argument(
        "--config",
        default="configs/inference.yaml",
        help="Path to inference.yaml",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_merge(config: dict, adapter_path: Path, mock: bool = False, verify: bool = True) -> Path:
    """Run Stage 1: LoRA adapter merge.

    Args:
        config:       Merged config dict.
        adapter_path: Path to LoRA adapter directory.
        mock:         If True, write stub files without GPU ops.
        verify:       If True, run logit comparison after merge.

    Returns:
        Path to merged model directory.
    """
    out = Path(config.get("merge", {}).get("output_dir", "outputs/merged_model"))
    if _stage_already_done(out, "config.json"):
        logger.info("Merge output already exists at %s — skipping", out)
        return out

    if mock:
        return _mock_merge(out)

    from drivesense.inference.merge_lora import LoRAMerger  # noqa: PLC0415
    merger = LoRAMerger(config)
    return merger.merge(adapter_path, verify=verify)


def run_quantize(
    config: dict,
    merged_model_dir: Path,
    mock: bool = False,
) -> Path:
    """Run Stage 2: AWQ 4-bit quantization.

    Args:
        config:           Merged config dict.
        merged_model_dir: Path to merged model (Stage 1 output).
        mock:             If True, write stub files without GPU ops.

    Returns:
        Path to quantized model directory.
    """
    out = Path(config.get("quantization", {}).get("output_dir", "outputs/quantized_model"))
    if _stage_already_done(out, "quant_config.json"):
        logger.info("Quantized model already exists at %s — skipping", out)
        return out

    if mock:
        return _mock_quantize(out)

    from drivesense.inference.quantize import AWQQuantizer  # noqa: PLC0415
    quantizer = AWQQuantizer(config)
    return quantizer.quantize(merged_model_dir)


def run_tensorrt(
    config: dict,
    model_dir: Path,
    mock: bool = False,
) -> dict:
    """Run Stage 3: TensorRT ViT compilation + benchmark.

    Args:
        config:    Merged config dict.
        model_dir: Path to merged model (used for ViT extraction).
        mock:      If True, write stub files without GPU ops.

    Returns:
        Benchmark results dict.
    """
    trt_cfg = config.get("tensorrt", {})
    out = Path(trt_cfg.get("output_dir", "outputs/tensorrt"))

    if _stage_already_done(out, "vit_benchmark.json"):
        logger.info("TensorRT artifacts already exist at %s — skipping", out)
        bm_path = out / "vit_benchmark.json"
        return json.loads(bm_path.read_text(encoding="utf-8")) if bm_path.exists() else {}

    if mock:
        return _mock_tensorrt(out)

    from drivesense.inference.tensorrt_vit import ViTExtractor  # noqa: PLC0415
    extractor = ViTExtractor(config)
    return extractor.full_pipeline(model_dir, out)


def run_benchmark_vit(config: dict, model_dir: Path, mock: bool = False) -> dict:
    """Run ViT benchmark only (no compilation).

    Args:
        config:    Merged config dict.
        model_dir: Path to model directory.
        mock:      If True, return mock benchmark data.

    Returns:
        Benchmark results dict.
    """
    if mock:
        return _mock_benchmark()

    trt_cfg = config.get("tensorrt", {})
    engine_path = Path(trt_cfg.get("engine_path", "outputs/tensorrt/vit.engine"))

    from drivesense.inference.tensorrt_vit import ViTExtractor  # noqa: PLC0415
    extractor = ViTExtractor(config)
    return extractor.benchmark_vit(model_dir, engine_path=engine_path)


def run_benchmark_quality(
    config: dict,
    merged_dir: Path,
    quantized_dir: Path,
    mock: bool = False,
) -> dict:
    """Run quantization quality benchmark.

    Args:
        config:        Merged config dict.
        merged_dir:    Path to full-precision merged model.
        quantized_dir: Path to quantized model.
        mock:          If True, return mock quality data.

    Returns:
        Quality benchmark dict.
    """
    if mock:
        return {
            "text_similarity": 0.97,
            "bbox_mae": 3.2,
            "label_agreement": 0.95,
            "size_reduction": 3.8,
            "original_size_gb": 4.1,
            "quantized_size_gb": 1.08,
        }

    from drivesense.inference.quantize import AWQQuantizer  # noqa: PLC0415
    quantizer = AWQQuantizer(config)
    return quantizer.benchmark_quality(merged_dir, quantized_dir, test_samples=[])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Require at least one stage flag
    if not any([args.all, args.merge, args.quantize, args.tensorrt,
                args.benchmark_vit, args.benchmark_quality]):
        logger.error(
            "No stage specified. Use --all or individual stage flags. "
            "Run with --help for usage."
        )
        sys.exit(1)

    # Load and merge all configs
    cfg_dir = Path(args.config).parent
    model_cfg = load_config(cfg_dir / "model.yaml")
    data_cfg = load_config(cfg_dir / "data.yaml")
    training_cfg = load_config(cfg_dir / "training.yaml")
    inference_cfg = load_config(args.config)
    config = merge_configs(model_cfg, data_cfg, training_cfg, inference_cfg)

    # Resolve paths
    adapter_path = Path(
        args.adapter_path
        or "outputs/training/lora_adapter"
    )
    merged_dir = Path(
        args.merged_model
        or config.get("merge", {}).get("output_dir", "outputs/merged_model")
    )
    model_dir = Path(
        args.model_dir
        or config.get("merge", {}).get("output_dir", "outputs/merged_model")
    )
    quantized_dir = Path(
        config.get("quantization", {}).get("output_dir", "outputs/quantized_model")
    )

    results: dict = {}

    if args.all or args.merge:
        logger.info("=== Stage 1: LoRA Merge ===")
        merged_dir = run_merge(config, adapter_path, mock=args.mock,
                               verify=not args.no_verify)
        results["merge"] = {"output_dir": str(merged_dir)}

    if args.all or args.quantize:
        logger.info("=== Stage 2: AWQ Quantization ===")
        quantized_dir = run_quantize(config, merged_dir, mock=args.mock)
        results["quantize"] = {"output_dir": str(quantized_dir)}

    if args.all or args.tensorrt:
        logger.info("=== Stage 3: TensorRT ViT Compilation ===")
        bm = run_tensorrt(config, model_dir, mock=args.mock)
        results["tensorrt"] = bm

    if args.benchmark_vit:
        logger.info("=== ViT Benchmark ===")
        bm = run_benchmark_vit(config, model_dir, mock=args.mock)
        results["benchmark_vit"] = bm

    if args.benchmark_quality:
        logger.info("=== Quality Benchmark ===")
        qm = run_benchmark_quality(config, merged_dir, quantized_dir, mock=args.mock)
        results["benchmark_quality"] = qm

    print("\n--- Optimization Pipeline Summary ---")
    print(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Mock helpers (for --mock mode)
# ---------------------------------------------------------------------------


def _stage_already_done(out_dir: Path, sentinel_file: str) -> bool:
    """Return True if sentinel_file exists in out_dir (idempotency check)."""
    return (out_dir / sentinel_file).exists()


def _mock_merge(out: Path) -> Path:
    """Write stub merged model files without loading any model."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_vl", "_mock": True}), encoding="utf-8"
    )
    (out / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 512}), encoding="utf-8"
    )
    # Stub safetensors index
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {}}), encoding="utf-8"
    )
    logger.info("[MOCK] Merge output written to %s", out)
    return out


def _mock_quantize(out: Path) -> Path:
    """Write stub quantized model files without running AWQ."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_vl", "quantization_config": {"bits": 4}}),
        encoding="utf-8",
    )
    (out / "quant_config.json").write_text(
        json.dumps({
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "version": "GEMM",
            "num_quantized_layers": 28,
        }),
        encoding="utf-8",
    )
    logger.info("[MOCK] Quantized model output written to %s", out)
    return out


def _mock_tensorrt(out: Path) -> dict:
    """Write stub TensorRT artifacts without GPU ops."""
    out.mkdir(parents=True, exist_ok=True)
    bm = _mock_benchmark()
    (out / "vit.onnx").write_bytes(b"mock_onnx")
    (out / "vit.engine").write_bytes(b"mock_engine")
    (out / "vit_benchmark.json").write_text(json.dumps(bm, indent=2), encoding="utf-8")
    (out / "fallback_info.json").write_text(
        json.dumps({"onnx_method": "mock", "trt_method": "mock"}), encoding="utf-8"
    )
    (out / "optimization_report.txt").write_text("[MOCK] Optimization report", encoding="utf-8")
    logger.info("[MOCK] TensorRT artifacts written to %s", out)
    return bm


def _mock_benchmark() -> dict:
    """Return realistic mock benchmark values for --mock mode."""
    return {
        "input_shape": [1, 3, 448, 672],
        "num_iterations": 100,
        "pytorch_eager": {
            "mean_ms": 45.2,
            "p50_ms": 44.8,
            "p95_ms": 48.1,
            "p99_ms": 51.3,
            "throughput_fps": 22.1,
        },
        "torch_compile": {
            "mean_ms": 28.7,
            "p50_ms": 28.2,
            "p95_ms": 30.4,
            "p99_ms": 32.1,
            "throughput_fps": 34.8,
        },
        "tensorrt": {
            "mean_ms": 12.4,
            "p50_ms": 12.1,
            "p95_ms": 13.2,
            "p99_ms": 14.0,
            "throughput_fps": 80.6,
        },
        "speedup_compile_vs_eager": 1.57,
        "speedup_tensorrt_vs_eager": 3.65,
    }


if __name__ == "__main__":
    main()
