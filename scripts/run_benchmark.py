#!/usr/bin/env python3
"""Phase 3c: Production inference benchmark CLI.

Measures end-to-end latency, throughput, and memory usage for the
DriveSense-VLM serving stack.  Supports local (transformers) and vLLM
backends; images can be supplied or generated synthetically.

Usage:
    # Local transformers backend (HF Spaces / CPU dev)
    python scripts/run_benchmark.py --local

    # ViT-only throughput (no LLM decoding)
    python scripts/run_benchmark.py --vit-only

    # Mock mode — no model loads, instant results for CI
    python scripts/run_benchmark.py --mock

    # Custom output path
    python scripts/run_benchmark.py --local --output outputs/benchmarks/my_bench.json

    # vLLM backend (HPC only, requires quantized model)
    python scripts/run_benchmark.py --vllm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
logger = logging.getLogger("run_benchmark")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="DriveSense-VLM: Inference Benchmark (Phase 3c)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--local", action="store_true", help="Benchmark local transformers backend")
    p.add_argument("--vllm", action="store_true", help="Benchmark vLLM backend (HPC only)")
    p.add_argument("--vit-only", action="store_true", help="Benchmark ViT encoder only")
    p.add_argument("--mock", action="store_true", help="Mock mode — no GPU ops (for CI)")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: outputs/benchmarks/benchmark_<timestamp>.json)",
    )
    p.add_argument(
        "--config",
        default="configs/inference.yaml",
        help="Path to inference.yaml",
    )
    p.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent worker threads for vLLM benchmark (default: 4)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations before timing (default: 3)",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help="Directory of .jpg/.png images to use; synthetic if omitted",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_local_benchmark(
    config: dict,
    num_iterations: int = 50,
    warmup: int = 3,
    image_dir: Path | None = None,
) -> dict:
    """Benchmark the local transformers backend.

    Args:
        config:         Merged config dict.
        num_iterations: Number of timed iterations.
        warmup:         Warmup passes before timing starts.
        image_dir:      Optional directory of real images.

    Returns:
        Benchmark result dict.
    """
    from drivesense.inference.serve import DriveSenseLocalInference  # noqa: PLC0415

    images = _load_or_create_images(config, image_dir, n=max(num_iterations, warmup + 1))
    logger.info("Local backend: warming up (%d iterations)…", warmup)
    server = DriveSenseLocalInference(config)

    for i in range(warmup):
        server.predict(images[i % len(images)])

    logger.info("Local backend: timing %d iterations…", num_iterations)
    latencies: list[float] = []
    for i in range(num_iterations):
        t0 = time.perf_counter()
        server.predict(images[i % len(images)])
        latencies.append((time.perf_counter() - t0) * 1000)

    from drivesense.inference.serve import (  # noqa: PLC0415
        _get_gpu_memory_stats,
        _get_gpu_name,
        _latency_stats,
    )
    result = _latency_stats(latencies)
    result["backend"] = "local"
    result["num_iterations"] = num_iterations
    result["warmup"] = warmup
    result["throughput_rps"] = round(num_iterations / (sum(latencies) / 1000), 2)
    result["gpu"] = _get_gpu_name()
    result["gpu_memory"] = _get_gpu_memory_stats()
    return result


def run_vllm_benchmark(
    config: dict,
    num_iterations: int = 50,
    concurrency: int = 4,
    warmup: int = 3,
    image_dir: Path | None = None,
) -> dict:
    """Benchmark the vLLM backend.

    Args:
        config:         Merged config dict.
        num_iterations: Total requests to issue.
        concurrency:    Parallel worker count.
        warmup:         Warmup passes before timing starts.
        image_dir:      Optional directory of real images.

    Returns:
        Benchmark result dict.
    """
    from drivesense.inference.serve import DriveSenseVLLMServer, _get_gpu_name  # noqa: PLC0415

    images = _load_or_create_images(config, image_dir, n=max(num_iterations, warmup + 1))
    server = DriveSenseVLLMServer(config)

    logger.info("vLLM backend: warming up (%d iterations)…", warmup)
    for i in range(warmup):
        server.predict(images[i % len(images)])

    logger.info(
        "vLLM backend: timing %d iterations, concurrency=%d…",
        num_iterations,
        concurrency,
    )
    result = server.benchmark(images, concurrency=concurrency, num_iterations=num_iterations)
    result["backend"] = "vllm"
    result["num_iterations"] = num_iterations
    result["warmup"] = warmup
    result["concurrency"] = concurrency
    result["gpu"] = _get_gpu_name()
    server.shutdown()
    return result


def run_vit_benchmark(
    config: dict,
    num_iterations: int = 100,
    warmup: int = 5,
    image_dir: Path | None = None,
) -> dict:
    """Benchmark the ViT encoder only (no LLM decoding).

    Args:
        config:         Merged config dict.
        num_iterations: Number of timed iterations.
        warmup:         Warmup passes.
        image_dir:      Optional directory of real images.

    Returns:
        Benchmark result dict.
    """
    from drivesense.inference.tensorrt_vit import ViTExtractor  # noqa: PLC0415

    trt_cfg = config.get("tensorrt", {})
    model_dir = Path(config.get("merge", {}).get("output_dir", "outputs/merged_model"))
    engine_path = Path(trt_cfg.get("engine_path", "outputs/tensorrt/vit.engine"))
    extractor = ViTExtractor(config)
    result = extractor.benchmark_vit(model_dir, engine_path=engine_path)
    result["backend"] = "vit_only"
    return result


def run_mock_benchmark() -> dict:
    """Return mock benchmark data for CI/testing."""
    return {
        "backend": "mock",
        "num_iterations": 50,
        "warmup": 3,
        "mean_ms": 38.4,
        "p50_ms": 37.9,
        "p95_ms": 42.1,
        "p99_ms": 45.6,
        "throughput_rps": 26.0,
        "gpu": "MOCK_A100",
        "gpu_memory": {
            "total_gb": 40.0,
            "used_gb": 8.2,
            "free_gb": 31.8,
        },
        "model_info": {
            "model_path": "outputs/quantized_model",
            "model_type": "qwen2_5_vl",
            "quantization": "awq",
        },
        "cold_start_ms": 4200.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if not any([args.local, args.vllm, args.vit_only, args.mock]):
        logger.error(
            "No stage specified. Use --local, --vllm, --vit-only, or --mock. "
            "Run with --help for usage."
        )
        sys.exit(1)

    # Load and merge configs
    cfg_dir = Path(args.config).parent
    model_cfg = load_config(cfg_dir / "model.yaml")
    data_cfg = load_config(cfg_dir / "data.yaml")
    training_cfg = load_config(cfg_dir / "training.yaml")
    inference_cfg = load_config(args.config)
    config = merge_configs(model_cfg, data_cfg, training_cfg, inference_cfg)

    image_dir = Path(args.image_dir) if args.image_dir else None

    results: dict = {}

    if args.mock:
        logger.info("=== Mock Benchmark ===")
        results = run_mock_benchmark()

    elif args.local:
        logger.info("=== Local Inference Benchmark ===")
        results = run_local_benchmark(
            config,
            num_iterations=args.num_iterations,
            warmup=args.warmup,
            image_dir=image_dir,
        )

    elif args.vllm:
        logger.info("=== vLLM Benchmark ===")
        results = run_vllm_benchmark(
            config,
            num_iterations=args.num_iterations,
            concurrency=args.concurrency,
            warmup=args.warmup,
            image_dir=image_dir,
        )

    elif args.vit_only:
        logger.info("=== ViT-Only Benchmark ===")
        results = run_vit_benchmark(
            config,
            num_iterations=args.num_iterations,
            warmup=args.warmup,
            image_dir=image_dir,
        )

    # Save results
    out_path = _resolve_output_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to: %s", out_path)

    print("\n--- Benchmark Results ---")
    print(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_output_path(output: str | None) -> Path:
    """Resolve output path, generating a timestamped default if omitted."""
    if output:
        return Path(output)
    ts = int(time.time())
    return Path(f"outputs/benchmarks/benchmark_{ts}.json")


def _load_or_create_images(
    config: dict,
    image_dir: Path | None,
    n: int,
) -> list:
    """Load real images or create synthetic PIL images for benchmarking.

    Args:
        config:    Config dict (reads ``demo.max_image_size``).
        image_dir: Optional directory of images.
        n:         Number of images needed.

    Returns:
        List of PIL Images (repeating if directory has fewer than ``n``).
    """
    try:
        from PIL import Image  # type: ignore[import]
    except ImportError:
        logger.warning("PIL not available — returning empty list")
        return []

    max_size = config.get("demo", {}).get("max_image_size", [672, 448])
    w, h = int(max_size[0]), int(max_size[1])

    if image_dir and image_dir.is_dir():
        paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        if paths:
            imgs = []
            for i in range(n):
                p = paths[i % len(paths)]
                try:
                    imgs.append(Image.open(p).convert("RGB").resize((w, h)))
                except Exception:  # noqa: BLE001
                    imgs.append(Image.new("RGB", (w, h), color=(128, 128, 128)))
            return imgs

    # Synthetic: random noise images
    import random
    images = []
    for _ in range(n):
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        images.append(Image.new("RGB", (w, h), color=(r, g, b)))
    return images


if __name__ == "__main__":
    main()
