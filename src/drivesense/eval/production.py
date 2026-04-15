"""Level 3: Production Readiness Evaluation.

Measures inference performance, quantization quality degradation, and resource
utilization to validate deployment readiness.

Metrics: E2E latency (T4/A100), ViT TensorRT latency, throughput, VRAM usage,
quantization degradation vs fp16 baseline.

Reads benchmark JSON files written by Phase 3c (run_benchmark.py and
run_optimize_model.py) — does NOT run live benchmarks.

Implemented in Phase 4b.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# ProductionEvaluator
# ---------------------------------------------------------------------------


class ProductionEvaluator:
    """Level 3 production readiness evaluation.

    Aggregates benchmark results from Phase 3c and computes degradation
    metrics by comparing quantized vs full-precision performance.

    All targets are read from ``config["production"]["targets"]``.

    Args:
        config: Merged config dict; reads ``config["production"]`` section.
    """

    def __init__(self, config: dict) -> None:
        prod_cfg = config.get("production", {})
        t = prod_cfg.get("targets", {})
        self._target_latency_t4: float = float(t.get("latency_t4_p50_ms", 500))
        self._target_latency_a100: float = float(t.get("latency_a100_p50_ms", 200))
        self._target_vit_latency: float = float(t.get("vit_tensorrt_latency_ms", 25))
        self._target_throughput: float = float(t.get("throughput_a100_fps", 8))
        self._target_vram_t4: float = float(t.get("vram_t4_gb", 6.0))
        self._target_quant_deg: float = float(t.get("quant_degradation_pct", 2.0))
        self._cfg = config

    # ── public API ──────────────────────────────────────────────────────────

    def load_benchmark_results(self, benchmark_dir: Path) -> dict:
        """Load and merge benchmark JSON files from Phase 3c output.

        Reads ``local_bench.json``, ``vllm_bench.json`` if present.

        Args:
            benchmark_dir: Directory containing Phase 3c benchmark JSONs.

        Returns:
            Dict with keys ``"local"`` and ``"vllm"`` (either may be None).
        """
        out: dict[str, dict | None] = {"local": None, "vllm": None}
        for name, key in (("local_bench.json", "local"), ("vllm_bench.json", "vllm")):
            p = benchmark_dir / name
            if p.exists():
                with contextlib.suppress(Exception):
                    out[key] = json.loads(p.read_text(encoding="utf-8"))
        return out

    def load_quality_comparison(self, comparison_path: Path) -> dict:
        """Load AWQ quality benchmark results from Phase 3a quantize output.

        Args:
            comparison_path: Path to quality comparison JSON.

        Returns:
            Quality comparison dict. Empty dict if file missing.
        """
        if not comparison_path.exists():
            logger.warning("Quality comparison file not found: %s", comparison_path)
            return {}
        with contextlib.suppress(Exception):
            return json.loads(comparison_path.read_text(encoding="utf-8"))
        return {}

    def compute_production_metrics(
        self,
        benchmark_results: dict,
        quality_comparison: dict,
        vit_benchmark: dict | None = None,
    ) -> dict:
        """Compute all Level 3 production metrics.

        Args:
            benchmark_results:  Output of ``load_benchmark_results()``.
            quality_comparison: Output of ``load_quality_comparison()``.
            vit_benchmark:      Optional ViT benchmark dict from TRT pipeline.

        Returns:
            Full Level 3 metrics dict with ``targets_met`` and ``overall_pass``.
        """
        latency = _extract_latency(benchmark_results, vit_benchmark)
        throughput = _extract_throughput(benchmark_results)
        memory = _extract_memory(benchmark_results)
        degradation = _extract_degradation(quality_comparison)

        targets_met = {
            "latency_t4": (
                latency["t4_e2e_p50_ms"] < self._target_latency_t4
                if latency["t4_e2e_p50_ms"] is not None else False
            ),
            "latency_a100": (
                latency["a100_e2e_p50_ms"] < self._target_latency_a100
                if latency["a100_e2e_p50_ms"] is not None else False
            ),
            "vit_latency": (
                latency["vit_tensorrt_p50_ms"] < self._target_vit_latency
                if latency["vit_tensorrt_p50_ms"] is not None else True  # optional
            ),
            "throughput_a100": throughput["a100_single_fps"] > self._target_throughput,
            "vram_t4": (
                memory["model_vram_gb"] < self._target_vram_t4
                if memory["model_vram_gb"] is not None else False
            ),
            "quant_degradation": (
                degradation["quant_degradation_pct"] < self._target_quant_deg
                if degradation["quant_degradation_pct"] is not None else True
            ),
        }
        overall_pass = all(targets_met.values())

        return {
            "latency": latency,
            "throughput": throughput,
            "memory": memory,
            "quantization_degradation": degradation,
            "targets_met": targets_met,
            "overall_pass": overall_pass,
        }

    def evaluate(self, results_dir: Path) -> dict:
        """Run complete Level 3 evaluation from stored Phase 3c results.

        Looks for benchmark JSON files in ``results_dir/benchmarks/`` and
        quality JSON in ``results_dir/quantized_model/``.

        Args:
            results_dir: Project outputs root (e.g. ``outputs/``).

        Returns:
            Full Level 3 metrics dict.
        """
        benchmark_dir = results_dir / "benchmarks"
        quality_path = results_dir / "quantized_model" / "quality_comparison.json"
        vit_bench_path = results_dir / "tensorrt" / "vit_benchmark.json"

        bench = self.load_benchmark_results(benchmark_dir)
        quality = self.load_quality_comparison(quality_path)
        vit_bm: dict | None = None
        if vit_bench_path.exists():
            with contextlib.suppress(Exception):
                vit_bm = json.loads(vit_bench_path.read_text(encoding="utf-8"))

        return self.compute_production_metrics(bench, quality, vit_benchmark=vit_bm)

    def generate_report(self, metrics: dict, output_dir: Path) -> Path:
        """Write Level 3 report files to ``output_dir``.

        Creates:
            - ``production_metrics.json``
            - ``production_report.txt``
            - ``targets_summary.json``

        Args:
            metrics:    Output of ``compute_production_metrics()``.
            output_dir: Destination directory.

        Returns:
            Path to ``production_report.txt``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "production_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (output_dir / "targets_summary.json").write_text(
            json.dumps(metrics.get("targets_met", {}), indent=2), encoding="utf-8"
        )

        report_path = output_dir / "production_report.txt"
        report_path.write_text(
            _format_production_report(metrics, self._get_targets()), encoding="utf-8"
        )
        logger.info("Level 3 report written to %s", output_dir)
        return report_path

    def log_to_wandb(self, metrics: dict) -> None:
        """Log Level 3 metrics to Weights & Biases.

        Args:
            metrics: Output of ``compute_production_metrics()``.
        """
        if not _WANDB_AVAILABLE or wandb is None:
            logger.debug("wandb not available — skipping log_to_wandb")
            return
        with contextlib.suppress(Exception):
            flat = _flatten_metrics(metrics, prefix="level3")
            wandb.log(flat)  # type: ignore[union-attr]
            logger.info("Level 3 metrics logged to W&B")

    # ── private ─────────────────────────────────────────────────────────────

    def _get_targets(self) -> dict:
        return {
            "latency_t4_p50_ms": self._target_latency_t4,
            "latency_a100_p50_ms": self._target_latency_a100,
            "vit_tensorrt_p50_ms": self._target_vit_latency,
            "throughput_a100_fps": self._target_throughput,
            "vram_t4_gb": self._target_vram_t4,
            "quant_degradation_pct": self._target_quant_deg,
        }


# ---------------------------------------------------------------------------
# Module-level functions (forward-compatible with original stub interface)
# ---------------------------------------------------------------------------


def benchmark_latency(
    server: object,
    test_images: list,
    warmup_iters: int = 10,
    measure_iters: int = 100,
) -> dict:
    """Measure end-to-end inference latency statistics.

    Args:
        server:        DriveSenseLocalInference or VLLMServer instance.
        test_images:   List of PIL Images to benchmark with.
        warmup_iters:  Warmup iterations (discarded).
        measure_iters: Timed iterations.

    Returns:
        Latency statistics dict: ``{p50_ms, p95_ms, p99_ms, mean_ms}``.
    """
    import itertools
    import time

    images = list(itertools.cycle(test_images))

    for i in range(warmup_iters):
        server.predict(images[i % len(images)])  # type: ignore[union-attr]

    latencies: list[float] = []
    for i in range(measure_iters):
        t0 = time.perf_counter()
        server.predict(images[i % len(images)])  # type: ignore[union-attr]
        latencies.append((time.perf_counter() - t0) * 1000)

    from drivesense.inference.serve import _latency_stats  # noqa: PLC0415
    return _latency_stats(latencies)


def measure_vram_usage(server: object, test_image: object) -> float:
    """Measure peak GPU VRAM during a single inference call.

    Args:
        server:     DriveSenseLocalInference instance.
        test_image: PIL Image input.

    Returns:
        Peak VRAM usage in GB, or 0.0 if torch/CUDA unavailable.
    """
    try:
        import torch  # type: ignore[import]
        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.reset_peak_memory_stats()
        server.predict(test_image)  # type: ignore[union-attr]
        peak_bytes = torch.cuda.max_memory_allocated()
        return round(peak_bytes / 1e9, 3)
    except ImportError:
        return 0.0


def compute_quant_degradation(
    fp16_metrics: dict,
    quant_metrics: dict,
    primary_metric: str = "iou_at_threshold",
) -> float:
    """Compute percentage quality drop from fp16 to AWQ quantized model.

    Args:
        fp16_metrics:   Grounding metrics for fp16 merged model.
        quant_metrics:  Grounding metrics for AWQ quantized model.
        primary_metric: Key to compare (default: ``"iou_at_threshold"``).

    Returns:
        Percentage degradation (e.g. 1.5 means 1.5% drop). 0.0 if unavailable.
    """
    fp16_val = float(fp16_metrics.get(primary_metric, 0.0))
    quant_val = float(quant_metrics.get(primary_metric, 0.0))
    if fp16_val <= 0.0:
        return 0.0
    return round((fp16_val - quant_val) / fp16_val * 100, 3)


def run_production_benchmark(config: dict, server: object, test_data: list[dict]) -> dict:
    """Run the full Level 3 production readiness benchmark suite.

    Args:
        config:    Config dict (reads ``config["production"]``).
        server:    DriveSenseLocalInference or VLLMServer instance.
        test_data: List of test dicts; each needs a ``"image"`` field.

    Returns:
        Level 3 metrics dict with pass/fail status.
    """
    images = [d["image"] for d in test_data if "image" in d]
    if not images:
        return {"error": "no images in test_data"}

    prod_cfg = config.get("production", {}).get("benchmark", {})
    warmup = int(prod_cfg.get("warmup_iterations", 10))
    measure = int(prod_cfg.get("measure_iterations", 100))

    latency_stats = benchmark_latency(server, images, warmup, measure)
    vram_gb = measure_vram_usage(server, images[0])

    evaluator = ProductionEvaluator(config)
    bench = {"local": {**latency_stats, "gpu_memory": {"used_gb": vram_gb}}}
    return evaluator.compute_production_metrics(bench, {})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_latency(benchmark_results: dict, vit_benchmark: dict | None) -> dict:
    """Extract latency values from benchmark results."""
    local = benchmark_results.get("local") or {}
    vllm = benchmark_results.get("vllm") or {}

    # T4: local transformers backend (usually slower / less optimised)
    # A100: vLLM backend (production throughput)
    t4_p50 = _get_p50(local)
    t4_p95 = local.get("p95_ms")
    a100_p50 = _get_p50(vllm) or t4_p50
    a100_p95 = vllm.get("p95_ms") or t4_p95

    vit_trt_p50: float | None = None
    vit_compile_p50: float | None = None
    if vit_benchmark:
        trt = vit_benchmark.get("tensorrt", {})
        compile_ = vit_benchmark.get("torch_compile", {})
        vit_trt_p50 = trt.get("p50_ms")
        vit_compile_p50 = compile_.get("p50_ms")

    return {
        "t4_e2e_p50_ms": t4_p50,
        "t4_e2e_p95_ms": t4_p95,
        "a100_e2e_p50_ms": a100_p50,
        "a100_e2e_p95_ms": a100_p95,
        "vit_tensorrt_p50_ms": vit_trt_p50,
        "vit_torch_compile_p50_ms": vit_compile_p50,
    }


def _extract_throughput(benchmark_results: dict) -> dict:
    """Extract throughput values from benchmark results."""
    local = benchmark_results.get("local") or {}
    vllm = benchmark_results.get("vllm") or {}

    # rps → fps (single-frame inference; tokens/sec estimated at avg 200 tokens)
    local_rps = float(local.get("throughput_rps", 0.0))
    vllm_rps = float(vllm.get("throughput_rps", local_rps))
    batched_fps = float(vllm.get("throughput_rps", local_rps))
    tokens_per_sec = round(vllm_rps * 200, 1)

    return {
        "a100_single_fps": round(local_rps, 2),
        "a100_batched_fps": round(batched_fps, 2),
        "a100_tokens_per_sec": tokens_per_sec,
    }


def _extract_memory(benchmark_results: dict) -> dict:
    """Extract memory values from benchmark results."""
    local = benchmark_results.get("local") or {}
    vllm = benchmark_results.get("vllm") or {}
    mem = local.get("gpu_memory") or vllm.get("gpu_memory") or {}

    used_gb: float | None = mem.get("used_gb")
    total_gb = float(mem.get("total_gb", 16.0))
    peak_gb = used_gb
    t4_headroom = round(16.0 - peak_gb, 2) if peak_gb is not None else None

    return {
        "model_vram_gb": used_gb,
        "peak_inference_vram_gb": peak_gb,
        "t4_headroom_gb": t4_headroom,
    }


def _extract_degradation(quality_comparison: dict) -> dict:
    """Extract quantization degradation metrics."""
    bbox_mae = quality_comparison.get("bbox_mae")
    label_agreement = quality_comparison.get("label_agreement")
    text_similarity = quality_comparison.get("text_similarity")
    size_reduction = quality_comparison.get("size_reduction")

    # Compute degradation pct from label_agreement drop
    # quality_comparison may also have explicit degradation_pct
    deg_pct: float | None = quality_comparison.get("degradation_pct")
    if deg_pct is None and label_agreement is not None:
        # Estimate: 100% - label_agreement*100 is error rate; use as proxy
        deg_pct = round(max(0.0, (1.0 - float(label_agreement)) * 100), 2)

    return {
        "bbox_mae": bbox_mae,
        "label_agreement_pct": round(float(label_agreement) * 100, 2) if label_agreement else None,
        "text_similarity": text_similarity,
        "size_reduction_ratio": size_reduction,
        "quant_degradation_pct": deg_pct,
    }


def _get_p50(d: dict) -> float | None:
    """Extract p50_ms from a benchmark result dict."""
    val = d.get("p50_ms") or d.get("mean_ms")
    return float(val) if val is not None else None


def _format_production_report(metrics: dict, targets: dict) -> str:
    """Format a human-readable Level 3 report string."""
    lat = metrics.get("latency", {})
    thr = metrics.get("throughput", {})
    mem = metrics.get("memory", {})
    deg = metrics.get("quantization_degradation", {})
    tgt = metrics.get("targets_met", {})
    overall = metrics.get("overall_pass", False)

    def _fmt(val: object, unit: str = "") -> str:
        if val is None:
            return "N/A"
        return f"{val}{unit}"

    def _pass(ok: bool | None) -> str:
        return "PASS" if ok else "FAIL"

    lines = [
        "DriveSense-VLM — Level 3: Production Readiness",
        "=" * 52,
        "",
        "Latency",
        f"  T4  E2E p50:         {_fmt(lat.get('t4_e2e_p50_ms'), ' ms'):<12} "
        f"target <{targets['latency_t4_p50_ms']} ms   [{_pass(tgt.get('latency_t4'))}]",
        f"  T4  E2E p95:         {_fmt(lat.get('t4_e2e_p95_ms'), ' ms'):<12}",
        f"  A100 E2E p50:        {_fmt(lat.get('a100_e2e_p50_ms'), ' ms'):<12} "
        f"target <{targets['latency_a100_p50_ms']} ms  [{_pass(tgt.get('latency_a100'))}]",
        f"  A100 E2E p95:        {_fmt(lat.get('a100_e2e_p95_ms'), ' ms'):<12}",
        f"  ViT TensorRT p50:    {_fmt(lat.get('vit_tensorrt_p50_ms'), ' ms'):<12} "
        f"target <{targets['vit_tensorrt_p50_ms']} ms   [{_pass(tgt.get('vit_latency'))}]",
        "",
        "Throughput",
        f"  Single-request fps:  {_fmt(thr.get('a100_single_fps'), ' fps'):<12} "
        f"target >{targets['throughput_a100_fps']} fps   [{_pass(tgt.get('throughput_a100'))}]",
        f"  Concurrent fps:      {_fmt(thr.get('a100_batched_fps'), ' fps'):<12}",
        f"  Tokens/sec:          {_fmt(thr.get('a100_tokens_per_sec'), ' tok/s'):<12}",
        "",
        "Memory",
        f"  Model VRAM:          {_fmt(mem.get('model_vram_gb'), ' GB'):<12} "
        f"target <{targets['vram_t4_gb']} GB    [{_pass(tgt.get('vram_t4'))}]",
        f"  Peak VRAM:           {_fmt(mem.get('peak_inference_vram_gb'), ' GB'):<12}",
        f"  T4 headroom:         {_fmt(mem.get('t4_headroom_gb'), ' GB'):<12}",
        "",
        "Quantization Quality (AWQ vs fp16)",
        f"  BBox MAE:            {_fmt(deg.get('bbox_mae')):<12}",
        f"  Label agreement:     {_fmt(deg.get('label_agreement_pct'), '%'):<12}",
        f"  Text similarity:     {_fmt(deg.get('text_similarity')):<12}",
        f"  Size reduction:      {_fmt(deg.get('size_reduction_ratio'), 'x'):<12}",
        f"  Degradation:         {_fmt(deg.get('quant_degradation_pct'), '%'):<12} "
        f"target <{targets['quant_degradation_pct']}%    [{_pass(tgt.get('quant_degradation'))}]",
        "",
        "=" * 52,
        f"Overall: {'ALL TARGETS MET' if overall else 'ONE OR MORE TARGETS MISSED'}",
    ]
    return "\n".join(lines) + "\n"


def _flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    """Flatten nested metrics dict for W&B logging."""
    out: dict = {}
    for k, v in metrics.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, key))
        elif v is not None:
            out[key] = v
    return out
