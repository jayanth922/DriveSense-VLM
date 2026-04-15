"""Tests for Level 3 Production Readiness evaluation (Phase 4b).

All tests use mocks — no GPU, no model loading, no API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prod_config() -> dict:
    """Minimal config for ProductionEvaluator."""
    return {
        "production": {
            "targets": {
                "latency_t4_p50_ms": 500,
                "latency_a100_p50_ms": 200,
                "vit_tensorrt_latency_ms": 25,
                "throughput_a100_fps": 8,
                "vram_t4_gb": 6.0,
                "quant_degradation_pct": 2.0,
            }
        }
    }


@pytest.fixture()
def passing_benchmark() -> dict:
    """Benchmark results that meet all targets."""
    return {
        "local": {
            "p50_ms": 400.0, "p95_ms": 470.0, "mean_ms": 410.0, "p99_ms": 495.0,
            "throughput_rps": 9.5,
            "gpu_memory": {"total_gb": 16.0, "used_gb": 3.1, "free_gb": 12.9},
        },
        "vllm": {
            "p50_ms": 180.0, "p95_ms": 210.0, "mean_ms": 185.0, "p99_ms": 220.0,
            "throughput_rps": 9.5,
            "gpu_memory": {"total_gb": 80.0, "used_gb": 5.2, "free_gb": 74.8},
        },
    }


@pytest.fixture()
def failing_benchmark() -> dict:
    """Benchmark results that fail the T4 latency target."""
    return {
        "local": {
            "p50_ms": 620.0, "p95_ms": 720.0, "mean_ms": 630.0, "p99_ms": 740.0,
            "throughput_rps": 5.0,
            "gpu_memory": {"total_gb": 16.0, "used_gb": 7.5, "free_gb": 8.5},
        },
        "vllm": None,
    }


@pytest.fixture()
def passing_quality() -> dict:
    """Quality comparison that meets degradation target."""
    return {
        "text_similarity": 0.97,
        "bbox_mae": 3.2,
        "label_agreement": 0.987,
        "size_reduction": 3.8,
    }


@pytest.fixture()
def passing_vit_benchmark() -> dict:
    """ViT benchmark showing TRT latency < 25ms."""
    return {
        "tensorrt": {"mean_ms": 12.4, "p50_ms": 12.1, "p95_ms": 13.5, "p99_ms": 14.2},
        "torch_compile": {"mean_ms": 28.7, "p50_ms": 28.3, "p95_ms": 30.1, "p99_ms": 31.5},
        "pytorch_eager": {"mean_ms": 45.2, "p50_ms": 44.8, "p95_ms": 48.0, "p99_ms": 51.2},
    }


# ---------------------------------------------------------------------------
# test_production_metrics_format
# ---------------------------------------------------------------------------


class TestProductionMetricsFormat:
    """Verify compute_production_metrics returns all expected top-level keys."""

    def test_top_level_keys(
        self, prod_config: dict, passing_benchmark: dict, passing_quality: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(passing_benchmark, passing_quality)
        expected = {"latency", "throughput", "memory", "quantization_degradation",
                    "targets_met", "overall_pass"}
        assert expected.issubset(m.keys())

    def test_latency_keys(
        self, prod_config: dict, passing_benchmark: dict, passing_quality: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(passing_benchmark, passing_quality)
        expected = {"t4_e2e_p50_ms", "t4_e2e_p95_ms", "a100_e2e_p50_ms",
                    "a100_e2e_p95_ms", "vit_tensorrt_p50_ms"}
        assert expected.issubset(m["latency"].keys())

    def test_targets_met_keys(
        self, prod_config: dict, passing_benchmark: dict, passing_quality: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(passing_benchmark, passing_quality)
        expected = {"latency_t4", "latency_a100", "vit_latency",
                    "throughput_a100", "vram_t4", "quant_degradation"}
        assert expected.issubset(m["targets_met"].keys())

    def test_overall_pass_is_bool(
        self, prod_config: dict, passing_benchmark: dict, passing_quality: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(passing_benchmark, passing_quality)
        assert isinstance(m["overall_pass"], bool)


# ---------------------------------------------------------------------------
# test_target_evaluation
# ---------------------------------------------------------------------------


class TestTargetEvaluation:
    """Known metrics, verify correct pass/fail determination."""

    def test_all_targets_pass(
        self, prod_config: dict, passing_benchmark: dict, passing_quality: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(passing_benchmark, passing_quality)
        assert m["targets_met"]["latency_t4"] is True
        assert m["targets_met"]["latency_a100"] is True
        assert m["overall_pass"] is True

    def test_failing_benchmark_fails_targets(
        self, prod_config: dict, failing_benchmark: dict, passing_quality: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(failing_benchmark, passing_quality)
        assert m["targets_met"]["latency_t4"] is False
        assert m["overall_pass"] is False


# ---------------------------------------------------------------------------
# test_latency_target_pass / fail
# ---------------------------------------------------------------------------


class TestLatencyTarget:
    """400ms < 500ms → pass; 600ms > 500ms → fail."""

    def test_latency_400ms_passes(self, prod_config: dict) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        bench = {"local": {"p50_ms": 400.0, "throughput_rps": 9.0,
                            "gpu_memory": {"used_gb": 3.0}}, "vllm": None}
        m = ev.compute_production_metrics(bench, {})
        assert m["targets_met"]["latency_t4"] is True

    def test_latency_600ms_fails(self, prod_config: dict) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        bench = {"local": {"p50_ms": 600.0, "throughput_rps": 9.0,
                            "gpu_memory": {"used_gb": 3.0}}, "vllm": None}
        m = ev.compute_production_metrics(bench, {})
        assert m["targets_met"]["latency_t4"] is False

    def test_latency_exactly_at_threshold_fails(self, prod_config: dict) -> None:
        """p50 == target is not strictly less-than → fail."""
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        bench = {"local": {"p50_ms": 500.0, "throughput_rps": 9.0,
                            "gpu_memory": {"used_gb": 3.0}}, "vllm": None}
        m = ev.compute_production_metrics(bench, {})
        assert m["targets_met"]["latency_t4"] is False


# ---------------------------------------------------------------------------
# test_quant_degradation
# ---------------------------------------------------------------------------


class TestQuantDegradation:
    """1.5% < 2% → pass; 2.5% > 2% → fail."""

    def test_low_degradation_passes(self, prod_config: dict, passing_benchmark: dict) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        quality = {"label_agreement": 0.985}  # 1.5% degradation
        m = ev.compute_production_metrics(passing_benchmark, quality)
        assert m["targets_met"]["quant_degradation"] is True

    def test_high_degradation_fails(self, prod_config: dict, passing_benchmark: dict) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        quality = {"label_agreement": 0.97}  # 3% degradation
        m = ev.compute_production_metrics(passing_benchmark, quality)
        assert m["targets_met"]["quant_degradation"] is False


# ---------------------------------------------------------------------------
# test_missing_benchmark_handled
# ---------------------------------------------------------------------------


class TestMissingBenchmarkHandled:
    """Missing ViT benchmark → graceful None, vit_latency target passes (optional)."""

    def test_missing_vit_benchmark_is_none(
        self, prod_config: dict, passing_benchmark: dict, passing_quality: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(passing_benchmark, passing_quality)
        assert m["latency"]["vit_tensorrt_p50_ms"] is None

    def test_missing_vit_target_passes_optional(
        self, prod_config: dict, passing_benchmark: dict, passing_quality: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(passing_benchmark, passing_quality)
        assert m["targets_met"]["vit_latency"] is True

    def test_vit_benchmark_provided(
        self,
        prod_config: dict,
        passing_benchmark: dict,
        passing_quality: dict,
        passing_vit_benchmark: dict,
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(
            passing_benchmark, passing_quality, vit_benchmark=passing_vit_benchmark
        )
        assert m["latency"]["vit_tensorrt_p50_ms"] == pytest.approx(12.1)
        assert m["targets_met"]["vit_latency"] is True

    def test_load_benchmark_results_empty_dir(
        self, prod_config: dict, tmp_path: Path
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        result = ev.load_benchmark_results(tmp_path)
        assert result["local"] is None
        assert result["vllm"] is None

    def test_load_benchmark_results_reads_files(
        self, prod_config: dict, tmp_path: Path, passing_benchmark: dict
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        (tmp_path / "local_bench.json").write_text(
            json.dumps(passing_benchmark["local"]), encoding="utf-8"
        )
        ev = ProductionEvaluator(prod_config)
        result = ev.load_benchmark_results(tmp_path)
        assert result["local"] is not None
        assert result["local"]["p50_ms"] == pytest.approx(400.0)

    def test_generate_report_creates_files(
        self, prod_config: dict, passing_benchmark: dict, passing_quality: dict, tmp_path: Path
    ) -> None:
        from drivesense.eval.production import ProductionEvaluator

        ev = ProductionEvaluator(prod_config)
        m = ev.compute_production_metrics(passing_benchmark, passing_quality)
        ev.generate_report(m, tmp_path)
        assert (tmp_path / "production_metrics.json").exists()
        assert (tmp_path / "production_report.txt").exists()
        assert (tmp_path / "targets_summary.json").exists()
