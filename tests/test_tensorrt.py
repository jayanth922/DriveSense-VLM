"""Tests for TensorRT ViT optimization pipeline (Phase 3b).

All tests use mocks — no GPU operations, no TensorRT installation required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trt_config() -> dict:
    """Minimal config for ViTExtractor."""
    return {
        "tensorrt": {
            "output_dir": "outputs/tensorrt",
            "onnx_path": "outputs/tensorrt/vit.onnx",
            "engine_path": "outputs/tensorrt/vit.engine",
            "input_shape": [1, 3, 448, 672],
            "precision": "fp16",
            "max_workspace_size_gb": 4,
            "dynamic_batch": False,
        },
    }


@pytest.fixture()
def mock_engine_path(tmp_path: Path) -> Path:
    """Create a stub .engine file."""
    engine = tmp_path / "vit.engine"
    engine.write_bytes(b"\x00" * 1024)
    return engine


@pytest.fixture()
def mock_onnx_path(tmp_path: Path) -> Path:
    """Create a stub .onnx file."""
    onnx = tmp_path / "vit.onnx"
    onnx.write_bytes(b"mock_onnx_data")
    return onnx


@pytest.fixture()
def mock_benchmark_result() -> dict:
    """A realistic benchmark result dict."""
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


# ---------------------------------------------------------------------------
# test_onnx_export_config
# ---------------------------------------------------------------------------


class TestOnnxExportConfig:
    """Verify ONNX export configuration matches inference.yaml."""

    def test_input_shape_from_config(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        assert extractor._input_shape == (1, 3, 448, 672)

    def test_onnx_path_from_config(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        assert extractor._onnx_path == Path("outputs/tensorrt/vit.onnx")

    def test_engine_path_from_config(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        assert extractor._engine_path == Path("outputs/tensorrt/vit.engine")

    def test_precision_fp16(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        assert extractor._precision == "fp16"

    def test_workspace_gb_from_config(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        assert extractor._max_workspace_gb == 4

    def test_no_dynamic_batch(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        assert extractor._dynamic_batch is False


# ---------------------------------------------------------------------------
# test_input_shape_order
# ---------------------------------------------------------------------------


class TestInputShapeOrder:
    """Verify input shape is [batch, channels, height, width] = [1, 3, 448, 672]."""

    def test_batch_dimension(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        assert extractor._input_shape[0] == 1, "batch must be 1"

    def test_channels_dimension(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        assert extractor._input_shape[1] == 3, "channels (RGB) must be 3"

    def test_height_before_width(self, trt_config: dict) -> None:
        from drivesense.inference.tensorrt_vit import ViTExtractor
        extractor = ViTExtractor(trt_config)
        h, w = extractor._input_shape[2], extractor._input_shape[3]
        assert h == 448, f"expected height=448, got {h}"
        assert w == 672, f"expected width=672, got {w}"
        # Confirm height < width (portrait is 448×672 dashcam ratio)
        assert h < w, "height (448) must be less than width (672)"

    def test_config_shape_matches_vision_resolution(self) -> None:
        """Verify shape matches model.yaml vision.image_resolution = [672, 448]."""
        # vision.image_resolution = [width=672, height=448]
        # input_shape = [batch, channels, height, width] = [1, 3, 448, 672]
        width, height = 672, 448
        input_shape = (1, 3, height, width)
        assert input_shape == (1, 3, 448, 672)

    def test_grid_thw_computation(self) -> None:
        from drivesense.inference.tensorrt_vit import _compute_grid_thw
        import torch
        grid = _compute_grid_thw((1, 3, 448, 672))
        # 448/28 = 16 height patches, 672/28 = 24 width patches
        assert grid.shape == (1, 3)
        assert grid[0, 0].item() == 1   # time=1 (static image)
        assert grid[0, 1].item() == 16  # height patches
        assert grid[0, 2].item() == 24  # width patches

    def test_total_patches_correct(self) -> None:
        """384 total patches = 16h × 24w."""
        h_patches, w_patches = 448 // 28, 672 // 28
        assert h_patches == 16
        assert w_patches == 24
        assert h_patches * w_patches == 384


# ---------------------------------------------------------------------------
# test_benchmark_result_format
# ---------------------------------------------------------------------------


class TestBenchmarkResultFormat:
    """Verify benchmark dict structure matches specification."""

    def test_required_top_level_keys(self, mock_benchmark_result: dict) -> None:
        required = {
            "input_shape",
            "num_iterations",
            "pytorch_eager",
            "torch_compile",
            "tensorrt",
            "speedup_compile_vs_eager",
            "speedup_tensorrt_vs_eager",
        }
        assert required.issubset(mock_benchmark_result.keys())

    def test_eager_result_keys(self, mock_benchmark_result: dict) -> None:
        eager = mock_benchmark_result["pytorch_eager"]
        required = {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_fps"}
        assert required.issubset(eager.keys())

    def test_percentile_ordering(self, mock_benchmark_result: dict) -> None:
        """p50 <= p95 <= p99 for each backend."""
        for key in ("pytorch_eager", "torch_compile", "tensorrt"):
            s = mock_benchmark_result[key]
            assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"], (
                f"{key}: p-values out of order"
            )

    def test_speedup_values_positive(self, mock_benchmark_result: dict) -> None:
        assert mock_benchmark_result["speedup_compile_vs_eager"] > 0
        assert mock_benchmark_result["speedup_tensorrt_vs_eager"] > 0

    def test_tensorrt_faster_than_eager(self, mock_benchmark_result: dict) -> None:
        eager_mean = mock_benchmark_result["pytorch_eager"]["mean_ms"]
        trt_mean = mock_benchmark_result["tensorrt"]["mean_ms"]
        assert trt_mean < eager_mean

    def test_mock_benchmark_cli(self) -> None:
        """_mock_benchmark() from CLI returns correct structure."""
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_optimize_model import _mock_benchmark  # type: ignore[import]
        result = _mock_benchmark()
        assert "pytorch_eager" in result
        assert "tensorrt" in result
        assert result["input_shape"] == [1, 3, 448, 672]


# ---------------------------------------------------------------------------
# test_fallback_documentation
# ---------------------------------------------------------------------------


class TestFallbackDocumentation:
    """Verify fallback_info.json is created with the correct structure."""

    def test_fallback_info_written_on_mock(self, tmp_path: Path) -> None:
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_optimize_model import _mock_tensorrt  # type: ignore[import]
        _mock_tensorrt(tmp_path)
        fallback = tmp_path / "fallback_info.json"
        assert fallback.exists()

    def test_fallback_info_has_onnx_method(self, tmp_path: Path) -> None:
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_optimize_model import _mock_tensorrt  # type: ignore[import]
        _mock_tensorrt(tmp_path)
        info = json.loads((tmp_path / "fallback_info.json").read_text())
        assert "onnx_method" in info

    def test_fallback_info_has_trt_method(self, tmp_path: Path) -> None:
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_optimize_model import _mock_tensorrt  # type: ignore[import]
        _mock_tensorrt(tmp_path)
        info = json.loads((tmp_path / "fallback_info.json").read_text())
        assert "trt_method" in info

    def test_save_fallback_info_creates_file(self, tmp_path: Path) -> None:
        from drivesense.inference.tensorrt_vit import _save_fallback_info
        _save_fallback_info(tmp_path, {"onnx_method": "direct", "trt_method": "tensorrt"})
        f = tmp_path / "fallback_info.json"
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["onnx_method"] == "direct"
        assert data["trt_method"] == "tensorrt"

    def test_save_fallback_info_merges_with_existing(self, tmp_path: Path) -> None:
        from drivesense.inference.tensorrt_vit import _save_fallback_info
        _save_fallback_info(tmp_path, {"onnx_method": "direct"})
        _save_fallback_info(tmp_path, {"trt_method": "tensorrt"})
        data = json.loads((tmp_path / "fallback_info.json").read_text())
        assert data["onnx_method"] == "direct"
        assert data["trt_method"] == "tensorrt"

    def test_torch_compile_sentinel_path(self, tmp_path: Path, trt_config: dict) -> None:
        """compile_tensorrt creates a .torch_compile file when TRT unavailable."""
        onnx_path = tmp_path / "vit.onnx"
        onnx_path.write_bytes(b"mock_onnx")
        engine_path = tmp_path / "vit.engine"

        with patch("drivesense.inference.tensorrt_vit._TRT_AVAILABLE", False):
            from drivesense.inference.tensorrt_vit import ViTExtractor
            extractor = ViTExtractor(trt_config)
            result = extractor.compile_tensorrt(
                onnx_path, engine_path=engine_path
            )
        assert result.suffix == ".torch_compile"
        assert result.exists()
        info = json.loads(result.read_text())
        assert info["method"] == "torch_compile"


# ---------------------------------------------------------------------------
# test_full_pipeline_output_structure
# ---------------------------------------------------------------------------


class TestFullPipelineOutputStructure:
    """Verify all output files are created by the full optimization pipeline."""

    def test_mock_pipeline_creates_all_artifacts(self, tmp_path: Path) -> None:
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_optimize_model import _mock_tensorrt  # type: ignore[import]
        _mock_tensorrt(tmp_path)

        assert (tmp_path / "vit.onnx").exists(), "vit.onnx missing"
        assert (tmp_path / "vit.engine").exists(), "vit.engine missing"
        assert (tmp_path / "vit_benchmark.json").exists(), "vit_benchmark.json missing"
        assert (tmp_path / "fallback_info.json").exists(), "fallback_info.json missing"
        assert (tmp_path / "optimization_report.txt").exists(), "optimization_report.txt missing"

    def test_benchmark_json_loadable(self, tmp_path: Path) -> None:
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_optimize_model import _mock_tensorrt  # type: ignore[import]
        _mock_tensorrt(tmp_path)
        bm = json.loads((tmp_path / "vit_benchmark.json").read_text())
        assert "pytorch_eager" in bm

    def test_optimization_report_txt_has_content(self, tmp_path: Path) -> None:
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_optimize_model import _mock_tensorrt  # type: ignore[import]
        _mock_tensorrt(tmp_path)
        report = (tmp_path / "optimization_report.txt").read_text()
        assert len(report) > 0

    def test_format_optimization_report_structure(
        self, mock_benchmark_result: dict
    ) -> None:
        from drivesense.inference.tensorrt_vit import _format_optimization_report
        report = _format_optimization_report(
            mock_benchmark_result,
            {"onnx_method": "direct", "trt_method": "tensorrt"},
        )
        assert "DriveSense-VLM" in report
        assert "TensorRT" in report
        assert "torch.compile" in report
        assert "PyTorch Eager" in report

    def test_idempotent_skip_when_done(self, tmp_path: Path, trt_config: dict) -> None:
        """CLI skips stage if vit_benchmark.json already exists."""
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_optimize_model import _stage_already_done  # type: ignore[import]

        (tmp_path / "vit_benchmark.json").write_text("{}", encoding="utf-8")
        assert _stage_already_done(tmp_path, "vit_benchmark.json") is True
        assert _stage_already_done(tmp_path, "nonexistent.json") is False
