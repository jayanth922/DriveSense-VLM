"""Tests for DriveSense-VLM serving layer (Phase 3c).

All tests use mocks — no GPU operations, no model downloads required.
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
def serve_config() -> dict:
    """Minimal config for serve module."""
    return {
        "vllm": {
            "model_path": "outputs/quantized_model",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "max_model_len": 2048,
        },
        "demo": {
            "model_path": "outputs/quantized_model",
            "device": "cpu",
            "max_image_size": [672, 448],
        },
    }


@pytest.fixture()
def mock_image():
    """Create a minimal synthetic PIL Image."""
    pytest.importorskip("PIL")
    from PIL import Image
    return Image.new("RGB", (672, 448), color=(100, 150, 200))


@pytest.fixture()
def sample_annotation() -> dict:
    """Realistic hazard annotation dict."""
    return {
        "hazards": [
            {
                "label": "pedestrian_in_path",
                "bbox_2d": [120, 80, 350, 280],
                "severity": "high",
                "reasoning": "Pedestrian stepping into crosswalk with fast ego vehicle.",
                "action": "yield",
            }
        ],
        "scene_summary": "Urban intersection, pedestrian crossing.",
        "ego_context": {
            "weather": "clear",
            "time_of_day": "day",
            "road_type": "urban",
        },
    }


# ---------------------------------------------------------------------------
# test_local_inference_init
# ---------------------------------------------------------------------------


class TestLocalInferenceInit:
    """Verify DriveSenseLocalInference reads config correctly."""

    def test_model_path_from_config(self, serve_config: dict) -> None:
        from drivesense.inference.serve import DriveSenseLocalInference
        inf = DriveSenseLocalInference(serve_config)
        assert inf._model_path == "outputs/quantized_model"

    def test_device_from_config(self, serve_config: dict) -> None:
        from drivesense.inference.serve import DriveSenseLocalInference
        inf = DriveSenseLocalInference(serve_config)
        assert inf._device == "cpu"

    def test_max_image_size_from_config(self, serve_config: dict) -> None:
        from drivesense.inference.serve import DriveSenseLocalInference
        inf = DriveSenseLocalInference(serve_config)
        assert inf._max_image_size == (672, 448)

    def test_model_not_loaded_at_init(self, serve_config: dict) -> None:
        """Model must be lazy-loaded — not loaded at __init__ time."""
        from drivesense.inference.serve import DriveSenseLocalInference
        inf = DriveSenseLocalInference(serve_config)
        assert inf._model is None

    def test_processor_not_loaded_at_init(self, serve_config: dict) -> None:
        from drivesense.inference.serve import DriveSenseLocalInference
        inf = DriveSenseLocalInference(serve_config)
        assert inf._processor is None


# ---------------------------------------------------------------------------
# test_local_inference_predict_format
# ---------------------------------------------------------------------------


class TestLocalInferencePredictFormat:
    """Verify predict() calls _load() and returns a dict."""

    def test_predict_returns_dict(self, serve_config: dict, mock_image: object) -> None:
        from drivesense.inference.serve import DriveSenseLocalInference

        inf = DriveSenseLocalInference(serve_config)
        mock_result = {"hazards": [], "scene_summary": "clear road"}

        with patch.object(inf, "_load"), patch.object(
            inf, "_run_inference", return_value=mock_result
        ):
            result = inf.predict(mock_image)

        assert isinstance(result, dict)
        assert result == mock_result

    def test_predict_calls_load(self, serve_config: dict, mock_image: object) -> None:
        from drivesense.inference.serve import DriveSenseLocalInference

        inf = DriveSenseLocalInference(serve_config)
        load_called = []

        def _fake_load():
            load_called.append(True)

        with patch.object(inf, "_load", side_effect=_fake_load), patch.object(
            inf, "_run_inference", return_value={}
        ):
            inf.predict(mock_image)

        assert len(load_called) == 1


# ---------------------------------------------------------------------------
# test_local_inference_predict_with_viz
# ---------------------------------------------------------------------------


class TestLocalInferencePredictWithViz:
    """Verify predict_with_visualization returns (PIL Image, dict)."""

    def test_returns_tuple(self, serve_config: dict, mock_image: object) -> None:
        from drivesense.inference.serve import DriveSenseLocalInference

        inf = DriveSenseLocalInference(serve_config)
        annotation = {"hazards": [], "scene_summary": "empty road"}

        with patch.object(inf, "predict", return_value=annotation), patch(
            "drivesense.inference.serve.draw_hazard_boxes",
            return_value=mock_image,
        ):
            img, ann = inf.predict_with_visualization(mock_image)

        assert ann == annotation
        assert img is mock_image

    def test_draw_called_with_annotation(
        self, serve_config: dict, mock_image: object
    ) -> None:
        from drivesense.inference.serve import DriveSenseLocalInference

        inf = DriveSenseLocalInference(serve_config)
        annotation = {"hazards": [{"label": "debris", "bbox_2d": [100, 100, 200, 200]}]}

        drawn_args = []

        def _fake_draw(img, ann, **kwargs):
            drawn_args.append((img, ann))
            return img

        with patch.object(inf, "predict", return_value=annotation), patch(
            "drivesense.inference.serve.draw_hazard_boxes", side_effect=_fake_draw
        ):
            inf.predict_with_visualization(mock_image)

        assert len(drawn_args) == 1
        assert drawn_args[0][1] == annotation


# ---------------------------------------------------------------------------
# test_vllm_server_config
# ---------------------------------------------------------------------------


class TestVLLMServerConfig:
    """Verify DriveSenseVLLMServer reads config correctly (mocked engine)."""

    def test_model_path_from_config(self, serve_config: dict) -> None:
        mock_llm = MagicMock()
        with patch("drivesense.inference.serve._VLLM_AVAILABLE", True), patch(
            "drivesense.inference.serve._LLM", return_value=mock_llm
        ):
            from drivesense.inference.serve import DriveSenseVLLMServer

            server = DriveSenseVLLMServer(serve_config)
            assert server._model_path == "outputs/quantized_model"

    def test_tensor_parallel_from_config(self, serve_config: dict) -> None:
        mock_llm = MagicMock()
        with patch("drivesense.inference.serve._VLLM_AVAILABLE", True), patch(
            "drivesense.inference.serve._LLM", return_value=mock_llm
        ):
            from drivesense.inference.serve import DriveSenseVLLMServer

            server = DriveSenseVLLMServer(serve_config)
            assert server._tensor_parallel_size == 1

    def test_gpu_memory_utilization(self, serve_config: dict) -> None:
        mock_llm = MagicMock()
        with patch("drivesense.inference.serve._VLLM_AVAILABLE", True), patch(
            "drivesense.inference.serve._LLM", return_value=mock_llm
        ):
            from drivesense.inference.serve import DriveSenseVLLMServer

            server = DriveSenseVLLMServer(serve_config)
            assert server._gpu_memory_utilization == pytest.approx(0.85)

    def test_max_model_len_from_config(self, serve_config: dict) -> None:
        mock_llm = MagicMock()
        with patch("drivesense.inference.serve._VLLM_AVAILABLE", True), patch(
            "drivesense.inference.serve._LLM", return_value=mock_llm
        ):
            from drivesense.inference.serve import DriveSenseVLLMServer

            server = DriveSenseVLLMServer(serve_config)
            assert server._max_model_len == 2048

    def test_vllm_unavailable_raises(self, serve_config: dict) -> None:
        with patch("drivesense.inference.serve._VLLM_AVAILABLE", False):
            from drivesense.inference.serve import DriveSenseVLLMServer

            with pytest.raises(ImportError, match="vLLM"):
                DriveSenseVLLMServer(serve_config)


# ---------------------------------------------------------------------------
# test_benchmark_result_format
# ---------------------------------------------------------------------------


class TestBenchmarkResultFormat:
    """Verify _latency_stats and benchmark produce correctly shaped dicts."""

    def test_latency_stats_keys(self) -> None:
        from drivesense.inference.serve import _latency_stats

        result = _latency_stats([10.0, 20.0, 30.0, 40.0, 50.0])
        assert {"mean_ms", "p50_ms", "p95_ms", "p99_ms"}.issubset(result.keys())

    def test_latency_stats_empty(self) -> None:
        from drivesense.inference.serve import _latency_stats

        result = _latency_stats([])
        assert result["mean_ms"] == 0.0

    def test_latency_stats_ordering(self) -> None:
        from drivesense.inference.serve import _latency_stats

        latencies = [float(i) for i in range(1, 101)]
        result = _latency_stats(latencies)
        assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]

    def test_latency_mean_correct(self) -> None:
        from drivesense.inference.serve import _latency_stats

        result = _latency_stats([10.0, 20.0, 30.0])
        assert result["mean_ms"] == pytest.approx(20.0, abs=0.1)


# ---------------------------------------------------------------------------
# test_mock_benchmark (run_benchmark.py)
# ---------------------------------------------------------------------------


class TestMockBenchmark:
    """Verify run_benchmark.py mock mode returns correctly shaped output."""

    def test_mock_returns_dict(self) -> None:
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_benchmark import run_mock_benchmark  # type: ignore[import]

        result = run_mock_benchmark()
        assert isinstance(result, dict)

    def test_mock_has_required_keys(self) -> None:
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_benchmark import run_mock_benchmark  # type: ignore[import]

        result = run_mock_benchmark()
        required = {"backend", "num_iterations", "mean_ms", "p50_ms", "p95_ms", "p99_ms"}
        assert required.issubset(result.keys())

    def test_mock_backend_label(self) -> None:
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_benchmark import run_mock_benchmark  # type: ignore[import]

        result = run_mock_benchmark()
        assert result["backend"] == "mock"

    def test_mock_latencies_positive(self) -> None:
        scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_benchmark import run_mock_benchmark  # type: ignore[import]

        result = run_mock_benchmark()
        assert result["mean_ms"] > 0
        assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]
