"""Tests for AWQ quantization pipeline (Phase 3a).

All tests use mocks — no GPU operations, no model downloads.
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
def quant_config() -> dict:
    """Minimal config for AWQQuantizer."""
    return {
        "quantization": {
            "output_dir": "outputs/quantized_model",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "calibration_samples": 16,
        },
        "annotation": {
            "sft_output_dir": "outputs/data/sft_ready",
        },
    }


@pytest.fixture()
def sft_jsonl(tmp_path: Path) -> Path:
    """Create a minimal SFT training JSONL for calibration tests."""
    fpath = tmp_path / "sft_train.jsonl"
    records = []
    for i in range(30):
        records.append({
            "frame_id": f"frame_{i:04d}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are DriveSense-VLM.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyse dashcam frame {i} for hazards."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "hazards": [],
                        "scene_summary": f"Clear road, frame {i}.",
                        "ego_context": {
                            "weather": "clear",
                            "time_of_day": "day",
                            "road_type": "urban",
                        },
                    }),
                },
            ],
        })
    with fpath.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return fpath


@pytest.fixture()
def mock_quantized_dir(tmp_path: Path) -> Path:
    """Create a fake quantized model directory."""
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_vl", "quantization_config": {"bits": 4}}),
        encoding="utf-8",
    )
    (tmp_path / "quant_config.json").write_text(
        json.dumps({
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "version": "GEMM",
            "num_quantized_layers": 28,
        }),
        encoding="utf-8",
    )
    # Fake quantized weight file (much smaller than fp16)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 1024 * 1024)
    return tmp_path


# ---------------------------------------------------------------------------
# test_quantization_config
# ---------------------------------------------------------------------------


class TestQuantizationConfig:
    """Verify AWQ parameters match config values."""

    def test_bits_is_4(self, quant_config: dict) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        assert q._bits == 4

    def test_group_size_is_128(self, quant_config: dict) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        assert q._group_size == 128

    def test_zero_point_is_true(self, quant_config: dict) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        assert q._zero_point is True

    def test_output_dir_from_config(self, quant_config: dict) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        assert q._output_dir == Path("outputs/quantized_model")

    def test_calibration_samples_from_config(self, quant_config: dict) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        assert q._calibration_samples == 16

    def test_custom_bits_config(self) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        cfg = {"quantization": {"bits": 8, "group_size": 64, "zero_point": False,
                                 "calibration_samples": 64}}
        q = AWQQuantizer(cfg)
        assert q._bits == 8
        assert q._group_size == 64
        assert q._zero_point is False


# ---------------------------------------------------------------------------
# test_calibration_data_loading
# ---------------------------------------------------------------------------


class TestCalibrationDataLoading:
    """Verify correct number and format of calibration samples."""

    def test_loads_correct_number_of_samples(
        self, quant_config: dict, sft_jsonl: Path
    ) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        texts = q.prepare_calibration_data(sft_jsonl, num_samples=10)
        assert len(texts) == 10

    def test_returns_strings(self, quant_config: dict, sft_jsonl: Path) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        texts = q.prepare_calibration_data(sft_jsonl, num_samples=5)
        assert all(isinstance(t, str) for t in texts)

    def test_non_empty_strings(self, quant_config: dict, sft_jsonl: Path) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        texts = q.prepare_calibration_data(sft_jsonl, num_samples=5)
        assert all(len(t) > 0 for t in texts)

    def test_fallback_when_file_missing(
        self, quant_config: dict, tmp_path: Path
    ) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        texts = q.prepare_calibration_data(tmp_path / "nonexistent.jsonl", num_samples=8)
        assert len(texts) == 8

    def test_pads_to_requested_count_when_few_records(
        self, quant_config: dict, tmp_path: Path
    ) -> None:
        """If JSONL has fewer records than requested, pads with fallback texts."""
        fpath = tmp_path / "small.jsonl"
        with fpath.open("w") as fh:
            for i in range(3):
                fh.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": f"frame {i}"}]}
                    ]
                }) + "\n")
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        texts = q.prepare_calibration_data(fpath, num_samples=10)
        assert len(texts) == 10

    def test_extract_text_from_record(self) -> None:
        from drivesense.inference.quantize import _extract_text_from_record
        rec = {
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": [{"type": "text", "text": "Analyse frame."}]},
                {"role": "assistant", "content": "The road is clear."},
            ]
        }
        text = _extract_text_from_record(rec)
        assert "You are an assistant." in text
        assert "Analyse frame." in text
        # Assistant content should NOT appear in calibration text
        assert "The road is clear." not in text


# ---------------------------------------------------------------------------
# test_quantization_stats_format
# ---------------------------------------------------------------------------


class TestQuantizationStatsFormat:
    """Verify get_quantization_stats returns the expected schema."""

    def test_all_required_keys(
        self, quant_config: dict, mock_quantized_dir: Path
    ) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        stats = q.get_quantization_stats(mock_quantized_dir)
        required = {
            "model_size_bytes",
            "quantized_size_gb",
            "compression_ratio",
            "quantized_layers",
            "total_weight_files",
        }
        assert required.issubset(stats.keys())

    def test_model_size_positive(
        self, quant_config: dict, mock_quantized_dir: Path
    ) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        stats = q.get_quantization_stats(mock_quantized_dir)
        assert stats["model_size_bytes"] > 0
        assert stats["quantized_size_gb"] > 0.0

    def test_quantized_layers_from_quant_config(
        self, quant_config: dict, mock_quantized_dir: Path
    ) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        stats = q.get_quantization_stats(mock_quantized_dir)
        assert stats["quantized_layers"] == 28  # From quant_config.json fixture

    def test_compression_ratio_above_one(
        self, quant_config: dict, mock_quantized_dir: Path
    ) -> None:
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        stats = q.get_quantization_stats(mock_quantized_dir)
        assert stats["compression_ratio"] >= 1.0


# ---------------------------------------------------------------------------
# test_quality_benchmark_format
# ---------------------------------------------------------------------------


class TestQualityBenchmarkFormat:
    """Verify benchmark_quality returns the expected metrics schema."""

    def test_all_required_keys_returned(self, quant_config: dict, tmp_path: Path) -> None:
        from drivesense.inference.quantize import AWQQuantizer, _empty_quality_metrics
        metrics = _empty_quality_metrics(4.1, 1.08, 3.8)
        required = {
            "text_similarity",
            "bbox_mae",
            "label_agreement",
            "size_reduction",
            "original_size_gb",
            "quantized_size_gb",
        }
        assert required.issubset(metrics.keys())

    def test_size_reduction_is_ratio(self, quant_config: dict) -> None:
        from drivesense.inference.quantize import _empty_quality_metrics
        metrics = _empty_quality_metrics(4.0, 1.0, 4.0)
        assert metrics["size_reduction"] == 4.0

    def test_benchmark_skips_inference_when_no_samples(
        self, quant_config: dict, mock_quantized_dir: Path, tmp_path: Path
    ) -> None:
        """Empty test_samples returns zero metrics without running inference."""
        from drivesense.inference.quantize import AWQQuantizer
        q = AWQQuantizer(quant_config)
        result = q.benchmark_quality(
            merged_model_dir=mock_quantized_dir,
            quantized_model_dir=mock_quantized_dir,
            test_samples=[],
        )
        assert "text_similarity" in result
        assert "bbox_mae" in result


# ---------------------------------------------------------------------------
# test_vision_encoder_excluded
# ---------------------------------------------------------------------------


class TestVisionEncoderExcluded:
    """Verify ViT modules are excluded from AWQ quantization."""

    def test_discover_vision_modules_finds_visual(self) -> None:
        from drivesense.inference.quantize import _discover_vision_modules
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("visual", MagicMock()),
            ("visual.patch_embed", MagicMock()),
            ("visual.blocks.0.attn", MagicMock()),
            ("model.layers.0.self_attn.q_proj", MagicMock()),
            ("model.layers.0.self_attn.v_proj", MagicMock()),
            ("lm_head", MagicMock()),
        ]
        excluded = _discover_vision_modules(mock_model)
        assert "visual" in excluded

    def test_llm_layers_not_excluded(self) -> None:
        from drivesense.inference.quantize import _discover_vision_modules
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("visual", MagicMock()),
            ("model", MagicMock()),
            ("model.layers.0", MagicMock()),
            ("lm_head", MagicMock()),
        ]
        excluded = _discover_vision_modules(mock_model)
        assert "model" not in excluded
        assert "lm_head" not in excluded

    def test_fallback_to_visual_when_no_vit_found(self) -> None:
        from drivesense.inference.quantize import _discover_vision_modules
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("transformer.h.0", MagicMock()),
            ("lm_head", MagicMock()),
        ]
        excluded = _discover_vision_modules(mock_model)
        assert excluded == ["visual"]

    def test_modules_to_not_convert_in_quant_config(self) -> None:
        """Verify the quant_config dict contains modules_to_not_convert."""
        # Simulates what quantize() builds before calling model.quantize()
        from drivesense.inference.quantize import _discover_vision_modules
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("visual", MagicMock()),
            ("model.layers.0.q_proj", MagicMock()),
        ]
        excluded = _discover_vision_modules(mock_model)
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
            "modules_to_not_convert": excluded,
        }
        assert "visual" in quant_config["modules_to_not_convert"]
        assert quant_config["w_bit"] == 4
        assert quant_config["q_group_size"] == 128

    def test_multiple_vit_prefixes_detected(self) -> None:
        from drivesense.inference.quantize import _discover_vision_modules
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [
            ("vision_model", MagicMock()),
            ("vision_model.encoder.layer.0", MagicMock()),
            ("model.decoder.layers.0", MagicMock()),
        ]
        excluded = _discover_vision_modules(mock_model)
        assert "vision_model" in excluded
        assert "model" not in excluded
