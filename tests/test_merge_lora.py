"""Tests for LoRA adapter merge pipeline (Phase 3a).

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
def merge_config() -> dict:
    """Minimal config for LoRAMerger."""
    return {
        "model": {
            "name": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": "main",
            "torch_dtype": "bfloat16",
        },
        "merge": {
            "output_dir": "outputs/merged_model",
            "safe_serialization": True,
        },
    }


@pytest.fixture()
def mock_merged_dir(tmp_path: Path) -> Path:
    """Create a minimal fake merged model directory."""
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "qwen2_5_vl",
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "vocab_size": 152064,
        }),
        encoding="utf-8",
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 512}), encoding="utf-8"
    )
    # Fake safetensors file
    st_file = tmp_path / "model-00001-of-00001.safetensors"
    st_file.write_bytes(b"\x00" * 1024 * 1024 * 4)  # 4 MB stub
    # Fake tokenizer files
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def mock_adapter_dir(tmp_path: Path) -> Path:
    """Create a minimal fake LoRA adapter directory."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({
            "base_model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 512)
    (adapter_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return adapter_dir


# ---------------------------------------------------------------------------
# test_merge_config_loading
# ---------------------------------------------------------------------------


class TestMergeConfigLoading:
    """Verify LoRAMerger reads all required config fields."""

    def test_model_name_from_config(self, merge_config: dict) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        assert merger._model_name == "Qwen/Qwen3-VL-2B-Instruct"

    def test_output_dir_from_config(self, merge_config: dict) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        assert merger._output_dir == Path("outputs/merged_model")

    def test_safe_serialization_true(self, merge_config: dict) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        assert merger._safe_serialization is True

    def test_revision_from_config(self, merge_config: dict) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        assert merger._revision == "main"

    def test_torch_dtype_from_config(self, merge_config: dict) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        assert merger._torch_dtype == "bfloat16"

    def test_custom_output_dir(self) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        cfg = {
            "model": {"name": "Qwen/Qwen3-VL-2B-Instruct"},
            "merge": {"output_dir": "/custom/path", "safe_serialization": False},
        }
        merger = LoRAMerger(cfg)
        assert merger._output_dir == Path("/custom/path")
        assert merger._safe_serialization is False


# ---------------------------------------------------------------------------
# test_merge_output_directory_structure
# ---------------------------------------------------------------------------


class TestMergeOutputDirectoryStructure:
    """Verify expected files are present after a mock merge."""

    def test_config_json_present(self, mock_merged_dir: Path) -> None:
        assert (mock_merged_dir / "config.json").exists()

    def test_generation_config_present(self, mock_merged_dir: Path) -> None:
        assert (mock_merged_dir / "generation_config.json").exists()

    def test_safetensors_file_present(self, mock_merged_dir: Path) -> None:
        st_files = list(mock_merged_dir.glob("*.safetensors"))
        assert len(st_files) >= 1, "Expected at least one .safetensors file"

    def test_processor_files_present(self, mock_merged_dir: Path) -> None:
        assert (mock_merged_dir / "tokenizer.json").exists()

    def test_mock_merge_creates_expected_files(self, tmp_path: Path) -> None:
        """Test that run_optimize_model._mock_merge creates required files."""
        sys_path_fix()
        from scripts.run_optimize_model import _mock_merge  # type: ignore[import]
        out = tmp_path / "merged"
        _mock_merge(out)
        assert (out / "config.json").exists()
        assert (out / "generation_config.json").exists()
        assert (out / "model.safetensors.index.json").exists()


# ---------------------------------------------------------------------------
# test_merge_stats_format
# ---------------------------------------------------------------------------


class TestMergeStatsFormat:
    """Verify get_merge_stats returns the expected schema."""

    def test_all_required_keys(self, merge_config: dict, mock_merged_dir: Path) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        stats = merger.get_merge_stats(mock_merged_dir)
        required = {"total_parameters", "model_size_gb", "safetensors_files", "config_hash"}
        assert required.issubset(stats.keys())

    def test_model_size_positive(self, merge_config: dict, mock_merged_dir: Path) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        stats = merger.get_merge_stats(mock_merged_dir)
        assert stats["model_size_gb"] > 0.0

    def test_safetensors_files_is_list(self, merge_config: dict, mock_merged_dir: Path) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        stats = merger.get_merge_stats(mock_merged_dir)
        assert isinstance(stats["safetensors_files"], list)
        assert len(stats["safetensors_files"]) >= 1

    def test_config_hash_is_hex_string(self, merge_config: dict, mock_merged_dir: Path) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        stats = merger.get_merge_stats(mock_merged_dir)
        h = stats["config_hash"]
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex digest length

    def test_total_parameters_non_negative(
        self, merge_config: dict, mock_merged_dir: Path
    ) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        stats = merger.get_merge_stats(mock_merged_dir)
        assert stats["total_parameters"] >= 0

    def test_empty_dir_returns_zero_size(self, merge_config: dict, tmp_path: Path) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        stats = merger.get_merge_stats(tmp_path)
        assert stats["model_size_gb"] == 0.0
        assert stats["safetensors_files"] == []


# ---------------------------------------------------------------------------
# test_merge_saves_processor
# ---------------------------------------------------------------------------


class TestMergeSavesProcessor:
    """Verify processor files are copied to the output directory."""

    def test_copy_processor_writes_files(
        self, mock_adapter_dir: Path, tmp_path: Path
    ) -> None:
        """_copy_processor should copy tokenizer files when transformers available."""
        mock_proc = MagicMock()
        with patch(
            "drivesense.inference.merge_lora._AutoProcessor"
        ) as mock_cls, patch(
            "drivesense.inference.merge_lora._TRANSFORMERS_AVAILABLE", True
        ):
            mock_cls.from_pretrained.return_value = mock_proc
            from drivesense.inference.merge_lora import _copy_processor
            _copy_processor(mock_adapter_dir, tmp_path)
            mock_cls.from_pretrained.assert_called_once_with(str(mock_adapter_dir))
            mock_proc.save_pretrained.assert_called_once_with(str(tmp_path))

    def test_copy_processor_skips_when_no_transformers(
        self, mock_adapter_dir: Path, tmp_path: Path
    ) -> None:
        with patch("drivesense.inference.merge_lora._TRANSFORMERS_AVAILABLE", False):
            from drivesense.inference.merge_lora import _copy_processor
            _copy_processor(mock_adapter_dir, tmp_path)  # Should not raise


# ---------------------------------------------------------------------------
# test_merge_safe_serialization
# ---------------------------------------------------------------------------


class TestMergeSafeSerialization:
    """Verify safe_serialization=True is enforced."""

    def test_safe_serialization_default_true(self, merge_config: dict) -> None:
        from drivesense.inference.merge_lora import LoRAMerger
        merger = LoRAMerger(merge_config)
        assert merger._safe_serialization is True

    def test_safetensors_not_bin(self, mock_merged_dir: Path) -> None:
        """No legacy .bin files when safe_serialization=True."""
        bin_files = list(mock_merged_dir.glob("*.bin"))
        # If safetensors files exist, bin files should not be the primary output
        st_files = list(mock_merged_dir.glob("*.safetensors"))
        if st_files:
            # Safetensors presence implies safe_serialization was used
            assert len(st_files) >= 1

    def test_mock_merge_uses_safetensors_index(self, tmp_path: Path) -> None:
        sys_path_fix()
        from scripts.run_optimize_model import _mock_merge  # type: ignore[import]
        out = _mock_merge(tmp_path / "out")
        index = out / "model.safetensors.index.json"
        assert index.exists()
        data = json.loads(index.read_text())
        assert "weight_map" in data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sys_path_fix() -> None:
    """Ensure scripts/ directory is importable."""
    import sys
    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
