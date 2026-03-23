"""Tests for YAML config loading, required keys, and data types.

Verifies that all five config files:
  - Parse without error
  - Contain the expected top-level and nested keys
  - Have correct data types for critical hyperparameters
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Ensure src/ is on path for editable installs not yet registered
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(name: str) -> dict:
    """Load a config YAML file by filename."""
    path = CONFIGS_DIR / name
    with path.open() as f:
        return yaml.safe_load(f)


# ── model.yaml ─────────────────────────────────────────────────────────────────

class TestModelConfig:
    """Tests for configs/model.yaml."""

    def setup_method(self) -> None:
        self.config = load_yaml("model.yaml")

    def test_parses_without_error(self) -> None:
        assert isinstance(self.config, dict)

    def test_required_top_level_keys(self) -> None:
        for key in ("model", "lora", "quantization", "vision"):
            assert key in self.config, f"Missing top-level key: {key}"

    def test_model_name_is_string(self) -> None:
        assert isinstance(self.config["model"]["name"], str)
        assert "Qwen" in self.config["model"]["name"]

    def test_lora_rank_is_int(self) -> None:
        assert isinstance(self.config["lora"]["rank"], int)
        assert self.config["lora"]["rank"] > 0

    def test_lora_alpha_is_int(self) -> None:
        assert isinstance(self.config["lora"]["alpha"], int)

    def test_lora_target_modules_is_list(self) -> None:
        assert isinstance(self.config["lora"]["target_modules"], list)
        assert len(self.config["lora"]["target_modules"]) > 0

    def test_quant_bits_is_int(self) -> None:
        assert isinstance(self.config["quantization"]["bits"], int)
        assert self.config["quantization"]["bits"] in (4, 8)

    def test_vision_resolution_is_list_of_two_ints(self) -> None:
        res = self.config["vision"]["image_resolution"]
        assert isinstance(res, list)
        assert len(res) == 2
        assert all(isinstance(v, int) for v in res)


# ── data.yaml ──────────────────────────────────────────────────────────────────

class TestDataConfig:
    """Tests for configs/data.yaml."""

    def setup_method(self) -> None:
        self.config = load_yaml("data.yaml")

    def test_parses_without_error(self) -> None:
        assert isinstance(self.config, dict)

    def test_required_top_level_keys(self) -> None:
        for key in ("paths", "splits", "nuscenes", "dada2000", "annotation", "preprocessing"):
            assert key in self.config, f"Missing top-level key: {key}"

    def test_split_ratios_sum_to_one(self) -> None:
        s = self.config["splits"]
        total = s["train"] + s["val"] + s["test"]
        assert abs(total - 1.0) < 1e-6, f"Split ratios sum to {total}, expected 1.0"

    def test_split_ratios_are_floats(self) -> None:
        s = self.config["splits"]
        for key in ("train", "val", "test"):
            assert isinstance(s[key], float), f"splits.{key} should be float"

    def test_nuscenes_cameras_is_list(self) -> None:
        assert isinstance(self.config["nuscenes"]["cameras"], list)

    def test_rarity_thresholds_are_numeric(self) -> None:
        rarity = self.config["nuscenes"]["rarity"]
        assert isinstance(rarity["proximity_threshold_m"], (int, float))
        assert isinstance(rarity["min_rarity_score"], int)

    def test_annotation_temperature_is_float(self) -> None:
        assert isinstance(self.config["annotation"]["temperature"], float)


# ── training.yaml ──────────────────────────────────────────────────────────────

class TestTrainingConfig:
    """Tests for configs/training.yaml."""

    def setup_method(self) -> None:
        self.config = load_yaml("training.yaml")

    def test_parses_without_error(self) -> None:
        assert isinstance(self.config, dict)

    def test_required_top_level_keys(self) -> None:
        for key in ("training", "wandb", "early_stopping"):
            assert key in self.config, f"Missing top-level key: {key}"

    def test_num_epochs_is_int(self) -> None:
        assert isinstance(self.config["training"]["num_epochs"], int)
        assert self.config["training"]["num_epochs"] > 0

    def test_learning_rate_is_float(self) -> None:
        lr = self.config["training"]["learning_rate"]
        assert isinstance(lr, float)
        assert 0.0 < lr < 1.0

    def test_batch_size_is_int(self) -> None:
        assert isinstance(self.config["training"]["per_device_train_batch_size"], int)

    def test_bf16_is_bool(self) -> None:
        assert isinstance(self.config["training"]["bf16"], bool)

    def test_wandb_project_is_string(self) -> None:
        assert isinstance(self.config["wandb"]["project"], str)

    def test_early_stopping_patience_is_int(self) -> None:
        assert isinstance(self.config["early_stopping"]["patience"], int)


# ── inference.yaml ─────────────────────────────────────────────────────────────

class TestInferenceConfig:
    """Tests for configs/inference.yaml."""

    def setup_method(self) -> None:
        self.config = load_yaml("inference.yaml")

    def test_parses_without_error(self) -> None:
        assert isinstance(self.config, dict)

    def test_required_top_level_keys(self) -> None:
        for key in ("merge", "quantization", "tensorrt", "vllm", "demo"):
            assert key in self.config, f"Missing top-level key: {key}"

    def test_tensorrt_input_shape_is_list_of_four_ints(self) -> None:
        shape = self.config["tensorrt"]["input_shape"]
        assert isinstance(shape, list)
        assert len(shape) == 4
        assert all(isinstance(v, int) for v in shape)

    def test_vllm_port_is_int(self) -> None:
        assert isinstance(self.config["vllm"]["port"], int)

    def test_quant_bits_is_int(self) -> None:
        assert isinstance(self.config["quantization"]["bits"], int)


# ── eval.yaml ──────────────────────────────────────────────────────────────────

class TestEvalConfig:
    """Tests for configs/eval.yaml."""

    def setup_method(self) -> None:
        self.config = load_yaml("eval.yaml")

    def test_parses_without_error(self) -> None:
        assert isinstance(self.config, dict)

    def test_required_top_level_keys(self) -> None:
        for key in ("grounding", "reasoning", "production", "robustness"):
            assert key in self.config, f"Missing top-level key: {key}"

    def test_iou_threshold_is_float(self) -> None:
        assert isinstance(self.config["grounding"]["iou_threshold"], float)
        assert 0.0 < self.config["grounding"]["iou_threshold"] <= 1.0

    def test_reasoning_scale_is_list_of_two(self) -> None:
        scale = self.config["reasoning"]["judge"]["scale"]
        assert isinstance(scale, list)
        assert len(scale) == 2

    def test_robustness_stratify_by_is_list(self) -> None:
        assert isinstance(self.config["robustness"]["stratify_by"], list)
        assert len(self.config["robustness"]["stratify_by"]) > 0
