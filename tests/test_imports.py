"""Tests that all drivesense modules import cleanly and expose expected attributes.

Verifies:
  - Every module in src/drivesense/ imports without ImportError
  - Package version is a non-empty string
  - utils.config provides load_config and merge_configs callables
  - Stub functions are callable (even if they raise NotImplementedError at runtime)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable for both editable and non-installed environments
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ── Module Import Tests ────────────────────────────────────────────────────────

ALL_MODULES = [
    "drivesense",
    "drivesense.data",
    "drivesense.data.nuscenes_loader",
    "drivesense.data.dada_loader",
    "drivesense.data.annotation",
    "drivesense.data.dataset",
    "drivesense.data.transforms",
    "drivesense.training",
    "drivesense.training.sft_trainer",
    "drivesense.training.callbacks",
    "drivesense.inference",
    "drivesense.inference.merge_lora",
    "drivesense.inference.quantize",
    "drivesense.inference.tensorrt_vit",
    "drivesense.inference.serve",
    "drivesense.eval",
    "drivesense.eval.grounding",
    "drivesense.eval.reasoning",
    "drivesense.eval.production",
    "drivesense.eval.robustness",
    "drivesense.utils",
    "drivesense.utils.config",
    "drivesense.utils.logging",
    "drivesense.utils.visualization",
]


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports_without_error(module_name: str) -> None:
    """Each drivesense module should import cleanly (no ImportError).

    GPU-only dependencies (torch, peft, vllm, etc.) are guarded with
    try/except ImportError in each module, so this test passes on macOS
    without any GPU packages installed.
    """
    mod = importlib.import_module(module_name)
    assert mod is not None


# ── Package Version Test ───────────────────────────────────────────────────────

class TestPackageVersion:
    """Tests for the drivesense package version attribute."""

    def test_version_is_accessible(self) -> None:
        import drivesense
        assert hasattr(drivesense, "__version__")

    def test_version_is_non_empty_string(self) -> None:
        import drivesense
        assert isinstance(drivesense.__version__, str)
        assert len(drivesense.__version__) > 0

    def test_version_format(self) -> None:
        """Version should be semver-like: MAJOR.MINOR.PATCH."""
        import drivesense
        parts = drivesense.__version__.split(".")
        assert len(parts) >= 2, "Version should have at least MAJOR.MINOR"
        assert all(part.isdigit() for part in parts), "All version parts should be numeric"


# ── Config Utility Tests ───────────────────────────────────────────────────────

class TestConfigUtils:
    """Tests for the config utilities that are fully implemented (not stubs)."""

    def test_load_config_is_callable(self) -> None:
        from drivesense.utils.config import load_config
        assert callable(load_config)

    def test_merge_configs_is_callable(self) -> None:
        from drivesense.utils.config import merge_configs
        assert callable(merge_configs)

    def test_merge_configs_shallow(self) -> None:
        """Merging two flat dicts should produce the union (later wins)."""
        from drivesense.utils.config import merge_configs
        a = {"x": 1, "y": 2}
        b = {"y": 99, "z": 3}
        result = merge_configs(a, b)
        assert result == {"x": 1, "y": 99, "z": 3}

    def test_merge_configs_deep(self) -> None:
        """Nested dicts should merge recursively."""
        from drivesense.utils.config import merge_configs
        base = {"model": {"rank": 16, "alpha": 32}}
        override = {"model": {"rank": 32}}
        result = merge_configs(base, override)
        assert result["model"]["rank"] == 32
        assert result["model"]["alpha"] == 32  # preserved from base

    def test_merge_configs_does_not_mutate_inputs(self) -> None:
        """Input dicts should not be modified by merge."""
        from drivesense.utils.config import merge_configs
        a = {"x": 1}
        b = {"x": 2}
        merge_configs(a, b)
        assert a["x"] == 1  # unchanged

    def test_load_config_raises_on_missing_file(self) -> None:
        from drivesense.utils.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_loads_model_yaml(self) -> None:
        from drivesense.utils.config import load_config
        config_path = Path(__file__).parent.parent / "configs" / "model.yaml"
        config = load_config(config_path)
        assert isinstance(config, dict)
        assert "model" in config
        assert "lora" in config


# ── Logging Utility Tests ──────────────────────────────────────────────────────

class TestLoggingUtils:
    """Tests for the logging utilities."""

    def test_get_logger_returns_logger(self) -> None:
        from drivesense.utils.logging import get_logger
        import logging
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_name_matches(self) -> None:
        from drivesense.utils.logging import get_logger
        logger = get_logger("drivesense.test")
        assert logger.name == "drivesense.test"
