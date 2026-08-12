"""Regression tests for the --resume bug: both CLI entry points mutated a
LOCAL config dict and then called ``train(config_path)`` — a bare path string —
so ``train()``'s own fresh reload from disk silently discarded the mutation and
training always started from scratch. These tests patch --resume end-to-end
and assert the checkpoint actually reaches the trainer, so this class of bug
(a CLI flag that never reaches the function it's supposed to affect) can't
regress silently again.

No torch/transformers/peft required: sft_trainer.py guards all GPU imports, and
the mocked-internals test below patches every GPU-touching call.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.training.sft_trainer import (  # noqa: E402
    _build_callbacks,
    _resolve_checkpoint,
    _sorted_checkpoints,
)


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _resolve_checkpoint / _sorted_checkpoints — numeric vs lexicographic sort
# ---------------------------------------------------------------------------


def _touch_checkpoints(base: Path, steps: list[int]) -> None:
    for s in steps:
        (base / f"checkpoint-{s}").mkdir(parents=True)


class TestResolveCheckpoint:
    def test_latest_with_no_checkpoints_is_none(self, tmp_path):
        assert _resolve_checkpoint({"resume_from_checkpoint": "latest"}, tmp_path) is None

    def test_explicit_path_passes_through(self, tmp_path):
        assert _resolve_checkpoint({"resume_from_checkpoint": "/some/path"}, tmp_path) \
            == "/some/path"

    def test_absent_or_null_returns_none(self, tmp_path):
        assert _resolve_checkpoint({}, tmp_path) is None
        assert _resolve_checkpoint({"resume_from_checkpoint": None}, tmp_path) is None

    def test_latest_picks_highest_step_across_digit_boundary(self, tmp_path):
        # The exact scenario that bit this user: checkpoint-904 exists alongside
        # later checkpoints whose step count crosses into 4 digits. A plain
        # string sort puts "checkpoint-1356" BEFORE "checkpoint-904" ('1' < '9'),
        # so the old code would silently resume from the STALE 904 checkpoint.
        _touch_checkpoints(tmp_path, [452, 904, 1356, 1808, 2260])
        resolved = _resolve_checkpoint({"resume_from_checkpoint": "latest"}, tmp_path)
        assert resolved == str(tmp_path / "checkpoint-2260")

    def test_sorted_checkpoints_is_numeric_not_lexicographic(self, tmp_path):
        _touch_checkpoints(tmp_path, [904, 1356])
        ordered = [p.name for p in _sorted_checkpoints(tmp_path)]
        assert ordered == ["checkpoint-904", "checkpoint-1356"]  # numeric order
        # (a lexicographic sort would put checkpoint-1356 first)
        assert sorted(ordered) != ordered


# ---------------------------------------------------------------------------
# CLI wiring: --resume must reach train() as an explicit override, not a
# mutated-and-discarded local dict.
# ---------------------------------------------------------------------------


class TestRunTrainingCLIResumeWiring:
    def test_resume_flag_passes_latest_override_to_train(self):
        mod = _load_module("scripts/run_training.py", "run_training_resume_test")
        fake_train = MagicMock(return_value={})
        with patch("drivesense.training.sft_trainer.train", fake_train), \
             patch.object(sys, "argv", ["run_training.py", "--resume"]):
            mod.main()
        fake_train.assert_called_once()
        _, kwargs = fake_train.call_args
        assert kwargs.get("resume_override") == "latest"

    def test_without_resume_flag_override_is_none(self):
        mod = _load_module("scripts/run_training.py", "run_training_resume_test2")
        fake_train = MagicMock(return_value={})
        with patch("drivesense.training.sft_trainer.train", fake_train), \
             patch.object(sys, "argv", ["run_training.py"]):
            mod.main()
        _, kwargs = fake_train.call_args
        assert kwargs.get("resume_override") is None


class TestMainEntrypointResumeWiring:
    def test_resume_flag_passes_latest_override_to_train(self):
        mod = _load_module("src/drivesense/training/__main__.py", "training_main_resume_test")
        fake_train = MagicMock(return_value={})
        with patch("drivesense.training.sft_trainer.train", fake_train), \
             patch.object(sys, "argv", ["__main__.py", "--resume"]):
            mod.main()
        fake_train.assert_called_once()
        _, kwargs = fake_train.call_args
        assert kwargs.get("resume_override") == "latest"

    def test_without_resume_flag_override_is_none(self):
        mod = _load_module("src/drivesense/training/__main__.py", "training_main_resume_test2")
        fake_train = MagicMock(return_value={})
        with patch("drivesense.training.sft_trainer.train", fake_train), \
             patch.object(sys, "argv", ["__main__.py"]):
            mod.main()
        _, kwargs = fake_train.call_args
        assert kwargs.get("resume_override") is None


# ---------------------------------------------------------------------------
# Deep integration: train(config_path, resume_override="latest") end-to-end,
# with only the GPU-touching internals mocked — proves the REAL checkpoint
# path (not None) reaches trainer.train(resume_from_checkpoint=...).
# ---------------------------------------------------------------------------


def _write_min_configs(cfg_dir: Path, output_dir: Path) -> Path:
    (cfg_dir / "model.yaml").write_text(json.dumps({
        "model": {"name": "fake/model"}, "lora": {}, "vision": {}, "quantization": {},
    }))
    (cfg_dir / "data.yaml").write_text(json.dumps({"paths": {}, "annotation": {}}))
    training_path = cfg_dir / "training.yaml"
    training_path.write_text(json.dumps({
        "training": {"output_dir": str(output_dir), "resume_from_checkpoint": None},
        "wandb": {}, "early_stopping": {},
    }))
    return training_path


class TestTrainResumeIntegration:
    def test_resume_override_reaches_trainer_train_with_real_checkpoint(self, tmp_path):
        import drivesense.training.sft_trainer as st

        output_dir = tmp_path / "out"
        _touch_checkpoints(output_dir, [452, 904, 1356])  # "latest" -> 1356
        config_path = _write_min_configs(tmp_path, output_dir)

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(training_loss=0.5, metrics={"epoch": 3.0})
        fake_trainer.evaluate.return_value = {"eval_loss": 0.6}

        with patch.object(st, "_require_gpu_deps", lambda: None), \
             patch.object(st, "_init_wandb", lambda cfg: None), \
             patch.object(st, "setup_model_and_processor",
                          return_value=(MagicMock(), MagicMock(), MagicMock())), \
             patch.object(st, "_load_datasets", return_value=(MagicMock(), MagicMock())), \
             patch.object(st, "setup_training_args", return_value=MagicMock()), \
             patch.object(st, "_build_callbacks", return_value=[]), \
             patch.object(st, "setup_trainer", return_value=fake_trainer):
            st.train(config_path, resume_override="latest")

        fake_trainer.train.assert_called_once()
        _, kwargs = fake_trainer.train.call_args
        assert kwargs["resume_from_checkpoint"] == str(output_dir / "checkpoint-1356")

    def test_no_override_starts_from_scratch(self, tmp_path):
        import drivesense.training.sft_trainer as st

        output_dir = tmp_path / "out"
        _touch_checkpoints(output_dir, [452, 904])  # present, but resume not requested
        config_path = _write_min_configs(tmp_path, output_dir)

        fake_trainer = MagicMock()
        fake_trainer.train.return_value = MagicMock(training_loss=0.5, metrics={"epoch": 1.0})
        fake_trainer.evaluate.return_value = {"eval_loss": 0.6}

        with patch.object(st, "_require_gpu_deps", lambda: None), \
             patch.object(st, "_init_wandb", lambda cfg: None), \
             patch.object(st, "setup_model_and_processor",
                          return_value=(MagicMock(), MagicMock(), MagicMock())), \
             patch.object(st, "_load_datasets", return_value=(MagicMock(), MagicMock())), \
             patch.object(st, "setup_training_args", return_value=MagicMock()), \
             patch.object(st, "_build_callbacks", return_value=[]), \
             patch.object(st, "setup_trainer", return_value=fake_trainer):
            st.train(config_path, resume_override=None)

        _, kwargs = fake_trainer.train.call_args
        assert kwargs["resume_from_checkpoint"] is None


# ---------------------------------------------------------------------------
# ResumeGradientFixCallback must actually be wired into every training run —
# the resume grad-disconnection bug fix is a no-op if it's built but not used.
# ---------------------------------------------------------------------------


def test_build_callbacks_includes_resume_gradient_fix():
    from drivesense.training.callbacks import ResumeGradientFixCallback

    callbacks = _build_callbacks({"early_stopping": {}}, MagicMock(), MagicMock())
    assert any(isinstance(cb, ResumeGradientFixCallback) for cb in callbacks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
