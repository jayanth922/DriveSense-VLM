"""Tests for ResumeGradientFixCallback — the fix for the resume grad-disconnection bug.

No torch/transformers required: callbacks.py guards ``TrainerCallback = object``
when transformers is absent, so the class is importable and testable with a
lightweight fake model (no real tensors needed — only ``requires_grad_``,
``train()``, and ``enable_input_require_grads`` need to be callable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.training.callbacks import ResumeGradientFixCallback  # noqa: E402


class _FakeParam:
    """Stand-in for a torch.nn.Parameter — only requires_grad_ is needed."""

    def __init__(self, requires_grad: bool = False) -> None:
        self.requires_grad = requires_grad

    def requires_grad_(self, value: bool = True) -> "_FakeParam":
        self.requires_grad = value
        return self


class _FakeModelBase:
    """Minimal fake model: named_parameters() + train() tracking."""

    def __init__(self, params: dict[str, _FakeParam]) -> None:
        self._params = params
        self.train_called = False

    def named_parameters(self):
        return list(self._params.items())

    def train(self) -> None:
        self.train_called = True


class _FakeModelWithHook(_FakeModelBase):
    def __init__(self, params: dict[str, _FakeParam]) -> None:
        super().__init__(params)
        self.enable_input_require_grads_called = False

    def enable_input_require_grads(self) -> None:
        self.enable_input_require_grads_called = True


def _make_params() -> dict[str, _FakeParam]:
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": _FakeParam(False),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": _FakeParam(False),
        "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": _FakeParam(False),
        "base_model.model.embed_tokens.weight": _FakeParam(False),
    }


class TestResumeGradientFixCallback:
    def test_reenables_only_lora_params(self) -> None:
        model = _FakeModelWithHook(_make_params())
        ResumeGradientFixCallback().on_train_begin(None, None, None, model=model)
        for name, p in model._params.items():
            if "lora_" in name:
                assert p.requires_grad is True
            else:
                assert p.requires_grad is False  # untouched — base stays frozen

    def test_sets_model_to_train_mode(self) -> None:
        model = _FakeModelWithHook(_make_params())
        assert not model.train_called
        ResumeGradientFixCallback().on_train_begin(None, None, None, model=model)
        assert model.train_called

    def test_calls_enable_input_require_grads_when_present(self) -> None:
        model = _FakeModelWithHook(_make_params())
        ResumeGradientFixCallback().on_train_begin(None, None, None, model=model)
        assert model.enable_input_require_grads_called

    def test_skips_enable_input_require_grads_when_absent(self) -> None:
        # No enable_input_require_grads attribute at all — must not raise
        # (hasattr-gated), covering a non-PEFT / plain model edge case.
        model = _FakeModelBase(_make_params())
        ResumeGradientFixCallback().on_train_begin(None, None, None, model=model)
        assert model.train_called  # still ran the rest

    def test_noop_when_model_is_none(self) -> None:
        # Trainer's on_train_begin signature always passes model=..., but guard
        # against a None model rather than crashing the training loop.
        ResumeGradientFixCallback().on_train_begin(None, None, None, model=None)

    def test_idempotent_on_a_fresh_run_already_trainable(self) -> None:
        # Simulates the fresh (non-resumed) path: LoRA params already trainable
        # from get_peft_model(). Re-running the callback must be a harmless no-op.
        params = _make_params()
        for name, p in params.items():
            if "lora_" in name:
                p.requires_grad_(True)
        model = _FakeModelWithHook(params)
        ResumeGradientFixCallback().on_train_begin(None, None, None, model=model)
        assert all(p.requires_grad for name, p in params.items() if "lora_" in name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
