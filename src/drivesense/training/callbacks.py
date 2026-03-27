"""Custom HuggingFace Trainer callbacks for DriveSense-VLM training.

Provides W&B metric logging, GPU memory monitoring, sample-prediction
inspection, and early stopping.

Phase 2a implementation.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace Trainer callbacks — HPC only
# ---------------------------------------------------------------------------
try:
    from transformers import (  # type: ignore[import]
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    TrainerCallback = object  # type: ignore[assignment, misc]
    TrainerControl = None  # type: ignore[assignment]
    TrainerState = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False

try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# GPUMemoryCallback
# ---------------------------------------------------------------------------


class GPUMemoryCallback(TrainerCallback):  # type: ignore[misc]
    """Logs GPU memory usage to W&B at each training step.

    Tracks allocated, reserved, and max-allocated memory and warns when
    utilisation exceeds 90 % of available VRAM.
    """

    _WARN_THRESHOLD: float = 0.90

    def on_step_end(
        self,
        args: Any,  # noqa: ANN401
        state: Any,  # noqa: ANN401
        control: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Log GPU memory metrics after every training step.

        Args:
            args: ``TrainingArguments`` from the Trainer.
            state: ``TrainerState`` with current global step.
            control: ``TrainerControl`` (not modified).
            **kwargs: Ignored extra keyword arguments.
        """
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        metrics = _gpu_memory_metrics()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        if metrics["gpu/memory_allocated_gb"] / total > self._WARN_THRESHOLD:
            logger.warning(
                "GPU memory > 90%%: %.2f / %.2f GB",
                metrics["gpu/memory_allocated_gb"],
                total,
            )
        if _WANDB_AVAILABLE and wandb is not None:
            with contextlib.suppress(Exception):
                wandb.log(metrics, step=state.global_step)


def _gpu_memory_metrics() -> dict[str, float]:
    """Return current GPU memory stats as a W&B-friendly dict."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_alloc = torch.cuda.max_memory_allocated() / 1e9
    return {
        "gpu/memory_allocated_gb": round(allocated, 3),
        "gpu/memory_reserved_gb": round(reserved, 3),
        "gpu/memory_max_allocated_gb": round(max_alloc, 3),
    }


# ---------------------------------------------------------------------------
# SamplePredictionCallback
# ---------------------------------------------------------------------------


class SamplePredictionCallback(TrainerCallback):  # type: ignore[misc]
    """Generates predictions on fixed validation samples at each eval step.

    Logs predicted vs. ground-truth JSON to W&B as text artifacts so you
    can watch the model improve qualitatively during training.

    Args:
        processor: Qwen3-VL processor for decoding generated tokens.
        val_dataset: Dataset to draw fixed samples from.
        num_samples: Number of samples to inspect (default 3).
    """

    def __init__(
        self,
        processor: Any,  # noqa: ANN401
        val_dataset: Any,  # noqa: ANN401
        num_samples: int = 3,
    ) -> None:
        self._processor = processor
        self._num_samples = num_samples
        self._samples: list[dict] = []
        if val_dataset and len(val_dataset) > 0:
            self._samples = [
                val_dataset[i] for i in range(min(num_samples, len(val_dataset)))
            ]

    def on_evaluate(
        self,
        args: Any,  # noqa: ANN401
        state: Any,  # noqa: ANN401
        control: Any,  # noqa: ANN401
        model: Any = None,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Generate sample predictions and log them to W&B.

        Args:
            args: ``TrainingArguments``.
            state: ``TrainerState`` with current step.
            control: ``TrainerControl``.
            model: The model being trained.
            **kwargs: Ignored extras.
        """
        if model is None or not self._samples:
            return
        if not _TORCH_AVAILABLE or not _WANDB_AVAILABLE or wandb is None:
            return
        model.eval()
        for i, sample in enumerate(self._samples):
            pred_text = _generate_prediction(model, self._processor, sample)
            gt_text = _extract_ground_truth(sample, self._processor)
            with contextlib.suppress(Exception):
                wandb.log(
                    {
                        f"samples/sample_{i}_pred": pred_text,
                        f"samples/sample_{i}_gt": gt_text,
                    },
                    step=state.global_step,
                )
        model.train()


def _generate_prediction(model: Any, processor: Any, sample: dict) -> str:  # noqa: ANN401
    """Run one inference step and return decoded text."""
    try:
        device = next(model.parameters()).device
        inputs = {
            k: v.unsqueeze(0).to(device)
            for k, v in sample.items()
            if k != "labels" and hasattr(v, "unsqueeze")
        }
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]
    except Exception as exc:  # noqa: BLE001
        return f"<generation error: {exc}>"


def _extract_ground_truth(sample: dict, processor: Any) -> str:  # noqa: ANN401
    """Decode ground-truth labels (non -100 tokens) as a string."""
    try:
        labels = sample["labels"]
        valid_ids = labels[labels != -100]
        return processor.batch_decode([valid_ids], skip_special_tokens=True)[0]
    except Exception:  # noqa: BLE001
        return "<unavailable>"


# ---------------------------------------------------------------------------
# TrainingMetricsCallback
# ---------------------------------------------------------------------------


class TrainingMetricsCallback(TrainerCallback):  # type: ignore[misc]
    """Enhanced training metrics logging beyond default Trainer behavior.

    Tracks best validation loss, gradient norm statistics, and running
    loss mean; logs extras to W&B at each logging or evaluation event.
    """

    def __init__(self) -> None:
        self._best_eval_loss: float = float("inf")
        self._grad_norms: list[float] = []

    def on_log(
        self,
        args: Any,  # noqa: ANN401
        state: Any,  # noqa: ANN401
        control: Any,  # noqa: ANN401
        logs: dict | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Log gradient norm statistics to W&B.

        Args:
            args: ``TrainingArguments``.
            state: ``TrainerState``.
            control: ``TrainerControl``.
            logs: Metrics dict from Trainer.
            **kwargs: Ignored extras.
        """
        if logs is None or not _WANDB_AVAILABLE or wandb is None:
            return
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            self._grad_norms.append(float(grad_norm))
            with contextlib.suppress(Exception):
                wandb.log(
                    {"train/grad_norm_mean": sum(self._grad_norms) / len(self._grad_norms)},
                    step=state.global_step,
                )

    def on_evaluate(
        self,
        args: Any,  # noqa: ANN401
        state: Any,  # noqa: ANN401
        control: Any,  # noqa: ANN401
        metrics: dict | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Track best validation loss.

        Args:
            args: ``TrainingArguments``.
            state: ``TrainerState``.
            control: ``TrainerControl``.
            metrics: Eval metrics from Trainer.
            **kwargs: Ignored extras.
        """
        if metrics is None or not _WANDB_AVAILABLE or wandb is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and float(eval_loss) < self._best_eval_loss:
            self._best_eval_loss = float(eval_loss)
            with contextlib.suppress(Exception):
                wandb.log(
                    {"eval/best_loss": self._best_eval_loss},
                    step=state.global_step,
                )


# ---------------------------------------------------------------------------
# EarlyStoppingCallback
# ---------------------------------------------------------------------------


class EarlyStoppingCallback(TrainerCallback):  # type: ignore[misc]
    """Stops training when eval_loss stagnates for ``patience`` evaluations.

    Args:
        patience: Number of evaluations without improvement before halting.
        threshold: Minimum improvement to reset the patience counter.
    """

    def __init__(self, patience: int = 2, threshold: float = 0.001) -> None:
        self._patience = patience
        self._threshold = threshold
        self._best_loss: float = float("inf")
        self._counter: int = 0

    def on_evaluate(
        self,
        args: Any,  # noqa: ANN401
        state: Any,  # noqa: ANN401
        control: Any,  # noqa: ANN401
        metrics: dict | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Check eval_loss and signal early stop if patience is exhausted.

        Args:
            args: ``TrainingArguments``.
            state: ``TrainerState`` with current epoch.
            control: ``TrainerControl`` — sets ``should_training_stop`` to halt.
            metrics: Eval metrics dict from Trainer.
            **kwargs: Ignored extras.
        """
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        if float(eval_loss) < self._best_loss - self._threshold:
            self._best_loss = float(eval_loss)
            self._counter = 0
            logger.info("Early stopping: new best loss %.5f", self._best_loss)
        else:
            self._counter += 1
            logger.info(
                "Early stopping: no improvement (%d/%d)", self._counter, self._patience
            )
            if self._counter >= self._patience and control is not None:
                control.should_training_stop = True
                logger.info("Early stopping triggered — halting training.")


# ---------------------------------------------------------------------------
# Legacy alias — kept for backward compat with old stub import name
# ---------------------------------------------------------------------------

WandBHazardCallback = TrainingMetricsCallback
