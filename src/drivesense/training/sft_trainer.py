"""SFT training entry point for DriveSense-VLM.

Implements Phase 2a: loads config, initialises the Qwen3-VL-2B-Instruct base model,
applies LoRA adapters via PEFT, and runs supervised fine-tuning with HuggingFace Trainer.

Usage (HPC only):
    python -m drivesense.training.sft_trainer --config configs/training.yaml

Or via SLURM:
    sbatch slurm/train.sbatch

Implementation target: Phase 2a
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# GPU-only training dependencies — guarded for local macOS dev
try:
    import torch  # type: ignore[import]
    from peft import LoraConfig, get_peft_model  # type: ignore[import]
    from transformers import (  # type: ignore[import]  # type: ignore[import]
        AutoModelForCausalLM,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    torch = None  # type: ignore[assignment]
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]


def train(config_path: str | Path) -> dict:
    """Main SFT training entry point.

    Loads all configs (model, data, training), builds the model with LoRA,
    and runs the full training loop with W&B tracking.

    Args:
        config_path: Path to configs/training.yaml (other configs auto-discovered).

    Returns:
        Dict of final training metrics:
        ``{"train_loss": float, "eval_loss": float, "epochs_trained": int,
           "best_checkpoint": str}``.
    """
    raise NotImplementedError("Phase 2a: load configs, build model+LoRA, run Trainer.train()")


def setup_lora_model(model: object, lora_config: dict) -> object:
    """Apply LoRA adapters to the model using PEFT.

    Args:
        model: HuggingFace PreTrainedModel instance (Qwen3-VL-2B-Instruct).
        lora_config: LoRA config dict from configs/model.yaml ['lora'] section.

    Returns:
        PeftModel with LoRA adapters applied and non-adapter parameters frozen.
    """
    raise NotImplementedError("Phase 2a: build LoraConfig from dict and apply via get_peft_model")


def build_training_args(config: dict) -> TrainingArguments:  # type: ignore[name-defined]
    """Construct HuggingFace TrainingArguments from training config dict.

    Args:
        config: Training config dict from configs/training.yaml ['training'] section.

    Returns:
        TrainingArguments instance ready to pass to Trainer.
    """
    raise NotImplementedError("Phase 2a: map config keys to TrainingArguments fields")
