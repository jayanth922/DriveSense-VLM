"""LoRA adapter merge — fuse LoRA weights into base model for deployment.

Implements Phase 3a: loads the LoRA checkpoint produced by Phase 2a, merges
adapter weights into the base Qwen3-VL-2B parameters using PEFT's
``merge_and_unload()``, and saves the merged model in safetensors format.

Usage (HPC only):
    python -m drivesense.inference.merge_lora --config configs/inference.yaml

Implementation target: Phase 3a
"""

from __future__ import annotations

from pathlib import Path

# PEFT and Transformers — HPC only
try:
    from peft import PeftModel  # type: ignore[import]
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import]
except ImportError:
    PeftModel = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]


def merge_lora_checkpoint(
    base_model_name: str,
    lora_checkpoint_dir: Path,
    output_dir: Path,
    safe_serialization: bool = True,
) -> Path:
    """Merge LoRA adapter weights into the base model and save.

    Args:
        base_model_name: HuggingFace model ID (e.g. "Qwen/Qwen3-VL-2B-Instruct").
        lora_checkpoint_dir: Path to the LoRA checkpoint directory (from Phase 2a).
        output_dir: Directory to save the merged model.
        safe_serialization: If True, save as .safetensors (recommended).

    Returns:
        Path to the saved merged model directory.
    """
    raise NotImplementedError("Phase 3a: load base + LoRA, call merge_and_unload(), save model")


def verify_merge(merged_model_dir: Path, test_prompt: str = "Describe this scene.") -> bool:
    """Sanity-check the merged model with a short generation.

    Args:
        merged_model_dir: Path to the merged model directory.
        test_prompt: Short text prompt for generation verification.

    Returns:
        True if the model generates a non-empty response; False otherwise.
    """
    raise NotImplementedError("Phase 3a: load merged model and run a short generation check")
