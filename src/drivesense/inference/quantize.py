"""AWQ 4-bit quantization of the DriveSense-VLM LLM decoder.

Implements Phase 3b: applies AutoAWQ weight quantization to the merged LLM
decoder from Phase 3a. The Vision Transformer (ViT) is excluded from
quantization and remains in fp16 for accuracy preservation.

Calibration uses a small subset of the training data. Quantized model is
saved in AutoAWQ format compatible with vLLM serving.

Usage (HPC only):
    python -m drivesense.inference.quantize --config configs/inference.yaml

Implementation target: Phase 3b
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# AutoAWQ — HPC only
try:
    from awq import AutoAWQForCausalLM  # type: ignore[import]
except ImportError:
    AutoAWQForCausalLM = None  # type: ignore[assignment]


def quantize_model(
    merged_model_dir: Path,
    output_dir: Path,
    config: dict,
    calibration_data: list[str] | None = None,
) -> Path:
    """Apply AWQ 4-bit quantization to the LLM decoder of the merged model.

    Args:
        merged_model_dir: Path to the merged model from Phase 3a.
        output_dir: Directory to save the quantized model.
        config: Inference config dict from configs/inference.yaml ['quantization'].
        calibration_data: Optional list of calibration text strings.
                          If None, uses default AWQ calibration from Pile dataset.

    Returns:
        Path to the saved quantized model directory.
    """
    raise NotImplementedError("Phase 3b: apply AutoAWQ quantize_model() to LLM decoder only")


def load_calibration_data(data_dir: Path, n_samples: int = 128) -> list[str]:
    """Load calibration samples from the training dataset for AWQ.

    Args:
        data_dir: Path to the DriveSense training data directory.
        n_samples: Number of samples to use for AWQ weight search.

    Returns:
        List of formatted prompt strings for calibration.
    """
    raise NotImplementedError("Phase 3b: load n_samples prompts from training JSONL")
