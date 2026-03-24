"""Unified SFT dataset combining nuScenes rare frames and DADA-2000 accident frames.

Implements Phase 1c: wraps annotated frame data as a PyTorch Dataset compatible
with HuggingFace Trainer. Each item returns a dict with image, formatted prompt
text, target response text, and metadata for evaluation.

The conversation format follows Qwen3-VL-2B chat template:
    System: "You are a hazard detection assistant for autonomous vehicles..."
    User:   [Image] "Describe any hazards visible in this dashcam frame..."
    Assistant: (structured JSON response)

Implementation target: Phase 1c
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# torch.utils.data.Dataset — HPC only; guarded for local macOS dev
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    torch = None  # type: ignore[assignment]
    TorchDataset = object  # type: ignore[assignment,misc]


class DriveSenseDataset(TorchDataset):  # type: ignore[misc]
    """SFT dataset for DriveSense-VLM training.

    Loads annotated frame JSONL files produced by the annotation pipeline
    and returns processed training examples ready for the Qwen3-VL processor.

    Args:
        data_dir: Directory containing metadata.json and images/.
        split: One of "train", "val", or "test".
        config: Data config dict loaded from configs/data.yaml.
        processor: Qwen3-VL processor instance (optional; can be set post-init).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        config: dict,
        processor: object | None = None,
    ) -> None:
        raise NotImplementedError("Phase 1c: load and split annotated JSONL data")

    def __len__(self) -> int:
        """Return the number of examples in this split.

        Returns:
            Number of training examples.
        """
        raise NotImplementedError("Phase 1c: return len of loaded split")

    def __getitem__(self, idx: int) -> dict:
        """Return a single training example.

        Args:
            idx: Integer index into the dataset split.

        Returns:
            Dict with keys:
            - ``"image"``: PIL.Image.Image — dashcam frame
            - ``"input_text"``: str — formatted system + user prompt
            - ``"target_text"``: str — ground-truth JSON annotation string
            - ``"metadata"``: dict — source, split, rarity score, etc.
        """
        raise NotImplementedError("Phase 1c: load image and format Qwen3-VL prompt")

    def get_collate_fn(self) -> object:
        """Return a collate function for use with DataLoader.

        Returns:
            Callable that batches a list of __getitem__ dicts using the
            Qwen3-VL processor's ``process_batch`` method.
        """
        raise NotImplementedError("Phase 1c: implement processor-aware collate function")
