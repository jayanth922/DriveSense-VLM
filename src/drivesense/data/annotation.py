"""LLM-based annotation and counterfactual augmentation pipeline.

Implements Phase 1c: call an LLM (Anthropic Claude or OpenAI) to generate
structured JSON annotations for rare-hazard frames. For nuScenes frames,
also generates counterfactual scene descriptions for augmentation.

Output annotation schema (matches CLAUDE.md output spec):
    {
        "bbox_2d": [x1, y1, x2, y2],
        "hazard_class": str,
        "severity": int (1–5),
        "reasoning": str,
        "action": str
    }

Implementation target: Phase 1c
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# Anthropic client — optional at import time; required at runtime in Phase 1c
try:
    import anthropic  # type: ignore[import]
except ImportError:
    anthropic = None  # type: ignore[assignment]


class AnnotationPipeline:
    """Generates structured JSON annotations using an LLM judge.

    Supports Anthropic Claude and OpenAI GPT-4o as annotation backends.
    Counterfactual augmentation creates alternative scene descriptions for
    frames that already have a primary annotation.

    Args:
        config: Data config dict loaded from configs/data.yaml.
    """

    def __init__(self, config: dict) -> None:
        raise NotImplementedError("Phase 1c: initialise LLM client from config llm_provider")

    def annotate_frame(self, image: Image.Image, metadata: dict) -> dict:
        """Call the LLM to generate a structured hazard annotation for one frame.

        Args:
            image: PIL Image of the dashcam frame.
            metadata: Dict of available context (source, location, weather, etc.).

        Returns:
            Structured annotation dict matching the output schema above.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        raise NotImplementedError("Phase 1c: build prompt, call LLM, parse JSON response")

    def generate_counterfactual(self, annotation: dict, metadata: dict) -> dict:
        """Generate a counterfactual scene variant for an annotated frame.

        Prompts the LLM to imagine an alternative scenario (e.g., pedestrian
        stepped further into the lane) and returns a modified annotation dict.

        Args:
            annotation: Existing annotation for the frame.
            metadata: Frame metadata for context.

        Returns:
            Modified annotation dict representing the counterfactual scenario.
        """
        raise NotImplementedError("Phase 1c: generate counterfactual annotation via LLM")

    def annotate_dataset(self, frames: list[dict], output_path: Path) -> Path:
        """Annotate a list of frame dicts and write results to a JSONL file.

        Applies counterfactual augmentation to
        ``config['annotation']['counterfactual_ratio']`` fraction of frames.

        Args:
            frames: List of frame dicts from NuScenesRarityFilter or DADALoader.
            output_path: Path to output JSONL file.

        Returns:
            Path to the written JSONL file.
        """
        raise NotImplementedError("Phase 1c: batch annotate frames and write JSONL")
