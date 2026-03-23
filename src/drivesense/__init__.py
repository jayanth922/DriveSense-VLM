"""DriveSense-VLM: SFT-optimized vision-language model for AV rare hazard detection.

Built on Qwen3-VL-2B-Instruct with LoRA fine-tuning, trained on nuScenes and DADA-2000.
Outputs structured JSON with bounding boxes, hazard class, severity, reasoning, and action.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "DriveSense-VLM Contributors"
__license__ = "Apache-2.0"

__all__ = ["__version__", "__author__", "__license__"]
