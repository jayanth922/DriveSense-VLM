"""SFT training modules for DriveSense-VLM.

Submodules:
    sft_trainer: Main LoRA SFT training entry point using HuggingFace Trainer (Phase 2a).
    callbacks:   Custom W&B logging and early stopping callbacks (Phase 2a).
"""

from __future__ import annotations
