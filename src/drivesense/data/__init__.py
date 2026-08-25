"""Data loading, preprocessing, and annotation modules for DriveSense-VLM.

Submodules:
    nuscenes_loader: nuScenes rarity scoring and rare-frame filtering (Phase 1a).
    annotation:      LLM-based counterfactual annotation pipeline (Phase 1c).
    dataset:         nuScenes split-manifest dataset builder (Phase 1b).
    transforms:      Image augmentation and preprocessing transforms (Phase 1a).
"""

from __future__ import annotations
