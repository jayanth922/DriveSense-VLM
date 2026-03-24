"""Data loading, preprocessing, and annotation modules for DriveSense-VLM.

Submodules:
    nuscenes_loader: nuScenes rarity scoring and rare-frame filtering (Phase 1a).
    dada_loader:     DADA-2000 critical moment frame extraction (Phase 1b).
    annotation:      LLM-based counterfactual annotation pipeline (Phase 1c).
    dataset:         Unified SFT dataset combining all sources (Phase 1c).
    transforms:      Image augmentation and preprocessing transforms (Phase 1a).
"""

from __future__ import annotations
