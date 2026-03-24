"""DADA-2000 dataset loader for critical-moment accident frame extraction.

Implements Phase 1b: locate annotated accident timestamps in the DADA-2000
dataset, extract the critical moment frame plus N context frames before it,
and export them in the standard DriveSense frame format.

DADA-2000 directory structure (after download + extraction):
    dada2000/
        videos/
            <category>/
                <clip_id>/
                    frames/    # pre-extracted or extracted by this loader
        annotations/
            <clip_id>.json     # accident timestamp, category, ego-actions

Implementation target: Phase 1b
"""

from __future__ import annotations

from pathlib import Path


class DADALoader:
    """Loads and extracts critical accident frames from DADA-2000.

    Args:
        dada_root: Path to the DADA-2000 dataset root directory.
        config: Data config dict loaded from configs/data.yaml.
    """

    def __init__(self, dada_root: Path, config: dict) -> None:
        raise NotImplementedError("Phase 1b: scan DADA-2000 directory structure and load config")

    def list_clips(self) -> list[dict]:
        """Return metadata for all annotated video clips in the dataset.

        Returns:
            List of dicts, each containing:
            ``{"clip_id": str, "category": str, "accident_frame": int,
               "video_path": Path, "annotation_path": Path}``.
        """
        raise NotImplementedError("Phase 1b: scan dada_root for clips and annotations")

    def extract_critical_frames(self, clip_id: str) -> list[dict]:
        """Extract critical moment frame and preceding context frames for a clip.

        Extracts the frame at the annotated accident timestamp, plus
        ``config['dada2000']['frame_extraction']['additional_context_frames']``
        frames immediately before it.

        Args:
            clip_id: DADA-2000 clip identifier string.

        Returns:
            List of frame dicts ordered chronologically, each containing:
            ``{"frame_index": int, "is_critical": bool, "image": PIL.Image,
               "clip_id": str, "timestamp_ms": int}``.
        """
        raise NotImplementedError("Phase 1b: extract frames at accident timestamp from video")

    def export_dataset(self, output_dir: Path) -> Path:
        """Export all extracted frames as a structured directory with metadata JSON.

        Respects ``config['dada2000']['max_frames']`` total frame limit.

        Args:
            output_dir: Directory to write images and metadata.json.

        Returns:
            Path to the written metadata.json file.
        """
        raise NotImplementedError("Phase 1b: export frames and metadata JSON to output_dir")
