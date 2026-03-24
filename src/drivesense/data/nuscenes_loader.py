"""nuScenes dataset loader with composite rarity scoring for rare hazard frame filtering.

Implements Phase 1a: scans nuScenes scenes, computes a 0-6 composite rarity score per
sample keyframe, and exports frames scoring >= min_rarity_score as a curated dataset
ready for LLM annotation in Phase 1c.

Rarity signals (each contributes +1 to the score):
    1. proximity         — any pedestrian/cyclist within proximity_threshold_m of ego
    2. occlusion         — any annotation with visibility in [occ_min, occ_max]%
    3. density           — frame has >= min_agents_for_density annotations
    4. weather           — scene description contains adverse weather/night keywords
    5. vulnerable_road_user — frame contains at least one pedestrian annotation
    6. cyclist           — frame contains at least one bicycle/motorcycle annotation

Maximum score: 6.  Default minimum threshold to keep a frame: 3 (from configs/data.yaml).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Guard nuScenes devkit — not available on macOS local dev installs.
try:
    from nuscenes.nuscenes import NuScenes  # type: ignore[import]
    _NUSCENES_AVAILABLE = True
except ImportError:
    NuScenes = None  # type: ignore[assignment, misc]
    _NUSCENES_AVAILABLE = False

# Visibility level → (min_pct, max_pct) mapping per nuScenes schema.
_VISIBILITY_RANGES: dict[str, tuple[int, int]] = {
    "1": (0, 40),
    "2": (40, 60),
    "3": (60, 80),
    "4": (80, 100),
}

_PEDESTRIAN_PREFIX = "human.pedestrian"
# nuScenes cyclist categories: vehicle.bicycle, vehicle.motorcycle — both contain "cycle".
_CYCLIST_SUBSTR = "cycle"


class NuScenesRarityFilter:
    """Filters nuScenes frames by composite rarity score to identify safety-critical scenarios.

    The rarity score is the sum of binary signals — higher score means rarer and more
    interesting for SFT training data curation.

    Args:
        nuscenes_root: Path to the nuScenes dataset root directory.
        config: Data config dict loaded from configs/data.yaml.

    Raises:
        ImportError: If nuScenes devkit is not installed.
    """

    def __init__(self, nuscenes_root: Path, config: dict) -> None:
        if not _NUSCENES_AVAILABLE:
            raise ImportError(
                "nuScenes devkit not installed. Run: pip install nuscenes-devkit"
            )
        rarity = config["nuscenes"]["rarity"]
        self._proximity_threshold: float = float(rarity["proximity_threshold_m"])
        self._occ_min: int = int(rarity["occlusion_min_visibility"])
        self._occ_max: int = int(rarity["occlusion_max_visibility"])
        self._density_threshold: int = int(rarity["min_agents_for_density"])
        self._weather_keywords: list[str] = [
            kw.lower() for kw in rarity["adverse_weather_keywords"]
        ]
        self._min_rarity_score: int = int(rarity["min_rarity_score"])
        self._intersection_keywords: list[str] = [
            kw.lower() for kw in rarity.get("intersection_keywords", [])
        ]

        version: str = config["nuscenes"].get("version", "v1.0-mini")
        logger.info("Loading nuScenes %s from %s ...", version, nuscenes_root)
        self.nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)
        logger.info(
            "Loaded %d scenes, %d samples.", len(self.nusc.scene), len(self.nusc.sample)
        )

        # Populated by filter_rare_frames(); queried by get_score_distribution / export.
        self._all_scores: list[dict] = []
        self._filtered_scores: list[dict] = []

    # ------------------------------------------------------------------
    # Private signal helpers — each returns (active: bool, details)
    # ------------------------------------------------------------------

    def _compute_proximity_score(
        self, sample_token: str
    ) -> tuple[bool, list[dict]]:
        """Check if any pedestrian/cyclist is within proximity_threshold_m of ego.

        Args:
            sample_token: nuScenes sample token.

        Returns:
            Tuple of (has_proximity, nearby_agents). Each agent dict contains
            ``{"category": str, "distance_m": float, "token": str}``.
        """
        sample = self.nusc.get("sample", sample_token)
        cam_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
        ego_xy = np.array(ego_pose["translation"][:2])

        nearby: list[dict] = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]
            if not (cat.startswith(_PEDESTRIAN_PREFIX) or _CYCLIST_SUBSTR in cat):
                continue
            ann_xy = np.array(ann["translation"][:2])
            dist = float(np.linalg.norm(ego_xy - ann_xy))
            if dist <= self._proximity_threshold:
                nearby.append(
                    {"category": cat, "distance_m": round(dist, 3), "token": ann_token}
                )
        return bool(nearby), nearby

    def _compute_occlusion_score(
        self, sample_token: str
    ) -> tuple[bool, list[dict]]:
        """Check if any annotation has visibility within the configured low-vis range.

        Args:
            sample_token: nuScenes sample token.

        Returns:
            Tuple of (has_occlusion, occluded_agents) where each dict contains
            ``{"category": str, "visibility_level": str, "token": str}``.
        """
        sample = self.nusc.get("sample", sample_token)
        occluded: list[dict] = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            vis = self.nusc.get("visibility", ann["visibility_token"])
            level = str(vis.get("level", "4"))
            lo, hi = _VISIBILITY_RANGES.get(level, (80, 100))
            # Visibility range overlaps [occ_min, occ_max] → qualifies.
            if lo <= self._occ_max and hi >= self._occ_min:
                occluded.append(
                    {
                        "category": ann["category_name"],
                        "visibility_level": level,
                        "token": ann_token,
                    }
                )
        return bool(occluded), occluded

    def _compute_density_score(self, sample_token: str) -> tuple[bool, int]:
        """Check if annotation count exceeds the density threshold.

        Args:
            sample_token: nuScenes sample token.

        Returns:
            Tuple of (is_dense, agent_count).
        """
        sample = self.nusc.get("sample", sample_token)
        count = len(sample["anns"])
        return count >= self._density_threshold, count

    def _compute_weather_score(self, scene_token: str) -> tuple[bool, str]:
        """Check if scene description contains adverse weather or night keywords.

        Args:
            scene_token: nuScenes scene token.

        Returns:
            Tuple of (is_adverse, matched_keyword). Keyword is "" when inactive.
        """
        scene = self.nusc.get("scene", scene_token)
        desc = scene.get("description", "").lower()
        for kw in self._weather_keywords:
            if kw in desc:
                return True, kw
        return False, ""

    def _compute_vulnerable_road_user_score(
        self, sample_token: str
    ) -> tuple[bool, int]:
        """Check if frame contains pedestrians AND ego is at/near an intersection.

        Both conditions must be true for the signal to fire (+1), matching the spec:
        "pedestrian present AND scene description or map data suggests intersection."
        Intersection detection uses keyword matching on scene['description'] as a
        lightweight proxy for map data (no map expansion package required).

        Args:
            sample_token: nuScenes sample token.

        Returns:
            Tuple of (active, ped_count). ``ped_count`` is the number of pedestrian
            annotations regardless of intersection context (useful for diagnostics).
        """
        sample = self.nusc.get("sample", sample_token)
        ped_count = sum(
            1
            for t in sample["anns"]
            if self.nusc.get("sample_annotation", t)["category_name"].startswith(
                _PEDESTRIAN_PREFIX
            )
        )
        if ped_count == 0:
            return False, 0

        scene = self.nusc.get("scene", sample["scene_token"])
        desc = scene.get("description", "").lower()
        at_intersection = any(kw in desc for kw in self._intersection_keywords)
        return at_intersection, ped_count

    def _compute_cyclist_score(self, sample_token: str) -> tuple[bool, int]:
        """Check if frame contains bicycle or motorcycle annotations.

        Args:
            sample_token: nuScenes sample token.

        Returns:
            Tuple of (has_cyclist, cyclist_count).
        """
        sample = self.nusc.get("sample", sample_token)
        count = sum(
            1
            for t in sample["anns"]
            if _CYCLIST_SUBSTR in self.nusc.get("sample_annotation", t)["category_name"]
        )
        return count > 0, count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_rarity_score(self, sample_token: str) -> dict:
        """Compute the composite rarity score (0-6) for a single keyframe.

        Args:
            sample_token: nuScenes sample token identifying the keyframe.

        Returns:
            Dict with keys: ``sample_token``, ``scene_token``, ``rarity_score``,
            ``signals``, ``cam_front_path``, ``cam_front_token``, ``timestamp``,
            ``scene_description``, ``ego_pose``, ``num_annotations``.
        """
        sample = self.nusc.get("sample", sample_token)
        scene_token: str = sample["scene_token"]
        scene = self.nusc.get("scene", scene_token)
        cam_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

        prox_active, prox_details = self._compute_proximity_score(sample_token)
        occ_active, occ_details = self._compute_occlusion_score(sample_token)
        dense_active, agent_count = self._compute_density_score(sample_token)
        weather_active, weather_kw = self._compute_weather_score(scene_token)
        vru_active, vru_count = self._compute_vulnerable_road_user_score(sample_token)
        cyc_active, cyc_count = self._compute_cyclist_score(sample_token)

        score = sum(
            [prox_active, occ_active, dense_active, weather_active, vru_active, cyc_active]
        )

        return {
            "sample_token": sample_token,
            "scene_token": scene_token,
            "rarity_score": score,
            "signals": {
                "proximity": {"active": prox_active, "details": prox_details},
                "occlusion": {"active": occ_active, "details": occ_details},
                "density": {"active": dense_active, "count": agent_count},
                "weather": {"active": weather_active, "keyword": weather_kw},
                "vulnerable_road_user": {"active": vru_active, "count": vru_count},
                "cyclist": {"active": cyc_active, "count": cyc_count},
            },
            "cam_front_path": self.nusc.get_sample_data_path(cam_data["token"]),
            "cam_front_token": cam_data["token"],
            "timestamp": sample["timestamp"],
            "scene_description": scene.get("description", ""),
            "ego_pose": {
                "translation": ego_pose["translation"],
                "rotation": ego_pose["rotation"],
            },
            "num_annotations": len(sample["anns"]),
        }

    def filter_rare_frames(self, min_score: int | None = None) -> list[dict]:
        """Score all samples and return those meeting the minimum rarity threshold.

        Iterates all scenes in the dataset, scores every sample keyframe, caches
        all results internally, and returns filtered results sorted by score descending.
        Logs score distribution and per-signal statistics.

        Args:
            min_score: Override minimum rarity score. Defaults to config value.

        Returns:
            List of rarity score dicts for frames that meet the threshold.
        """
        threshold = min_score if min_score is not None else self._min_rarity_score

        # Collect all sample tokens across all scenes, preserving temporal order.
        all_tokens: list[str] = []
        for scene in self.nusc.scene:
            token: str = scene["first_sample_token"]
            while token:
                all_tokens.append(token)
                token = self.nusc.get("sample", token)["next"]

        logger.info("Scoring %d samples (min_score=%d)...", len(all_tokens), threshold)
        self._all_scores = [
            self.compute_rarity_score(t)
            for t in tqdm(all_tokens, desc="Scoring frames", unit="frame")
        ]

        self._filtered_scores = sorted(
            [s for s in self._all_scores if s["rarity_score"] >= threshold],
            key=lambda x: x["rarity_score"],
            reverse=True,
        )

        dist = self.get_score_distribution()
        logger.info("Score distribution: %s", dist)
        logger.info(
            "Frames passing threshold=%d: %d / %d (%.1f%%)",
            threshold,
            len(self._filtered_scores),
            len(self._all_scores),
            100 * len(self._filtered_scores) / max(len(self._all_scores), 1),
        )
        for signal in (
            "proximity", "occlusion", "density", "weather",
            "vulnerable_road_user", "cyclist",
        ):
            active_count = sum(
                1 for s in self._all_scores if s["signals"][signal]["active"]
            )
            logger.info("  %-25s %d frames active", signal, active_count)

        return self._filtered_scores

    def get_score_distribution(self) -> dict[int, int]:
        """Return the count of frames at each rarity score level across all scored frames.

        Returns:
            Dict mapping score (0-6) to frame count, e.g. ``{0: 300, 1: 80, ...}``.
            Returns empty dict if filter_rare_frames() has not been called yet.
        """
        if not self._all_scores:
            logger.warning("No scores computed yet — call filter_rare_frames() first.")
            return {}
        dist: dict[int, int] = dict.fromkeys(range(7), 0)
        for s in self._all_scores:
            dist[s["rarity_score"]] += 1
        return dist

    def export_filtered_dataset(self, output_dir: Path) -> Path:
        """Export filtered frames to a structured output directory.

        Copies CAM_FRONT images to ``output_dir/images/`` and writes three JSON files:
        ``metadata.json`` (per-frame rarity records), ``score_distribution.json``,
        and ``summary.json`` with aggregate statistics.

        Args:
            output_dir: Target directory for exported dataset.

        Returns:
            Path to the output directory.

        Raises:
            RuntimeError: If filter_rare_frames() has not been called.
        """
        if not self._all_scores:
            raise RuntimeError(
                "No scores computed. Call filter_rare_frames() before export."
            )

        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        exported: list[dict] = []
        for entry in tqdm(self._filtered_scores, desc="Exporting images", unit="frame"):
            src = Path(entry["cam_front_path"])
            dst = images_dir / src.name
            shutil.copy2(src, dst)
            exported.append({**entry, "exported_image_path": str(dst.relative_to(output_dir))})

        (output_dir / "metadata.json").write_text(json.dumps(exported, indent=2))
        (output_dir / "score_distribution.json").write_text(
            json.dumps(self.get_score_distribution(), indent=2)
        )

        signal_counts = {
            sig: sum(1 for s in self._filtered_scores if s["signals"][sig]["active"])
            for sig in (
                "proximity", "occlusion", "density", "weather",
                "vulnerable_road_user", "cyclist",
            )
        }
        summary = {
            "total_frames_scanned": len(self._all_scores),
            "frames_exported": len(exported),
            "filter_rate_pct": round(
                100 * len(exported) / max(len(self._all_scores), 1), 1
            ),
            "min_rarity_score_used": self._min_rarity_score,
            "score_distribution": self.get_score_distribution(),
            "signal_breakdown_in_filtered": signal_counts,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        logger.info("Exported %d frames to %s", len(exported), output_dir)
        return output_dir
