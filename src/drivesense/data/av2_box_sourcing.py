"""Annotation v2 (AV2 variant): source tight hazard boxes from Argoverse 2 GT.

Mirrors ``drivesense.data.box_sourcing``'s role for nuScenes — map a dataset's
native 3D annotations to our 7-class hazard taxonomy, project each to a tight
2D box, and apply the SAME hard box-quality filter (``box_reject_reason`` /
``filter_frame_boxes`` / ``BOX_EXEMPT_LABELS`` are imported directly from
``box_sourcing``, not reimplemented — they operate purely on ``[0, 1000]``
bbox coordinates and hazard dicts, with no nuScenes-specific assumptions).

⚠️ TAXONOMY CONFIDENCE: ``AV2_TO_HAZARD_CLASS`` below is a BEST-EFFORT mapping
from Argoverse 2's documented 30-class Sensor Dataset taxonomy, built from
general familiarity with the public category list — this session's live
research into ``av2-api`` source was interrupted before the category list
could be independently re-confirmed against source (e.g.
``av2.utils.metadata`` or the official taxonomy docs). **Cross-check this
table against the installed ``av2`` package before trusting it in any real
pipeline run.** See ``docs/AV2_INTEGRATION.md``.
"""

from __future__ import annotations

import logging
from typing import Any

from drivesense.data.av2_box_projection import get_2d_bbox_from_cuboid
from drivesense.data.box_sourcing import BOX_EXEMPT_LABELS, filter_frame_boxes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category taxonomy — AV2 category name -> our 7-class hazard schema.
#
# UNVERIFIED (best-effort, see module docstring). A value of None means
# "deliberately unmapped" — e.g. ordinary vehicles aren't hazards in this
# taxonomy any more than nuScenes' vehicle.car is (box_sourcing.py returns
# None for it too). Categories not present in this dict AT ALL (vs mapped to
# None) are logged and counted separately as "unrecognized" so a taxonomy
# drift (AV2 renames/adds a category) is visible, not silently dropped.
# ---------------------------------------------------------------------------
AV2_TO_HAZARD_CLASS: dict[str, str | None] = {
    # Vulnerable road users
    "PEDESTRIAN": "jaywalking",
    "WHEELCHAIR": "jaywalking",
    "STROLLER": "jaywalking",
    "OFFICIAL_SIGNALER": "jaywalking",
    "BICYCLIST": "cyclist_proximity",
    "BICYCLE": "cyclist_proximity",
    "MOTORCYCLIST": "cyclist_proximity",
    "MOTORCYCLE": "cyclist_proximity",
    "WHEELED_RIDER": "cyclist_proximity",
    "WHEELED_DEVICE": "cyclist_proximity",
    # Construction / roadwork
    "CONSTRUCTION_CONE": "construction_zone",
    "CONSTRUCTION_BARREL": "construction_zone",
    "BOLLARD": "construction_zone",
    "MESSAGE_BOARD_TRAILER": "construction_zone",
    "TRAFFIC_LIGHT_TRAILER": "construction_zone",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": "construction_zone",
    # Unusual objects on/near the road
    "DOG": "unusual_object",
    "ANIMAL": "unusual_object",
    "STOP_SIGN": "unusual_object",
    "SIGN": "unusual_object",
    # Ordinary vehicles are not, by themselves, hazards in this taxonomy —
    # deliberately unmapped, mirroring nuscenes_category_to_hazard's vehicle.car.
    "REGULAR_VEHICLE": None,
    "LARGE_VEHICLE": None,
    "BUS": None,
    "SCHOOL_BUS": None,
    "ARTICULATED_BUS": None,
    "BOX_TRUCK": None,
    "TRUCK": None,
    "TRUCK_CAB": None,
    "VEHICULAR_TRAILER": None,
    "RAILED_VEHICLE": None,
}

# Occlusion signal name if the caller's cuboid/track dict carries one (AV2's
# own occlusion field name is UNVERIFIED this session — see docs). Callers may
# pass any dict; this key is only consulted if present.
_OCCLUSION_KEY = "num_interior_pts"  # heuristic proxy: very few lidar returns ~ occluded
_OCCLUDED_MAX_POINTS = 5


def av2_category_to_hazard(category: str, occlusion_signal: int | None = None) -> str | None:
    """Map an AV2 category name to our 7-class hazard taxonomy, or ``None``.

    ``high_density`` and ``no_hazard`` are NOT produced here — they are
    scene-level, handled by the caller (mirrors ``nuscenes_category_to_hazard``).

    Args:
        category: AV2 ``category`` string (e.g. ``"PEDESTRIAN"``).
        occlusion_signal: Optional proxy for occlusion (e.g. lidar interior
            point count); pedestrians below :data:`_OCCLUDED_MAX_POINTS` map
            to ``occluded_pedestrian`` instead of ``jaywalking``, mirroring
            nuScenes' visibility-based split. ``None`` skips this refinement.

    Returns:
        A hazard label, or ``None`` if unmapped/unrecognized.
    """
    cat = category.upper()
    if cat not in AV2_TO_HAZARD_CLASS:
        logger.debug("av2_category_to_hazard: unrecognized category %r (not in taxonomy "
                     "table — may indicate AV2 taxonomy drift)", category)
        return None
    label = AV2_TO_HAZARD_CLASS[cat]
    if (label == "jaywalking" and cat == "PEDESTRIAN"
            and occlusion_signal is not None and occlusion_signal < _OCCLUDED_MAX_POINTS):
        return "occluded_pedestrian"
    return label


def source_boxes_for_av2_frame(
    cuboids: list[Any],  # noqa: ANN401 — list[av2.structures.cuboid.Cuboid]
    camera: Any,  # noqa: ANN401 — av2.geometry.camera.pinhole_camera.PinholeCamera
    city_SE3_ego: Any = None,  # noqa: ANN401 — av2.geometry.se3.SE3
    density_threshold: int = 15,
) -> tuple[list[dict], list[dict]]:
    """Build filtered, GT-sourced hazards for one AV2 camera frame.

    Mirrors ``box_sourcing.source_boxes_for_frame``'s exact role/return shape:
    for each cuboid, map its category to a hazard label, project to a 2D box
    via :func:`get_2d_bbox_from_cuboid`, and apply the SAME hard box filter
    used for nuScenes (:func:`box_sourcing.filter_frame_boxes`).

    Args:
        cuboids:            This frame's ``Cuboid`` list (already loaded).
        camera:              The target camera's ``PinholeCamera``.
        city_SE3_ego:        Ego pose, if cuboids are in city frame (see
                             ``av2_box_projection.cuboid_vertices_in_ego_frame``).
        density_threshold:   Cuboid count at/above which ``high_density`` fires.

    Returns:
        ``(kept_hazards, rejected_hazards)`` — same shape as
        ``box_sourcing.source_boxes_for_frame``.
    """
    candidates: list[dict] = []
    for cuboid in cuboids:
        label = av2_category_to_hazard(getattr(cuboid, "category", ""))
        if label is None:
            continue
        bbox = get_2d_bbox_from_cuboid(cuboid, camera, city_SE3_ego)
        if bbox is None:  # behind camera / outside frustum
            continue
        candidates.append({
            "label": label,
            "bbox_2d": bbox,
            "box_source": "av2_gt",
            "category": getattr(cuboid, "category", ""),
        })

    kept, rejected = filter_frame_boxes(candidates)

    if len(cuboids) >= density_threshold:
        kept.append({"label": "high_density", "box_source": "scene_level"})  # no bbox

    logger.debug("av2 frame: %d cuboids -> %d hazard candidates -> %d kept, %d rejected",
                len(cuboids), len(candidates), len(kept), len(rejected))
    return kept, rejected


def taxonomy_coverage(categories: list[str]) -> dict[str, int]:
    """Count how many of ``categories`` are mapped / deliberately-unmapped / unrecognized.

    Useful as a quick sanity check when pointed at a real log's category list
    — a large "unrecognized" count signals the taxonomy table needs updating.

    Args:
        categories: Raw AV2 category strings (e.g. from a batch of cuboids).

    Returns:
        ``{"mapped": n, "unmapped": n, "unrecognized": n}``.
    """
    counts = {"mapped": 0, "unmapped": 0, "unrecognized": 0}
    for cat in categories:
        key = cat.upper()
        if key not in AV2_TO_HAZARD_CLASS:
            counts["unrecognized"] += 1
        elif AV2_TO_HAZARD_CLASS[key] is None:
            counts["unmapped"] += 1
        else:
            counts["mapped"] += 1
    return counts


__all__ = [
    "AV2_TO_HAZARD_CLASS",
    "BOX_EXEMPT_LABELS",
    "av2_category_to_hazard",
    "source_boxes_for_av2_frame",
    "taxonomy_coverage",
]
