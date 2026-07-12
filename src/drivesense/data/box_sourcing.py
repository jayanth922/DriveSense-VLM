"""Annotation v2: source tight hazard boxes from nuScenes GT, not from a VLM.

The v1 loop asked a foundation VLM to invent bounding boxes, which produced
constant full-frame / center-blob boxes. v2 derives boxes from the dataset's
native 3D annotations via :func:`drivesense.data.transforms.get_2d_bbox_from_3d`
and demotes the VLM to describe-only.

Two classes are **box-exempt** (scene-level, no bbox): ``high_density`` and
``no_hazard``. All other hazards must carry a filtered GT box.

Design: see ``docs/annotation_v2.md``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Labels that legitimately carry NO bounding box (scene-level / absence).
BOX_EXEMPT_LABELS: frozenset[str] = frozenset({"high_density", "no_hazard"})

# nuScenes category prefixes/substrings used for the box-sourced hazard classes.
_PEDESTRIAN_PREFIX = "human.pedestrian"
_CYCLIST_SUBSTR = "cycle"  # vehicle.bicycle, vehicle.motorcycle
_CONSTRUCTION_CATEGORIES = (
    "movable_object.barrier",
    "movable_object.trafficcone",
    "vehicle.construction",
)
_DEBRIS_CATEGORY = "movable_object.debris"

# nuScenes visibility level 1 == 0–40% visible (occluded).
_OCCLUDED_VISIBILITY_LEVEL = 1

# Box filter thresholds (coordinates are in the [0, 1000] normalized space).
_COORD_MAX = 1000
_FRAME_AREA = float(_COORD_MAX * _COORD_MAX)
_MAX_AREA_FRAC = 0.40           # reject boxes covering > 40% of the frame
_MIN_SIDE = 0.015 * _COORD_MAX  # reject degenerate boxes (side < 1.5% of frame)
_ASPECT_MIN, _ASPECT_MAX = 0.15, 8.0
_EDGE_EPS = 0.01 * _COORD_MAX   # within 1% of a frame edge counts as "touching"


def nuscenes_category_to_hazard(
    category_name: str, visibility_level: int
) -> str | None:
    """Map a nuScenes annotation to a box-sourced hazard class, or ``None``.

    ``high_density`` and ``no_hazard`` are NOT produced here — they are
    scene-level and handled by the caller.

    Args:
        category_name:    nuScenes ``category_name`` (e.g. ``human.pedestrian.adult``).
        visibility_level: nuScenes visibility level (1–4; 1 == 0–40% visible).

    Returns:
        A hazard label with a GT box, or ``None`` if the instance is not a
        box-sourced hazard (e.g. an ordinary parked car).
    """
    cat = category_name.lower()
    if cat.startswith(_PEDESTRIAN_PREFIX):
        # v1 descope: no map/crosswalk gate. Occluded → occluded_pedestrian,
        # otherwise a generic pedestrian hazard labelled jaywalking.
        return "occluded_pedestrian" if visibility_level == _OCCLUDED_VISIBILITY_LEVEL \
            else "jaywalking"
    if _CYCLIST_SUBSTR in cat:
        return "cyclist_proximity"
    if cat == _DEBRIS_CATEGORY:
        return "unusual_object"
    if cat in _CONSTRUCTION_CATEGORIES:
        return "construction_zone"
    return None


def box_reject_reason(bbox: list[float]) -> str | None:
    """Return why a box should be rejected, or ``None`` if it passes.

    Rules (per frame, box-classes only): oversized (> 40% area), degenerate
    (side < 1.5%), bad aspect ratio, or catch-all (touches >= 3 frame edges).

    Args:
        bbox: ``[x1, y1, x2, y2]`` in [0, 1000] coordinates.

    Returns:
        A short reason string, or ``None`` if the box is acceptable.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "malformed"
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return "inverted_or_zero"
    if (w * h) / _FRAME_AREA > _MAX_AREA_FRAC:
        return "oversized_gt_40pct"
    if w < _MIN_SIDE or h < _MIN_SIDE:
        return "degenerate_tiny"
    aspect = w / h
    if aspect < _ASPECT_MIN or aspect > _ASPECT_MAX:
        return "bad_aspect"
    edges = sum([
        x1 <= _EDGE_EPS, y1 <= _EDGE_EPS,
        x2 >= _COORD_MAX - _EDGE_EPS, y2 >= _COORD_MAX - _EDGE_EPS,
    ])
    if edges >= 3:
        return "catch_all_edges"
    return None


def filter_frame_boxes(hazards: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a frame's hazards into (kept, rejected) by the hard box filter.

    Box-exempt labels (``high_density`` / ``no_hazard``) bypass the filter and
    are always kept. Rejected hazards get a ``reject_reason`` field for logging.

    Args:
        hazards: Hazard dicts, each with ``label`` and (for box classes) ``bbox_2d``.

    Returns:
        ``(kept, rejected)``.
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    for hz in hazards:
        if hz.get("label") in BOX_EXEMPT_LABELS:
            kept.append(hz)
            continue
        reason = box_reject_reason(hz.get("bbox_2d", []))
        if reason is None:
            kept.append(hz)
        else:
            rejected.append({**hz, "reject_reason": reason})
    return kept, rejected


def source_boxes_for_frame(
    nusc: Any,  # noqa: ANN401 — nuScenes SDK object
    sample_token: str,
    density_threshold: int = 15,
) -> tuple[list[dict], list[dict]]:
    """Build filtered, GT-sourced hazards for one nuScenes keyframe.

    Projects each hazard-bearing annotation to a tight 2D box via
    :func:`get_2d_bbox_from_3d`, applies the hard box filter, and adds a
    scene-level ``high_density`` label (box-exempt) when the frame is dense.

    Args:
        nusc:             Initialised nuScenes SDK object.
        sample_token:     nuScenes sample (keyframe) token.
        density_threshold: Agent count at/above which ``high_density`` fires.

    Returns:
        ``(kept_hazards, rejected_hazards)``. ``kept_hazards`` is the v2 label
        for the frame; ``rejected_hazards`` is the reject log.
    """
    from drivesense.data.transforms import get_2d_bbox_from_3d  # noqa: PLC0415

    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"]["CAM_FRONT"]

    candidates: list[dict] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        vis = nusc.get("visibility", ann["visibility_token"])
        try:
            vis_level = int(vis.get("level", "4"))
        except (ValueError, TypeError):
            vis_level = 4
        label = nuscenes_category_to_hazard(ann["category_name"], vis_level)
        if label is None:
            continue
        bbox = get_2d_bbox_from_3d(nusc, ann_token, cam_token)
        if bbox is None:  # entirely behind the camera plane
            continue
        candidates.append({
            "label": label,
            "bbox_2d": bbox,
            "box_source": "nuscenes_gt",
            "ann_token": ann_token,
            "visibility_level": vis_level,
        })

    kept, rejected = filter_frame_boxes(candidates)

    if len(sample["anns"]) >= density_threshold:
        kept.append({"label": "high_density", "box_source": "scene_level"})  # no bbox

    logger.debug(
        "frame %s: %d GT candidates → %d kept, %d rejected",
        sample_token, len(candidates), len(kept), len(rejected),
    )
    return kept, rejected
