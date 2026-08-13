"""3D cuboid → 2D box projection for Argoverse 2 (AV2).

Built on AV2's OWN confirmed projection primitives — nothing here reimplements
pinhole-camera geometry from scratch. Operates purely by DUCK TYPING on objects
matching the documented AV2 shapes below, so this module needs no ``av2``
import and no real dataset to be tested (see ``tests/test_av2_box_projection.py``).

CONFIRMED (fetched directly from ``github.com/argoverse/av2-api`` source,
2026-08-13 — see ``docs/AV2_INTEGRATION.md`` for the full research log):

- ``av2.geometry.se3.SE3`` — ``rotation`` (3,3 ndarray), ``translation``
  (3,) ndarray, ``transform_from(point_cloud)`` applies ``pts @ R.T + t``,
  ``inverse()``, ``compose()``.
- ``av2.structures.cuboid.Cuboid`` — ``dst_SE3_object: SE3``,
  ``length_m``/``width_m``/``height_m``, ``vertices_m`` (cached property,
  (8, 3) ndarray of box corners in the cuboid's "destination frame").
- ``av2.geometry.camera.pinhole_camera.PinholeCamera`` — ``ego_SE3_cam: SE3``,
  ``intrinsics: Intrinsics``; **``project_ego_to_img(points_ego, remove_nan=False)
  -> (uv, points_cam, valid_mask)``** is AV2's own projection utility (extrinsic
  transform → intrinsic matmul → perspective divide → frustum validity mask) —
  this is what's reused here, not reimplemented.
- ``Intrinsics`` — ``fx_px``, ``fy_px``, ``cx_px``, ``cy_px``, ``width_px``,
  ``height_px``.

UNVERIFIED (this session's research was interrupted before confirming): whether
a log's LOADED cuboid annotations are natively in ego-vehicle frame or city
frame — the ``Cuboid`` docstring says "typically ego-vehicle or city frame"
without specifying which applies to the standard annotation-loading path. This
module takes an explicit, optional ``city_SE3_ego`` so the caller states which
frame applies rather than this code silently assuming one — see
``docs/AV2_INTEGRATION.md`` for what needs confirming against real data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from drivesense.data.transforms import normalize_bbox_to_1000

logger = logging.getLogger(__name__)

# Minimum in-frustum corners required to accept a projected box. A cuboid with
# only 0-1 valid corners can't form a meaningful axis-aligned box.
MIN_VALID_CORNERS = 2


def cuboid_vertices_in_ego_frame(
    vertices_dst: np.ndarray, city_SE3_ego: Any = None,  # noqa: ANN401 — av2.geometry.se3.SE3
) -> np.ndarray:
    """Return cuboid vertices in the ego-vehicle frame.

    Args:
        vertices_dst: (8, 3) cuboid vertices from ``Cuboid.vertices_m``, in
            whatever frame the cuboid's ``dst_SE3_object`` used.
        city_SE3_ego: Pass the ego pose at this timestamp if ``vertices_dst``
            is in CITY frame, to convert to ego frame via
            ``city_SE3_ego.inverse().transform_from(...)``. Pass ``None`` if
            the vertices are already in ego frame (no-op).

    Returns:
        (8, 3) vertices in ego frame.
    """
    if city_SE3_ego is None:
        return vertices_dst
    return city_SE3_ego.inverse().transform_from(vertices_dst)


def project_cuboid_to_2d_bbox(
    vertices_ego: np.ndarray,
    camera: Any,  # noqa: ANN401 — av2.geometry.camera.pinhole_camera.PinholeCamera
    min_valid_corners: int = MIN_VALID_CORNERS,
) -> list[float] | None:
    """Project a cuboid's ego-frame vertices to a 2D pixel bbox via AV2's PinholeCamera.

    Uses ``camera.project_ego_to_img`` (AV2's own projection utility) to get
    per-corner image coordinates plus a validity mask, then takes the
    axis-aligned bounding box of the VALID corners only.

    This is a coarser approximation than the nuScenes-side edge-clipping in
    ``drivesense.data.transforms.project_box_to_2d`` (which clips each edge at
    its near-plane crossing point): a corner behind the camera is dropped
    entirely here rather than clipped. Whether this materially matters for
    AV2's typical cuboid distances is UNVERIFIED pending real data — see
    ``docs/AV2_INTEGRATION.md``.

    Args:
        vertices_ego:      (8, 3) cuboid corners in ego frame.
        camera:             A ``PinholeCamera``-shaped object exposing
                            ``project_ego_to_img`` and ``intrinsics.width_px``/
                            ``height_px``.
        min_valid_corners:  Minimum in-frustum corners required to accept the box.

    Returns:
        ``[x1, y1, x2, y2]`` pixel bbox, or ``None`` if too few corners are valid
        or the resulting box is degenerate.
    """
    uv, _, valid_mask = camera.project_ego_to_img(vertices_ego, remove_nan=True)
    valid_uv = np.asarray(uv)[np.asarray(valid_mask)]
    if len(valid_uv) < min_valid_corners:
        return None

    x1, y1 = float(valid_uv[:, 0].min()), float(valid_uv[:, 1].min())
    x2, y2 = float(valid_uv[:, 0].max()), float(valid_uv[:, 1].max())
    w, h = float(camera.intrinsics.width_px), float(camera.intrinsics.height_px)
    x1, x2 = max(0.0, x1), min(w, x2)
    y1, y2 = max(0.0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def get_2d_bbox_from_cuboid(
    cuboid: Any,  # noqa: ANN401 — av2.structures.cuboid.Cuboid
    camera: Any,  # noqa: ANN401 — av2.geometry.camera.pinhole_camera.PinholeCamera
    city_SE3_ego: Any = None,  # noqa: ANN401 — av2.geometry.se3.SE3
) -> list[int] | None:
    """Full pipeline: an AV2 ``Cuboid`` → a normalized [0, 1000] 2D bbox.

    Mirrors the role of ``drivesense.data.transforms.get_2d_bbox_from_3d`` for
    nuScenes, adapted to AV2's confirmed API surface.

    Args:
        cuboid:        A ``Cuboid``-shaped object exposing ``vertices_m``.
        camera:        The target camera's ``PinholeCamera``-shaped object.
        city_SE3_ego:  Pass if ``cuboid.vertices_m`` is in CITY frame (see
            :func:`cuboid_vertices_in_ego_frame`); ``None`` if already ego frame.

    Returns:
        ``[x1, y1, x2, y2]`` normalized to [0, 1000], or ``None`` if the
        cuboid doesn't project to a valid box (behind camera / outside frustum).
    """
    vertices_ego = cuboid_vertices_in_ego_frame(cuboid.vertices_m, city_SE3_ego)
    bbox_px = project_cuboid_to_2d_bbox(vertices_ego, camera)
    if bbox_px is None:
        return None
    return normalize_bbox_to_1000(
        bbox_px, int(camera.intrinsics.width_px), int(camera.intrinsics.height_px))
