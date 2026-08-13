"""Synthetic AV2 test doubles matching the CONFIRMED shapes from real
av2-api source (see docs/AV2_INTEGRATION.md) — no ``av2`` package needed.

``_FakePinholeCamera.project_ego_to_img`` implements the exact algorithm
confirmed from ``pinhole_camera.py`` (extrinsic transform -> intrinsic matmul
-> perspective divide -> z>0 validity), so tests built on these doubles
exercise real, internally-consistent pinhole geometry, not a stub.
"""

from __future__ import annotations

import numpy as np


class FakeSE3:
    """Matches the confirmed av2.geometry.se3.SE3 contract exactly."""

    def __init__(self, rotation, translation) -> None:
        self.rotation = np.asarray(rotation, dtype=float)
        self.translation = np.asarray(translation, dtype=float)

    @property
    def transform_matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def transform_from(self, point_cloud: np.ndarray) -> np.ndarray:
        return np.asarray(point_cloud, dtype=float) @ self.rotation.T + self.translation

    def inverse(self) -> "FakeSE3":
        return FakeSE3(self.rotation.T, self.rotation.T.dot(-self.translation))

    def compose(self, right: "FakeSE3") -> "FakeSE3":
        m = self.transform_matrix @ right.transform_matrix
        return FakeSE3(m[:3, :3], m[:3, 3])


IDENTITY_SE3 = FakeSE3(np.eye(3), np.zeros(3))


class FakeIntrinsics:
    """Matches the confirmed Intrinsics field names."""

    def __init__(self, fx=1000.0, fy=1000.0, cx=960.0, cy=600.0, width=1920, height=1200) -> None:
        self.fx_px, self.fy_px, self.cx_px, self.cy_px = fx, fy, cx, cy
        self.width_px, self.height_px = width, height

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx_px, 0, self.cx_px],
                         [0, self.fy_px, self.cy_px],
                         [0, 0, 1.0]])


class FakePinholeCamera:
    """Implements the CONFIRMED project_ego_to_img algorithm for testing."""

    def __init__(self, ego_SE3_cam: FakeSE3, intrinsics: FakeIntrinsics) -> None:
        self.ego_SE3_cam = ego_SE3_cam
        self.intrinsics = intrinsics

    @property
    def extrinsics(self) -> np.ndarray:
        return self.ego_SE3_cam.inverse().transform_matrix

    def project_ego_to_img(self, points_ego: np.ndarray, remove_nan: bool = False):
        pts = np.asarray(points_ego, dtype=float)
        n = pts.shape[0]
        hom = np.hstack([pts, np.ones((n, 1))])
        cam_pts = (self.extrinsics @ hom.T).T[:, :3]
        z = cam_pts[:, 2]
        valid = z > 0.0
        uv_hom = (self.intrinsics.K @ cam_pts.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uv_hom[:, :2] / uv_hom[:, 2:3]
        if remove_nan:
            uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
        return uv, cam_pts, valid


class FakeCuboid:
    """Matches the confirmed Cuboid shape: .category, .vertices_m (8,3)."""

    def __init__(self, category: str, vertices_m) -> None:
        self.category = category
        self.vertices_m = np.asarray(vertices_m, dtype=float)


def box_vertices(center, half_extents) -> list[list[float]]:
    """8 axis-aligned corners of a box (no rotation) — order-independent for our AABB use."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    signs = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
             (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
    return [[cx + sx * hx, cy + sy * hy, cz + sz * hz] for sx, sy, sz in signs]
