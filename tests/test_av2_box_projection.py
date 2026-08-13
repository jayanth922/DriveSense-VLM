"""Tests for drivesense.data.av2_box_projection, built on synthetic doubles
that implement the CONFIRMED av2-api pinhole-camera algorithm (see
tests/_av2_fakes.py and docs/AV2_INTEGRATION.md). No av2 package needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for p in (_SRC, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from drivesense.data.av2_box_projection import (  # noqa: E402
    cuboid_vertices_in_ego_frame,
    get_2d_bbox_from_cuboid,
    project_cuboid_to_2d_bbox,
)
from _av2_fakes import (  # noqa: E402
    FakeCuboid,
    FakeIntrinsics,
    FakePinholeCamera,
    FakeSE3,
    IDENTITY_SE3,
    box_vertices,
)


def _camera() -> FakePinholeCamera:
    return FakePinholeCamera(IDENTITY_SE3, FakeIntrinsics())


class TestCuboidVerticesInEgoFrame:
    def test_none_pose_is_noop(self):
        verts = np.array(box_vertices((0, 0, 10), (1, 1, 1)))
        out = cuboid_vertices_in_ego_frame(verts, None)
        assert np.allclose(out, verts)

    def test_city_frame_conversion(self):
        # ego is 5m ahead of city origin along x (city_SE3_ego translation=(5,0,0)).
        city_SE3_ego = FakeSE3(np.eye(3), np.array([5.0, 0.0, 0.0]))
        city_pt = np.array([[15.0, 0.0, 10.0]])
        ego_pt = cuboid_vertices_in_ego_frame(city_pt, city_SE3_ego)
        assert np.allclose(ego_pt, [[10.0, 0.0, 10.0]])


class TestProjectCuboidTo2DBbox:
    def test_cuboid_in_front_projects_near_center(self):
        cam = _camera()
        verts = np.array(box_vertices((0, 0, 10), (1, 1, 1)))
        bbox = project_cuboid_to_2d_bbox(verts, cam)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 < cam.intrinsics.cx_px < x2
        assert y1 < cam.intrinsics.cy_px < y2
        assert x2 > x1 and y2 > y1

    def test_cuboid_behind_camera_is_none(self):
        cam = _camera()
        verts = np.array(box_vertices((0, 0, -10), (1, 1, 1)))  # all z < 0
        assert project_cuboid_to_2d_bbox(verts, cam) is None

    def test_straddling_camera_uses_valid_corners_only(self):
        cam = _camera()
        # Half-extent 0.5 centered at z=0.05: some corners z>0 (valid), some z<=0.
        verts = np.array(box_vertices((0, 0, 0.05), (0.5, 0.5, 0.5)))
        bbox = project_cuboid_to_2d_bbox(verts, cam)
        assert bbox is not None  # >=2 valid corners -> still produces a box

    def test_farther_cuboid_projects_smaller(self):
        cam = _camera()
        near = project_cuboid_to_2d_bbox(np.array(box_vertices((0, 0, 5), (1, 1, 1))), cam)
        far = project_cuboid_to_2d_bbox(np.array(box_vertices((0, 0, 50), (1, 1, 1))), cam)
        near_area = (near[2] - near[0]) * (near[3] - near[1])
        far_area = (far[2] - far[0]) * (far[3] - far[1])
        assert far_area < near_area  # perspective: farther = smaller in pixels

    def test_min_valid_corners_guard(self):
        cam = _camera()
        verts = np.array(box_vertices((0, 0, 10), (1, 1, 1)))
        assert project_cuboid_to_2d_bbox(verts, cam, min_valid_corners=100) is None


class TestGet2DBboxFromCuboid:
    def test_end_to_end_normalized_output(self):
        cam = _camera()
        cuboid = FakeCuboid("PEDESTRIAN", box_vertices((0, 0, 10), (0.5, 0.5, 1)))
        bbox = get_2d_bbox_from_cuboid(cuboid, cam)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert all(isinstance(v, int) for v in bbox)
        assert 0 <= x1 < x2 <= 1000
        assert 0 <= y1 < y2 <= 1000

    def test_behind_camera_returns_none(self):
        cam = _camera()
        cuboid = FakeCuboid("PEDESTRIAN", box_vertices((0, 0, -10), (0.5, 0.5, 1)))
        assert get_2d_bbox_from_cuboid(cuboid, cam) is None

    def test_city_frame_cuboid_shifts_projection(self):
        cam = _camera()
        # Cuboid at ego(0,0,10) directly, vs the SAME cuboid expressed in a city
        # frame offset by (2,0,0) with city_SE3_ego correcting it back -> should
        # project identically once the frame is corrected.
        cuboid_ego = FakeCuboid("PEDESTRIAN", box_vertices((0, 0, 10), (0.5, 0.5, 1)))
        city_SE3_ego = FakeSE3(np.eye(3), np.array([2.0, 0.0, 0.0]))
        cuboid_city = FakeCuboid("PEDESTRIAN", box_vertices((2, 0, 10), (0.5, 0.5, 1)))

        bbox_ego = get_2d_bbox_from_cuboid(cuboid_ego, cam, city_SE3_ego=None)
        bbox_corrected = get_2d_bbox_from_cuboid(cuboid_city, cam, city_SE3_ego=city_SE3_ego)
        assert bbox_ego == bbox_corrected

        # Without the frame correction, the same "city" vertices misproject.
        bbox_uncorrected = get_2d_bbox_from_cuboid(cuboid_city, cam, city_SE3_ego=None)
        assert bbox_uncorrected != bbox_ego


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
