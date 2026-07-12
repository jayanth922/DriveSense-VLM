"""Tests for near-plane–clipped 3D→2D box projection (frustum clipping)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

pytest.importorskip("PIL")  # transforms imports PIL at module load

from drivesense.data.transforms import project_box_to_2d  # noqa: E402

# Simple pinhole camera: 1920x1080, focal 1000, principal point at centre.
_K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
_W, _H = 1920, 1080


def _cuboid(front_z: float, back_z: float, hx: float = 0.5, hy: float = 0.5) -> np.ndarray:
    """Build (3,8) corners in nuScenes order: front face 0-3 (front_z), back 4-7."""
    xs = [hx, hx, -hx, -hx, hx, hx, -hx, -hx]
    ys = [hy, -hy, -hy, hy, hy, -hy, -hy, hy]
    zs = [front_z, front_z, front_z, front_z, back_z, back_z, back_z, back_z]
    return np.array([xs, ys, zs], dtype=float)


class TestProjectBoxTouD:
    def test_fully_in_front_is_tight(self) -> None:
        # Box well in front → normal tight box, well inside the frame.
        corners = _cuboid(front_z=10.0, back_z=12.0)
        bb = project_box_to_2d(corners, _K, _W, _H)
        assert bb is not None
        x1, y1, x2, y2 = bb
        assert x1 < x2 and y1 < y2
        assert 0 < x1 and x2 < _W and 0 < y1 and y2 < _H  # not touching edges

    def test_straddling_near_plane_yields_valid_box(self) -> None:
        # Corners on BOTH sides of the near plane: OLD code (front corners only)
        # would invert/zero; near-plane clipping must yield a valid, sizeable box.
        corners = _cuboid(front_z=2.0, back_z=-1.0)
        bb = project_box_to_2d(corners, _K, _W, _H)
        assert bb is not None
        x1, y1, x2, y2 = bb
        assert x1 < x2 and y1 < y2, "straddling box must not invert/zero"
        assert (x2 - x1) > 100 and (y2 - y1) > 100, "must not be a sliver"
        assert 0 <= x1 and x2 <= _W and 0 <= y1 and y2 <= _H

    def test_single_edge_crossing_still_valid(self) -> None:
        # Most of the box behind, only the front face in front → still valid.
        corners = _cuboid(front_z=1.5, back_z=-0.5, hx=0.3, hy=0.9)
        bb = project_box_to_2d(corners, _K, _W, _H)
        assert bb is not None
        x1, y1, x2, y2 = bb
        assert x1 < x2 and y1 < y2

    def test_fully_behind_is_none(self) -> None:
        corners = _cuboid(front_z=-5.0, back_z=-8.0)
        assert project_box_to_2d(corners, _K, _W, _H) is None

    def test_near_plane_boundary(self) -> None:
        # Everything just behind the near plane → None (nothing to project).
        corners = _cuboid(front_z=0.05, back_z=-0.05)
        assert project_box_to_2d(corners, _K, _W, _H) is None
