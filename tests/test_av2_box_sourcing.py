"""Tests for drivesense.data.av2_box_sourcing. No av2 package needed —
uses the synthetic doubles in tests/_av2_fakes.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for p in (_SRC, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from drivesense.data.av2_box_sourcing import (  # noqa: E402
    av2_category_to_hazard,
    source_boxes_for_av2_frame,
    taxonomy_coverage,
)
from drivesense.data.box_sourcing import BOX_EXEMPT_LABELS  # noqa: E402
from _av2_fakes import (  # noqa: E402
    FakeCuboid, FakeIntrinsics, FakePinholeCamera, IDENTITY_SE3, box_vertices,
)


def _camera() -> FakePinholeCamera:
    return FakePinholeCamera(IDENTITY_SE3, FakeIntrinsics())


def _cuboid(category: str, center=(0, 0, 10), half_extents=(0.5, 0.5, 1.0)) -> FakeCuboid:
    return FakeCuboid(category, box_vertices(center, half_extents))


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------


class TestAv2CategoryToHazard:
    def test_pedestrian_maps_to_jaywalking_by_default(self):
        assert av2_category_to_hazard("PEDESTRIAN") == "jaywalking"

    def test_pedestrian_with_low_occlusion_signal_is_occluded(self):
        assert av2_category_to_hazard("PEDESTRIAN", occlusion_signal=2) == "occluded_pedestrian"

    def test_pedestrian_with_high_occlusion_signal_stays_jaywalking(self):
        assert av2_category_to_hazard("PEDESTRIAN", occlusion_signal=50) == "jaywalking"

    def test_cyclist_family_maps_correctly(self):
        assert av2_category_to_hazard("BICYCLIST") == "cyclist_proximity"
        assert av2_category_to_hazard("MOTORCYCLE") == "cyclist_proximity"

    def test_construction_family_maps_correctly(self):
        assert av2_category_to_hazard("CONSTRUCTION_CONE") == "construction_zone"
        assert av2_category_to_hazard("BOLLARD") == "construction_zone"

    def test_ordinary_vehicle_is_deliberately_unmapped(self):
        assert av2_category_to_hazard("REGULAR_VEHICLE") is None
        assert av2_category_to_hazard("BUS") is None

    def test_case_insensitive(self):
        assert av2_category_to_hazard("pedestrian") == "jaywalking"

    def test_unrecognized_category_is_none(self):
        assert av2_category_to_hazard("SOME_FUTURE_CATEGORY_XYZ") is None


class TestTaxonomyCoverage:
    def test_counts_mapped_unmapped_unrecognized(self):
        cats = ["PEDESTRIAN", "BICYCLIST", "REGULAR_VEHICLE", "BUS", "TOTALLY_UNKNOWN"]
        counts = taxonomy_coverage(cats)
        assert counts == {"mapped": 2, "unmapped": 2, "unrecognized": 1}


# ---------------------------------------------------------------------------
# source_boxes_for_av2_frame — mirrors box_sourcing.source_boxes_for_frame
# ---------------------------------------------------------------------------


class TestSourceBoxesForAv2Frame:
    def test_kept_hazards_carry_label_and_box(self):
        cam = _camera()
        cuboids = [_cuboid("PEDESTRIAN"), _cuboid("BICYCLIST", center=(1, 0, 8))]
        kept, rejected = source_boxes_for_av2_frame(cuboids, cam)
        labels = {h["label"] for h in kept}
        assert labels == {"jaywalking", "cyclist_proximity"}
        assert all("bbox_2d" in h for h in kept)
        assert rejected == []

    def test_unmapped_categories_produce_no_hazard(self):
        cam = _camera()
        cuboids = [_cuboid("REGULAR_VEHICLE"), _cuboid("BUS")]
        kept, rejected = source_boxes_for_av2_frame(cuboids, cam)
        assert kept == []
        assert rejected == []

    def test_behind_camera_cuboid_dropped_not_rejected(self):
        # Behind-camera means "no box at all" (frustum), not a box-quality
        # rejection — it never becomes a candidate in the first place.
        cam = _camera()
        cuboids = [_cuboid("PEDESTRIAN", center=(0, 0, -10))]
        kept, rejected = source_boxes_for_av2_frame(cuboids, cam)
        assert kept == [] and rejected == []

    def test_high_density_fires_at_threshold(self):
        cam = _camera()
        cuboids = [_cuboid("REGULAR_VEHICLE") for _ in range(15)]  # unmapped, but count matters
        kept, _ = source_boxes_for_av2_frame(cuboids, cam, density_threshold=15)
        assert any(h["label"] == "high_density" for h in kept)
        hd = next(h for h in kept if h["label"] == "high_density")
        assert "bbox_2d" not in hd  # box-exempt
        assert hd["label"] in BOX_EXEMPT_LABELS

    def test_high_density_does_not_fire_below_threshold(self):
        cam = _camera()
        cuboids = [_cuboid("REGULAR_VEHICLE") for _ in range(5)]
        kept, _ = source_boxes_for_av2_frame(cuboids, cam, density_threshold=15)
        assert not any(h["label"] == "high_density" for h in kept)

    def test_oversized_box_rejected_via_reused_filter(self):
        # A cuboid huge enough to cover >40% of frame area must be rejected by
        # the SAME box_reject_reason nuScenes uses (reused, not reimplemented).
        cam = _camera()
        huge = FakeCuboid("PEDESTRIAN", box_vertices((0, 0, 1.5), (50, 50, 50)))
        kept, rejected = source_boxes_for_av2_frame([huge], cam)
        assert kept == []
        assert len(rejected) == 1
        assert rejected[0]["reject_reason"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
