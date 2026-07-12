"""Tests for annotation-v2 box sourcing (class mapping + hard box filter)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data.box_sourcing import (  # noqa: E402
    BOX_EXEMPT_LABELS,
    box_reject_reason,
    filter_frame_boxes,
    nuscenes_category_to_hazard,
    visibility_level_of,
)


class TestVisibilityLevel:
    """The integer level lives in visibility_token, not the level string."""

    def test_reads_visibility_token(self) -> None:
        assert visibility_level_of({"visibility_token": "1"}) == 1
        assert visibility_level_of({"visibility_token": "4"}) == 4

    def test_ignores_level_string(self) -> None:
        # A record whose token is "1" is occluded even though 'level' is "v0-40".
        ann = {"visibility_token": "1", "level": "v0-40"}
        assert visibility_level_of(ann) == 1

    def test_missing_defaults_to_visible(self) -> None:
        assert visibility_level_of({}) == 4

    def test_occluded_pedestrian_end_to_end(self) -> None:
        # The bug: token "1" must yield occluded_pedestrian, not jaywalking.
        vl = visibility_level_of({"visibility_token": "1"})
        assert nuscenes_category_to_hazard("human.pedestrian.adult", vl) == "occluded_pedestrian"


class TestCategoryMapping:
    def test_occluded_pedestrian(self) -> None:
        assert nuscenes_category_to_hazard("human.pedestrian.adult", 1) == "occluded_pedestrian"

    def test_visible_pedestrian_is_jaywalking(self) -> None:
        # v1 descope: non-occluded pedestrian → generic jaywalking box, no map gate.
        assert nuscenes_category_to_hazard("human.pedestrian.adult", 4) == "jaywalking"

    def test_bicycle_and_motorcycle(self) -> None:
        assert nuscenes_category_to_hazard("vehicle.bicycle", 4) == "cyclist_proximity"
        assert nuscenes_category_to_hazard("vehicle.motorcycle", 3) == "cyclist_proximity"

    def test_debris_is_unusual_object(self) -> None:
        assert nuscenes_category_to_hazard("movable_object.debris", 4) == "unusual_object"

    def test_construction_items(self) -> None:
        for cat in ("movable_object.barrier", "movable_object.trafficcone", "vehicle.construction"):
            assert nuscenes_category_to_hazard(cat, 4) == "construction_zone"

    def test_ordinary_vehicle_is_none(self) -> None:
        assert nuscenes_category_to_hazard("vehicle.car", 4) is None
        assert nuscenes_category_to_hazard("vehicle.bus.rigid", 4) is None

    def test_high_density_not_produced_here(self) -> None:
        # high_density is scene-level; never emitted from a single instance.
        assert "high_density" in BOX_EXEMPT_LABELS


class TestBoxRejectReason:
    def test_good_box_passes(self) -> None:
        assert box_reject_reason([400, 300, 550, 600]) is None

    def test_full_frame_is_oversized(self) -> None:
        assert box_reject_reason([0, 0, 1000, 1000]) == "oversized_gt_40pct"

    def test_over_40pct_area_rejected(self) -> None:
        # 700 x 700 = 49% of frame
        assert box_reject_reason([100, 100, 800, 800]) == "oversized_gt_40pct"

    def test_degenerate_tiny_rejected(self) -> None:
        assert box_reject_reason([500, 500, 505, 560]) == "degenerate_tiny"

    def test_bad_aspect_rejected(self) -> None:
        # very wide, thin sliver (aspect >> 8) but under 40% area
        assert box_reject_reason([100, 500, 900, 520]) == "bad_aspect"

    def test_catch_all_edges_rejected(self) -> None:
        # touches left/top/right but not full area (height 30%): 3 edges
        assert box_reject_reason([0, 0, 1000, 300]) in ("catch_all_edges", "bad_aspect")

    def test_inverted_rejected(self) -> None:
        assert box_reject_reason([600, 600, 400, 400]) == "inverted_or_zero"

    def test_malformed_rejected(self) -> None:
        assert box_reject_reason([1, 2, 3]) == "malformed"


class TestFilterFrameBoxes:
    def test_exempt_labels_bypass(self) -> None:
        hazards = [{"label": "high_density"}, {"label": "no_hazard"}]
        kept, rejected = filter_frame_boxes(hazards)
        assert len(kept) == 2 and rejected == []

    def test_splits_good_and_bad(self) -> None:
        hazards = [
            {"label": "jaywalking", "bbox_2d": [400, 300, 550, 600]},   # good
            {"label": "cyclist_proximity", "bbox_2d": [0, 0, 1000, 1000]},  # oversized
        ]
        kept, rejected = filter_frame_boxes(hazards)
        assert [h["label"] for h in kept] == ["jaywalking"]
        assert rejected[0]["reject_reason"] == "oversized_gt_40pct"
        assert rejected[0]["label"] == "cyclist_proximity"
