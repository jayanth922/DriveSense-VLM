"""Tests for drivesense.data.av2_loader.

Includes cross-pipeline integration checks proving the key design constraint:
run_label_validation.py and merge_sft_v2.py's functions work UNCHANGED on
AV2-sourced records, because build_av2_sft_record reuses SFTDataFormatter
directly rather than reimplementing the schema.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for p in (_SRC, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import drivesense.data.av2_loader as av2_loader  # noqa: E402
from drivesense.data.av2_loader import build_av2_annotation, build_av2_sft_record  # noqa: E402
from _av2_fakes import (  # noqa: E402
    FakeCuboid, FakeIntrinsics, FakePinholeCamera, IDENTITY_SE3, box_vertices,
)


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _camera() -> FakePinholeCamera:
    return FakePinholeCamera(IDENTITY_SE3, FakeIntrinsics())


def _cuboids() -> list[FakeCuboid]:
    return [
        FakeCuboid("PEDESTRIAN", box_vertices((0, 0, 10), (0.5, 0.5, 1))),
        FakeCuboid("BICYCLIST", box_vertices((2, 0, 8), (0.5, 0.5, 1))),
        FakeCuboid("REGULAR_VEHICLE", box_vertices((-2, 0, 15), (1, 1, 1))),  # unmapped
    ]


# ---------------------------------------------------------------------------
# AV2LogReader — guard behavior + explicitly-unverified stubs
# ---------------------------------------------------------------------------


class TestAV2LogReaderGuard:
    def test_raises_importerror_when_av2_not_installed(self, tmp_path):
        # av2 genuinely isn't installed in this environment -> this proves the
        # guard fires correctly, not just that it's present in the source.
        assert av2_loader._AV2_AVAILABLE is False
        with pytest.raises(ImportError, match="av2 package required"):
            av2_loader.AV2LogReader(tmp_path, "fake-log-id")


class TestAV2LogReaderUnverifiedStubs:
    """With av2 'available' (patched), the loader constructs; its genuinely
    unverified methods must raise NotImplementedError, not silently guess."""

    def _reader(self, tmp_path):
        with patch.object(av2_loader, "_AV2_AVAILABLE", True), \
             patch.object(av2_loader, "AV2SensorDataLoader", MagicMock()):
            return av2_loader.AV2LogReader(tmp_path, "fake-log-id")

    def test_list_timestamps_not_implemented(self, tmp_path):
        reader = self._reader(tmp_path)
        with pytest.raises(NotImplementedError, match="unverified"):
            reader.list_timestamps()

    def test_get_cuboids_not_implemented(self, tmp_path):
        reader = self._reader(tmp_path)
        with pytest.raises(NotImplementedError, match="unverified"):
            reader.get_cuboids(12345)

    def test_get_ego_pose_not_implemented(self, tmp_path):
        reader = self._reader(tmp_path)
        with pytest.raises(NotImplementedError, match="unverified"):
            reader.get_ego_pose(12345)

    def test_get_image_path_not_implemented(self, tmp_path):
        reader = self._reader(tmp_path)
        with pytest.raises(NotImplementedError, match="unverified"):
            reader.get_image_path(12345)


# ---------------------------------------------------------------------------
# build_av2_annotation / build_av2_sft_record — fully buildable/testable today
# ---------------------------------------------------------------------------


class TestBuildAv2Annotation:
    def test_hazards_from_mapped_cuboids_only(self):
        ann = build_av2_annotation(_cuboids(), _camera())
        labels = {h["label"] for h in ann["hazards"]}
        assert labels == {"jaywalking", "cyclist_proximity"}  # vehicle unmapped, dropped

    def test_placeholder_fields_present_pending_describe_pass(self):
        ann = build_av2_annotation(_cuboids(), _camera())
        for h in ann["hazards"]:
            assert h["severity"] == "medium"
            assert h["action"] == "reduce speed"
            assert "bbox_2d" in h

    def test_ego_context_defaults_and_passthrough(self):
        default = build_av2_annotation(_cuboids(), _camera())
        assert default["ego_context"]["weather"] == "unknown"
        custom = build_av2_annotation(_cuboids(), _camera(),
                                      ego_context={"weather": "rain", "time_of_day": "night",
                                                  "road_type": "urban"})
        assert custom["ego_context"]["weather"] == "rain"


class TestBuildAv2SftRecord:
    def test_schema_matches_existing_sft_format(self):
        rec = build_av2_sft_record("/fake/path.jpg", _cuboids(), _camera(), frame_id="log1_1000")
        assert set(rec) == {"messages", "images", "frame_id", "source"}
        assert rec["frame_id"] == "log1_1000"
        assert rec["source"] == "av2"
        assert rec["images"] == ["/fake/path.jpg"]
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Cross-pipeline integration: the actual downstream tools, unmodified,
# consuming AV2-sourced records — proves requirement #4 for real.
# ---------------------------------------------------------------------------


class TestDownstreamPipelineCompatibility:
    def test_run_label_validation_gate_accepts_av2_records(self):
        gate = _load_module("scripts/run_label_validation.py", "gate_for_av2_test")
        records = [
            build_av2_sft_record(f"/fake/{i}.jpg", _cuboids(), _camera(), frame_id=f"log1_{i}")
            for i in range(10)
        ]
        stats = gate.collect_stats(records)
        # Real hazards present, box-diversity computable — the gate doesn't
        # need to know these came from AV2 rather than nuScenes.
        assert stats["total_boxes"] > 0
        assert 0.0 <= stats["unique_box_ratio"] <= 1.0

    def test_merge_sft_v2_functions_accept_av2_records_once_scene_token_stamped(self):
        # build_av2_sft_record deliberately doesn't stamp scene_token/split —
        # that's the annotation-orchestration layer's job (mirrors how
        # regenerate_annotations_v2_colab.py adds it AFTER format_single_example
        # for nuScenes too). Demonstrate the merge functions work once a caller
        # does the one-line stamp, exactly as documented in AV2_INTEGRATION.md.
        merge = _load_module("scripts/merge_sft_v2.py", "merge_for_av2_test")
        records = []
        for i in range(6):
            rec = build_av2_sft_record(f"/fake/{i}.jpg", _cuboids(), _camera(),
                                       frame_id=f"log1_{i}")
            rec["scene_token"] = "log1"  # AV2's log_id is the scene_token analogue
            records.append(rec)
        merge.require_keys(records)  # must not raise
        deduped = merge.dedup_by_frame_id(records)
        assert len(deduped) == 6
        merge.assign_scene_split(deduped, seed=42)
        merge.verify_no_scene_leak(deduped)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
