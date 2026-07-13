"""Tests for the annotation-v2 label validation gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT / "src", _ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_label_validation as gate  # noqa: E402


def _args(min_unique=0.5, max_freq=0.02, max_dup=5) -> argparse.Namespace:
    return argparse.Namespace(
        min_unique_ratio=min_unique, max_single_box_freq=max_freq, max_dup_frames=max_dup
    )


def _rec(fid: str, hazards: list[dict]) -> dict:
    return {"frame_id": fid, "hazards": hazards}


class TestGatePasses:
    def test_diverse_boxes_pass(self) -> None:
        records = [
            _rec(f"f{i}", [{"label": "jaywalking", "bbox_2d": [i, i, i + 100, i + 150]}])
            for i in range(50)
        ]
        stats = gate.collect_stats(records)
        assert stats["total_boxes"] == 50
        assert gate.evaluate_gate(stats, _args()) == []

    def test_small_all_unique_split_passes(self) -> None:
        # 8 unique boxes (a tiny val/test split): 1/8 = 0.125 must NOT fail,
        # because every box appears exactly once (top_box_count == 1).
        records = [
            _rec(f"f{i}", [{"label": "occluded_pedestrian", "bbox_2d": [i, i, i + 30, i + 80]}])
            for i in range(8)
        ]
        stats = gate.collect_stats(records)
        assert stats["unique_box_ratio"] == 1.0
        assert stats["top_box_count"] == 1
        assert stats["max_single_box_freq"] == 0.125  # 1/8
        assert gate.evaluate_gate(stats, _args()) == []  # passes despite 0.125 > 0.02

    def test_capped_duplicates_pass(self) -> None:
        # A static object across 3 kept keyframes (== dedup cap, <= max_dup_frames)
        # is benign and must PASS even though 3/N is a big ratio on a small split.
        dup = [253, 506, 277, 583]
        records = [_rec(f"d{i}", [{"label": "construction_zone", "bbox_2d": dup}]) for i in range(3)]
        records += [_rec(f"u{i}", [{"label": "occluded_pedestrian", "bbox_2d": [i, i, i + 30, i + 80]}])
                    for i in range(10)]
        stats = gate.collect_stats(records)
        assert stats["top_box_count"] == 3
        assert gate.evaluate_gate(stats, _args()) == []   # 3 <= max_dup_frames(5) → benign

    def test_high_density_box_exempt_ignored(self) -> None:
        records = [
            _rec("f1", [{"label": "high_density"}]),  # no bbox, legitimately
            _rec("f2", [{"label": "jaywalking", "bbox_2d": [10, 10, 120, 160]}]),
        ]
        stats = gate.collect_stats(records)
        assert stats["total_boxes"] == 1  # high_density excluded
        assert stats["exempt_labels_with_box"] == 0


class TestGateFails:
    def test_constant_box_collapse_fails(self) -> None:
        # The v1 signature: same box on every frame.
        records = [
            _rec(f"f{i}", [{"label": "occluded_pedestrian", "bbox_2d": [400, 350, 450, 500]}])
            for i in range(40)
        ]
        stats = gate.collect_stats(records)
        failures = gate.evaluate_gate(stats, _args())
        assert stats["unique_box_ratio"] < 0.5
        assert stats["max_single_box_freq"] > 0.02
        assert any("unique_box_ratio" in f for f in failures)
        assert any("max_single_box_freq" in f for f in failures)
        assert any("frames" in f for f in failures)  # cross-frame dup

    def test_oversized_box_fails(self) -> None:
        records = [_rec("f1", [{"label": "jaywalking", "bbox_2d": [0, 0, 1000, 1000]}])]
        stats = gate.collect_stats(records)
        assert stats["oversized_gt_40pct"] == 1
        assert any("40% frame area" in f for f in gate.evaluate_gate(stats, _args()))

    def test_no_hazard_with_box_fails(self) -> None:
        records = [_rec("f1", [{"label": "no_hazard", "bbox_2d": [0, 0, 1000, 1000]}])]
        stats = gate.collect_stats(records)
        assert stats["exempt_labels_with_box"] == 1
        assert any("carry a bbox" in f for f in gate.evaluate_gate(stats, _args()))

    def test_sft_messages_format_supported(self) -> None:
        import json
        ann = {"hazards": [{"label": "jaywalking", "bbox_2d": [400, 350, 450, 500]}]}
        records = [
            {"frame_id": f"f{i}",
             "messages": [{"role": "assistant", "content": json.dumps(ann)}]}
            for i in range(30)
        ]
        stats = gate.collect_stats(records)
        assert stats["total_boxes"] == 30
        assert stats["max_single_box_freq"] == 1.0  # all identical → fails
        assert gate.evaluate_gate(stats, _args()) != []
