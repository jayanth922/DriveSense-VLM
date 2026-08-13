"""Tests for drivesense.eval.failure_stratification and
scripts/analyze_failure_stratification.py. Pure data/logic — no GPU/torch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.eval.failure_stratification import (  # noqa: E402
    aspect_ratio,
    box_area_pct,
    build_hazard_rows,
    build_report,
    compute_bucket_metrics,
    cross_tabulate,
    load_stratified_ground_truth,
    rank_buckets,
    size_tier_of,
)


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gt(fid: str, hazards: list[dict], weather="clear", time_of_day="day",
       location="urban") -> dict:
    return {"frame_id": fid, "weather": weather, "time_of_day": time_of_day,
           "location": location, "hazards": hazards}


def _pred(fid: str, hazards: list[dict], parse_failure: bool = False) -> dict:
    return {"frame_id": fid, "hazards": hazards, "parse_failure": parse_failure}


def _haz(label: str, bbox: list[float]) -> dict:
    return {"label": label, "bbox_2d": bbox}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_box_area_pct(self):
        # A box covering 10% width x 10% height of a 1000x1000 frame = 1% area.
        assert box_area_pct([0, 0, 100, 100]) == pytest.approx(1.0)
        assert box_area_pct([0, 0, 1000, 1000]) == pytest.approx(100.0)

    def test_aspect_ratio(self):
        assert aspect_ratio([0, 0, 200, 100]) == pytest.approx(2.0)
        assert aspect_ratio([0, 0, 100, 100]) == pytest.approx(1.0)

    def test_aspect_ratio_degenerate_is_zero(self):
        assert aspect_ratio([0, 0, 100, 0]) == 0.0

    @pytest.mark.parametrize("area,expected", [
        (0.0, "tiny"), (0.5, "tiny"), (0.999, "tiny"),
        (1.0, "small"), (3.0, "small"), (4.999, "small"),
        (5.0, "medium"), (10.0, "medium"), (14.999, "medium"),
        (15.0, "large"), (50.0, "large"), (100.0, "large"),
    ])
    def test_size_tier_boundaries(self, area, expected):
        assert size_tier_of(area) == expected


# ---------------------------------------------------------------------------
# Stratified GT loading — must preserve weather/time_of_day/location
# ---------------------------------------------------------------------------


class TestLoadStratifiedGroundTruth:
    def test_direct_format_preserves_strata(self, tmp_path):
        p = tmp_path / "gt.jsonl"
        rec = _gt("f1", [_haz("jaywalking", [0, 0, 100, 100])], weather="rain",
                 time_of_day="night", location="intersection")
        p.write_text(json.dumps(rec) + "\n")
        loaded = load_stratified_ground_truth(p)
        assert loaded[0]["weather"] == "rain"
        assert loaded[0]["time_of_day"] == "night"
        assert loaded[0]["location"] == "intersection"
        assert loaded[0]["hazards"][0]["label"] == "jaywalking"

    def test_sft_messages_format_preserves_strata(self, tmp_path):
        p = tmp_path / "gt.jsonl"
        ann = {"hazards": [{"label": "cyclist_proximity", "bbox_2d": [10, 10, 50, 90]}]}
        rec = {"frame_id": "f2", "weather": "fog", "time_of_day": "day",
               "location": "urban",
               "messages": [{"role": "assistant", "content": json.dumps(ann)}]}
        p.write_text(json.dumps(rec) + "\n")
        loaded = load_stratified_ground_truth(p)
        assert loaded[0]["weather"] == "fog"
        assert loaded[0]["hazards"][0]["label"] == "cyclist_proximity"

    def test_missing_strata_default_unknown(self, tmp_path):
        p = tmp_path / "gt.jsonl"
        p.write_text(json.dumps({"frame_id": "f3", "hazards": []}) + "\n")
        loaded = load_stratified_ground_truth(p)
        assert loaded[0]["weather"] == "unknown"


# ---------------------------------------------------------------------------
# build_hazard_rows — reuses compute_iou; per-hazard, not per-frame
# ---------------------------------------------------------------------------


class TestBuildHazardRows:
    def test_best_iou_is_max_over_all_pred_pairs_in_frame(self):
        gt = [_gt("f1", [_haz("jaywalking", [100, 100, 200, 200])])]
        preds = [_pred("f1", [
            _haz("jaywalking", [500, 500, 600, 600]),  # no overlap -> iou 0
            _haz("jaywalking", [100, 100, 200, 200]),  # perfect overlap -> iou 1
        ])]
        rows = build_hazard_rows(preds, gt)
        assert len(rows) == 1
        assert rows[0]["best_iou"] == pytest.approx(1.0)

    def test_missing_frame_in_predictions_gives_zero_iou(self):
        gt = [_gt("f1", [_haz("jaywalking", [0, 0, 100, 100])])]
        rows = build_hazard_rows([], gt)
        assert rows[0]["best_iou"] == 0.0

    def test_parse_failure_prediction_gives_zero_iou(self):
        gt = [_gt("f1", [_haz("jaywalking", [0, 0, 100, 100])])]
        preds = [_pred("f1", [_haz("jaywalking", [0, 0, 100, 100])], parse_failure=True)]
        rows = build_hazard_rows(preds, gt)
        assert rows[0]["best_iou"] == 0.0

    def test_box_exempt_labels_excluded(self):
        gt = [_gt("f1", [{"label": "high_density"}, _haz("jaywalking", [0, 0, 100, 100])])]
        rows = build_hazard_rows([], gt)
        assert len(rows) == 1  # high_density has no bbox and is dropped
        assert rows[0]["label"] == "jaywalking"

    def test_hazard_without_bbox_is_skipped(self):
        gt = [_gt("f1", [{"label": "jaywalking"}])]  # no bbox_2d
        rows = build_hazard_rows([], gt)
        assert rows == []

    def test_size_tier_and_area_attached(self):
        gt = [_gt("f1", [_haz("jaywalking", [0, 0, 1000, 1000])])]  # 100% area -> large
        rows = build_hazard_rows([], gt)
        assert rows[0]["size_tier"] == "large"
        assert rows[0]["area_pct"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# compute_bucket_metrics / cross_tabulate / rank_buckets
# ---------------------------------------------------------------------------


class TestBucketMetrics:
    def test_empty_rows(self):
        m = compute_bucket_metrics([])
        assert m["n"] == 0
        assert m["mean_best_pair_iou"] == 0.0

    def test_detection_rate_by_iou(self):
        rows = [{"best_iou": 0.6}, {"best_iou": 0.2}, {"best_iou": 0.0}, {"best_iou": 0.4}]
        m = compute_bucket_metrics(rows, thresholds=(0.1, 0.3, 0.5))
        assert m["n"] == 4
        assert m["mean_best_pair_iou"] == pytest.approx(0.3)
        assert m["detection_rate_by_iou"]["0.1"] == pytest.approx(0.75)  # 3/4 >= 0.1
        assert m["detection_rate_by_iou"]["0.3"] == pytest.approx(0.5)   # 2/4 >= 0.3
        assert m["detection_rate_by_iou"]["0.5"] == pytest.approx(0.25)  # 1/4 >= 0.5


class TestCrossTabulate:
    def test_produces_cell_and_marginal_rows(self):
        rows = [
            {"size_tier": "tiny", "weather": "clear", "best_iou": 0.5},
            {"size_tier": "tiny", "weather": "rain", "best_iou": 0.0},
            {"size_tier": "large", "weather": "clear", "best_iou": 0.9},
        ]
        tab = cross_tabulate(rows, "weather")
        assert tab["tiny|clear"]["n"] == 1
        assert tab["tiny|rain"]["n"] == 1
        assert tab["tiny|ALL"]["n"] == 2       # marginal over weather, fixed tier
        assert tab["ALL|clear"]["n"] == 2      # marginal over tier, fixed weather
        assert "small|ALL" in tab              # present even with n=0
        assert tab["small|ALL"]["n"] == 0


class TestRankBuckets:
    def test_worst_is_lowest_iou_best_is_highest(self):
        tab = {
            "a": {"n": 10, "mean_best_pair_iou": 0.1, "detection_rate_by_iou": {}},
            "b": {"n": 10, "mean_best_pair_iou": 0.9, "detection_rate_by_iou": {}},
            "c": {"n": 10, "mean_best_pair_iou": 0.5, "detection_rate_by_iou": {}},
        }
        worst, best = rank_buckets(tab, min_samples=1, top_n=2)
        assert [w["bucket"] for w in worst] == ["a", "c"]
        assert [b["bucket"] for b in best] == ["b", "c"]

    def test_min_samples_excludes_small_buckets(self):
        tab = {
            "tiny_but_noisy": {"n": 2, "mean_best_pair_iou": 0.0, "detection_rate_by_iou": {}},
            "reliable": {"n": 50, "mean_best_pair_iou": 0.3, "detection_rate_by_iou": {}},
        }
        worst, _ = rank_buckets(tab, min_samples=5, top_n=5)
        assert [w["bucket"] for w in worst] == ["reliable"]


# ---------------------------------------------------------------------------
# build_report — the exact diagnostic scenario: "large" tier fails uniformly
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_worst_bucket_correctly_identifies_the_failing_size_tier(self):
        # "large" hazards never match (iou 0); "small" hazards match well.
        # Predictions echo GT exactly for small, are absent for large.
        gt, preds = [], []
        for i in range(10):
            gt.append(_gt(f"large{i}", [_haz("jaywalking", [0, 0, 500, 500])]))  # 25% area -> large
            preds.append(_pred(f"large{i}", []))  # never detected
        for i in range(10):
            box = [i, i, i + 50, i + 50]  # 0.25% area -> tiny
            gt.append(_gt(f"tiny{i}", [_haz("jaywalking", box)]))
            preds.append(_pred(f"tiny{i}", [_haz("jaywalking", box)]))  # perfect match

        report = build_report(preds, gt, dimensions=("weather",), min_samples=5)
        assert report["size_tier_summary"]["large"]["mean_best_pair_iou"] == 0.0
        assert report["size_tier_summary"]["tiny"]["mean_best_pair_iou"] == pytest.approx(1.0)
        # The #1 worst bucket must point at the large tier, not tiny.
        assert "large" in report["worst_buckets"][0]["bucket"]
        assert "tiny" in report["best_buckets"][0]["bucket"]

    def test_report_shape(self):
        gt = [_gt("f1", [_haz("jaywalking", [0, 0, 100, 100])])]
        preds = [_pred("f1", [_haz("jaywalking", [0, 0, 100, 100])])]
        report = build_report(preds, gt)
        assert set(report) >= {"n_frames", "n_hazards", "overall", "size_tier_summary",
                               "cross_tabs", "worst_buckets", "best_buckets"}
        assert report["n_frames"] == 1
        assert report["n_hazards"] == 1


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestAnalyzeCLI:
    def test_runs_end_to_end_and_writes_report(self, tmp_path, capsys):
        mod = _load_module("scripts/analyze_failure_stratification.py", "analyze_cli_test")
        gt_path = tmp_path / "gt.jsonl"
        pred_path = tmp_path / "preds.jsonl"
        gt_path.write_text("\n".join(
            json.dumps(_gt(f"f{i}", [_haz("jaywalking", [0, 0, 100, 100])])) for i in range(6)))
        pred_path.write_text("\n".join(
            json.dumps(_pred(f"f{i}", [_haz("jaywalking", [0, 0, 100, 100])])) for i in range(6)))
        out = tmp_path / "report.json"
        sys.argv = ["analyze", "--predictions", str(pred_path), "--ground-truth", str(gt_path),
                   "--output", str(out), "--min-samples", "1"]
        mod.main()
        assert "FAILURE STRATIFICATION REPORT" in capsys.readouterr().out
        assert json.loads(out.read_text())["n_hazards"] == 6

    def test_missing_file_exits_two(self, tmp_path):
        mod = _load_module("scripts/analyze_failure_stratification.py", "analyze_cli_test2")
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["analyze", "--predictions", str(tmp_path / "nope.jsonl"),
                       "--ground-truth", str(tmp_path / "nope2.jsonl")]
            mod.main()
        assert exc.value.code == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
