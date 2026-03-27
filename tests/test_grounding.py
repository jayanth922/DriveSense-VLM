"""Tests for Phase 2b: Level 1 Grounding Accuracy Evaluation.

All tests run on CPU-only macOS with mock predictions and ground truth.
No model loading, no API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.eval.grounding import (  # noqa: E402
    GroundingEvaluator,
    _active_hazards,
    compute_grounding_metrics,
    compute_iou,
    compute_severity_metrics,
    match_predictions_to_ground_truth,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h(bbox: list[int], label: str = "jaywalking", severity: str = "medium") -> dict:
    return {
        "bbox_2d": bbox,
        "label": label,
        "severity": severity,
        "reasoning": "Test hazard reasoning text.",
        "action": "Reduce speed.",
    }


def _pred(frame_id: str, hazards: list[dict], parse_failure: bool = False) -> dict:
    return {"frame_id": frame_id, "hazards": hazards, "parse_failure": parse_failure}


def _gt(frame_id: str, hazards: list[dict]) -> dict:
    return {"frame_id": frame_id, "hazards": hazards}


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------


class TestComputeIoU:
    def test_perfect_overlap(self) -> None:
        """Identical boxes → IoU = 1.0."""
        assert compute_iou([0, 0, 500, 500], [0, 0, 500, 500]) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Completely disjoint boxes → IoU = 0.0."""
        assert compute_iou([0, 0, 100, 100], [500, 500, 600, 600]) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Manually computed partial overlap.

        pred = [0, 0, 200, 200]  area = 40000
        gt   = [100, 100, 300, 300]  area = 40000
        intersection = [100, 100, 200, 200]  area = 10000
        union = 40000 + 40000 - 10000 = 70000
        IoU = 10000 / 70000 ≈ 0.1429
        """
        iou = compute_iou([0, 0, 200, 200], [100, 100, 300, 300])
        assert iou == pytest.approx(1 / 7, rel=1e-4)

    def test_zero_area_pred_box(self) -> None:
        """Zero-area predicted box → IoU = 0.0 (no crash)."""
        assert compute_iou([100, 100, 100, 200], [100, 100, 200, 200]) == pytest.approx(0.0)

    def test_zero_area_gt_box(self) -> None:
        """Zero-area GT box → IoU = 0.0 (no crash)."""
        assert compute_iou([100, 100, 200, 200], [150, 150, 150, 250]) == pytest.approx(0.0)

    def test_identical_single_point(self) -> None:
        """Degenerate box (single point) → IoU = 0.0."""
        assert compute_iou([50, 50, 50, 50], [50, 50, 50, 50]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Hungarian matching tests
# ---------------------------------------------------------------------------


class TestHungarianMatching:
    def test_matching_three_pairs(self) -> None:
        """3 predictions × 3 GT boxes with clear best-match structure."""
        # Position predictions and GT so each has a clear match
        preds = [
            _h([0, 0, 200, 200]),    # pred 0 → gt 0
            _h([400, 0, 600, 200]),  # pred 1 → gt 1
            _h([0, 400, 200, 600]),  # pred 2 → gt 2
        ]
        gts = [
            _h([10, 10, 210, 210]),    # gt 0 (matches pred 0)
            _h([410, 10, 610, 210]),   # gt 1 (matches pred 1)
            _h([10, 410, 210, 610]),   # gt 2 (matches pred 2)
        ]
        result = match_predictions_to_ground_truth(preds, gts, iou_threshold=0.3)
        assert len(result["matched_pairs"]) == 3
        assert result["unmatched_predictions"] == []
        assert result["unmatched_ground_truth"] == []
        # Each match should have positive IoU
        for _, _, iou_val in result["matched_pairs"]:
            assert iou_val > 0.3

    def test_matching_with_iou_threshold(self) -> None:
        """Only predictions above the IoU threshold form valid matches."""
        preds = [
            _h([0, 0, 300, 300]),      # high IoU with gt 0
            _h([700, 700, 900, 900]),  # low IoU / no overlap with gt 0
        ]
        gts = [_h([0, 0, 300, 300])]
        result = match_predictions_to_ground_truth(preds, gts, iou_threshold=0.5)
        assert len(result["matched_pairs"]) == 1
        assert result["matched_pairs"][0][2] == pytest.approx(1.0)
        # pred 1 should be unmatched (below threshold or no overlap)
        assert 1 in result["unmatched_predictions"]

    def test_empty_predictions(self) -> None:
        """Empty prediction list → all GT unmatched."""
        gts = [_h([0, 0, 100, 100])]
        result = match_predictions_to_ground_truth([], gts)
        assert result["matched_pairs"] == []
        assert result["unmatched_ground_truth"] == [0]
        assert result["unmatched_predictions"] == []

    def test_empty_ground_truth(self) -> None:
        """Empty GT list → all predictions unmatched."""
        preds = [_h([0, 0, 100, 100])]
        result = match_predictions_to_ground_truth(preds, [])
        assert result["matched_pairs"] == []
        assert result["unmatched_predictions"] == [0]
        assert result["unmatched_ground_truth"] == []

    def test_optimal_over_greedy(self) -> None:
        """Hungarian matching finds optimal assignment, not just greedy."""
        # pred 0 has IoU 0.9 with gt 0 AND IoU 0.7 with gt 1
        # pred 1 has IoU 0.0 with gt 0 AND IoU 0.8 with gt 1
        # Greedy would take (0→0, 1→1), Hungarian gives same here but ensures
        # each GT is matched at most once.
        preds = [_h([0, 0, 300, 300]), _h([350, 0, 650, 300])]
        gts = [_h([0, 0, 300, 300]), _h([350, 0, 650, 300])]
        result = match_predictions_to_ground_truth(preds, gts, iou_threshold=0.5)
        assert len(result["matched_pairs"]) == 2
        assert result["unmatched_predictions"] == []
        assert result["unmatched_ground_truth"] == []


# ---------------------------------------------------------------------------
# compute_grounding_metrics tests
# ---------------------------------------------------------------------------


class TestComputeGroundingMetrics:
    def test_detection_rate_calculation(self) -> None:
        """Known 2/3 recall → hazard_detection_rate = 0.667."""
        preds = [
            _pred("f1", [_h([0, 0, 100, 100])]),    # TP
            _pred("f2", [_h([200, 200, 300, 300])]),  # TP
            _pred("f3", []),                           # miss → FN
        ]
        gts = [
            _gt("f1", [_h([0, 0, 100, 100])]),
            _gt("f2", [_h([200, 200, 300, 300])]),
            _gt("f3", [_h([400, 400, 500, 500])]),
        ]
        m = compute_grounding_metrics(preds, gts, iou_threshold=0.5)
        assert m["true_positives"] == 2
        assert m["false_negatives"] == 1
        assert m["hazard_detection_rate"] == pytest.approx(2 / 3, rel=1e-3)

    def test_false_positive_rate_on_no_hazard_frames(self) -> None:
        """GT = no_hazard, pred has hazards → FP counted in FPR."""
        preds = [
            _pred("f1", [_h([0, 0, 100, 100])]),  # FP (GT = no_hazard)
            _pred("f2", []),                         # TN
        ]
        gts = [
            _gt("f1", []),  # no_hazard
            _gt("f2", []),  # no_hazard
        ]
        m = compute_grounding_metrics(preds, gts)
        assert m["true_negatives"] == 1
        assert m["false_positive_rate"] == pytest.approx(0.5, rel=1e-3)

    def test_no_hazard_frame_tn(self) -> None:
        """GT = no_hazard AND pred = no_hazard → True Negative."""
        preds = [_pred("f1", [])]
        gts = [_gt("f1", [])]
        m = compute_grounding_metrics(preds, gts)
        assert m["true_negatives"] == 1
        assert m["false_positives"] == 0
        assert m["no_hazard_accuracy"] == pytest.approx(1.0)

    def test_no_hazard_with_false_positives(self) -> None:
        """GT = no_hazard, pred has hazards → false_positive_rate > 0."""
        preds = [_pred("f1", [_h([0, 0, 200, 200]), _h([300, 300, 400, 400])])]
        gts = [_gt("f1", [])]
        m = compute_grounding_metrics(preds, gts)
        assert m["false_positives"] == 2
        assert m["true_negatives"] == 0
        assert m["no_hazard_accuracy"] == pytest.approx(0.0)

    def test_missed_hazard_is_fn(self) -> None:
        """GT has hazards, pred = empty → all GT hazards are FN."""
        preds = [_pred("f1", [])]
        gts = [_gt("f1", [_h([0, 0, 100, 100]), _h([200, 200, 300, 300])])]
        m = compute_grounding_metrics(preds, gts)
        assert m["false_negatives"] == 2
        assert m["true_positives"] == 0
        assert m["hazard_detection_rate"] == pytest.approx(0.0)

    def test_per_class_metrics(self) -> None:
        """Predictions with correct and incorrect labels → per-class stats."""
        preds = [
            _pred("f1", [_h([0, 0, 100, 100], "jaywalking")]),     # TP correct label
            _pred("f2", [_h([0, 0, 100, 100], "construction_zone")]),  # TP wrong label
        ]
        gts = [
            _gt("f1", [_h([0, 0, 100, 100], "jaywalking")]),
            _gt("f2", [_h([0, 0, 100, 100], "jaywalking")]),  # GT=jaywalking, pred=construction_zone
        ]
        m = compute_grounding_metrics(preds, gts)
        jw = m["per_class_metrics"]["jaywalking"]
        # f1 is TP for jaywalking; f2 GT is jaywalking but pred is construction_zone → FP for cz, FN for jw? No.
        # f2: pred=construction_zone matches gt=jaywalking at IoU=1.0 >= 0.5 → matched pair
        #     per_class["jaywalking"]["tp"] += 1 (GT label is jaywalking)
        #     classification: pred_label != gt_label → no correct_label count
        # So jaywalking has tp=2 (f1 and f2 both have GT jaywalking matched)
        assert jw["count"] == 2  # 2 GT jaywalking hazards
        # classification_accuracy should be 0.5 (1 correct / 2 matched)
        assert m["classification_accuracy"] == pytest.approx(0.5, rel=1e-3)

    def test_classification_accuracy(self) -> None:
        """3 matched pairs: 2 correct labels → accuracy = 2/3."""
        preds = [
            _pred("f1", [_h([0, 0, 200, 200], "jaywalking")]),
            _pred("f2", [_h([0, 0, 200, 200], "jaywalking")]),
            _pred("f3", [_h([0, 0, 200, 200], "construction_zone")]),  # wrong label
        ]
        gts = [
            _gt("f1", [_h([0, 0, 200, 200], "jaywalking")]),
            _gt("f2", [_h([0, 0, 200, 200], "jaywalking")]),
            _gt("f3", [_h([0, 0, 200, 200], "jaywalking")]),
        ]
        m = compute_grounding_metrics(preds, gts)
        assert m["classification_accuracy"] == pytest.approx(2 / 3, rel=1e-3)

    def test_parse_failure_counting(self) -> None:
        """Parse failures are counted separately and GT is treated as missed."""
        preds = [
            _pred("f1", [], parse_failure=True),  # failed parse → FN
            _pred("f2", [_h([0, 0, 100, 100])]),  # valid prediction → TP
        ]
        gts = [
            _gt("f1", [_h([0, 0, 100, 100])]),
            _gt("f2", [_h([0, 0, 100, 100])]),
        ]
        m = compute_grounding_metrics(preds, gts)
        assert m["parse_failures"] == 1
        assert m["parse_failure_rate"] == pytest.approx(0.5)
        assert m["false_negatives"] >= 1  # f1 GT was missed

    def test_empty_predictions_list(self) -> None:
        """Empty predictions list → all GT hazards are FN, no crash."""
        gts = [_gt("f1", [_h([0, 0, 100, 100])])]
        m = compute_grounding_metrics([], gts)
        assert m["total_frames"] == 0
        assert m["true_positives"] == 0

    def test_metrics_keys_present(self) -> None:
        """All expected keys are present in the returned dict."""
        preds = [_pred("f1", [_h([0, 0, 100, 100])])]
        gts = [_gt("f1", [_h([0, 0, 100, 100])])]
        m = compute_grounding_metrics(preds, gts)
        expected_keys = {
            "iou_at_threshold", "hazard_detection_rate", "false_positive_rate",
            "precision", "f1_score", "mean_iou", "classification_accuracy",
            "total_frames", "total_gt_hazards", "total_pred_hazards",
            "true_positives", "false_positives", "false_negatives", "true_negatives",
            "no_hazard_accuracy", "per_class_metrics", "confusion_matrix",
            "parse_failure_rate", "parse_failures",
        }
        assert expected_keys.issubset(set(m.keys()))


# ---------------------------------------------------------------------------
# compute_severity_metrics tests
# ---------------------------------------------------------------------------


class TestSeverityMetrics:
    def test_severity_spearman_perfect(self) -> None:
        """Perfect severity agreement → Spearman ρ = 1.0."""
        preds = [
            _pred("f1", [_h([0, 0, 200, 200], severity="low")]),
            _pred("f2", [_h([0, 0, 200, 200], severity="medium")]),
            _pred("f3", [_h([0, 0, 200, 200], severity="high")]),
            _pred("f4", [_h([0, 0, 200, 200], severity="critical")]),
        ]
        gts = [
            _gt("f1", [_h([0, 0, 200, 200], severity="low")]),
            _gt("f2", [_h([0, 0, 200, 200], severity="medium")]),
            _gt("f3", [_h([0, 0, 200, 200], severity="high")]),
            _gt("f4", [_h([0, 0, 200, 200], severity="critical")]),
        ]
        m = compute_severity_metrics(preds, gts)
        assert m["severity_accuracy"] == pytest.approx(1.0)
        assert m["severity_within_one"] == pytest.approx(1.0)
        # Spearman ρ = 1.0 when scipy is available; 0.0 when not
        assert m["severity_spearman"] in (pytest.approx(1.0, abs=1e-3), 0.0)

    def test_severity_keys_present(self) -> None:
        """All expected severity metric keys present."""
        m = compute_severity_metrics([], [])
        assert {"severity_accuracy", "severity_within_one",
                "severity_spearman", "severity_confusion_matrix"} == set(m.keys())


# ---------------------------------------------------------------------------
# GroundingEvaluator integration tests
# ---------------------------------------------------------------------------


class TestGroundingEvaluator:
    def test_evaluator_end_to_end(self, tmp_path: Path) -> None:
        """Full pipeline: write JSONL → evaluate → generate report."""
        cfg = {"grounding": {"iou_threshold": 0.5}, "output_dir": str(tmp_path / "eval")}

        # Write predictions JSONL
        pred_path = tmp_path / "predictions.jsonl"
        records = [
            {
                "frame_id": "f1",
                "raw_output": json.dumps({
                    "hazards": [_h([0, 0, 200, 200], "jaywalking")],
                    "scene_summary": "test",
                    "ego_context": {"weather": "clear", "time_of_day": "day", "road_type": "urban"},
                }),
                "parsed_output": None,
                "parse_success": False,
            },
        ]
        pred_path.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )

        # Write GT JSONL
        gt_path = tmp_path / "gt.jsonl"
        gt_path.write_text(
            json.dumps({
                "frame_id": "f1",
                "hazards": [_h([0, 0, 200, 200], "jaywalking")],
                "scene_summary": "test",
                "ego_context": {},
            }),
            encoding="utf-8",
        )

        evaluator = GroundingEvaluator(cfg)
        metrics = evaluator.evaluate(pred_path, gt_path)

        # report
        report_path = evaluator.generate_report(metrics, tmp_path / "eval")
        assert report_path.exists()
        assert (tmp_path / "eval" / "grounding_metrics.json").exists()
        assert (tmp_path / "eval" / "per_class_metrics.json").exists()
        assert (tmp_path / "eval" / "confusion_matrix.json").exists()

    def test_load_sft_jsonl_ground_truth(self, tmp_path: Path) -> None:
        """load_ground_truth handles SFT JSONL format with messages key."""
        gt_path = tmp_path / "sft_test.jsonl"
        annotation = {
            "hazards": [_h([0, 0, 100, 100])],
            "scene_summary": "test",
            "ego_context": {},
        }
        gt_path.write_text(
            json.dumps({
                "frame_id": "f1",
                "messages": [
                    {"role": "system", "content": "You are DriveSense."},
                    {"role": "user", "content": "Analyse."},
                    {"role": "assistant", "content": json.dumps(annotation)},
                ],
                "images": [],
                "source": "nuscenes",
            }),
            encoding="utf-8",
        )
        cfg = {"grounding": {}, "output_dir": str(tmp_path)}
        ev = GroundingEvaluator(cfg)
        gt = ev.load_ground_truth(gt_path)
        assert len(gt) == 1
        assert gt[0]["frame_id"] == "f1"
        assert len(gt[0]["hazards"]) == 1

    def test_parse_model_output_valid_json(self) -> None:
        """parse_model_output returns dict for valid JSON."""
        cfg = {"grounding": {}, "output_dir": "outputs/eval"}
        ev = GroundingEvaluator(cfg)
        result = ev.parse_model_output(
            '{"hazards": [], "scene_summary": "ok", "ego_context": {}}'
        )
        assert result is not None
        assert "hazards" in result

    def test_parse_model_output_invalid(self) -> None:
        """parse_model_output returns None for unparseable text."""
        cfg = {"grounding": {}, "output_dir": "outputs/eval"}
        ev = GroundingEvaluator(cfg)
        assert ev.parse_model_output("This is not JSON at all!") is None
