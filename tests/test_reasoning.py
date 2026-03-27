"""Tests for Phase 2b: Level 2 Reasoning Quality Evaluation.

All tests use MockLLMJudge exclusively — no API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.eval.reasoning import (  # noqa: E402
    MockLLMJudge,
    ReasoningEvaluator,
    _build_judge_prompt,
    compute_reasoning_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIMS = ["correctness", "completeness", "action_relevance"]

_ANNOTATION = {
    "hazards": [{
        "bbox_2d": [100, 200, 400, 600],
        "label": "jaywalking",
        "severity": "high",
        "reasoning": "Pedestrian crossing mid-block outside crosswalk.",
        "action": "Apply brakes and yield.",
    }],
    "scene_summary": "Urban intersection with pedestrian activity.",
    "ego_context": {"weather": "clear", "time_of_day": "day", "road_type": "urban"},
}


def _pred(frame_id: str = "f1", hazards: list | None = None) -> dict:
    return {
        "frame_id": frame_id,
        "hazards": hazards if hazards is not None else _ANNOTATION["hazards"],
        "scene_summary": _ANNOTATION["scene_summary"],
    }


def _gt(frame_id: str = "f1") -> dict:
    return {"frame_id": frame_id, **_ANNOTATION}


# ---------------------------------------------------------------------------
# LLM judge prompt tests
# ---------------------------------------------------------------------------


class TestJudgePrompt:
    def test_prompt_contains_gt_content(self) -> None:
        """Judge prompt includes key GT fields."""
        prompt = _build_judge_prompt(_pred(), _gt(), "correctness", "Is it correct?")
        assert "jaywalking" in prompt
        assert "GROUND TRUTH" in prompt

    def test_prompt_contains_prediction(self) -> None:
        """Judge prompt includes prediction content."""
        pred = _pred("f1", [{"bbox_2d": [0, 0, 100, 100], "label": "high_density",
                              "severity": "low", "reasoning": "x", "action": "y"}])
        prompt = _build_judge_prompt(pred, _gt(), "correctness", "Is it correct?")
        assert "MODEL PREDICTION" in prompt
        assert "high_density" in prompt

    def test_prompt_contains_dimension(self) -> None:
        """Judge prompt names the dimension being evaluated."""
        prompt = _build_judge_prompt(_pred(), _gt(), "action_relevance", "Is action relevant?")
        assert "action_relevance" in prompt

    def test_prompt_contains_criteria(self) -> None:
        """Judge prompt includes the dimension criteria text."""
        criteria = "Custom criteria for this test."
        prompt = _build_judge_prompt(_pred(), _gt(), "correctness", criteria)
        assert criteria in prompt


# ---------------------------------------------------------------------------
# MockLLMJudge tests
# ---------------------------------------------------------------------------


class TestMockLLMJudge:
    def test_judge_single_returns_valid_score(self) -> None:
        """judge_single returns score in [1, 5] for all dimensions."""
        judge = MockLLMJudge()
        for dim in _DIMS:
            result = judge.judge_single(_pred(), _gt(), dim)
            assert result["score"] in range(1, 6)
            assert result["dimension"] == dim
            assert "justification" in result

    def test_judge_single_all_dimensions(self) -> None:
        """judge_single works for each standard dimension."""
        judge = MockLLMJudge()
        for dim in _DIMS:
            result = judge.judge_single(_pred(), _gt(), dim)
            assert isinstance(result["score"], int)

    def test_judge_batch_returns_all_dims(self) -> None:
        """judge_batch returns scores for all dimensions per example."""
        judge = MockLLMJudge()
        preds = [_pred("f1"), _pred("f2")]
        gts = [_gt("f1"), _gt("f2")]
        results = judge.judge_batch(preds, gts)
        assert len(results) == 2
        for r in results:
            assert "scores" in r
            for dim in _DIMS:
                assert dim in r["scores"]
                assert r["scores"][dim]["score"] in range(1, 6)

    def test_judge_batch_respects_custom_dimensions(self) -> None:
        """judge_batch only scores specified dimensions."""
        judge = MockLLMJudge()
        results = judge.judge_batch([_pred()], [_gt()], dimensions=["correctness"])
        assert "correctness" in results[0]["scores"]
        assert "completeness" not in results[0]["scores"]

    def test_judge_batch_preserves_frame_ids(self) -> None:
        """judge_batch result includes frame_id from predictions."""
        judge = MockLLMJudge()
        results = judge.judge_batch([_pred("frame_42")], [_gt("frame_42")])
        assert results[0]["frame_id"] == "frame_42"


# ---------------------------------------------------------------------------
# compute_reasoning_metrics tests
# ---------------------------------------------------------------------------


class TestComputeReasoningMetrics:
    def _make_results(self, scores_per_frame: list[dict]) -> list[dict]:
        """Build judge_results list from per-frame score maps."""
        return [
            {"frame_id": f"f{i}", "scores": {
                dim: {"score": scores_per_frame[i].get(dim, 3), "justification": ""}
                for dim in _DIMS
            }}
            for i in range(len(scores_per_frame))
        ]

    def test_mean_score_calculation(self) -> None:
        """Mean score computed correctly from known values."""
        results = self._make_results([
            {"correctness": 4, "completeness": 3, "action_relevance": 5},
            {"correctness": 2, "completeness": 4, "action_relevance": 3},
        ])
        m = compute_reasoning_metrics(results)
        assert m["correctness"]["mean"] == pytest.approx(3.0)
        assert m["completeness"]["mean"] == pytest.approx(3.5)
        assert m["action_relevance"]["mean"] == pytest.approx(4.0)

    def test_overall_score(self) -> None:
        """Overall score is mean across all dimensions and examples."""
        # All scores = 4 → overall = 4.0
        results = self._make_results([
            {"correctness": 4, "completeness": 4, "action_relevance": 4}
        ])
        m = compute_reasoning_metrics(results)
        assert m["overall_score"] == pytest.approx(4.0)

    def test_pass_rate_all_above_threshold(self) -> None:
        """All examples ≥ 3.5 on every dim → pass_rate = 1.0."""
        results = self._make_results([
            {"correctness": 4, "completeness": 4, "action_relevance": 4},
            {"correctness": 5, "completeness": 5, "action_relevance": 5},
        ])
        m = compute_reasoning_metrics(results)
        assert m["pass_rate"] == pytest.approx(1.0)

    def test_pass_rate_some_below_threshold(self) -> None:
        """Examples below 3.5 on any dim don't pass."""
        # f1: all >= 3.5 (passes)
        # f2: completeness = 2 (fails)
        results = self._make_results([
            {"correctness": 4, "completeness": 4, "action_relevance": 4},
            {"correctness": 4, "completeness": 2, "action_relevance": 4},
        ])
        m = compute_reasoning_metrics(results)
        assert m["pass_rate"] == pytest.approx(0.5)

    def test_distribution_counts(self) -> None:
        """Score distribution counts are correct."""
        results = self._make_results([
            {"correctness": 3, "completeness": 3, "action_relevance": 3},
            {"correctness": 5, "completeness": 5, "action_relevance": 5},
        ])
        m = compute_reasoning_metrics(results)
        dist = m["correctness"]["distribution"]
        assert dist[3] == 1
        assert dist[5] == 1
        assert dist[1] == 0
        assert dist[2] == 0

    def test_empty_results(self) -> None:
        """Empty judge results → zero metrics, no crash."""
        m = compute_reasoning_metrics([])
        assert m["overall_score"] == 0.0
        assert m["total_judged"] == 0
        assert m["pass_rate"] == 0.0

    def test_metrics_keys_present(self) -> None:
        """All expected top-level keys are present."""
        judge = MockLLMJudge()
        results = judge.judge_batch([_pred()], [_gt()])
        m = compute_reasoning_metrics(results)
        expected = {
            "correctness", "completeness", "action_relevance",
            "overall_score", "pass_rate", "total_judged", "judge_failures",
        }
        assert expected.issubset(set(m.keys()))

    def test_per_dimension_stats_keys(self) -> None:
        """Each dimension dict has mean, median, std, distribution."""
        judge = MockLLMJudge()
        results = judge.judge_batch([_pred()], [_gt()])
        m = compute_reasoning_metrics(results)
        for dim in _DIMS:
            d = m[dim]
            assert "mean" in d
            assert "median" in d
            assert "std" in d
            assert "distribution" in d


# ---------------------------------------------------------------------------
# ReasoningEvaluator end-to-end test
# ---------------------------------------------------------------------------


class TestReasoningEvaluatorEndToEnd:
    def test_full_pipeline_with_mock(self, tmp_path: Path) -> None:
        """Full evaluate + generate_report with MockLLMJudge."""
        cfg = {
            "output_dir": str(tmp_path / "eval"),
            "grounding": {"iou_threshold": 0.5},
            "reasoning": {
                "judge": {
                    "dimensions": _DIMS,
                    "max_concurrent": 1,
                },
            },
        }

        # Write predictions JSONL
        pred_path = tmp_path / "predictions.jsonl"
        pred_records = [
            {
                "frame_id": "f1",
                "raw_output": "",
                "parsed_output": {
                    "hazards": _ANNOTATION["hazards"],
                    "scene_summary": _ANNOTATION["scene_summary"],
                    "ego_context": _ANNOTATION["ego_context"],
                },
                "parse_success": True,
                "generation_time_ms": 100,
            }
        ]
        pred_path.write_text(
            "\n".join(json.dumps(r) for r in pred_records), encoding="utf-8"
        )

        # Write GT JSONL
        gt_path = tmp_path / "gt.jsonl"
        gt_path.write_text(
            json.dumps({
                "frame_id": "f1",
                "hazards": _ANNOTATION["hazards"],
                "scene_summary": _ANNOTATION["scene_summary"],
                "ego_context": _ANNOTATION["ego_context"],
            }),
            encoding="utf-8",
        )

        evaluator = ReasoningEvaluator(cfg, use_mock=True)
        metrics = evaluator.evaluate(pred_path, gt_path)

        assert metrics["total_judged"] == 1
        assert metrics["overall_score"] > 0

        # generate report
        report_path = evaluator.generate_report(metrics, tmp_path / "eval" / "level2")
        assert report_path.exists()
        assert (tmp_path / "eval" / "level2" / "reasoning_metrics.json").exists()
        assert (tmp_path / "eval" / "level2" / "per_dimension_scores.json").exists()
        assert (tmp_path / "eval" / "level2" / "judge_results_raw.json").exists()
