"""Tests for Level 4 Robustness evaluation (Phase 4b).

All tests use mocks — no GPU, no model loading, no API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rob_config() -> dict:
    """Minimal config for RobustnessEvaluator."""
    return {
        "robustness": {
            "stratify_by": ["time_of_day", "weather", "location", "ego_speed_bucket"],
            "targets": {
                "max_day_night_gap": 0.10,
                "max_weather_gap": 0.15,
                "max_location_gap": 0.10,
                "ood_relative_performance": 0.70,
            },
        }
    }


def _make_gt(
    frame_id: str,
    time_of_day: str = "day",
    weather: str = "clear",
    location: str = "boston",
    ego_speed_kmh: float = 25.0,
    source: str = "nuscenes",
    has_hazard: bool = True,
) -> dict:
    """Build a minimal GT annotation record."""
    hazards = (
        [{"label": "pedestrian_in_path", "bbox_2d": [100, 100, 300, 400], "severity": "high"}]
        if has_hazard else []
    )
    return {
        "frame_id": frame_id,
        "hazards": hazards,
        "ego_context": {"time_of_day": time_of_day, "weather": weather},
        "metadata": {"location": location, "ego_speed_kmh": ego_speed_kmh, "source": source},
        "source": source,
    }


def _make_pred(frame_id: str, has_hazard: bool = True) -> dict:
    """Build a minimal prediction record."""
    hazards = (
        [{"label": "pedestrian_in_path", "bbox_2d": [110, 110, 290, 390]}]
        if has_hazard else []
    )
    return {"frame_id": frame_id, "hazards": hazards, "parse_failure": False}


@pytest.fixture()
def day_night_data() -> tuple[list[dict], list[dict]]:
    """GT + predictions with day/night split."""
    gt = [
        _make_gt("d1", time_of_day="day"),
        _make_gt("d2", time_of_day="day"),
        _make_gt("d3", time_of_day="day"),
        _make_gt("n1", time_of_day="night"),
        _make_gt("n2", time_of_day="night"),
    ]
    preds = [
        _make_pred("d1"), _make_pred("d2"), _make_pred("d3"),
        _make_pred("n1"), _make_pred("n2"),
    ]
    return preds, gt


@pytest.fixture()
def source_data() -> tuple[list[dict], list[dict]]:
    """GT + predictions with nuscenes/dada split."""
    gt = [
        _make_gt("ns1", source="nuscenes"),
        _make_gt("ns2", source="nuscenes"),
        _make_gt("ns3", source="nuscenes"),
        _make_gt("da1", source="dada2000"),
        _make_gt("da2", source="dada2000"),
    ]
    preds = [
        _make_pred("ns1"), _make_pred("ns2"), _make_pred("ns3"),
        _make_pred("da1"), _make_pred("da2"),
    ]
    return preds, gt


# ---------------------------------------------------------------------------
# test_stratification_by_time
# ---------------------------------------------------------------------------


class TestStratificationByTime:
    """Verify day/night split is correct."""

    def test_groups_exist(self, rob_config: dict, day_night_data: tuple) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        preds, gt = day_night_data
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "time_of_day")
        assert "day" in groups
        assert "night" in groups

    def test_day_count(self, rob_config: dict, day_night_data: tuple) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        preds, gt = day_night_data
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "time_of_day")
        _, day_gt = groups["day"]
        assert len(day_gt) == 3

    def test_night_count(self, rob_config: dict, day_night_data: tuple) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        preds, gt = day_night_data
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "time_of_day")
        _, night_gt = groups["night"]
        assert len(night_gt) == 2

    def test_predictions_matched_by_frame_id(
        self, rob_config: dict, day_night_data: tuple
    ) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        preds, gt = day_night_data
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "time_of_day")
        day_preds, _ = groups["day"]
        assert len(day_preds) == 3


# ---------------------------------------------------------------------------
# test_stratification_by_weather
# ---------------------------------------------------------------------------


class TestStratificationByWeather:
    """Verify weather groups are created correctly."""

    def test_weather_groups(self, rob_config: dict) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        gt = [
            _make_gt("c1", weather="clear"), _make_gt("c2", weather="clear"),
            _make_gt("r1", weather="rain"),
        ]
        preds = [_make_pred("c1"), _make_pred("c2"), _make_pred("r1")]
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "weather")
        assert "clear" in groups
        assert "rain" in groups
        assert len(groups["clear"][1]) == 2
        assert len(groups["rain"][1]) == 1


# ---------------------------------------------------------------------------
# test_stratification_by_source
# ---------------------------------------------------------------------------


class TestStratificationBySource:
    """Verify nuscenes/dada2000 split."""

    def test_source_groups_exist(self, rob_config: dict, source_data: tuple) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        preds, gt = source_data
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "source")
        assert "nuscenes" in groups
        assert "dada2000" in groups

    def test_nuscenes_count(self, rob_config: dict, source_data: tuple) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        preds, gt = source_data
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "source")
        _, ns_gt = groups["nuscenes"]
        assert len(ns_gt) == 3

    def test_dada_count(self, rob_config: dict, source_data: tuple) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        preds, gt = source_data
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "source")
        _, da_gt = groups["dada2000"]
        assert len(da_gt) == 2

    def test_frame_id_prefix_inference(self, rob_config: dict) -> None:
        """frame_ids starting with 'dada' → source=dada2000."""
        from drivesense.eval.robustness import _infer_source

        assert _infer_source("dada2000_seq01_frame0042") == "dada2000"
        assert _infer_source("nuscenes_scene123_token456") == "nuscenes"


# ---------------------------------------------------------------------------
# test_gap_calculation
# ---------------------------------------------------------------------------


class TestGapCalculation:
    """Gap = max(detection_rate) - min(detection_rate) across groups."""

    def test_gap_equals_max_minus_min(self) -> None:
        from drivesense.eval.robustness import compute_condition_gap

        stratum = {
            "day":   {"hazard_detection_rate": 0.85, "n_frames": 50},
            "night": {"hazard_detection_rate": 0.72, "n_frames": 30},
        }
        result = compute_condition_gap(stratum, "hazard_detection_rate")
        assert result["gap"] == pytest.approx(0.85 - 0.72, abs=1e-4)
        assert result["best_group"] == "day"
        assert result["worst_group"] == "night"

    def test_single_group_gap_is_zero(self) -> None:
        from drivesense.eval.robustness import compute_condition_gap

        stratum = {"day": {"hazard_detection_rate": 0.80, "n_frames": 50}}
        result = compute_condition_gap(stratum, "hazard_detection_rate")
        assert result["gap"] == 0.0

    def test_three_group_gap(self) -> None:
        from drivesense.eval.robustness import compute_condition_gap

        stratum = {
            "clear": {"hazard_detection_rate": 0.87, "n_frames": 100},
            "rain":  {"hazard_detection_rate": 0.71, "n_frames": 40},
            "fog":   {"hazard_detection_rate": 0.78, "n_frames": 20},
        }
        result = compute_condition_gap(stratum, "hazard_detection_rate")
        assert result["gap"] == pytest.approx(0.87 - 0.71, abs=1e-4)


# ---------------------------------------------------------------------------
# test_ood_performance_ratio
# ---------------------------------------------------------------------------


class TestOODPerformanceRatio:
    """dada2000 DR / nuscenes DR = OOD ratio."""

    def test_ratio_computed_correctly(self) -> None:
        from drivesense.eval.robustness import _compute_all_gaps

        strata = {
            "time_of_day": {},
            "weather": {},
            "location": {},
            "ego_speed_bucket": {},
            "source": {
                "nuscenes": {"hazard_detection_rate": 0.80, "n_frames": 200},
                "dada2000": {"hazard_detection_rate": 0.60, "n_frames": 40},
            },
        }
        gaps = _compute_all_gaps(strata)
        assert gaps["ood_relative_performance"] == pytest.approx(0.75, abs=1e-3)

    def test_zero_nuscenes_dr_no_division_error(self) -> None:
        from drivesense.eval.robustness import _compute_all_gaps

        strata = {
            "time_of_day": {}, "weather": {}, "location": {}, "ego_speed_bucket": {},
            "source": {
                "nuscenes": {"hazard_detection_rate": 0.0, "n_frames": 10},
                "dada2000": {"hazard_detection_rate": 0.5, "n_frames": 5},
            },
        }
        gaps = _compute_all_gaps(strata)
        assert gaps["ood_relative_performance"] == 0.0


# ---------------------------------------------------------------------------
# test_target_evaluation
# ---------------------------------------------------------------------------


class TestTargetEvaluation:
    """Known gaps — verify targets_met pass/fail."""

    def test_small_gap_passes(self, rob_config: dict) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        ev = RobustnessEvaluator(rob_config)
        gaps = {
            "day_night_detection_rate_gap": 0.05,  # < 0.10 → pass
            "weather_detection_rate_gap": 0.10,    # < 0.15 → pass
            "location_detection_rate_gap": 0.04,   # < 0.10 → pass
            "ood_relative_performance": 0.85,      # > 0.70 → pass
        }
        strata = {
            "time_of_day": {}, "weather": {}, "location": {},
            "ego_speed_bucket": {}, "source": {},
        }
        tgt = ev._evaluate_targets(strata, gaps)
        assert all(tgt.values())

    def test_large_gap_fails(self, rob_config: dict) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        ev = RobustnessEvaluator(rob_config)
        gaps = {
            "day_night_detection_rate_gap": 0.20,  # > 0.10 → fail
            "weather_detection_rate_gap": 0.05,
            "location_detection_rate_gap": 0.04,
            "ood_relative_performance": 0.80,
        }
        strata = {
            "time_of_day": {}, "weather": {}, "location": {},
            "ego_speed_bucket": {}, "source": {},
        }
        tgt = ev._evaluate_targets(strata, gaps)
        assert tgt["day_night_gap"] is False

    def test_ood_below_threshold_fails(self, rob_config: dict) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        ev = RobustnessEvaluator(rob_config)
        gaps = {
            "day_night_detection_rate_gap": 0.05,
            "weather_detection_rate_gap": 0.05,
            "location_detection_rate_gap": 0.05,
            "ood_relative_performance": 0.60,   # < 0.70 → fail
        }
        strata = {
            "time_of_day": {}, "weather": {}, "location": {},
            "ego_speed_bucket": {}, "source": {},
        }
        tgt = ev._evaluate_targets(strata, gaps)
        assert tgt["ood_performance"] is False


# ---------------------------------------------------------------------------
# test_empty_group_handled
# ---------------------------------------------------------------------------


class TestEmptyGroupHandled:
    """A stratification group with 0 frames must not raise."""

    def test_empty_predictions_group(self, rob_config: dict) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        gt = [_make_gt("d1", time_of_day="day")]
        preds = []  # no predictions at all
        ev = RobustnessEvaluator(rob_config)
        groups = ev.stratify_predictions(preds, gt, "time_of_day")
        assert "day" in groups
        day_preds, day_gt = groups["day"]
        assert len(day_gt) == 1
        assert len(day_preds) == 0

    def test_compute_stratified_metrics_empty_group(self, rob_config: dict) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        gt = [_make_gt("d1", time_of_day="day")]
        preds = []
        ev = RobustnessEvaluator(rob_config)
        metrics = ev.compute_stratified_metrics(preds, gt)
        assert "by_time_of_day" in metrics
        assert isinstance(metrics["gaps"], dict)


# ---------------------------------------------------------------------------
# test_overall_pass
# ---------------------------------------------------------------------------


class TestOverallPass:
    """All targets met → overall_pass = True; any fail → False."""

    def test_all_pass_gives_overall_true(self, rob_config: dict) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        ev = RobustnessEvaluator(rob_config)
        metrics = {
            "gaps": {
                "day_night_detection_rate_gap": 0.05,
                "weather_detection_rate_gap": 0.08,
                "location_detection_rate_gap": 0.03,
                "ood_relative_performance": 0.85,
            },
            "targets_met": {
                "day_night_gap": True, "weather_gap": True,
                "location_gap": True, "ood_performance": True,
            },
            "overall_pass": True,
        }
        assert metrics["overall_pass"] is True

    def test_one_fail_gives_overall_false(self, rob_config: dict) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        ev = RobustnessEvaluator(rob_config)
        targets_met = {
            "day_night_gap": True, "weather_gap": False,
            "location_gap": True, "ood_performance": True,
        }
        assert not all(targets_met.values())

    def test_generate_report_creates_files(
        self, rob_config: dict, tmp_path: Path
    ) -> None:
        from drivesense.eval.robustness import RobustnessEvaluator

        ev = RobustnessEvaluator(rob_config)
        metrics = {
            "by_time_of_day": {"day": {"hazard_detection_rate": 0.8, "n_frames": 10}},
            "by_weather": {}, "by_location": {}, "by_ego_speed_bucket": {}, "by_source": {},
            "gaps": {
                "day_night_detection_rate_gap": 0.05,
                "weather_detection_rate_gap": 0.05,
                "location_detection_rate_gap": 0.03,
                "ood_relative_performance": 0.80,
            },
            "targets_met": {
                "day_night_gap": True, "weather_gap": True,
                "location_gap": True, "ood_performance": True,
            },
            "overall_pass": True,
        }
        ev.generate_report(metrics, tmp_path)
        assert (tmp_path / "robustness_metrics.json").exists()
        assert (tmp_path / "robustness_report.txt").exists()
        assert (tmp_path / "gap_analysis.json").exists()
        assert (tmp_path / "stratified_results" / "by_time_of_day.json").exists()
