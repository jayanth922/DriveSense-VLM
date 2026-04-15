"""Level 4: Robustness Evaluation.

Tests model generalization across driving conditions and datasets to ensure
the model doesn't overfit to specific scenarios.

Stratifies Level 1 grounding metrics by: time_of_day, weather, location,
ego_speed_bucket, and evaluates on OOD DADA-2000 holdout.

Implemented in Phase 4b.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# RobustnessEvaluator
# ---------------------------------------------------------------------------


class RobustnessEvaluator:
    """Level 4 robustness evaluation via stratified analysis.

    Takes Level 1 grounding metrics and re-computes them for each
    stratification condition, measuring performance gaps.

    Args:
        config: Merged config dict; reads ``config["robustness"]`` section.
    """

    def __init__(self, config: dict) -> None:
        rob_cfg = config.get("robustness", {})
        t = rob_cfg.get("targets", {})
        self._target_day_night: float = float(t.get("max_day_night_gap", 0.10))
        self._target_weather: float = float(t.get("max_weather_gap", 0.15))
        self._target_location: float = float(t.get("max_location_gap", 0.10))
        self._target_ood: float = float(t.get("ood_relative_performance", 0.70))
        self._stratify_keys: list[str] = list(
            rob_cfg.get("stratify_by", ["time_of_day", "weather", "location", "ego_speed_bucket"])
        )
        self._cfg = config

    # ── public API ──────────────────────────────────────────────────────────

    def stratify_predictions(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
        stratify_by: str,
    ) -> dict[str, tuple[list[dict], list[dict]]]:
        """Split predictions and GT into groups by stratification key.

        Metadata is extracted from GT records; predictions are matched by
        ``frame_id``.

        Args:
            predictions:  List of prediction dicts (must have ``frame_id``).
            ground_truth: List of GT dicts (must have ``frame_id``).
            stratify_by:  One of ``"time_of_day"``, ``"weather"``,
                          ``"location"``, ``"ego_speed_bucket"``, ``"source"``.

        Returns:
            ``{group_name: (preds_in_group, gt_in_group)}``
        """
        gt_by_id = {g["frame_id"]: g for g in ground_truth if "frame_id" in g}
        pred_by_id = {p["frame_id"]: p for p in predictions if "frame_id" in p}

        groups: dict[str, tuple[list[dict], list[dict]]] = {}

        for frame_id, gt_rec in gt_by_id.items():
            group = _extract_stratum_value(gt_rec, stratify_by)
            if group not in groups:
                groups[group] = ([], [])
            if frame_id in pred_by_id:
                groups[group][0].append(pred_by_id[frame_id])
            groups[group][1].append(gt_rec)

        return groups

    def compute_stratified_metrics(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
    ) -> dict:
        """Compute Level 1 grounding metrics stratified by all conditions.

        Args:
            predictions:  List of prediction dicts.
            ground_truth: List of GT annotation dicts.

        Returns:
            Full Level 4 metrics dict with ``gaps``, ``targets_met``,
            ``overall_pass``.
        """
        from drivesense.eval.grounding import compute_grounding_metrics  # noqa: PLC0415

        strata: dict[str, dict] = {}
        for key in ("time_of_day", "weather", "location", "ego_speed_bucket", "source"):
            groups = self.stratify_predictions(predictions, ground_truth, key)
            strata[key] = _compute_per_group(groups, compute_grounding_metrics)

        gaps = _compute_all_gaps(strata)
        targets_met = self._evaluate_targets(strata, gaps)

        return {
            "by_time_of_day": strata["time_of_day"],
            "by_weather": strata["weather"],
            "by_location": strata["location"],
            "by_ego_speed_bucket": strata["ego_speed_bucket"],
            "by_source": strata["source"],
            "gaps": gaps,
            "targets_met": targets_met,
            "overall_pass": all(targets_met.values()),
        }

    def evaluate(
        self,
        predictions_path: Path,
        ground_truth_path: Path,
    ) -> dict:
        """Run complete Level 4 evaluation.

        Loads predictions and GT from JSONL files and runs
        ``compute_stratified_metrics``.

        Args:
            predictions_path:  Path to predictions JSONL.
            ground_truth_path: Path to ground truth JSONL.

        Returns:
            Full Level 4 metrics dict.
        """
        from drivesense.eval.grounding import GroundingEvaluator  # noqa: PLC0415

        g_eval = GroundingEvaluator(self._cfg)
        predictions = g_eval.load_predictions(predictions_path)
        ground_truth = g_eval.load_ground_truth(ground_truth_path)
        return self.compute_stratified_metrics(predictions, ground_truth)

    def generate_report(self, metrics: dict, output_dir: Path) -> Path:
        """Write Level 4 report files.

        Creates::

            output_dir/
            ├── robustness_metrics.json
            ├── robustness_report.txt
            ├── stratified_results/
            │   ├── by_time_of_day.json
            │   ├── by_weather.json
            │   ├── by_location.json
            │   ├── by_speed.json
            │   └── by_source.json
            └── gap_analysis.json

        Args:
            metrics:    Output of ``compute_stratified_metrics()``.
            output_dir: Destination directory.

        Returns:
            Path to ``robustness_report.txt``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        strat_dir = output_dir / "stratified_results"
        strat_dir.mkdir(exist_ok=True)

        (output_dir / "robustness_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        (output_dir / "gap_analysis.json").write_text(
            json.dumps(metrics.get("gaps", {}), indent=2), encoding="utf-8"
        )

        for key, fname in (
            ("by_time_of_day", "by_time_of_day.json"),
            ("by_weather", "by_weather.json"),
            ("by_location", "by_location.json"),
            ("by_ego_speed_bucket", "by_speed.json"),
            ("by_source", "by_source.json"),
        ):
            (strat_dir / fname).write_text(
                json.dumps(metrics.get(key, {}), indent=2), encoding="utf-8"
            )

        report_path = output_dir / "robustness_report.txt"
        report_path.write_text(
            _format_robustness_report(metrics, self._get_targets()), encoding="utf-8"
        )
        logger.info("Level 4 report written to %s", output_dir)
        return report_path

    def log_to_wandb(self, metrics: dict) -> None:
        """Log Level 4 metrics to Weights & Biases.

        Args:
            metrics: Output of ``compute_stratified_metrics()``.
        """
        if not _WANDB_AVAILABLE or wandb is None:
            logger.debug("wandb not available — skipping log_to_wandb")
            return
        with contextlib.suppress(Exception):
            flat = _flatten_for_wandb(metrics, prefix="level4")
            wandb.log(flat)  # type: ignore[union-attr]
            logger.info("Level 4 metrics logged to W&B")

    # ── private ─────────────────────────────────────────────────────────────

    def _evaluate_targets(self, strata: dict, gaps: dict) -> dict:
        """Check whether gap targets are met."""
        day_night_gap = gaps.get("day_night_detection_rate_gap", 0.0)
        weather_gap = gaps.get("weather_detection_rate_gap", 0.0)
        location_gap = gaps.get("location_detection_rate_gap", 0.0)
        ood_ratio = gaps.get("ood_relative_performance", 1.0)

        return {
            "day_night_gap": day_night_gap <= self._target_day_night,
            "weather_gap": weather_gap <= self._target_weather,
            "location_gap": location_gap <= self._target_location,
            "ood_performance": ood_ratio >= self._target_ood,
        }

    def _get_targets(self) -> dict:
        return {
            "max_day_night_gap": self._target_day_night,
            "max_weather_gap": self._target_weather,
            "max_location_gap": self._target_location,
            "ood_relative_performance": self._target_ood,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (keeps original stub interface)
# ---------------------------------------------------------------------------


def stratify_by_condition(
    test_data: list[dict],
    stratify_keys: list[str],
) -> dict[str, dict[str, list[dict]]]:
    """Group test examples by the specified metadata condition keys.

    Args:
        test_data:     List of GT dicts with metadata fields.
        stratify_keys: Metadata keys to stratify by.

    Returns:
        ``{key: {value: [records]}}``.
    """
    result: dict[str, dict[str, list[dict]]] = {}
    for key in stratify_keys:
        groups: dict[str, list[dict]] = {}
        for record in test_data:
            group = _extract_stratum_value(record, key)
            groups.setdefault(group, []).append(record)
        result[key] = groups
    return result


def compute_stratum_metrics(
    stratified_data: dict[str, list[dict]],
    predictions: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute grounding metrics for each condition stratum.

    Args:
        stratified_data: ``{group_name: [gt_records]}``.
        predictions:     Full list of predictions (matched by frame_id).
        iou_threshold:   IoU threshold for grounding.

    Returns:
        ``{group_name: grounding_metrics_dict}``.
    """
    from drivesense.eval.grounding import compute_grounding_metrics  # noqa: PLC0415

    pred_by_id = {p["frame_id"]: p for p in predictions if "frame_id" in p}
    result: dict[str, dict] = {}
    for group, gt_list in stratified_data.items():
        frame_ids = {g["frame_id"] for g in gt_list if "frame_id" in g}
        group_preds = [p for p in predictions if p.get("frame_id") in frame_ids]
        result[group] = compute_grounding_metrics(group_preds, gt_list, iou_threshold)
        result[group]["n_frames"] = len(gt_list)
    _ = pred_by_id  # used implicitly via closure
    return result


def compute_condition_gap(stratum_metrics: dict, metric_key: str = "hazard_detection_rate") -> dict:
    """Compute max performance gap between conditions within each stratum.

    Args:
        stratum_metrics: ``{group: metrics_dict}``.
        metric_key:      Metric to compare across conditions.

    Returns:
        ``{"gap": float, "best_group": str, "worst_group": str}``.
    """
    values = {
        g: float(m.get(metric_key, 0.0))
        for g, m in stratum_metrics.items()
        if m.get("n_frames", m.get("total_frames", 1)) > 0
    }
    if len(values) < 2:
        return {"gap": 0.0, "best_group": "", "worst_group": ""}
    best = max(values, key=lambda k: values[k])
    worst = min(values, key=lambda k: values[k])
    return {
        "gap": round(values[best] - values[worst], 4),
        "best_group": best,
        "worst_group": worst,
    }


def run_robustness_evaluation(
    config: dict,
    test_data: list[dict],
    ood_data: list[dict],
    predictions: list[dict],
    ood_predictions: list[dict],
) -> dict:
    """Run the full Level 4 robustness evaluation suite.

    Args:
        config:          Config dict (reads ``config["robustness"]``).
        test_data:       In-distribution GT records.
        ood_data:        Out-of-distribution GT records (DADA-2000).
        predictions:     Predictions for test_data.
        ood_predictions: Predictions for ood_data.

    Returns:
        Level 4 metrics dict with pass/fail status.
    """
    from drivesense.eval.grounding import compute_grounding_metrics  # noqa: PLC0415

    evaluator = RobustnessEvaluator(config)

    # Inject source tags so stratification by source works
    for rec in test_data:
        rec.setdefault("source", "nuscenes")
    for rec in ood_data:
        rec.setdefault("source", "dada2000")

    all_gt = test_data + ood_data
    all_preds = predictions + ood_predictions
    return evaluator.compute_stratified_metrics(all_preds, all_gt)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_stratum_value(gt_record: dict, key: str) -> str:
    """Extract a stratification group label from a GT record.

    Checks ``metadata``, ``ego_context``, and top-level fields.
    Falls back to ``"unknown"`` if the key is not found.
    """
    if key == "ego_speed_bucket":
        speed = (
            gt_record.get("ego_speed_kmh")
            or gt_record.get("metadata", {}).get("ego_speed_kmh")
            or 0.0
        )
        return _speed_bucket(float(speed))

    if key == "source":
        src = (
            gt_record.get("source")
            or gt_record.get("metadata", {}).get("source")
            or _infer_source(gt_record.get("frame_id", ""))
        )
        return str(src).lower()

    # General key: try metadata → ego_context → top-level
    val = (
        gt_record.get("metadata", {}).get(key)
        or gt_record.get("ego_context", {}).get(key)
        or gt_record.get(key)
    )
    return str(val).lower() if val else "unknown"


def _speed_bucket(speed_kmh: float) -> str:
    """Bin ego speed into a bucket string."""
    if speed_kmh < 20:
        return "0-20"
    if speed_kmh < 40:
        return "20-40"
    return "40+"


def _infer_source(frame_id: str) -> str:
    """Infer dataset source from frame_id prefix."""
    if frame_id.startswith("dada"):
        return "dada2000"
    return "nuscenes"


def _compute_per_group(
    groups: dict[str, tuple[list[dict], list[dict]]],
    compute_fn: object,
) -> dict[str, dict]:
    """Compute grounding metrics for each group in the stratification."""
    result: dict[str, dict] = {}
    for group, (preds, gts) in groups.items():
        if not gts:
            result[group] = _empty_group_metrics()
            continue
        with contextlib.suppress(Exception):
            m = compute_fn(preds, gts)  # type: ignore[operator]
            m["n_frames"] = len(gts)
            result[group] = m
            continue
        result[group] = _empty_group_metrics()
    return result


def _empty_group_metrics() -> dict:
    return {
        "hazard_detection_rate": 0.0,
        "false_positive_rate": 0.0,
        "mean_iou": 0.0,
        "iou_at_threshold": 0.0,
        "n_frames": 0,
    }


def _detection_rate_gap(stratum: dict) -> float:
    """Max − min detection_rate across non-empty groups."""
    rates = [
        m.get("hazard_detection_rate", 0.0)
        for m in stratum.values()
        if m.get("n_frames", 0) > 0
    ]
    if len(rates) < 2:
        return 0.0
    return round(max(rates) - min(rates), 4)


def _compute_all_gaps(strata: dict) -> dict:
    """Compute gap metrics and OOD relative performance."""
    day_night = strata.get("time_of_day", {})
    weather = strata.get("weather", {})
    location = strata.get("location", {})
    source = strata.get("source", {})

    nuscenes_dr = source.get("nuscenes", {}).get("hazard_detection_rate", 0.0)
    dada_dr = source.get("dada2000", {}).get("hazard_detection_rate", 0.0)
    ood_ratio = round(dada_dr / nuscenes_dr, 4) if nuscenes_dr > 0 else 0.0

    return {
        "day_night_detection_rate_gap": _detection_rate_gap(day_night),
        "weather_detection_rate_gap": _detection_rate_gap(weather),
        "location_detection_rate_gap": _detection_rate_gap(location),
        "source_detection_rate_gap": _detection_rate_gap(source),
        "speed_detection_rate_gap": _detection_rate_gap(strata.get("ego_speed_bucket", {})),
        "ood_relative_performance": ood_ratio,
    }


def _format_robustness_report(metrics: dict, targets: dict) -> str:
    """Format a human-readable Level 4 report string."""
    gaps = metrics.get("gaps", {})
    tgt = metrics.get("targets_met", {})
    overall = metrics.get("overall_pass", False)

    def _pass(ok: bool | None) -> str:
        return "PASS" if ok else "FAIL"

    def _pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    lines = [
        "DriveSense-VLM — Level 4: Robustness",
        "=" * 52,
        "",
        "Performance Gaps (detection rate)",
        f"  Day/Night:    {_pct(gaps.get('day_night_detection_rate_gap', 0)):<8} "
        f"target <{_pct(targets['max_day_night_gap'])}   [{_pass(tgt.get('day_night_gap'))}]",
        f"  Weather:      {_pct(gaps.get('weather_detection_rate_gap', 0)):<8} "
        f"target <{_pct(targets['max_weather_gap'])}   [{_pass(tgt.get('weather_gap'))}]",
        f"  Location:     {_pct(gaps.get('location_detection_rate_gap', 0)):<8} "
        f"target <{_pct(targets['max_location_gap'])}   [{_pass(tgt.get('location_gap'))}]",
        f"  Speed bucket: {_pct(gaps.get('speed_detection_rate_gap', 0)):<8}",
        "",
        "Out-of-Distribution (DADA-2000 vs nuScenes)",
        f"  OOD ratio:    {gaps.get('ood_relative_performance', 0):.3f}     "
        f"target >={targets['ood_relative_performance']:.2f}   [{_pass(tgt.get('ood_performance'))}]",
        "",
    ]

    # Per-stratum summaries
    for label, key in (
        ("By Time of Day", "by_time_of_day"),
        ("By Weather", "by_weather"),
        ("By Location", "by_location"),
        ("By Speed", "by_ego_speed_bucket"),
        ("By Source", "by_source"),
    ):
        stratum = metrics.get(key, {})
        if stratum:
            lines.append(f"{label}:")
            for g, m in stratum.items():
                dr = m.get("hazard_detection_rate", 0.0)
                n = m.get("n_frames", 0)
                lines.append(f"  {g:<16} DR={dr:.3f}  n={n}")
            lines.append("")

    lines += ["=" * 52, f"Overall: {'ALL TARGETS MET' if overall else 'ONE OR MORE TARGETS MISSED'}"]
    return "\n".join(lines) + "\n"


def _flatten_for_wandb(metrics: dict, prefix: str = "") -> dict:
    """Flatten nested metrics for W&B (skip large sub-dicts)."""
    out: dict = {}
    skip = {"by_time_of_day", "by_weather", "by_location", "by_ego_speed_bucket",
            "by_source", "targets_met"}
    for k, v in metrics.items():
        if k in skip:
            continue
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_for_wandb(v, key))
        elif v is not None:
            out[key] = v
    return out
