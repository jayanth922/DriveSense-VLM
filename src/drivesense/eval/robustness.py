"""Level 4 evaluation: Robustness across driving conditions and OOD data.

Implements Phase 4b: stratifies test set by time-of-day, weather, geographic
location, and ego speed bucket, then computes grounding metrics per stratum.
Also evaluates OOD performance using DADA-2000 as the out-of-distribution set.
Checks that no stratum gap exceeds the configured tolerance thresholds.

Implementation target: Phase 4b
"""

from __future__ import annotations


def stratify_by_condition(
    test_data: list[dict],
    stratify_keys: list[str],
) -> dict[str, list[dict]]:
    """Group test examples by the specified metadata condition keys.

    Args:
        test_data: List of test example dicts with metadata fields.
        stratify_keys: List of metadata keys to stratify by
                       (e.g., ["time_of_day", "weather"]).

    Returns:
        Nested dict mapping stratum_key -> condition_value -> list[dict].
        Example: ``{"time_of_day": {"day": [...], "night": [...]}}``.
    """
    raise NotImplementedError("Phase 4b: group test_data by each key in stratify_keys")


def compute_stratum_metrics(
    stratified_data: dict[str, list[dict]],
    predictions: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute grounding metrics for each condition stratum.

    Args:
        stratified_data: Output of stratify_by_condition().
        predictions: Full list of model predictions (matched by image_id).
        iou_threshold: IoU threshold for grounding accuracy.

    Returns:
        Dict mapping stratum condition to grounding metrics dict.
        Example: ``{"day": {"iou_at_50": 0.65, ...}, "night": {...}}``.
    """
    raise NotImplementedError("Phase 4b: call compute_grounding_metrics per stratum")


def compute_condition_gap(stratum_metrics: dict, metric_key: str = "iou_at_50") -> dict:
    """Compute max performance gap between conditions within each stratum key.

    Args:
        stratum_metrics: Output of compute_stratum_metrics().
        metric_key: Metric to compare across conditions.

    Returns:
        Dict mapping stratum_key to the max gap between its conditions.
        Example: ``{"time_of_day": 0.08, "weather": 0.12}``.
    """
    raise NotImplementedError("Phase 4b: compute max(metric) - min(metric) per stratum key")


def run_robustness_evaluation(
    config: dict,
    test_data: list[dict],
    ood_data: list[dict],
    predictions: list[dict],
    ood_predictions: list[dict],
) -> dict:
    """Run the full Level 4 robustness evaluation suite.

    Args:
        config: Eval config dict from configs/eval.yaml ['robustness'].
        test_data: In-distribution test examples.
        ood_data: Out-of-distribution examples (DADA-2000).
        predictions: Predictions for test_data.
        ood_predictions: Predictions for ood_data.

    Returns:
        Dict of robustness metrics with pass/fail status against config targets.
    """
    raise NotImplementedError("Phase 4b: run stratification + OOD eval and check target gaps")
