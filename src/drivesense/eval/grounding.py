"""Level 1 evaluation: Grounding accuracy via bounding box IoU metrics.

Implements Phase 4b: computes IoU@50, mean IoU, hazard detection rate, and
false positive rate by comparing model-predicted bboxes against ground truth
annotations from the test split.

All bounding boxes are expected in [x1, y1, x2, y2] pixel coordinate format.

Implementation target: Phase 4b
"""

from __future__ import annotations


def compute_iou(pred_box: list[float], gt_box: list[float]) -> float:
    """Compute Intersection over Union between two axis-aligned bounding boxes.

    Args:
        pred_box: Predicted box as [x1, y1, x2, y2] in pixel coordinates.
        gt_box: Ground truth box as [x1, y1, x2, y2] in pixel coordinates.

    Returns:
        Float IoU value in [0.0, 1.0]. Returns 0.0 if there is no intersection.
    """
    raise NotImplementedError("Phase 4b: compute intersection area / union area")


def compute_grounding_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute full grounding accuracy metrics over the test set.

    Matches each prediction to the best overlapping ground truth box.
    A prediction is a True Positive (TP) if IoU >= iou_threshold.

    Args:
        predictions: List of prediction dicts, each containing:
                     ``{"image_id": str, "bbox_2d": [x1, y1, x2, y2], ...}``.
        ground_truth: List of ground truth dicts in the same format.
        iou_threshold: IoU threshold for TP classification. Default: 0.5.

    Returns:
        Dict of grounding metrics:
        ``{"iou_at_50": float, "detection_rate": float,
           "false_positive_rate": float, "mean_iou": float}``.
    """
    raise NotImplementedError("Phase 4b: match predictions to GT boxes and compute TP/FP/FN")


def compute_per_class_iou(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute IoU@threshold broken down by hazard class.

    Args:
        predictions: List of prediction dicts with 'hazard_class' field.
        ground_truth: List of GT dicts with 'hazard_class' field.
        iou_threshold: IoU threshold for TP classification.

    Returns:
        Dict mapping hazard_class string to its IoU@threshold value.
    """
    raise NotImplementedError(
        "Phase 4b: group by hazard_class and compute_grounding_metrics per group"
    )
