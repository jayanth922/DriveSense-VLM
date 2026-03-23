"""Level 2 evaluation: Reasoning quality via LLM-as-judge scoring.

Implements Phase 4b: sends model predictions to an LLM judge (Claude) alongside
the ground truth annotations and rates them on correctness, completeness, and
action relevance using a 1–5 Likert scale. Also computes classification accuracy
and severity Spearman rank correlation.

Implementation target: Phase 4b
"""

from __future__ import annotations

# Anthropic client — optional at import time; required at eval runtime
try:
    import anthropic  # type: ignore[import]
except ImportError:
    anthropic = None  # type: ignore[assignment]

# scipy for Spearman correlation
try:
    from scipy import stats  # type: ignore[import]
except ImportError:
    stats = None  # type: ignore[assignment]


def judge_reasoning(
    prediction: dict,
    ground_truth: dict,
    config: dict,
) -> dict:
    """Call LLM judge to score a single prediction on reasoning quality.

    Args:
        prediction: Model output dict with 'reasoning', 'hazard_class',
                    'severity', and 'action' fields.
        ground_truth: Reference annotation dict in the same format.
        config: Eval config dict from configs/eval.yaml ['reasoning']['judge'].

    Returns:
        Dict of per-dimension scores:
        ``{"correctness": float, "completeness": float,
           "action_relevance": float, "raw_response": str}``.
    """
    raise NotImplementedError("Phase 4b: build judge prompt, call LLM, parse dimension scores")


def compute_reasoning_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    config: dict,
) -> dict:
    """Compute all Level 2 reasoning quality metrics over the test set.

    Args:
        predictions: List of model output dicts.
        ground_truth: List of reference annotation dicts.
        config: Eval config dict from configs/eval.yaml ['reasoning'].

    Returns:
        Dict of reasoning metrics:
        ``{"correctness_score": float, "completeness_score": float,
           "action_relevance_score": float, "classification_accuracy": float,
           "severity_spearman": float}``.
    """
    raise NotImplementedError(
        "Phase 4b: aggregate judge_reasoning scores + classification + spearman"
    )


def compute_severity_spearman(
    pred_severities: list[int],
    gt_severities: list[int],
) -> float:
    """Compute Spearman rank correlation between predicted and GT severity scores.

    Args:
        pred_severities: List of predicted severity values (1–5).
        gt_severities: List of ground truth severity values (1–5).

    Returns:
        Spearman rank correlation coefficient in [-1.0, 1.0].
    """
    raise NotImplementedError("Phase 4b: call scipy.stats.spearmanr on severity lists")
