"""Cross-run Level-1 metric comparison — shared by the CI regression gate
(``scripts/run_regression_gate.py``) and the eval comparison report
(``scripts/compare_eval_runs.py``).

Operates purely on ``eval_summary.json`` dicts (as written by
``scripts/run_evaluation.py``: ``{"level1": {...grounding metrics...}, "level2": {...}}``).
No GPU/torch — pure data comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Baseline values below this are treated as "effectively zero" — a relative
# percentage against a near-zero baseline is not meaningful, so these trigger
# the absolute-epsilon fallback in relative_change() instead.
_ZERO_EPS = 1e-9

# name -> {path: dotted lookup under summary["level1"], direction, tolerance}.
# direction: "higher_better" (detection/accuracy metrics) or "lower_better"
# (parse_failure_rate — fewer failures is better).
DEFAULT_METRICS: dict[str, dict] = {
    "hazard_detection_rate": {
        "path": "hazard_detection_rate", "direction": "higher_better", "tolerance": 0.10,
    },
    "detection_rate_by_iou@0.1": {
        "path": "detection_rate_by_iou.0.1", "direction": "higher_better", "tolerance": 0.10,
    },
    "detection_rate_by_iou@0.3": {
        "path": "detection_rate_by_iou.0.3", "direction": "higher_better", "tolerance": 0.10,
    },
    "detection_rate_by_iou@0.5": {
        "path": "detection_rate_by_iou.0.5", "direction": "higher_better", "tolerance": 0.10,
    },
    "mean_best_pair_iou": {
        "path": "mean_best_pair_iou", "direction": "higher_better", "tolerance": 0.10,
    },
    "parse_failure_rate": {
        "path": "parse_failure_rate", "direction": "lower_better", "tolerance": 0.10,
    },
    "classification_accuracy": {
        "path": "classification_accuracy", "direction": "higher_better", "tolerance": 0.10,
    },
}


def load_eval_summary(path: str | Path) -> dict:
    """Load an ``eval_summary.json`` file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_metric(summary: dict, dotted_path: str) -> float | None:
    """Look up a metric under ``summary['level1']``, one level of nesting only.

    Tolerates being handed a bare Level-1 metrics dict directly (no
    ``"level1"`` wrapper), so this also works against
    ``GroundingEvaluator.evaluate()``'s raw return value in tests.

    ``dotted_path`` splits on the FIRST ``.`` only (``str.split(".", 1)``), not
    every ``.`` — ``detection_rate_by_iou``'s own keys are the literal strings
    ``"0.1"``/``"0.3"``/``"0.5"``, which themselves contain a dot. Splitting on
    every ``.`` would shred ``"0.1"`` into ``["0", "1"]`` and never find it.

    Args:
        summary:     A full ``eval_summary.json`` dict, or a bare level1 dict.
        dotted_path: A top-level key, or ``"dict_key.sub_key"`` for one level
                     of nesting (e.g. ``"detection_rate_by_iou.0.1"``).

    Returns:
        The metric value, or ``None`` if the path segment(s) are missing.
    """
    node: Any = summary.get("level1", summary) if isinstance(summary, dict) else summary
    top, _, rest = dotted_path.partition(".")
    if not isinstance(node, dict) or top not in node:
        return None
    node = node[top]
    if rest:
        if not isinstance(node, dict) or rest not in node:
            return None
        node = node[rest]
    return float(node) if isinstance(node, (int, float)) else None


def relative_change(direction: str, baseline: float, new: float) -> float:
    """"Badness" of ``new`` vs ``baseline``, oriented so POSITIVE always means worse.

    ``higher_better`` metrics: positive = a drop (``baseline > new``).
    ``lower_better`` metrics (e.g. ``parse_failure_rate``): positive = an increase.

    A near-zero baseline makes a relative percentage meaningless (division
    blows up), so that case returns a ``1.0`` ("100% worse") sentinel if the
    metric moved in the bad direction at all, else ``0.0`` — there is nothing
    to regress from below a floor of zero.

    Args:
        direction: ``"higher_better"`` or ``"lower_better"``.
        baseline:  The reference value.
        new:       The value being compared against it.

    Returns:
        A signed float; ``> 0`` means the metric got worse.
    """
    if direction == "higher_better":
        if baseline > _ZERO_EPS:
            return (baseline - new) / baseline
        return 0.0 if new >= baseline else 1.0
    if direction == "lower_better":
        if baseline > _ZERO_EPS:
            return (new - baseline) / baseline
        return 0.0 if new <= baseline + 0.01 else 1.0  # absolute epsilon: baseline was ~0
    raise ValueError(f"Unknown direction: {direction!r}")


def evaluate_metric(
    name: str, spec: dict, baseline_val: float | None, new_val: float | None,
) -> dict:
    """Compare one metric between baseline and new; tolerance-gated pass/fail.

    Args:
        name:         Metric display name.
        spec:         Entry from ``DEFAULT_METRICS`` (or a custom override).
        baseline_val: Baseline value, or ``None`` if unavailable.
        new_val:      New-run value, or ``None`` if unavailable.

    Returns:
        A result dict with ``status`` in ``{"REGRESSED", "IMPROVED", "FLAT",
        "MISSING"}`` and a boolean ``regressed`` (the CI-gate-relevant field).
    """
    result = {
        "metric": name, "path": spec["path"], "direction": spec["direction"],
        "tolerance": spec["tolerance"], "baseline": baseline_val, "new": new_val,
        "status": "MISSING", "relative_change": None, "regressed": False,
    }
    if baseline_val is None or new_val is None:
        return result
    rel = relative_change(spec["direction"], baseline_val, new_val)
    result["relative_change"] = round(rel, 4)
    result["regressed"] = rel > spec["tolerance"]
    result["status"] = (
        "REGRESSED" if result["regressed"] else ("IMPROVED" if rel < -1e-9 else "FLAT")
    )
    return result


def compare_summaries(
    baseline: dict, new: dict, metrics: dict[str, dict] | None = None,
) -> list[dict]:
    """Evaluate every metric in ``metrics`` (default: :data:`DEFAULT_METRICS`)."""
    metrics = metrics or DEFAULT_METRICS
    return [
        evaluate_metric(name, spec, get_metric(baseline, spec["path"]),
                        get_metric(new, spec["path"]))
        for name, spec in metrics.items()
    ]


def verdict_label(direction: str, prev_val: float | None, new_val: float | None,
                  eps: float = 0.005) -> str:
    """Sensitive (not tolerance-gated) improved/regressed/flat label for reports.

    Distinct from :func:`evaluate_metric`'s ``regressed`` flag, which is gated by
    a metric's CI tolerance (e.g. 10%) — a comparison report wants to show ANY
    real movement between runs, not just movement large enough to fail CI.

    Args:
        direction: ``"higher_better"`` or ``"lower_better"``.
        prev_val:  Previous run's value, or ``None``.
        new_val:   This run's value, or ``None``.
        eps:       Minimum relative change to call a real move (vs rounding noise).

    Returns:
        ``"↑ improved"``, ``"↓ regressed"``, ``"→ flat"``, or ``"n/a"``.
    """
    if prev_val is None or new_val is None:
        return "n/a"
    rel = relative_change(direction, prev_val, new_val)
    if rel > eps:
        return "↓ regressed"
    if rel < -eps:
        return "↑ improved"
    return "→ flat"


def build_comparison_rows(
    runs: list[tuple[str, dict]], metrics: dict[str, dict] | None = None,
) -> list[dict]:
    """Build one row per metric, with a per-run value + verdict-vs-previous-run.

    Args:
        runs:    Ordered ``(label, summary)`` pairs (comparison order matters —
                 each run's verdict is relative to the PREVIOUS run with a value,
                 not to the first/baseline run).
        metrics: Metric spec dict (default: :data:`DEFAULT_METRICS`).

    Returns:
        ``[{"metric": name, "cells": [{"label", "value", "verdict"}, ...]}, ...]``.
    """
    metrics = metrics or DEFAULT_METRICS
    rows: list[dict] = []
    for name, spec in metrics.items():
        cells: list[dict] = []
        prev_val: float | None = None
        for label, summary in runs:
            val = get_metric(summary, spec["path"])
            verdict = verdict_label(spec["direction"], prev_val, val) if cells else "n/a"
            cells.append({"label": label, "value": val, "verdict": verdict})
            if val is not None:
                prev_val = val
        rows.append({"metric": name, "cells": cells})
    return rows
