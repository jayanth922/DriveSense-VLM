"""Level 1: Grounding Accuracy Evaluation.

Measures whether the model correctly identifies, localises, and classifies
hazards using bounding-box IoU metrics and Hungarian assignment.

Metrics: IoU@threshold, Hazard Detection Rate (Recall), False Positive Rate,
Mean IoU, Localization Accuracy, Classification Accuracy.

Phase 2b implementation.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — guarded for macOS CPU dev
# ---------------------------------------------------------------------------

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore[import]
    _SCIPY_AVAILABLE = True
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

try:
    from scipy import stats as _scipy_stats  # type: ignore[import]
    _SCIPY_STATS_AVAILABLE = True
except ImportError:
    _scipy_stats = None  # type: ignore[assignment]
    _SCIPY_STATS_AVAILABLE = False

try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

from drivesense.data.annotation import AnnotationValidator  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LABELS: list[str] = AnnotationValidator.VALID_LABELS

SEVERITY_TO_INT: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


def compute_iou(pred_box: list[int], gt_box: list[int]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        pred_box: [x1, y1, x2, y2] in [0, 1000] normalised coordinates.
        gt_box:   [x1, y1, x2, y2] in [0, 1000] normalised coordinates.

    Returns:
        IoU value in [0.0, 1.0].
    """
    ix1 = max(pred_box[0], gt_box[0])
    iy1 = max(pred_box[1], gt_box[1])
    ix2 = min(pred_box[2], gt_box[2])
    iy2 = min(pred_box[3], gt_box[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    pred_area = max(0.0, pred_box[2] - pred_box[0]) * max(0.0, pred_box[3] - pred_box[1])
    gt_area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
    union = pred_area + gt_area - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# match_predictions_to_ground_truth
# ---------------------------------------------------------------------------


def match_predictions_to_ground_truth(
    pred_hazards: list[dict],
    gt_hazards: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Match predicted hazards to ground truth using Hungarian assignment.

    A match is valid only if IoU >= iou_threshold.  Each GT hazard can match
    at most one prediction (and vice versa).

    Args:
        pred_hazards: Predicted hazard dicts, each with a ``bbox_2d`` field.
        gt_hazards:   Ground-truth hazard dicts, each with a ``bbox_2d`` field.
        iou_threshold: Minimum IoU to accept a match.

    Returns:
        Dict with keys:
            ``matched_pairs`` — list of (pred_idx, gt_idx, iou) tuples,
            ``unmatched_predictions`` — indices of false-positive predictions,
            ``unmatched_ground_truth`` — indices of missed GT hazards.
    """
    if not pred_hazards or not gt_hazards:
        return {
            "matched_pairs": [],
            "unmatched_predictions": list(range(len(pred_hazards))),
            "unmatched_ground_truth": list(range(len(gt_hazards))),
        }

    n_p, n_g = len(pred_hazards), len(gt_hazards)
    cost = np.zeros((n_p, n_g), dtype=np.float64)
    for i, ph in enumerate(pred_hazards):
        for j, gh in enumerate(gt_hazards):
            cost[i, j] = compute_iou(
                ph.get("bbox_2d", [0, 0, 0, 0]),
                gh.get("bbox_2d", [0, 0, 0, 0]),
            )

    if _SCIPY_AVAILABLE and linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(-cost)
    else:
        # Greedy fallback: sort candidate pairs by descending IoU
        pairs = sorted(
            [(i, j) for i in range(n_p) for j in range(n_g)],
            key=lambda xy: -cost[xy[0], xy[1]],
        )
        used_p: set[int] = set()
        used_g: set[int] = set()
        row_ind_list: list[int] = []
        col_ind_list: list[int] = []
        for i, j in pairs:
            if i not in used_p and j not in used_g:
                row_ind_list.append(i)
                col_ind_list.append(j)
                used_p.add(i)
                used_g.add(j)
        row_ind, col_ind = row_ind_list, col_ind_list  # type: ignore[assignment]

    matched: list[tuple[int, int, float]] = []
    unmatched_p = set(range(n_p))
    unmatched_g = set(range(n_g))

    for r, c in zip(row_ind, col_ind):  # noqa: B905
        iou_val = float(cost[r, c])
        if iou_val >= iou_threshold:
            matched.append((int(r), int(c), iou_val))
            unmatched_p.discard(int(r))
            unmatched_g.discard(int(c))

    return {
        "matched_pairs": matched,
        "unmatched_predictions": sorted(unmatched_p),
        "unmatched_ground_truth": sorted(unmatched_g),
    }


# ---------------------------------------------------------------------------
# compute_grounding_metrics
# ---------------------------------------------------------------------------


def compute_grounding_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute all Level 1 grounding metrics across the test set.

    Args:
        predictions:  List of prediction dicts.  Each dict must have a
                      ``frame_id`` string and a ``hazards`` list.  An
                      optional ``parse_failure: True`` flag marks frames
                      where the model output could not be parsed.
        ground_truth: List of GT annotation dicts in the same format.
        iou_threshold: IoU threshold for a valid detection.

    Returns:
        Full metrics dict (see module docstring for field descriptions).
    """
    gt_by_frame = {g["frame_id"]: g for g in ground_truth if "frame_id" in g}

    tp = fp = fn = tn_frames = fp_no_hazard_frames = parse_failures = 0
    iou_values: list[float] = []
    total_gt = total_pred = 0
    correct_labels = matched_count = 0

    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # type: ignore[assignment]

    for pred in predictions:
        frame_id = pred.get("frame_id", "")
        gt = gt_by_frame.get(frame_id, {})
        gt_hazards = _active_hazards(gt.get("hazards", []))

        if pred.get("parse_failure", False):
            parse_failures += 1
            fn += len(gt_hazards)
            for h in gt_hazards:
                per_class[h.get("label", "")]["fn"] += 1
            continue

        pred_hazards = _active_hazards(pred.get("hazards", []))
        total_gt += len(gt_hazards)
        total_pred += len(pred_hazards)

        gt_empty = len(gt_hazards) == 0
        pred_empty = len(pred_hazards) == 0

        # ── no-hazard frame handling ─────────────────────────────────────
        if gt_empty:
            if pred_empty:
                tn_frames += 1
            else:
                fp_no_hazard_frames += 1
                fp += len(pred_hazards)
                for h in pred_hazards:
                    per_class[h.get("label", "")]["fp"] += 1
            continue

        # ── GT has hazards ───────────────────────────────────────────────
        if pred_empty:
            fn += len(gt_hazards)
            for h in gt_hazards:
                per_class[h.get("label", "")]["fn"] += 1
            continue

        result = match_predictions_to_ground_truth(pred_hazards, gt_hazards, iou_threshold)

        for pi, gi, iou_val in result["matched_pairs"]:
            tp += 1
            iou_values.append(iou_val)
            pl = pred_hazards[pi].get("label", "")
            gl = gt_hazards[gi].get("label", "")
            per_class[gl]["tp"] += 1
            confusion[gl][pl] += 1
            if pl == gl:
                correct_labels += 1
            matched_count += 1

        for pi in result["unmatched_predictions"]:
            fp += 1
            per_class[pred_hazards[pi].get("label", "")]["fp"] += 1

        for gi in result["unmatched_ground_truth"]:
            fn += 1
            per_class[gt_hazards[gi].get("label", "")]["fn"] += 1

    # ── aggregate ────────────────────────────────────────────────────────
    total_no_hazard = tn_frames + fp_no_hazard_frames
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
    # iou_at_threshold: Jaccard = TP / (TP + FP + FN)
    iou_at_thr = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    fpr = fp_no_hazard_frames / total_no_hazard if total_no_hazard > 0 else 0.0
    no_hazard_acc = tn_frames / total_no_hazard if total_no_hazard > 0 else 0.0
    class_acc = correct_labels / matched_count if matched_count > 0 else 0.0
    n_preds = len(predictions)
    parse_failure_rate = parse_failures / n_preds if n_preds > 0 else 0.0

    per_class_metrics: dict[str, dict] = {}
    for label in VALID_LABELS:
        if label == "no_hazard":
            continue
        c = per_class.get(label, {"tp": 0, "fp": 0, "fn": 0})
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class_metrics[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "count": c["tp"] + c["fn"],
        }

    confusion_dict: dict[str, dict[str, int]] = {
        gl: dict(pred_counts) for gl, pred_counts in confusion.items()
    }

    return {
        "iou_at_threshold": round(iou_at_thr, 4),
        "hazard_detection_rate": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "classification_accuracy": round(class_acc, 4),
        "total_frames": n_preds,
        "total_gt_hazards": total_gt,
        "total_pred_hazards": total_pred,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn_frames,
        "no_hazard_accuracy": round(no_hazard_acc, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_dict,
        "parse_failure_rate": round(parse_failure_rate, 4),
        "parse_failures": parse_failures,
    }


# ---------------------------------------------------------------------------
# compute_severity_metrics
# ---------------------------------------------------------------------------


def compute_severity_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """Evaluate severity prediction accuracy for matched hazard pairs.

    Args:
        predictions:  Prediction dicts with ``frame_id`` and ``hazards``.
        ground_truth: GT annotation dicts with ``frame_id`` and ``hazards``.

    Returns:
        Dict with severity_accuracy, severity_within_one, severity_spearman,
        and severity_confusion_matrix.
    """
    gt_by_frame = {g["frame_id"]: g for g in ground_truth if "frame_id" in g}

    pred_ints: list[int] = []
    gt_ints: list[int] = []
    sev_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # type: ignore[assignment]

    for pred in predictions:
        if pred.get("parse_failure", False):
            continue
        frame_id = pred.get("frame_id", "")
        gt = gt_by_frame.get(frame_id, {})
        pred_h = _active_hazards(pred.get("hazards", []))
        gt_h = _active_hazards(gt.get("hazards", []))
        if not pred_h or not gt_h:
            continue
        result = match_predictions_to_ground_truth(pred_h, gt_h, iou_threshold=0.5)
        for pi, gi, _ in result["matched_pairs"]:
            ps = pred_h[pi].get("severity", "")
            gs = gt_h[gi].get("severity", "")
            if ps and gs:
                sev_confusion[gs][ps] += 1
                p_int = SEVERITY_TO_INT.get(ps, 0)
                g_int = SEVERITY_TO_INT.get(gs, 0)
                if p_int > 0 and g_int > 0:
                    pred_ints.append(p_int)
                    gt_ints.append(g_int)

    if not pred_ints:
        return {
            "severity_accuracy": 0.0,
            "severity_within_one": 0.0,
            "severity_spearman": 0.0,
            "severity_confusion_matrix": {},
        }

    n = len(pred_ints)
    pairs = list(zip(pred_ints, gt_ints))  # noqa: B905
    exact = sum(1 for p, g in pairs if p == g)
    within_one = sum(1 for p, g in pairs if abs(p - g) <= 1)

    spearman = 0.0
    if _SCIPY_STATS_AVAILABLE and _scipy_stats is not None and n >= 3:
        corr = _scipy_stats.spearmanr(pred_ints, gt_ints).correlation
        spearman = float(corr) if not np.isnan(float(corr)) else 0.0

    return {
        "severity_accuracy": round(exact / n, 4),
        "severity_within_one": round(within_one / n, 4),
        "severity_spearman": round(spearman, 4),
        "severity_confusion_matrix": {
            gs: dict(pred_counts) for gs, pred_counts in sev_confusion.items()
        },
    }


# ---------------------------------------------------------------------------
# GroundingEvaluator
# ---------------------------------------------------------------------------


class GroundingEvaluator:
    """Complete Level 1 evaluation pipeline.

    Loads model predictions and ground truth, computes all metrics,
    generates report files, and optionally logs to W&B.

    Args:
        config: Merged eval config dict (from ``configs/eval.yaml``).
    """

    def __init__(self, config: dict) -> None:
        grounding_cfg = config.get("grounding", {})
        self._iou_threshold: float = float(grounding_cfg.get("iou_threshold", 0.5))
        self._output_dir = Path(config.get("output_dir", "outputs/eval"))
        self._cfg = config

    # ── data loading ────────────────────────────────────────────────────

    def load_predictions(self, predictions_path: Path) -> list[dict]:
        """Load and parse model predictions from a JSONL file.

        Each line may have:
            ``parse_success`` + ``parsed_output`` (preferred), or
            ``raw_output`` to be re-parsed by AnnotationValidator.

        Args:
            predictions_path: Path to predictions JSONL.

        Returns:
            List of normalised prediction dicts.
        """
        path = Path(predictions_path)
        if not path.exists():
            logger.warning("Predictions file not found: %s", path)
            return []

        preds: list[dict] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                entry = _record_to_pred_entry(rec)
                preds.append(entry)

        logger.info("Loaded %d predictions from %s", len(preds), path)
        return preds

    def load_ground_truth(self, manifest_path: Path) -> list[dict]:
        """Load ground truth annotations from an annotated manifest or SFT JSONL.

        Supports:
            * SFT JSONL (``messages`` key) — parses assistant content as GT.
            * Annotated manifest JSONL (direct ``hazards`` key).
            * Annotated manifest JSON array (``[{...}, ...]``).

        Args:
            manifest_path: Path to the ground truth file.

        Returns:
            List of GT dicts with ``frame_id`` and ``hazards``.
        """
        path = Path(manifest_path)
        if not path.exists():
            logger.warning("Ground truth file not found: %s", path)
            return []

        text = path.read_text(encoding="utf-8").strip()

        # JSON array format
        if text.startswith("["):
            raw_list = json.loads(text)
            return [_normalise_gt(r) for r in raw_list if isinstance(r, dict)]

        # JSONL — detect by first line content
        gt_list: list[dict] = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines:
            rec = json.loads(line)
            gt_list.append(_normalise_gt(rec))

        logger.info("Loaded %d GT examples from %s", len(gt_list), path)
        return gt_list

    def parse_model_output(self, raw_text: str) -> dict | None:
        """Parse raw model output text into a structured annotation dict.

        Reuses :meth:`AnnotationValidator.parse_llm_response`.

        Args:
            raw_text: Raw LLM-generated text.

        Returns:
            Parsed dict or ``None`` if unparseable.
        """
        return AnnotationValidator.parse_llm_response(raw_text)

    # ── evaluation entry point ───────────────────────────────────────────

    def evaluate(
        self,
        predictions_path: Path,
        ground_truth_path: Path,
    ) -> dict:
        """Run complete Level 1 evaluation.

        Args:
            predictions_path:  Path to predictions JSONL.
            ground_truth_path: Path to GT manifest / SFT JSONL.

        Returns:
            Full metrics dict including severity sub-metrics.
        """
        preds = self.load_predictions(predictions_path)
        gt = self.load_ground_truth(ground_truth_path)
        logger.info(
            "Level 1 eval: %d predictions, %d GT examples", len(preds), len(gt)
        )
        metrics = compute_grounding_metrics(preds, gt, self._iou_threshold)
        sev_metrics = compute_severity_metrics(preds, gt)
        metrics.update(sev_metrics)
        return metrics

    # ── report generation ────────────────────────────────────────────────

    def generate_report(self, metrics: dict, output_dir: Path) -> Path:
        """Write evaluation artefacts to *output_dir*.

        Creates::

            output_dir/
            ├── grounding_metrics.json
            ├── grounding_report.txt
            ├── per_class_metrics.json
            ├── confusion_matrix.json
            └── severity_confusion_matrix.json

        Args:
            metrics:    Full metrics dict returned by :meth:`evaluate`.
            output_dir: Directory to write artefacts into.

        Returns:
            Path to the plain-text report file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        _scalar_keys = {
            k for k, v in metrics.items()
            if not isinstance(v, (dict, list))
        }
        scalar_metrics = {k: metrics[k] for k in _scalar_keys}
        (out / "grounding_metrics.json").write_text(
            json.dumps(scalar_metrics, indent=2), encoding="utf-8"
        )
        (out / "per_class_metrics.json").write_text(
            json.dumps(metrics.get("per_class_metrics", {}), indent=2),
            encoding="utf-8",
        )
        (out / "confusion_matrix.json").write_text(
            json.dumps(metrics.get("confusion_matrix", {}), indent=2),
            encoding="utf-8",
        )
        (out / "severity_confusion_matrix.json").write_text(
            json.dumps(metrics.get("severity_confusion_matrix", {}), indent=2),
            encoding="utf-8",
        )

        report_path = out / "grounding_report.txt"
        report_path.write_text(_format_grounding_report(metrics), encoding="utf-8")
        logger.info("Level 1 report written to %s", report_path)
        return report_path

    # ── W&B logging ──────────────────────────────────────────────────────

    def log_to_wandb(self, metrics: dict) -> None:
        """Log grounding metrics to W&B.

        Args:
            metrics: Full metrics dict returned by :meth:`evaluate`.
        """
        if not _WANDB_AVAILABLE or wandb is None:
            return

        scalars = {
            "eval/iou_at_threshold": metrics.get("iou_at_threshold", 0),
            "eval/hazard_detection_rate": metrics.get("hazard_detection_rate", 0),
            "eval/false_positive_rate": metrics.get("false_positive_rate", 0),
            "eval/precision": metrics.get("precision", 0),
            "eval/f1_score": metrics.get("f1_score", 0),
            "eval/mean_iou": metrics.get("mean_iou", 0),
            "eval/classification_accuracy": metrics.get("classification_accuracy", 0),
            "eval/no_hazard_accuracy": metrics.get("no_hazard_accuracy", 0),
            "eval/severity_accuracy": metrics.get("severity_accuracy", 0),
            "eval/severity_spearman": metrics.get("severity_spearman", 0),
            "eval/parse_failure_rate": metrics.get("parse_failure_rate", 0),
        }
        with contextlib.suppress(Exception):
            wandb.log(scalars)

        # Confusion matrix table
        cm = metrics.get("confusion_matrix", {})
        if cm:
            rows = [
                [gt_l, pred_l, cnt]
                for gt_l, pred_dict in cm.items()
                for pred_l, cnt in pred_dict.items()
            ]
            with contextlib.suppress(Exception):
                wandb.log({
                    "eval/confusion_matrix": wandb.Table(
                        columns=["GT Label", "Pred Label", "Count"], data=rows
                    )
                })

        # Per-class F1 bar chart
        per_class = metrics.get("per_class_metrics", {})
        if per_class:
            data = [
                [cls, m["f1"]]
                for cls, m in per_class.items()
                if m["count"] > 0
            ]
            if data:
                with contextlib.suppress(Exception):
                    tbl = wandb.Table(data=data, columns=["Class", "F1"])
                    wandb.log({
                        "eval/per_class_f1": wandb.plot.bar(
                            tbl, "Class", "F1", title="F1 Score by Hazard Class"
                        )
                    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _active_hazards(hazards: list[dict]) -> list[dict]:
    """Return hazards that are not the sentinel no_hazard label."""
    return [h for h in hazards if h.get("label") != "no_hazard"]


def _record_to_pred_entry(rec: dict) -> dict:
    """Convert a raw predictions-JSONL record to a normalised eval dict."""
    frame_id = rec.get("frame_id", "")
    if rec.get("parse_success", False) and rec.get("parsed_output"):
        parsed: dict = rec["parsed_output"]
        return {
            "frame_id": frame_id,
            "hazards": parsed.get("hazards", []),
            "scene_summary": parsed.get("scene_summary", ""),
            "ego_context": parsed.get("ego_context", {}),
            "parse_failure": False,
        }

    # Fall back to re-parsing raw_output
    raw = rec.get("raw_output", "")
    if raw:
        parsed_raw = AnnotationValidator.parse_llm_response(raw)
        if parsed_raw:
            return {
                "frame_id": frame_id,
                "hazards": parsed_raw.get("hazards", []),
                "scene_summary": parsed_raw.get("scene_summary", ""),
                "ego_context": parsed_raw.get("ego_context", {}),
                "parse_failure": False,
            }

    return {"frame_id": frame_id, "hazards": [], "parse_failure": True}


def _normalise_gt(rec: dict) -> dict:
    """Normalise a GT record into a standard eval format."""
    # SFT JSONL format: extract assistant content
    if "messages" in rec:
        frame_id = rec.get("frame_id", "")
        for msg in rec.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                parsed = AnnotationValidator.parse_llm_response(
                    content if isinstance(content, str) else json.dumps(content)
                )
                if parsed:
                    return {
                        "frame_id": frame_id,
                        "hazards": parsed.get("hazards", []),
                        "scene_summary": parsed.get("scene_summary", ""),
                        "ego_context": parsed.get("ego_context", {}),
                    }
        return {"frame_id": frame_id, "hazards": []}

    # Direct annotation format
    return {
        "frame_id": rec.get("frame_id", ""),
        "hazards": rec.get("hazards", []),
        "scene_summary": rec.get("scene_summary", ""),
        "ego_context": rec.get("ego_context", {}),
    }


def _format_grounding_report(metrics: dict) -> str:
    """Format a human-readable Level 1 evaluation report."""
    lines = [
        "=" * 62,
        "  DriveSense-VLM — Level 1 Grounding Evaluation Report",
        "=" * 62,
        "",
        "  Detection Metrics",
        "  " + "-" * 40,
        f"  Recall (Hazard Detection Rate) : {metrics.get('hazard_detection_rate', 0):.4f}",
        f"  Precision                      : {metrics.get('precision', 0):.4f}",
        f"  F1 Score                       : {metrics.get('f1_score', 0):.4f}",
        f"  IoU @ Threshold (Jaccard)      : {metrics.get('iou_at_threshold', 0):.4f}",
        f"  Mean IoU (matched pairs)       : {metrics.get('mean_iou', 0):.4f}",
        "",
        "  Classification",
        "  " + "-" * 40,
        f"  Label Accuracy (matched)       : {metrics.get('classification_accuracy', 0):.4f}",
        f"  No-Hazard Frame Accuracy       : {metrics.get('no_hazard_accuracy', 0):.4f}",
        f"  False Positive Rate            : {metrics.get('false_positive_rate', 0):.4f}",
        "",
        "  Severity",
        "  " + "-" * 40,
        f"  Severity Accuracy (exact)      : {metrics.get('severity_accuracy', 0):.4f}",
        f"  Severity Within ±1             : {metrics.get('severity_within_one', 0):.4f}",
        f"  Severity Spearman ρ            : {metrics.get('severity_spearman', 0):.4f}",
        "",
        "  Counts",
        "  " + "-" * 40,
        f"  Total Frames                   : {metrics.get('total_frames', 0)}",
        f"  GT Hazards / Pred Hazards      : "
        f"{metrics.get('total_gt_hazards', 0)} / {metrics.get('total_pred_hazards', 0)}",
        f"  TP / FP / FN / TN              : "
        f"{metrics.get('true_positives', 0)} / {metrics.get('false_positives', 0)} / "
        f"{metrics.get('false_negatives', 0)} / {metrics.get('true_negatives', 0)}",
        f"  Parse Failures                 : "
        f"{metrics.get('parse_failures', 0)} "
        f"({metrics.get('parse_failure_rate', 0):.1%})",
        "",
        "  Per-Class Metrics (F1 / Precision / Recall / n)",
        "  " + "-" * 40,
    ]

    per_class = metrics.get("per_class_metrics", {})
    for label in VALID_LABELS:
        if label == "no_hazard":
            continue
        m = per_class.get(label)
        if m and m["count"] > 0:
            lines.append(
                f"  {label:<30}  F1={m['f1']:.3f}  "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  n={m['count']}"
            )

    lines += ["", "=" * 62]
    return "\n".join(lines)
