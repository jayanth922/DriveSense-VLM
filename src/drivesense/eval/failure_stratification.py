"""Failure-mode stratification for Level-1 grounding — the "why is grounding
near-zero everywhere" analysis.

Cross-tabulates per-GT-hazard localization quality by box SIZE TIER (a proxy
for distance — a closer/larger hazard occupies more of the frame) against the
test set's condition metadata (weather, time_of_day, location), so a flat,
uniform failure rate across hazard CLASSES can be re-examined along dimensions
that might actually explain it.

Reuses the existing grounding primitives rather than reimplementing them:
``compute_iou`` for all box overlap, ``GroundingEvaluator.load_predictions``
for prediction loading, ``AnnotationValidator.parse_llm_response`` for GT
assistant-turn parsing, and ``_box_hazards``/``BOX_EXEMPT_LABELS`` for the
box-exempt filter. Only the per-hazard stratified aggregation is new — the
existing ``compute_grounding_metrics`` aggregates ``detection_rate_by_iou``/
``mean_best_pair_iou`` at the FRAME level (one best-IoU value per frame), which
can't be sliced by individual hazard size when a frame holds hazards of
different sizes. This module computes the same underlying quantity — the best
IoU between a GT box and any predicted box in its frame, via the same
``compute_iou`` — at per-HAZARD granularity instead, which is what makes a
size-tier cross-tab possible at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from drivesense.data.annotation import AnnotationValidator
from drivesense.eval.grounding import BOX_EXEMPT_LABELS, compute_iou

logger = logging.getLogger(__name__)

# (tier_name, min_area_pct_inclusive, max_area_pct_exclusive_or_None)
SIZE_TIERS: tuple[tuple[str, float, float | None], ...] = (
    ("tiny", 0.0, 1.0),
    ("small", 1.0, 5.0),
    ("medium", 5.0, 15.0),
    ("large", 15.0, None),
)

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.1, 0.3, 0.5)
DEFAULT_DIMENSIONS: tuple[str, ...] = ("weather", "time_of_day", "location")

_FRAME_AREA = 1000.0 * 1000.0


# ---------------------------------------------------------------------------
# Geometry: size tier + aspect ratio
# ---------------------------------------------------------------------------


def box_area_pct(bbox_2d: list[float]) -> float:
    """Box area as a percentage of the frame (bbox in [0, 1000] coordinates)."""
    x1, y1, x2, y2 = bbox_2d
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return (w * h) / _FRAME_AREA * 100.0


def aspect_ratio(bbox_2d: list[float]) -> float:
    """Box width/height ratio (0.0 if degenerate)."""
    x1, y1, x2, y2 = bbox_2d
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return w / h if h > 0 else 0.0


def size_tier_of(area_pct: float) -> str:
    """Bucket a box's area percentage into a size tier (see :data:`SIZE_TIERS`)."""
    for name, lo, hi in SIZE_TIERS:
        if area_pct >= lo and (hi is None or area_pct < hi):
            return name
    return SIZE_TIERS[-1][0]  # unreachable given the tiers are contiguous+open-ended


# ---------------------------------------------------------------------------
# Stratification-preserving GT loading
#
# GroundingEvaluator.load_ground_truth() / _normalise_gt() intentionally drop
# every field except frame_id/hazards/scene_summary/ego_context — this loader
# keeps weather/time_of_day/location too, reusing the same JSON-extraction
# logic (AnnotationValidator.parse_llm_response) rather than reimplementing it.
# ---------------------------------------------------------------------------


def load_stratified_ground_truth(path: str | Path) -> list[dict]:
    """Load GT keeping stratification metadata alongside hazards.

    Args:
        path: SFT-enriched JSONL (``messages`` format) or a direct-format
            JSONL/JSON-array with a ``hazards`` key. Either way, top-level
            ``weather``/``time_of_day``/``location`` are preserved if present.

    Returns:
        ``[{"frame_id", "weather", "time_of_day", "location", "hazards"}, ...]``.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    records = json.loads(text) if text.startswith("[") else [
        json.loads(ln) for ln in text.splitlines() if ln.strip()
    ]
    return [_normalise_stratified_gt(r) for r in records if isinstance(r, dict)]


def _normalise_stratified_gt(rec: dict) -> dict:
    """Like ``grounding._normalise_gt`` but keeps stratification fields."""
    strata = {
        "weather": rec.get("weather", "unknown"),
        "time_of_day": rec.get("time_of_day", "unknown"),
        "location": rec.get("location", "unknown"),
    }
    if "messages" in rec:
        for msg in rec.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            parsed = AnnotationValidator.parse_llm_response(
                content if isinstance(content, str) else json.dumps(content))
            if parsed:
                return {"frame_id": rec.get("frame_id", ""), "hazards": parsed.get("hazards", []),
                       **strata}
        return {"frame_id": rec.get("frame_id", ""), "hazards": [], **strata}
    return {"frame_id": rec.get("frame_id", ""), "hazards": rec.get("hazards", []), **strata}


# ---------------------------------------------------------------------------
# Per-hazard rows (the unit of stratification)
# ---------------------------------------------------------------------------


def build_hazard_rows(predictions: list[dict], ground_truth: list[dict]) -> list[dict]:
    """Build one row per box-bearing GT hazard, with its best matching pred IoU.

    For each frame, a GT hazard's "best IoU" is the max ``compute_iou`` value
    against every predicted hazard in that same frame — the same quantity
    ``compute_grounding_metrics`` already uses for ``best_pair_ious``, just
    kept per-hazard instead of collapsed to one max-per-frame value, which is
    what makes slicing by individual hazard size possible.

    Args:
        predictions:  Normalised prediction dicts (frame_id, hazards, ...) —
            e.g. from ``GroundingEvaluator.load_predictions``.
        ground_truth: Stratified GT dicts from :func:`load_stratified_ground_truth`.

    Returns:
        Rows: ``{frame_id, weather, time_of_day, location, label, bbox_2d,
        area_pct, aspect_ratio, size_tier, best_iou}``.
    """
    pred_by_frame = {p.get("frame_id", ""): p for p in predictions}
    rows: list[dict] = []
    for gt in ground_truth:
        frame_id = gt.get("frame_id", "")
        gt_hazards = [h for h in gt.get("hazards", []) if h.get("label") not in BOX_EXEMPT_LABELS]
        pred = pred_by_frame.get(frame_id, {})
        if pred.get("parse_failure", False):
            pred_hazards: list[dict] = []
        else:
            pred_hazards = [h for h in pred.get("hazards", [])
                           if h.get("label") not in BOX_EXEMPT_LABELS]

        for h in gt_hazards:
            bbox = h.get("bbox_2d")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            best_iou = max(
                (compute_iou(bbox, ph.get("bbox_2d", [0, 0, 0, 0])) for ph in pred_hazards),
                default=0.0,
            )
            area = box_area_pct(bbox)
            rows.append({
                "frame_id": frame_id,
                "weather": gt.get("weather", "unknown"),
                "time_of_day": gt.get("time_of_day", "unknown"),
                "location": gt.get("location", "unknown"),
                "label": h.get("label", ""),
                "bbox_2d": list(bbox),
                "area_pct": round(area, 4),
                "aspect_ratio": round(aspect_ratio(bbox), 4),
                "size_tier": size_tier_of(area),
                "best_iou": round(best_iou, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Bucket metrics + cross-tabulation
# ---------------------------------------------------------------------------


def compute_bucket_metrics(
    rows: list[dict], thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict:
    """Aggregate per-hazard rows into detection_rate_by_iou + mean_best_pair_iou.

    Args:
        rows:       Hazard rows (or a filtered subset) from :func:`build_hazard_rows`.
        thresholds: IoU thresholds for ``detection_rate_by_iou``.

    Returns:
        ``{"n": int, "mean_best_pair_iou": float, "detection_rate_by_iou": {thr: rate}}``.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0, "mean_best_pair_iou": 0.0,
                "detection_rate_by_iou": {str(t): 0.0 for t in thresholds}}
    ious = [r["best_iou"] for r in rows]
    return {
        "n": n,
        "mean_best_pair_iou": round(sum(ious) / n, 4),
        "detection_rate_by_iou": {
            str(t): round(sum(1 for v in ious if v >= t) / n, 4) for t in thresholds
        },
    }


def cross_tabulate(
    rows: list[dict], dimension: str, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, dict]:
    """Cross-tab: for each size tier AND each value of ``dimension``, bucket metrics.

    Args:
        rows:       Hazard rows from :func:`build_hazard_rows`.
        dimension:  ``"weather"``, ``"time_of_day"``, or ``"location"``.
        thresholds: IoU thresholds for ``detection_rate_by_iou``.

    Returns:
        ``{"<size_tier>|<dim_value>": bucket_metrics, ...}`` plus marginal rows
        ``"<size_tier>|ALL"`` and ``"ALL|<dim_value>"``.
    """
    tiers = [t[0] for t in SIZE_TIERS]
    values = sorted({r[dimension] for r in rows})
    out: dict[str, dict] = {}
    for tier in tiers:
        tier_rows = [r for r in rows if r["size_tier"] == tier]
        out[f"{tier}|ALL"] = compute_bucket_metrics(tier_rows, thresholds)
        for val in values:
            cell_rows = [r for r in tier_rows if r[dimension] == val]
            out[f"{tier}|{val}"] = compute_bucket_metrics(cell_rows, thresholds)
    for val in values:
        val_rows = [r for r in rows if r[dimension] == val]
        out[f"ALL|{val}"] = compute_bucket_metrics(val_rows, thresholds)
    return out


def rank_buckets(
    cross_tab: dict[str, dict], min_samples: int = 5, top_n: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Rank cross-tab cells by ``mean_best_pair_iou``; return (worst, best).

    Cells with fewer than ``min_samples`` hazards are excluded — a bucket with
    2 examples isn't a reliable "this is the hardest condition" signal.

    Args:
        cross_tab:   Output of :func:`cross_tabulate`.
        min_samples: Minimum ``n`` for a cell to be eligible for ranking.
        top_n:       How many worst/best cells to return.

    Returns:
        ``(worst, best)`` — each a list of ``{"bucket": key, **metrics}``,
        sorted by ``mean_best_pair_iou`` ascending / descending.
    """
    eligible = [
        {"bucket": key, **metrics} for key, metrics in cross_tab.items()
        if metrics["n"] >= min_samples
    ]
    worst = sorted(eligible, key=lambda r: r["mean_best_pair_iou"])[:top_n]
    best = sorted(eligible, key=lambda r: -r["mean_best_pair_iou"])[:top_n]
    return worst, best


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def build_report(
    predictions: list[dict],
    ground_truth: list[dict],
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    min_samples: int = 5,
    top_n: int = 5,
) -> dict:
    """Build the full stratification report.

    Args:
        predictions:  Normalised prediction dicts.
        ground_truth: Stratified GT dicts from :func:`load_stratified_ground_truth`.
        dimensions:   Frame-level condition dimensions to cross with size tier.
        thresholds:   IoU thresholds for ``detection_rate_by_iou``.
        min_samples:  Minimum bucket size to be eligible for the worst/best ranking.
        top_n:        How many worst/best buckets to report.

    Returns:
        ``{"n_frames", "n_hazards", "overall", "size_tier_summary",
        "cross_tabs": {dim: {...}}, "worst_buckets", "best_buckets"}``.
    """
    rows = build_hazard_rows(predictions, ground_truth)
    overall = compute_bucket_metrics(rows, thresholds)

    size_tier_summary = {
        tier: compute_bucket_metrics([r for r in rows if r["size_tier"] == tier], thresholds)
        for tier, _, _ in SIZE_TIERS
    }

    cross_tabs = {dim: cross_tabulate(rows, dim, thresholds) for dim in dimensions}

    # Rank over the UNION of all cross-tab cells (across every dimension) so
    # "worst bucket" can point at whichever condition combo is actually worst,
    # not just the worst within one dimension.
    combined: dict[str, dict] = {}
    for dim, tab in cross_tabs.items():
        combined.update({f"{dim}:{k}": v for k, v in tab.items()})
    worst, best = rank_buckets(combined, min_samples=min_samples, top_n=top_n)

    return {
        "n_frames": len({r["frame_id"] for r in rows}),
        "n_hazards": len(rows),
        "overall": overall,
        "size_tier_summary": size_tier_summary,
        "cross_tabs": cross_tabs,
        "worst_buckets": worst,
        "best_buckets": best,
    }
