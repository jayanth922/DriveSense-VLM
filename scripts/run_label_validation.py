#!/usr/bin/env python3
"""Annotation v2 gate: reject a training-label set with degenerate boxes.

Runs BEFORE any training. Fails (exit 1) if the label boxes show the collapse
signature that sank v1 (constant / full-frame / template boxes). Box-exempt
labels (high_density, no_hazard) are excluded from the box statistics.

Usage:
    python scripts/run_label_validation.py --labels outputs/data/sft_ready/sft_train.jsonl
    python scripts/run_label_validation.py --labels <file> --min-unique-ratio 0.5 \
        --max-single-box-freq 0.02 --max-dup-frames 5

Fails on any of:
    * unique_box_ratio       < --min-unique-ratio        (default 0.50)
    * max_single_box_freq    > --max-single-box-freq      (default 0.02)
    * boxes with area        > 40% of frame               (any)
    * no_hazard / high_density hazards carrying a bbox     (any)
    * one identical box shared across  > max(--max-dup-frames, --max-box-frame-share × N)
      frames, where N is the number of frames — the cross-frame-dup limit SCALES with
      dataset size (a static roadside object naturally recurs more at scale), while the
      absolute floor still catches true collapse (v1 put one box on 780 frames).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data.box_sourcing import BOX_EXEMPT_LABELS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_label_validation")

_FRAME_AREA = 1000.0 * 1000.0
_MAX_AREA_FRAC = 0.40


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Validate annotation-v2 label boxes before training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--labels", required=True, help="Path to the labels JSONL/JSON.")
    p.add_argument("--min-unique-ratio", type=float, default=0.50)
    p.add_argument("--max-single-box-freq", type=float, default=0.02)
    p.add_argument("--max-dup-frames", type=int, default=5,
                   help="Absolute floor for frames sharing one identical box (small sets).")
    p.add_argument("--max-box-frame-share", type=float, default=0.005,
                   help="Cross-frame-dup limit as a fraction of frame count; effective "
                        "limit = max(--max-dup-frames, this*N). Scales the gate to set size.")
    return p.parse_args()


def _dup_limit(n_frames: int, args: argparse.Namespace) -> int:
    """Effective 'frames sharing one box' limit — scales with dataset size.

    ``max(--max-dup-frames, ceil(--max-box-frame-share × N))``: the absolute floor
    keeps small sets strict; the share term lets a legitimately recurring static
    object grow with the dataset while still tripping on true collapse (v1 had one
    box on 780 frames — far above 0.5% of any realistic frame count).
    """
    return max(args.max_dup_frames, math.ceil(args.max_box_frame_share * n_frames))


def _parse_json(text: object) -> dict | None:
    """Best-effort extract of the first JSON object from text/dict."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _hazards_of(record: dict) -> list[dict]:
    """Extract the hazard list from an SFT (messages) or direct-format record."""
    if "hazards" in record and isinstance(record["hazards"], list):
        return record["hazards"]
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            parsed = _parse_json(msg.get("content", ""))
            if parsed:
                return parsed.get("hazards", [])
    return []


def _read_records(path: Path) -> list[dict]:
    """Read a JSONL (one object per line) or JSON-array label file."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return [r for r in json.loads(text) if isinstance(r, dict)]
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def collect_stats(records: list[dict]) -> dict:
    """Compute box-diversity and schema-violation statistics.

    Args:
        records: Label records (SFT or direct format).

    Returns:
        Stats dict used by :func:`evaluate_gate`.
    """
    box_counter: Counter = Counter()               # bbox tuple -> count
    box_frames: dict[tuple, set] = {}              # bbox tuple -> {frame_id}
    n_frames = len(records)
    total_boxes = 0
    oversized = 0
    exempt_with_box = 0

    for rec in records:
        fid = rec.get("frame_id", id(rec))
        for hz in _hazards_of(rec):
            label = hz.get("label", "")
            bbox = hz.get("bbox_2d")
            has_box = isinstance(bbox, (list, tuple)) and len(bbox) == 4
            if label in BOX_EXEMPT_LABELS:
                if has_box:
                    exempt_with_box += 1
                continue
            if not has_box:
                continue
            key = tuple(bbox)
            box_counter[key] += 1
            box_frames.setdefault(key, set()).add(fid)
            total_boxes += 1
            x1, y1, x2, y2 = bbox
            if max(0.0, x2 - x1) * max(0.0, y2 - y1) / _FRAME_AREA > _MAX_AREA_FRAC:
                oversized += 1

    unique = len(box_counter)
    top_box, top_count = (box_counter.most_common(1)[0] if box_counter else ((), 0))
    max_dup_frames, dup_box = max(
        ((len(f), b) for b, f in box_frames.items()), default=(0, ())
    )
    return {
        "n_frames": n_frames,
        "total_boxes": total_boxes,
        "unique_boxes": unique,
        "unique_box_ratio": round(unique / total_boxes, 4) if total_boxes else 0.0,
        "max_single_box_freq": round(top_count / total_boxes, 4) if total_boxes else 0.0,
        "top_box": list(top_box), "top_box_count": top_count,
        "oversized_gt_40pct": oversized,
        "exempt_labels_with_box": exempt_with_box,
        "max_frames_sharing_one_box": max_dup_frames,
        "most_shared_box": list(dup_box),
        "top_5_boxes": [[list(b), c] for b, c in box_counter.most_common(5)],
    }


def evaluate_gate(stats: dict, args: argparse.Namespace) -> list[str]:
    """Return the list of gate failures (empty == pass)."""
    failures: list[str] = []
    dup_limit = _dup_limit(stats.get("n_frames", 0), args)
    logger.info("cross-frame dup limit: %d  (max(%d, %.2f%% of %d frames))",
                dup_limit, args.max_dup_frames, args.max_box_frame_share * 100,
                stats.get("n_frames", 0))
    # Schema violation is checked first — it is independent of box count.
    if stats["exempt_labels_with_box"] > 0:
        failures.append(
            f"{stats['exempt_labels_with_box']} high_density/no_hazard hazards carry a bbox"
        )
    if stats["total_boxes"] == 0:
        if not failures:
            failures.append("no boxes found in label set")
        return failures
    if stats["unique_box_ratio"] < args.min_unique_ratio:
        failures.append(
            f"unique_box_ratio {stats['unique_box_ratio']} < {args.min_unique_ratio}"
        )
    # Collapse = a box repeated across MANY frames. A handful of benign repeats
    # (a static object across a few kept keyframes) is fine, and on a small split
    # its k/N ratio would otherwise trip spuriously. Gate the ratio check on the
    # size-scaled ABSOLUTE repeat count so it is robust to split size.
    if (stats["top_box_count"] > dup_limit
            and stats["max_single_box_freq"] > args.max_single_box_freq):
        failures.append(
            f"max_single_box_freq {stats['max_single_box_freq']} > "
            f"{args.max_single_box_freq} (box {stats['top_box']} ×{stats['top_box_count']})"
        )
    if stats["oversized_gt_40pct"] > 0:
        failures.append(f"{stats['oversized_gt_40pct']} boxes exceed 40% frame area")
    # Cross-frame duplication limit SCALES with dataset size (see _dup_limit): a
    # static roadside object legitimately recurs on more keyframes as N grows.
    if stats["max_frames_sharing_one_box"] > dup_limit:
        failures.append(
            f"one box appears on {stats['max_frames_sharing_one_box']} frames > {dup_limit} "
            f"(limit = max({args.max_dup_frames}, {args.max_box_frame_share:.2%} of "
            f"{stats.get('n_frames', 0)} frames); box {stats['most_shared_box']})"
        )
    return failures


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    path = Path(args.labels)
    if not path.exists():
        logger.error("Labels file not found: %s", path)
        sys.exit(1)

    records = _read_records(path)
    stats = collect_stats(records)
    failures = evaluate_gate(stats, args)

    print("\n=== Label Validation Gate ===")
    print(json.dumps(stats, indent=2))
    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("\n✓ GATE PASSED — boxes are diverse, bounded, and schema-clean.")


if __name__ == "__main__":
    main()
