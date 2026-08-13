#!/usr/bin/env python3
"""Failure-mode stratification: which size tier / condition is grounding worst on?

Level-1 eval showed uniform near-zero box-grounding across ALL hazard classes
regardless of training-example count (24 to 841 examples per class) — ruling
out "just need more examples of class X." This cross-tabulates per-hazard
localization quality by GT box SIZE TIER (a distance proxy) against
weather/time_of_day/location, using the existing grounding.py compute_iou
logic (not reimplemented), to find out what's ACTUALLY hard.

Runs standalone on existing eval artifacts — no model, no GPU.

Usage:
    python scripts/analyze_failure_stratification.py \
        --predictions outputs/predictions/test_predictions.jsonl \
        --ground-truth outputs/data/sft_ready_v2_merged/sft_test_enriched.jsonl

    # Write the machine-readable report (feeds select_mining_targets.py):
    python scripts/analyze_failure_stratification.py --predictions ... --ground-truth ... \
        --output outputs/eval/failure_stratification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.eval.failure_stratification import (  # noqa: E402
    DEFAULT_DIMENSIONS,
    DEFAULT_THRESHOLDS,
    build_report,
    load_stratified_ground_truth,
)
from drivesense.eval.grounding import GroundingEvaluator  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Cross-tabulate grounding failure by box size tier x condition.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--predictions", required=True, help="predictions.jsonl")
    p.add_argument("--ground-truth", required=True, help="Enriched GT JSONL (has stratification)")
    p.add_argument("--dimensions", nargs="+", default=list(DEFAULT_DIMENSIONS),
                   choices=["weather", "time_of_day", "location"])
    p.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    p.add_argument("--min-samples", type=int, default=5,
                   help="Minimum hazards in a bucket to rank it as worst/best.")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--output", default=None, help="Write the full JSON report here")
    return p.parse_args()


def _fmt_rate(bucket: dict, thresholds: list[float]) -> str:
    dr = bucket["detection_rate_by_iou"]
    return " / ".join(f"{dr.get(str(t), 0.0):.1%}" for t in thresholds)


def print_report(report: dict, thresholds: list[float]) -> None:
    """Print the size-tier summary + worst/best buckets as tables."""
    thr_hdr = "/".join(f"@{t}" for t in thresholds)
    print("\n" + "=" * 78)
    print("  FAILURE STRATIFICATION REPORT")
    print("=" * 78)
    print(f"  frames={report['n_frames']}  hazards={report['n_hazards']}")
    o = report["overall"]
    print(f"  OVERALL: mean_best_pair_iou={o['mean_best_pair_iou']:.4f}  "
          f"det_rate{thr_hdr}={_fmt_rate(o, thresholds)}")

    print(f"\n  {'size tier':<12}{'n':>8}{'mean_iou':>12}{'det_rate' + thr_hdr:>24}")
    print("  " + "-" * 56)
    for tier, m in report["size_tier_summary"].items():
        print(f"  {tier:<12}{m['n']:>8}{m['mean_best_pair_iou']:>12.4f}"
              f"{_fmt_rate(m, thresholds):>24}")

    print(f"\n  WORST buckets (min {report.get('_min_samples', '?')} samples):")
    for b in report["worst_buckets"]:
        print(f"    {b['bucket']:<28} n={b['n']:<5} mean_iou={b['mean_best_pair_iou']:.4f}  "
              f"det_rate={_fmt_rate(b, thresholds)}")

    print("\n  BEST buckets:")
    for b in report["best_buckets"]:
        print(f"    {b['bucket']:<28} n={b['n']:<5} mean_iou={b['mean_best_pair_iou']:.4f}  "
              f"det_rate={_fmt_rate(b, thresholds)}")
    print("=" * 78)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    for label, path in (("--predictions", args.predictions), ("--ground-truth", args.ground_truth)):
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(2)

    evaluator = GroundingEvaluator({})
    predictions = evaluator.load_predictions(Path(args.predictions))
    ground_truth = load_stratified_ground_truth(args.ground_truth)

    report = build_report(
        predictions, ground_truth, dimensions=tuple(args.dimensions),
        thresholds=tuple(args.thresholds), min_samples=args.min_samples, top_n=args.top_n,
    )
    report["_min_samples"] = args.min_samples
    print_report(report, args.thresholds)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
