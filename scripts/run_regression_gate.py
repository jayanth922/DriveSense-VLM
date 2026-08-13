#!/usr/bin/env python3
"""CI regression gate: fail if a new model's eval regresses vs a baseline.

Compares Level-1 grounding metrics (hazard_detection_rate,
detection_rate_by_iou@0.1/0.3/0.5, mean_best_pair_iou, parse_failure_rate,
classification_accuracy) between a baseline ``eval_summary.json`` (the last
known-good checkpoint) and a new one, and fails (exit 1) if any metric
regressed beyond its tolerance. Meant to run in CI before promoting a new
checkpoint.

Usage:
    python scripts/run_regression_gate.py \
        --baseline outputs/eval/v2/eval_summary.json \
        --new      outputs/eval/v3/eval_summary.json

    # Tighten/loosen one metric's tolerance (repeatable):
    python scripts/run_regression_gate.py --baseline ... --new ... \
        --tolerance "detection_rate_by_iou@0.1=0.05"

    # Save the machine-readable report:
    python scripts/run_regression_gate.py --baseline ... --new ... \
        --output outputs/eval/regression_report.json

Exit code: 0 = pass, 1 = at least one metric regressed beyond tolerance,
2 = bad input (missing file, unknown --tolerance metric name).
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

from drivesense.eval.regression import (  # noqa: E402
    DEFAULT_METRICS,
    compare_summaries,
    load_eval_summary,
)

_COL = (10, 12, 12, 10, 12)  # baseline, new, Δ, tol, status column widths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="CI gate: fail if a new eval run regresses vs a baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--baseline", required=True, help="Baseline eval_summary.json")
    p.add_argument("--new", required=True, help="New (candidate) eval_summary.json")
    p.add_argument("--tolerance", action="append", default=[], metavar="NAME=FLOAT",
                   help="Override one metric's relative tolerance (repeatable).")
    p.add_argument("--output", default=None, help="Write the JSON report here")
    return p.parse_args()


def apply_tolerance_overrides(metrics: dict[str, dict], overrides: list[str]) -> dict[str, dict]:
    """Apply ``NAME=FLOAT`` tolerance overrides onto a copy of ``metrics``."""
    metrics = {k: dict(v) for k, v in metrics.items()}
    for item in overrides:
        name, sep, val = item.partition("=")
        if not sep:
            print(f"ERROR: --tolerance must be NAME=FLOAT, got {item!r}", file=sys.stderr)
            sys.exit(2)
        if name not in metrics:
            print(f"ERROR: unknown metric {name!r} for --tolerance. Valid: "
                  f"{sorted(metrics)}", file=sys.stderr)
            sys.exit(2)
        try:
            metrics[name]["tolerance"] = float(val)
        except ValueError:
            print(f"ERROR: --tolerance value must be a float, got {val!r}", file=sys.stderr)
            sys.exit(2)
    return metrics


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def print_report(results: list[dict]) -> None:
    """Print a clear pass/fail table with before/after values."""
    print("\n" + "=" * 78)
    print("  REGRESSION GATE REPORT")
    print("=" * 78)
    header = f"  {'metric':<28}{'baseline':>{_COL[0]}}{'new':>{_COL[1]}}" \
             f"{'Δ (rel)':>{_COL[2]}}{'tol':>{_COL[3]}}{'status':>{_COL[4]}}"
    print(header)
    print("-" * 78)
    for r in results:
        delta = "—" if r["relative_change"] is None else f"{r['relative_change']:+.1%}"
        mark = {"REGRESSED": "✗ REGRESSED", "IMPROVED": "✓ improved",
                "FLAT": "= flat", "MISSING": "? missing"}[r["status"]]
        print(f"  {r['metric']:<28}{_fmt(r['baseline']):>{_COL[0]}}{_fmt(r['new']):>{_COL[1]}}"
              f"{delta:>{_COL[2]}}{r['tolerance']:>{_COL[3]}.0%}{mark:>{_COL[4]}}")
    print("=" * 78)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    baseline_path, new_path = Path(args.baseline), Path(args.new)
    for p in (baseline_path, new_path):
        if not p.exists():
            print(f"ERROR: eval summary not found: {p}", file=sys.stderr)
            sys.exit(2)

    baseline = load_eval_summary(baseline_path)
    new = load_eval_summary(new_path)
    metrics = apply_tolerance_overrides(DEFAULT_METRICS, args.tolerance)
    results = compare_summaries(baseline, new, metrics)
    print_report(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nReport written to {out}")

    failed = [r for r in results if r["regressed"]]
    if failed:
        print(f"\n❌ REGRESSION GATE FAILED — {len(failed)} metric(s) regressed beyond "
              "tolerance:")
        for r in failed:
            print(f"   ✗ {r['metric']}: {_fmt(r['baseline'])} → {_fmt(r['new'])} "
                  f"({r['relative_change']:+.1%}, tolerance {r['tolerance']:.0%})")
        sys.exit(1)

    print("\n✅ REGRESSION GATE PASSED — no metric regressed beyond tolerance.")


if __name__ == "__main__":
    main()
