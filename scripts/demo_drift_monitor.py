#!/usr/bin/env python3
"""Concrete, runnable DriftMonitor demonstration — no production data required.

Splits an eval-format label set into two halves, treats one as the REFERENCE
distribution and the other as an INCOMING batch, and shows:

  Case A — incoming = the other (unmodified) half of the SAME split:
           expect NO dimension flagged as drifted (same underlying distribution).
  Case B — incoming = that half with weather forced to "rain" for every record
           (a deliberate, clearly-labeled synthetic skew): expect the
           `weather` dimension to be flagged as drifted; other dimensions
           should stay clean.

Runs standalone with no external file: if ``--labels`` isn't given (or the
path doesn't exist), a synthetic-but-realistic label set is generated in
memory so this demo is always runnable. Pass a real ``sft_test_enriched.jsonl``
via ``--labels`` to demo against actual eval data instead.

Usage:
    python scripts/demo_drift_monitor.py
    python scripts/demo_drift_monitor.py --labels outputs/data/sft_ready/sft_test_enriched.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.monitoring.drift import DEFAULT_DIMENSIONS, DriftMonitor  # noqa: E402

_WEATHERS = ["clear"] * 8 + ["rain"] * 1 + ["fog"] * 1
_TIMES = ["day"] * 7 + ["night"] * 3
_LOCATIONS = ["urban"] * 5 + ["intersection"] * 4 + ["highway"] * 1
_LABELS = ["jaywalking", "occluded_pedestrian", "cyclist_proximity",
           "construction_zone", "unusual_object", "no_hazard"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=None,
                   help="Real sft_*_enriched.jsonl to demo against (default: synthetic).")
    p.add_argument("--n-synthetic", type=int, default=400,
                   help="Number of synthetic records to generate if --labels is unset.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.20, help="PSI drift threshold.")
    return p.parse_args()


def make_synthetic_records(n: int, seed: int) -> list[dict]:
    """Generate a realistic-shaped, reproducible synthetic label set."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        n_hazards = rng.choice([0, 1, 1, 2])
        hazards = [{"label": rng.choice(_LABELS)} for _ in range(n_hazards)]
        records.append({
            "frame_id": f"synthetic-{i:04d}",
            "weather": rng.choice(_WEATHERS),
            "time_of_day": rng.choice(_TIMES),
            "location": rng.choice(_LOCATIONS),
            "messages": [{"role": "assistant", "content": json.dumps({"hazards": hazards})}],
        })
    return records


def load_records(path: str) -> list[dict]:
    """Load SFT-format records from a JSONL file."""
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def split_half(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic 50/50 shuffle-split."""
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]


def force_rain(records: list[dict]) -> list[dict]:
    """Return a copy of ``records`` with weather overridden to 'rain' for every record.

    This is a DELIBERATE SYNTHETIC OVERRIDE for demonstration purposes only —
    it does not reflect any real weather in the source data.
    """
    out = copy.deepcopy(records)
    for r in out:
        r["weather"] = "rain"
    return out


def print_case(title: str, report: dict, threshold: float) -> None:
    """Print one drift-check report as a table."""
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")
    drift_col = f"drifted (>= {threshold:.2f})"
    print(f"  {'dimension':<16}{'psi':>10}{'severity':>14}{drift_col:>18}")
    print("  " + "-" * 66)
    for dim, r in report.items():
        flag = "YES — DRIFT" if r["drifted"] else "no"
        print(f"  {dim:<16}{r['psi']:>10.4f}{r['severity']:>14}{flag:>18}")
    any_drift = DriftMonitor.any_drifted(report)
    print(f"\n  Verdict: {'⚠️  DRIFT DETECTED' if any_drift else '✓ no drift detected'}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if args.labels and Path(args.labels).exists():
        records = load_records(args.labels)
        print(f"Loaded {len(records)} real records from {args.labels}")
    else:
        if args.labels:
            print(f"'{args.labels}' not found — generating synthetic records instead.")
        records = make_synthetic_records(args.n_synthetic, args.seed)
        print(f"Generated {len(records)} synthetic records (seed={args.seed})")

    reference, incoming = split_half(records, args.seed)
    print(f"Split: {len(reference)} reference / {len(incoming)} incoming")

    monitor = DriftMonitor.from_records(reference, DEFAULT_DIMENSIONS, threshold=args.threshold)

    report_a = monitor.check(incoming)
    print_case("CASE A — incoming = same distribution (expect NO drift)", report_a,
              args.threshold)

    skewed = force_rain(incoming)
    print("\n  >>> SYNTHETIC OVERRIDE for Case B: weather forced to 'rain' for every "
          "incoming record. <<<")
    report_b = monitor.check(skewed)
    print_case("CASE B — incoming = weather-skewed (expect 'weather' DRIFT)", report_b,
               args.threshold)

    ok_a = not DriftMonitor.any_drifted(report_a)
    ok_b = report_b["weather"]["drifted"]
    print(f"\n{'=' * 70}")
    print(f"  Case A correctly showed no drift : {'PASS' if ok_a else 'FAIL'}")
    print(f"  Case B correctly flagged weather  : {'PASS' if ok_b else 'FAIL'}")
    print("=" * 70)
    if not (ok_a and ok_b):
        sys.exit(1)


if __name__ == "__main__":
    main()
