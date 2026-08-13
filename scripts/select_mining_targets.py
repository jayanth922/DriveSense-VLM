#!/usr/bin/env python3
"""Closed-loop mining target selection — failure analysis drives the NEXT mine.

Takes a failure-stratification report (from analyze_failure_stratification.py)
plus the global metadata.jsonl (all 34,149 nuScenes keyframes) and scores
un-mined candidate frames by how well they match the WORST-performing bucket,
producing a new shopping-list JSONL ready for run_streaming_miner.py. This
closes the loop: eval failure -> what's actually hard -> targeted mining -> (a
future) label -> gate -> train -> eval again.

Usage:
    python scripts/select_mining_targets.py \
        --report outputs/eval/failure_stratification.json \
        --metadata outputs/data/spark_processed/metadata.jsonl \
        --have-manifest outputs/data/have_basenames.txt \
        --output outputs/data/mining_shoppinglist.jsonl \
        --target-count 2000

IMPORTANT: after writing the new list, run the miner with --no-rebuild-list
(pointing at the SAME shoppinglist_path) — otherwise the miner's implicit
rebuild-when-populated default will silently regenerate a plain rarity-sampled
list and overwrite this targeted one:

    python scripts/run_streaming_miner.py --no-rebuild-list ...
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

from drivesense.data.mining_targets import (  # noqa: E402
    clean_target_spec,
    parse_bucket_key,
    score_histogram,
    select_targets,
    worst_bucket_to_target_spec,
)
from drivesense.data.streaming_miner import load_have_basenames, write_shoppinglist  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Select mining targets from a failure-stratification report.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--report", required=True, help="failure_stratification.json")
    p.add_argument("--metadata", required=True, help="Global metadata.jsonl")
    p.add_argument("--bucket", default=None,
                   help="Explicit bucket key to target instead of the report's #1 worst "
                        "(e.g. 'weather:tiny|rain'). Useful to target the 2nd/3rd worst.")
    p.add_argument("--have-manifest", default=None, help="Already-mined basenames file")
    p.add_argument("--cam-front-dir", default=None, help="Also exclude physically-present images")
    p.add_argument("--band", nargs=2, type=int, default=[3, 20], metavar=("LO", "HI"))
    p.add_argument("--count-mode", choices=["hazard_class", "num_annotations"],
                   default="hazard_class")
    p.add_argument("--target-count", type=int, default=2000)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/data/mining_shoppinglist.jsonl")
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    for label, path in (("--report", args.report), ("--metadata", args.metadata)):
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(2)

    report = json.loads(Path(args.report).read_text())
    target = (clean_target_spec(parse_bucket_key(args.bucket)) if args.bucket
             else worst_bucket_to_target_spec(report))
    if not target:
        print("ERROR: no usable (supported-dimension) target spec — the worst bucket may "
              "only carry 'location', which metadata.jsonl can't score. Pass --bucket "
              "explicitly, or re-run analyze_failure_stratification.py with more --dimensions.",
              file=sys.stderr)
        sys.exit(2)
    print(f"Target spec: {target}")

    already_have = load_have_basenames(args.have_manifest) if args.have_manifest else set()

    rows = select_targets(
        args.metadata, target, already_have=already_have, cam_front_dir=args.cam_front_dir,
        band=tuple(args.band), hazard_count_mode=args.count_mode,
        target_count=args.target_count, min_score=args.min_score, seed=args.seed,
    )
    if not rows:
        print("WARNING: 0 candidates matched — nothing written.", file=sys.stderr)
        sys.exit(1)

    write_shoppinglist(rows, args.output)
    hist = score_histogram(rows)
    print(f"\nSelected {len(rows)} frames -> {args.output}")
    print("mining_score distribution:")
    for bucket, n in hist.items():
        print(f"    {bucket:<14}{n}")
    print(f"\nNext: python scripts/run_streaming_miner.py --no-rebuild-list "
          f"(so it uses THIS list, not an auto-rebuilt one)")


if __name__ == "__main__":
    main()
