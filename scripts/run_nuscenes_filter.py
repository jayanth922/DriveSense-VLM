"""Phase 1a: Run the nuScenes rarity filtering pipeline.

Scores all keyframes in the nuScenes dataset using the composite rarity signal,
exports frames meeting the configured threshold, and optionally generates visual
inspection artifacts (distribution plot + sample grid).

Usage:
    # Development (mini dataset, ~400 frames, run in seconds)
    python scripts/run_nuscenes_filter.py

    # Full trainval dataset on HPC
    python scripts/run_nuscenes_filter.py --version v1.0-trainval

    # Lower threshold to get more frames
    python scripts/run_nuscenes_filter.py --min-score 2

    # Override data root (or set HPC_DATA_ROOT env var)
    python scripts/run_nuscenes_filter.py --nuscenes-root /scratch/data/nuscenes

    # Generate distribution plot and sample grid
    python scripts/run_nuscenes_filter.py --visualize

Output structure (default: outputs/data/nuscenes_filtered/):
    images/             CAM_FRONT images for exported frames
    metadata.json       Per-frame rarity records (list of dicts)
    score_distribution.json  {score: count} across all frames
    summary.json        Aggregate statistics
    filtering_log.json  Full run provenance record
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Allow running from the repo root without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from drivesense.data.nuscenes_loader import NuScenesRarityFilter  # noqa: E402
from drivesense.utils.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_nuscenes_filter")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="nuScenes rarity filtering pipeline (Phase 1a)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default="configs/data.yaml",
        help="Path to data config YAML (default: configs/data.yaml)",
    )
    p.add_argument(
        "--nuscenes-root",
        default=None,
        help="Override nuscenes_root from config. Can also set HPC_DATA_ROOT env var.",
    )
    p.add_argument(
        "--version",
        default=None,
        choices=["v1.0-mini", "v1.0-trainval"],
        help="nuScenes dataset version (default: from config, usually v1.0-mini for dev)",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=None,
        help="Minimum rarity score to keep a frame (default: from config, usually 3)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: outputs/data/nuscenes_filtered/)",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Generate distribution plot + sample grid after filtering",
    )
    return p.parse_args()


def _print_summary(
    filtered: list[dict],
    all_count: int,
    distribution: dict[int, int],
    threshold: int,
) -> None:
    """Print a human-readable pipeline summary to stdout."""
    print("\n" + "=" * 60)
    print("  nuScenes Rarity Filtering — Results")
    print("=" * 60)
    print(f"  Total frames scanned : {all_count}")
    print(f"  Threshold used       : score >= {threshold}")
    print(f"  Frames exported      : {len(filtered)}"
          f"  ({100 * len(filtered) / max(all_count, 1):.1f}%)")
    print()
    print("  Score Distribution:")
    for score in sorted(distribution):
        count = distribution[score]
        bar = "#" * min(40, count // max(1, all_count // 40))
        marker = " <-- threshold" if score == threshold else ""
        print(f"    {score}  [{bar:<40}] {count:>5}{marker}")
    print()
    print("  Signal Breakdown (across all frames):")
    signal_totals: dict[str, int] = {}
    for frame in filtered:
        for sig, info in frame["signals"].items():
            signal_totals[sig] = signal_totals.get(sig, 0) + int(info["active"])
    for sig, count in signal_totals.items():
        print(f"    {sig:<25} {count:>4} of {len(filtered)} exported frames")
    print("=" * 60 + "\n")


def _run_visualizations(
    filtered: list[dict],
    distribution: dict[int, int],
    output_dir: Path,
) -> None:
    """Generate distribution chart and sample grid. Skips gracefully if matplotlib absent."""
    try:
        from drivesense.utils.visualization import (
            create_rarity_distribution_plot,
            create_sample_grid,
        )
    except ImportError as exc:
        logger.warning("Skipping visualizations: %s", exc)
        return

    # Distribution bar chart.
    plot_path = output_dir / "rarity_distribution.png"
    try:
        create_rarity_distribution_plot(distribution, plot_path)
        print(f"  Distribution plot  → {plot_path}")
    except ImportError as exc:
        logger.warning("Skipping distribution plot: %s", exc)

    # Sample grid of top-scoring frames.
    grid_items = [
        {
            "image_path": output_dir / "images" / Path(f["cam_front_path"]).name,
            "score": f["rarity_score"],
            "signals": f["signals"],
        }
        for f in filtered[:9]  # top 9 for a 3×3 grid
    ]
    grid_path = output_dir / "sample_grid.png"
    try:
        create_sample_grid(grid_items, grid_path, grid_size=(3, 3))
        print(f"  Sample grid        → {grid_path}")
    except ImportError as exc:
        logger.warning("Skipping sample grid: %s", exc)


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    args = _parse_args()

    # Load and optionally override config values.
    config = load_config(args.config)
    if args.nuscenes_root:
        config["paths"]["nuscenes_root"] = args.nuscenes_root
    if args.version:
        config["nuscenes"]["version"] = args.version
    min_score: int = args.min_score or int(config["nuscenes"]["rarity"]["min_rarity_score"])

    nuscenes_root = Path(config["paths"]["nuscenes_root"]).expanduser()
    # Default output dir comes from config["paths"]["output_dir"] (never hardcoded).
    _default_out = Path(config["paths"]["output_dir"]) / "nuscenes_filtered"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_out

    print("\n[Phase 1a] nuScenes Rarity Filtering")
    print(f"  Dataset root  : {nuscenes_root}")
    print(f"  Version       : {config['nuscenes']['version']}")
    print(f"  Min score     : {min_score}")
    print(f"  Output dir    : {output_dir}\n")

    # Run pipeline.
    try:
        rarity_filter = NuScenesRarityFilter(nuscenes_root, config)
    except ImportError as exc:
        logger.error("nuScenes devkit not available: %s", exc)
        return 1

    filtered = rarity_filter.filter_rare_frames(min_score=min_score)
    distribution = rarity_filter.get_score_distribution()
    all_count = sum(distribution.values())

    _print_summary(filtered, all_count, distribution, min_score)

    # Export dataset.
    output_dir = rarity_filter.export_filtered_dataset(output_dir)
    print(f"  Dataset exported → {output_dir}")

    if args.visualize:
        _run_visualizations(filtered, distribution, output_dir)

    # Write provenance log (idempotent — overwrites on re-run).
    log_record = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "nuscenes_version": config["nuscenes"]["version"],
        "nuscenes_root": str(nuscenes_root),
        "min_score_used": min_score,
        "total_frames_scanned": all_count,
        "frames_exported": len(filtered),
        "score_distribution": distribution,
    }
    log_path = output_dir / "filtering_log.json"
    log_path.write_text(json.dumps(log_record, indent=2))
    print(f"  Provenance log   → {log_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
