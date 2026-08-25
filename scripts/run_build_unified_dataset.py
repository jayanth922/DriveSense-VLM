"""Phase 1b: Build unified DriveSense dataset from nuScenes manifests.

Reads rare nuScenes frames (from the Spark pipeline output), assigns
stratified train/val/test splits, and writes per-split ``*_manifest.jsonl``
files under the output directory. Originally also merged in a second source
(DADA-2000); that loader was scaffolding never used past Phase 1 and has
been removed, so this script is nuScenes-only.

Usage:
    python scripts/run_build_unified_dataset.py

    # Override source/output directories
    python scripts/run_build_unified_dataset.py \\
        --nuscenes-dir outputs/data/spark_processed \\
        --output-dir   outputs/data/unified

Output structure (default: outputs/data/unified/):
    train_manifest.jsonl       Training split records
    val_manifest.jsonl         Validation split records
    test_manifest.jsonl        Test split records
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from drivesense.utils.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_build_unified_dataset")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build unified DriveSense dataset manifest (Phase 1b)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default="configs/data.yaml",
        help="Path to data config YAML (default: configs/data.yaml)",
    )
    p.add_argument(
        "--nuscenes-dir", default=None,
        help="nuScenes pipeline output dir (default: from config or outputs/data/spark_processed)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output dir for manifest files (default: outputs/data/unified)",
    )
    p.add_argument(
        "--nuscenes-only", action="store_true",
        help="Kept for backward compatibility with existing notebook cells; this "
             "script is nuScenes-only regardless of the flag.",
    )
    return p.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    args = _parse_args()

    config = load_config(args.config)
    unified_cfg = config.get("unified", {})

    _default_nuscenes = Path(
        config.get("spark", {}).get("output_dir", "outputs/data/spark_processed")
    )
    _default_out = Path(unified_cfg.get("output_dir", "outputs/data/unified"))

    nuscenes_dir = Path(args.nuscenes_dir) if args.nuscenes_dir else _default_nuscenes
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_out

    print("\n[Phase 1b] Build Unified DriveSense Dataset")
    print(f"  nuScenes dir : {nuscenes_dir}")
    print(f"  Output dir   : {output_dir}\n")

    try:
        from drivesense.data.dataset import UnifiedDatasetBuilder  # noqa: E402
    except ImportError as exc:
        logger.error("Import failed: %s", exc)
        return 1

    try:
        builder = UnifiedDatasetBuilder(config)

        print("[1/3] Loading nuScenes frames …")
        n_nuscenes = builder.load_nuscenes_frames(nuscenes_dir)
        print(f"      → {n_nuscenes} frames loaded")

        if n_nuscenes == 0:
            logger.error("No frames loaded. Run the nuScenes Spark pipeline first.")
            return 1

        print(f"[2/3] Assigning splits (total {n_nuscenes} frames) …")
        builder.assign_splits()

        print("[3/3] Writing manifests …")
        paths = builder.build(output_dir)
        for split, path in paths.items():
            print(f"      {split:5s} → {path}")

        builder.print_statistics()

    except Exception as exc:  # noqa: BLE001
        logger.exception("Build failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
