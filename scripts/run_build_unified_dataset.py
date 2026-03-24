"""Phase 1b: Build unified DriveSense dataset from nuScenes + DADA-2000 manifests.

Merges rare nuScenes frames (from the Spark pipeline output) and DADA-2000
extracted frames into a single stratified-split dataset, writing per-split
``*_manifest.jsonl`` files under the output directory.

Usage:
    # Full build (both sources)
    python scripts/run_build_unified_dataset.py

    # Override source directories
    python scripts/run_build_unified_dataset.py \\
        --nuscenes-dir outputs/data/spark_processed \\
        --dada-dir     outputs/data/dada_extracted

    # nuScenes-only (skip DADA-2000)
    python scripts/run_build_unified_dataset.py --nuscenes-only

    # DADA-2000-only
    python scripts/run_build_unified_dataset.py --dada-only

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
        "--dada-dir", default=None,
        help="DADA-2000 extraction output dir (default: outputs/data/dada_extracted)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output dir for manifest files (default: outputs/data/unified)",
    )
    p.add_argument(
        "--nuscenes-only", action="store_true",
        help="Only load nuScenes frames; skip DADA-2000.",
    )
    p.add_argument(
        "--dada-only", action="store_true",
        help="Only load DADA-2000 frames; skip nuScenes.",
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
    _default_dada = Path("outputs/data/dada_extracted")
    _default_out = Path(unified_cfg.get("output_dir", "outputs/data/unified"))

    nuscenes_dir = Path(args.nuscenes_dir) if args.nuscenes_dir else _default_nuscenes
    dada_dir = Path(args.dada_dir) if args.dada_dir else _default_dada
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_out

    print("\n[Phase 1b] Build Unified DriveSense Dataset")
    print(f"  nuScenes dir : {nuscenes_dir}")
    print(f"  DADA-2000 dir: {dada_dir}")
    print(f"  Output dir   : {output_dir}\n")

    try:
        from drivesense.data.dataset import UnifiedDatasetBuilder  # noqa: E402
    except ImportError as exc:
        logger.error("Import failed: %s", exc)
        return 1

    try:
        builder = UnifiedDatasetBuilder(config)

        n_nuscenes = 0
        n_dada = 0

        if not args.dada_only:
            print("[1/4] Loading nuScenes frames …")
            n_nuscenes = builder.load_nuscenes_frames(nuscenes_dir)
            print(f"      → {n_nuscenes} frames loaded")
        else:
            print("[1/4] Skipping nuScenes (--dada-only)")

        if not args.nuscenes_only:
            print("[2/4] Loading DADA-2000 frames …")
            n_dada = builder.load_dada2000_frames(dada_dir)
            print(f"      → {n_dada} frames loaded")
        else:
            print("[2/4] Skipping DADA-2000 (--nuscenes-only)")

        total = n_nuscenes + n_dada
        if total == 0:
            logger.error(
                "No frames loaded. Run the nuScenes Spark pipeline and/or "
                "run_dada_extraction.py first."
            )
            return 1

        print(f"[3/4] Assigning splits (total {total} frames) …")
        builder.assign_splits()

        print("[4/4] Writing manifests …")
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
