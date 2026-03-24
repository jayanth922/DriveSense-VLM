"""Phase 1b: DADA-2000 critical-moment frame extraction CLI.

Scans the DADA-2000 dataset directory, extracts critical and context frames
for every sequence, resizes to 672×448, and exports a DriveSense-format
``metadata.jsonl`` with images under ``output_dir/``.

Usage:
    python scripts/run_dada_extraction.py

    # Override dataset root
    python scripts/run_dada_extraction.py --dada-root ~/data/dada2000

    # Limit to first 10 sequences (dev/debug)
    python scripts/run_dada_extraction.py --max-sequences 10

    # Custom output location
    python scripts/run_dada_extraction.py --output-dir outputs/data/dada_extracted

Output structure (default: outputs/data/dada_extracted/):
    images/                    Resized PNG frames
    metadata.jsonl             One record per frame (DriveSense format)
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
logger = logging.getLogger("run_dada_extraction")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DADA-2000 critical-moment frame extraction (Phase 1b)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default="configs/data.yaml",
        help="Path to data config YAML (default: configs/data.yaml)",
    )
    p.add_argument(
        "--dada-root", default=None,
        help="Override dada2000_root from config or HPC_DATA_ROOT env var.",
    )
    p.add_argument(
        "--max-sequences", type=int, default=None,
        help="Limit extraction to first N sequences (useful for debugging).",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: outputs/data/dada_extracted/)",
    )
    return p.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    args = _parse_args()

    config = load_config(args.config)
    if args.dada_root:
        config["paths"]["dada2000_root"] = args.dada_root

    dada_root = Path(config["paths"]["dada2000_root"]).expanduser()
    _default_out = Path("outputs/data/dada_extracted")
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_out

    print("\n[Phase 1b] DADA-2000 Critical-Moment Frame Extraction")
    print(f"  DADA root   : {dada_root}")
    print(f"  Output dir  : {output_dir}\n")

    try:
        from drivesense.data.dada_loader import DADA2000Loader  # noqa: E402
    except ImportError as exc:
        logger.error("Import failed: %s", exc)
        return 1

    try:
        loader = DADA2000Loader(dada_root, config)

        # Apply --max-sequences limit if requested.
        if args.max_sequences is not None:
            loader._sequences = loader._sequences[: args.max_sequences]  # noqa: SLF001

        stats = loader.get_summary_statistics()
        print("[1/3] Dataset summary:")
        print(f"      sequences  : {stats['total_sequences']}")
        print(f"      categories : {stats['total_categories']}")
        print(f"      annotated  : {stats['sequences_with_annotations']}")
        print(f"      max frames : {stats['max_frames_budget']}")

        print("[2/3] Extracting and exporting keyframes …")
        jsonl_path = loader.export_keyframes(output_dir)
        print(f"      → {jsonl_path}")

        print("[3/3] Category distribution:")
        dist = loader.get_category_distribution()
        for cat, count in list(dist.items())[:10]:
            print(f"      cat {cat}: {count} sequences")
        if len(dist) > 10:
            print(f"      … and {len(dist) - 10} more categories")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Extraction failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
