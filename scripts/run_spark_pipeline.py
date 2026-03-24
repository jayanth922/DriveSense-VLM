"""Phase 1a-spark: Distributed nuScenes rarity scoring via PySpark.

Runs the full ETL pipeline:
  1. Load data config.
  2. Extract nuScenes metadata to JSON Lines (skip with --skip-extraction).
  3. Read JSON Lines into a Spark DataFrame.
  4. Score all frames with SparkRarityScorer.
  5. Filter to rare frames and export Parquet.
  6. Run analytics (score distribution, co-occurrence, per-scene stats, etc.).
  7. Print summary report.

Usage:
    # Development (mini dataset)
    python scripts/run_spark_pipeline.py --version v1.0-mini

    # Full trainval on HPC
    python scripts/run_spark_pipeline.py --version v1.0-trainval

    # Skip extraction if JSON Lines already exists
    python scripts/run_spark_pipeline.py --skip-extraction

    # Analytics-only (requires existing scored Parquet)
    python scripts/run_spark_pipeline.py --analytics-only

Output structure (default: outputs/data/spark_processed/):
    metadata.jsonl              Flat nuScenes frame records
    scored/                     Full scored DataFrame (Parquet)
    rare_frames/                Filtered rare frames (Parquet)
    analytics/                  5 analytics Parquet tables
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
logger = logging.getLogger("run_spark_pipeline")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="nuScenes distributed rarity scoring pipeline (Phase 1a-spark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default="configs/data.yaml",
        help="Path to data config YAML (default: configs/data.yaml)",
    )
    p.add_argument(
        "--nuscenes-root", default=None,
        help="Override nuscenes_root from config or HPC_DATA_ROOT env var.",
    )
    p.add_argument(
        "--version", default=None,
        choices=["v1.0-mini", "v1.0-trainval"],
        help="nuScenes dataset version (default: from config)",
    )
    p.add_argument(
        "--min-score", type=int, default=None,
        help="Minimum rarity score to keep a frame (default: from config)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Base output directory (default: outputs/data/spark_processed/)",
    )
    p.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip metadata extraction; use existing metadata.jsonl",
    )
    p.add_argument(
        "--analytics-only", action="store_true",
        help="Run analytics over an existing scored Parquet; skip extraction+scoring",
    )
    return p.parse_args()


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    args = _parse_args()

    config = load_config(args.config)
    if args.nuscenes_root:
        config["paths"]["nuscenes_root"] = args.nuscenes_root
    if args.version:
        config["nuscenes"]["version"] = args.version

    min_score: int = (
        args.min_score
        or int(config["nuscenes"]["rarity"]["min_rarity_score"])
    )
    nuscenes_root = Path(config["paths"]["nuscenes_root"]).expanduser()
    spark_cfg = config.get("spark", {})
    _default_out = Path(
        spark_cfg.get("output_dir", "outputs/data/spark_processed")
    )
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_out
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "metadata.jsonl"
    scored_path = output_dir / "scored"
    rare_path = output_dir / "rare_frames"

    print("\n[Phase 1a-spark] Distributed nuScenes Rarity Scoring")
    print(f"  Dataset root  : {nuscenes_root}")
    print(f"  Version       : {config['nuscenes']['version']}")
    print(f"  Min score     : {min_score}")
    print(f"  Output dir    : {output_dir}\n")

    # Import here so failures are caught cleanly.
    try:
        from drivesense.data.spark_pipeline import (  # noqa: E402
            FRAME_SCHEMA,
            NuScenesMetadataExtractor,
            SparkAnalytics,
            SparkRarityScorer,
        )
    except ImportError as exc:
        logger.error("Import failed: %s", exc)
        return 1

    scorer: SparkRarityScorer | None = None
    try:
        # ------------------------------------------------------------------
        # Step 1-2: Extract metadata to JSON Lines
        # ------------------------------------------------------------------
        if not args.analytics_only and not args.skip_extraction:
            print("[1/7] Extracting nuScenes metadata …")
            try:
                extractor = NuScenesMetadataExtractor(nuscenes_root, config)
            except ImportError as exc:
                logger.error("nuScenes devkit not available: %s", exc)
                return 1
            extractor.extract_to_jsonl(jsonl_path)
            print(f"      → {jsonl_path}")
        elif args.skip_extraction or args.analytics_only:
            print(f"[1/7] Skipping extraction — using {jsonl_path}")

        if args.analytics_only:
            # ------------------------------------------------------------------
            # Analytics-only mode: load existing scored Parquet
            # ------------------------------------------------------------------
            print("[2/7] Analytics-only mode: loading scored Parquet …")
            scorer = SparkRarityScorer(config)
            scored_df = scorer.spark.read.parquet(str(scored_path))
            scorer._scored_df = scored_df.cache()  # noqa: SLF001
        else:
            # ------------------------------------------------------------------
            # Step 3: Load JSON Lines into Spark
            # ------------------------------------------------------------------
            print("[2/7] Loading metadata into Spark …")
            scorer = SparkRarityScorer(config)
            raw_df = scorer.spark.read.schema(FRAME_SCHEMA).json(str(jsonl_path))
            print(f"      → {raw_df.count()} frames loaded")

            # ------------------------------------------------------------------
            # Step 4: Score all frames
            # ------------------------------------------------------------------
            print("[3/7] Scoring frames …")
            scored_df = scorer.compute_all_scores(raw_df)

            # ------------------------------------------------------------------
            # Step 5: Persist scored DataFrame
            # ------------------------------------------------------------------
            print("[4/7] Writing scored DataFrame to Parquet …")
            scored_df.coalesce(4).write.mode("overwrite").parquet(str(scored_path))
            print(f"      → {scored_path}")

            # ------------------------------------------------------------------
            # Step 6: Filter and export rare frames
            # ------------------------------------------------------------------
            print(f"[5/7] Filtering rare frames (score >= {min_score}) …")
            rare_df = scorer.filter_by_threshold(min_score)
            rare_count = rare_df.count()
            total_count = scored_df.count()
            print(
                f"      → {rare_count} / {total_count} frames"
                f"  ({100 * rare_count / max(total_count, 1):.1f}%)"
            )
            rare_df.coalesce(2).write.mode("overwrite").parquet(str(rare_path))
            print(f"      → {rare_path}")

        # ------------------------------------------------------------------
        # Step 7: Analytics
        # ------------------------------------------------------------------
        print("[6/7] Running analytics …")
        analytics = SparkAnalytics(scorer._scored_df, config)  # noqa: SLF001
        analytics_dir = analytics.save_all_analytics(output_dir)
        print(f"      → {analytics_dir}")

        print("[7/7] Summary:")
        analytics.print_summary_report()

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        return 1
    finally:
        if scorer is not None:
            scorer.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
