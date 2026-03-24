"""PySpark ETL pipeline for distributed nuScenes rarity scoring (Phase 1a-spark).

Three-class pipeline:
    NuScenesMetadataExtractor — nuScenes SDK → JSON Lines flat records.
    SparkRarityScorer         — Distributed 6-signal rarity scoring via DataFrames.
    SparkAnalytics            — Score distributions, co-occurrence, temporal clustering.

Usage:
    from drivesense.data.spark_pipeline import (
        NuScenesMetadataExtractor, SparkRarityScorer, SparkAnalytics
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# nuScenes devkit guard (unavailable on macOS local dev)
# ---------------------------------------------------------------------------
try:
    from nuscenes.nuscenes import NuScenes  # type: ignore[import]
    _NUSCENES_AVAILABLE = True
except ImportError:
    NuScenes = None  # type: ignore[assignment, misc]
    _NUSCENES_AVAILABLE = False

# ---------------------------------------------------------------------------
# PySpark guard
# ---------------------------------------------------------------------------
try:
    from pyspark.sql import DataFrame, SparkSession, Window
    from pyspark.sql import functions as F  # noqa: N812
    from pyspark.sql.types import (
        ArrayType,
        DoubleType,
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
    )
    _SPARK_AVAILABLE = True
except ImportError:
    SparkSession = None  # type: ignore[assignment, misc]
    DataFrame = None  # type: ignore[assignment, misc]
    F = None  # type: ignore[assignment]
    Window = None  # type: ignore[assignment]
    _SPARK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared constants (mirrors nuscenes_loader.py)
# ---------------------------------------------------------------------------
_PEDESTRIAN_PREFIX = "human.pedestrian"
_CYCLIST_SUBSTR = "cycle"

_VISIBILITY_RANGES: dict[str, tuple[int, int]] = {
    "1": (0, 40),
    "2": (40, 60),
    "3": (60, 80),
    "4": (80, 100),
}

# ---------------------------------------------------------------------------
# Explicit Spark schemas (never use inferSchema)
# ---------------------------------------------------------------------------
if _SPARK_AVAILABLE:
    _ANNOTATION_SCHEMA = StructType([
        StructField("token", StringType(), True),
        StructField("category_name", StringType(), True),
        StructField("translation_x", DoubleType(), True),
        StructField("translation_y", DoubleType(), True),
        StructField("translation_z", DoubleType(), True),
        StructField("visibility_level", IntegerType(), True),
        StructField("distance_to_ego", DoubleType(), True),
    ])

    FRAME_SCHEMA = StructType([
        StructField("sample_token", StringType(), False),
        StructField("scene_token", StringType(), False),
        StructField("timestamp", LongType(), True),
        StructField("scene_description", StringType(), True),
        StructField("ego_x", DoubleType(), True),
        StructField("ego_y", DoubleType(), True),
        StructField("ego_z", DoubleType(), True),
        StructField("cam_front_path", StringType(), True),
        StructField("cam_front_token", StringType(), True),
        StructField("num_annotations", IntegerType(), True),
        StructField("annotations", ArrayType(_ANNOTATION_SCHEMA), True),
    ])
else:  # pragma: no cover
    _ANNOTATION_SCHEMA = None  # type: ignore[assignment]
    FRAME_SCHEMA = None  # type: ignore[assignment]


# ===========================================================================
# NuScenesMetadataExtractor
# ===========================================================================

class NuScenesMetadataExtractor:
    """Extract flat frame records from nuScenes SDK for Spark ingestion.

    Produces one JSON Lines file per call to ``extract_to_jsonl()``, where
    every line is a serialised ``FRAME_SCHEMA``-compatible dict.

    Args:
        nuscenes_root: Path to nuScenes dataset root.
        config: Data config dict loaded from configs/data.yaml.

    Raises:
        ImportError: If nuScenes devkit is not installed.
    """

    def __init__(self, nuscenes_root: Path, config: dict) -> None:
        if not _NUSCENES_AVAILABLE:
            raise ImportError(
                "nuScenes devkit not installed. Run: pip install nuscenes-devkit"
            )
        version: str = config["nuscenes"].get("version", "v1.0-mini")
        logger.info("Loading nuScenes %s from %s …", version, nuscenes_root)
        self.nusc = NuScenes(
            version=version, dataroot=str(nuscenes_root), verbose=False
        )
        logger.info(
            "Loaded %d scenes, %d samples.", len(self.nusc.scene), len(self.nusc.sample)
        )

    # ------------------------------------------------------------------

    def _visibility_level(self, ann_token: str) -> int:
        """Return integer visibility level (1-4) for an annotation."""
        ann = self.nusc.get("sample_annotation", ann_token)
        vis = self.nusc.get("visibility", ann["visibility_token"])
        try:
            return int(vis.get("level", "4"))
        except (ValueError, TypeError):
            return 4

    def _extract_sample(self, sample_token: str) -> dict:
        """Build a single FRAME_SCHEMA-compatible dict for one keyframe."""
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        cam_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
        ego_xy = np.array(ego_pose["translation"][:2])

        annotations: list[dict] = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            ann_xy = np.array(ann["translation"][:2])
            dist = float(np.linalg.norm(ego_xy - ann_xy))
            annotations.append({
                "token": ann_token,
                "category_name": ann["category_name"],
                "translation_x": float(ann["translation"][0]),
                "translation_y": float(ann["translation"][1]),
                "translation_z": float(ann["translation"][2]),
                "visibility_level": self._visibility_level(ann_token),
                "distance_to_ego": round(dist, 4),
            })

        return {
            "sample_token": sample_token,
            "scene_token": sample["scene_token"],
            "timestamp": int(sample["timestamp"]),
            "scene_description": scene.get("description", ""),
            "ego_x": float(ego_pose["translation"][0]),
            "ego_y": float(ego_pose["translation"][1]),
            "ego_z": float(ego_pose["translation"][2]),
            "cam_front_path": self.nusc.get_sample_data_path(cam_data["token"]),
            "cam_front_token": cam_data["token"],
            "num_annotations": len(annotations),
            "annotations": annotations,
        }

    def extract_to_jsonl(self, output_path: Path) -> Path:
        """Extract all keyframes and write one JSON record per line.

        Args:
            output_path: Destination ``.jsonl`` file path.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_tokens: list[str] = []
        for scene in self.nusc.scene:
            token: str = scene["first_sample_token"]
            while token:
                all_tokens.append(token)
                token = self.nusc.get("sample", token)["next"]

        logger.info("Extracting %d samples to %s …", len(all_tokens), output_path)
        with output_path.open("w") as fh:
            for token in all_tokens:
                record = self._extract_sample(token)
                fh.write(json.dumps(record) + "\n")

        logger.info("Extraction complete: %d records written.", len(all_tokens))
        return output_path


# ===========================================================================
# SparkRarityScorer
# ===========================================================================

class SparkRarityScorer:
    """Compute composite rarity scores over a nuScenes metadata DataFrame.

    Each of the 6 binary signals is computed as a separate DataFrame
    operation and left-joined back onto the base frame table.

    Args:
        config: Data config dict loaded from configs/data.yaml.

    Raises:
        ImportError: If PySpark is not installed.
    """

    def __init__(self, config: dict) -> None:
        if not _SPARK_AVAILABLE:
            raise ImportError("PySpark not installed. Run: pip install pyspark>=3.5")

        rarity = config["nuscenes"]["rarity"]
        self._proximity_threshold: float = float(rarity["proximity_threshold_m"])
        self._occ_min: int = int(rarity["occlusion_min_visibility"])
        self._occ_max: int = int(rarity["occlusion_max_visibility"])
        self._density_threshold: int = int(rarity["min_agents_for_density"])
        self._weather_keywords: list[str] = [
            kw.lower() for kw in rarity["adverse_weather_keywords"]
        ]
        self._intersection_keywords: list[str] = [
            kw.lower() for kw in rarity.get("intersection_keywords", [])
        ]
        self._min_rarity_score: int = int(rarity["min_rarity_score"])

        spark_cfg = config.get("spark", {})
        app_name: str = spark_cfg.get("app_name", "DriveSense-RarityScorer")
        master: str = spark_cfg.get("master", "local[*]")
        driver_memory: str = spark_cfg.get("driver_memory", "4g")
        shuffle_partitions: int = int(spark_cfg.get("shuffle_partitions", 8))
        log_level: str = spark_cfg.get("log_level", "WARN")

        self.spark: SparkSession = (
            SparkSession.builder
            .master(master)
            .appName(app_name)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.driver.memory", driver_memory)
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel(log_level)
        self._scored_df: DataFrame | None = None

    # ------------------------------------------------------------------
    # Signal helpers — each returns a DataFrame keyed on sample_token
    # ------------------------------------------------------------------

    def _proximity_signal(self, df: DataFrame) -> DataFrame:
        """Return DataFrame with ``sig_proximity`` (0/1) per sample_token."""
        prox = (
            df.select("sample_token", F.explode("annotations").alias("ann"))
            .filter(
                F.col("ann.category_name").startswith(_PEDESTRIAN_PREFIX)
                | F.col("ann.category_name").contains(_CYCLIST_SUBSTR)
            )
            .filter(F.col("ann.distance_to_ego") < self._proximity_threshold)
            .groupBy("sample_token")
            .agg(F.lit(1).cast(IntegerType()).alias("sig_proximity"))
        )
        return prox

    def _occlusion_signal(self, df: DataFrame) -> DataFrame:
        """Return DataFrame with ``sig_occlusion`` (0/1) per sample_token.

        Visibility level 1 maps to 0–40 % visibility, which is the only
        level that falls entirely within [occ_min, occ_max] = [0, 40].
        """
        occ = (
            df.select("sample_token", F.explode("annotations").alias("ann"))
            .filter(F.col("ann.visibility_level") == 1)
            .groupBy("sample_token")
            .agg(F.lit(1).cast(IntegerType()).alias("sig_occlusion"))
        )
        return occ

    def _density_signal(self, df: DataFrame) -> DataFrame:
        """Return DataFrame with ``sig_density`` (0/1) per sample_token."""
        dense = (
            df.select("sample_token", "num_annotations")
            .filter(F.col("num_annotations") >= self._density_threshold)
            .select("sample_token", F.lit(1).cast(IntegerType()).alias("sig_density"))
        )
        return dense

    def _weather_signal(self, df: DataFrame) -> DataFrame:
        """Return DataFrame with ``sig_weather`` (0/1) per sample_token."""
        condition = F.lit(False)
        for kw in self._weather_keywords:
            condition = condition | F.lower(F.col("scene_description")).contains(kw)
        weather = (
            df.select("sample_token", "scene_description")
            .filter(condition)
            .select("sample_token", F.lit(1).cast(IntegerType()).alias("sig_weather"))
        )
        return weather

    def _vru_signal(self, df: DataFrame) -> DataFrame:
        """Return DataFrame with ``sig_vru`` (0/1): pedestrian AND intersection."""
        has_ped = (
            df.select("sample_token", F.explode("annotations").alias("ann"))
            .filter(F.col("ann.category_name").startswith(_PEDESTRIAN_PREFIX))
            .select("sample_token")
            .distinct()
        )
        intersection_cond = F.lit(False)
        for kw in self._intersection_keywords:
            intersection_cond = (
                intersection_cond | F.lower(F.col("scene_description")).contains(kw)
            )
        at_intersection = (
            df.select("sample_token", "scene_description")
            .filter(intersection_cond)
            .select("sample_token")
        )
        vru = (
            has_ped.join(at_intersection, on="sample_token", how="inner")
            .select("sample_token", F.lit(1).cast(IntegerType()).alias("sig_vru"))
        )
        return vru

    def _cyclist_signal(self, df: DataFrame) -> DataFrame:
        """Return DataFrame with ``sig_cyclist`` (0/1) per sample_token."""
        cyc = (
            df.select("sample_token", F.explode("annotations").alias("ann"))
            .filter(F.col("ann.category_name").contains(_CYCLIST_SUBSTR))
            .groupBy("sample_token")
            .agg(F.lit(1).cast(IntegerType()).alias("sig_cyclist"))
        )
        return cyc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all_scores(self, df: DataFrame) -> DataFrame:
        """Join all 6 signals and sum to produce composite rarity scores.

        Args:
            df: Base metadata DataFrame conforming to ``FRAME_SCHEMA``.

        Returns:
            DataFrame with ``sample_token``, ``scene_token``, ``timestamp``,
            ``cam_front_path``, ``scene_description``, one ``sig_*`` column
            per signal (0/1), and ``rarity_score`` (0-6).
        """
        base = df.select(
            "sample_token", "scene_token", "timestamp",
            "cam_front_path", "scene_description", "num_annotations",
            "annotations",
        )

        signals = {
            "sig_proximity": self._proximity_signal(df),
            "sig_occlusion": self._occlusion_signal(df),
            "sig_density": self._density_signal(df),
            "sig_weather": self._weather_signal(df),
            "sig_vru": self._vru_signal(df),
            "sig_cyclist": self._cyclist_signal(df),
        }

        result = base
        for col_name, sig_df in signals.items():
            result = result.join(sig_df, on="sample_token", how="left")
            result = result.fillna({col_name: 0})

        sig_cols = list(signals.keys())
        score_expr = sum(F.col(c) for c in sig_cols)
        result = result.withColumn("rarity_score", score_expr.cast(IntegerType()))

        self._scored_df = result.cache()
        return self._scored_df

    def filter_by_threshold(self, min_score: int | None = None) -> DataFrame:
        """Return rows meeting the minimum rarity threshold.

        Args:
            min_score: Override minimum score; defaults to config value.

        Returns:
            Filtered DataFrame sorted by ``rarity_score`` descending.

        Raises:
            RuntimeError: If ``compute_all_scores()`` has not been called.
        """
        if self._scored_df is None:
            raise RuntimeError(
                "No scores computed. Call compute_all_scores() first."
            )
        threshold = min_score if min_score is not None else self._min_rarity_score
        return (
            self._scored_df
            .filter(F.col("rarity_score") >= threshold)
            .orderBy(F.col("rarity_score").desc())
        )

    def stop(self) -> None:
        """Stop the underlying SparkSession."""
        self.spark.stop()


# ===========================================================================
# SparkAnalytics
# ===========================================================================

class SparkAnalytics:
    """Analytical computations over a scored rarity DataFrame.

    Args:
        scored_df: Output of ``SparkRarityScorer.compute_all_scores()``.
        config: Data config dict loaded from configs/data.yaml.

    Raises:
        ImportError: If PySpark is not installed.
    """

    _SIGNAL_COLS = [
        "sig_proximity", "sig_occlusion", "sig_density",
        "sig_weather", "sig_vru", "sig_cyclist",
    ]

    def __init__(self, scored_df: Any, config: dict) -> None:
        if not _SPARK_AVAILABLE:
            raise ImportError("PySpark not installed. Run: pip install pyspark>=3.5")
        self._df: DataFrame = scored_df
        self._spark: SparkSession = scored_df.sparkSession
        self._output_dir = Path(
            config.get("spark", {}).get("output_dir", "outputs/data/spark_processed")
        )

    # ------------------------------------------------------------------

    def score_distribution(self) -> DataFrame:
        """Compute frame count and cumulative % at each rarity score level.

        Returns:
            DataFrame with columns: ``rarity_score``, ``frame_count``,
            ``cumulative_pct`` (ascending).
        """
        w = Window.orderBy("rarity_score").rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        total = self._df.count()
        dist = (
            self._df.groupBy("rarity_score")
            .agg(F.count("*").alias("frame_count"))
            .orderBy("rarity_score")
            .withColumn(
                "cumulative_pct",
                F.round(
                    100.0 * F.sum("frame_count").over(w) / F.lit(total),
                    2,
                ),
            )
        )
        return dist

    def signal_cooccurrence(self) -> DataFrame:
        """Compute a 6×6 co-occurrence matrix across signal columns.

        Each cell (i, j) is the number of frames where both signal_i
        and signal_j are active (== 1).

        Returns:
            DataFrame with columns: ``signal_a``, ``signal_b``, ``cooccurrence``.
        """
        rows: list[Any] = []
        for sig_a in self._SIGNAL_COLS:
            for sig_b in self._SIGNAL_COLS:
                count = (
                    self._df
                    .filter((F.col(sig_a) == 1) & (F.col(sig_b) == 1))
                    .count()
                )
                rows.append((sig_a, sig_b, int(count)))

        schema = StructType([
            StructField("signal_a", StringType(), False),
            StructField("signal_b", StringType(), False),
            StructField("cooccurrence", IntegerType(), False),
        ])
        return self._spark.createDataFrame(rows, schema=schema)

    def per_scene_stats(self) -> DataFrame:
        """Aggregate rarity statistics grouped by scene.

        Returns:
            DataFrame with columns: ``scene_token``, ``total_frames``,
            ``rare_frames`` (>= 3), ``mean_rarity_score``, ``max_rarity_score``,
            ``active_signals_total``.
        """
        sig_sum = sum(F.col(c) for c in self._SIGNAL_COLS)
        return (
            self._df
            .withColumn("active_signals", sig_sum)
            .groupBy("scene_token")
            .agg(
                F.count("*").alias("total_frames"),
                F.sum(
                    F.when(F.col("rarity_score") >= 3, 1).otherwise(0)
                ).alias("rare_frames"),
                F.round(F.avg("rarity_score"), 3).alias("mean_rarity_score"),
                F.max("rarity_score").alias("max_rarity_score"),
                F.sum("active_signals").alias("active_signals_total"),
            )
            .orderBy(F.col("mean_rarity_score").desc())
        )

    def category_breakdown(self) -> DataFrame:
        """Count annotations per category across all frames.

        Returns:
            DataFrame with columns: ``category_name``, ``annotation_count``,
            ordered by count descending.
        """
        return (
            self._df
            .select(F.explode("annotations").alias("ann"))
            .select(F.col("ann.category_name").alias("category_name"))
            .groupBy("category_name")
            .agg(F.count("*").alias("annotation_count"))
            .orderBy(F.col("annotation_count").desc())
        )

    def temporal_clustering(
        self, gap_threshold_us: int = 2_000_000
    ) -> DataFrame:
        """Assign burst IDs to rare frames based on temporal gaps within scenes.

        A new burst starts whenever consecutive rare frames within the same
        scene are separated by more than ``gap_threshold_us`` microseconds.

        Args:
            gap_threshold_us: Gap in microseconds that starts a new burst.
                Default 2 000 000 µs = 2 s.

        Returns:
            DataFrame of rare frames (rarity_score >= min_score) with added
            columns ``time_gap`` and ``burst_id``.
        """
        rare = self._df.filter(
            F.col("rarity_score") >= 3
        ).select("sample_token", "scene_token", "timestamp", "rarity_score")

        w_order = Window.partitionBy("scene_token").orderBy("timestamp")
        w_cumsum = (
            Window.partitionBy("scene_token")
            .orderBy("timestamp")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

        return (
            rare
            .withColumn("prev_timestamp", F.lag("timestamp").over(w_order))
            .withColumn(
                "time_gap",
                F.col("timestamp") - F.col("prev_timestamp"),
            )
            .withColumn(
                "new_burst",
                F.when(
                    F.col("time_gap").isNull()
                    | (F.col("time_gap") > gap_threshold_us),
                    1,
                ).otherwise(0),
            )
            .withColumn("burst_id", F.sum("new_burst").over(w_cumsum))
            .drop("new_burst")
        )

    def save_all_analytics(self, output_dir: Path | None = None) -> Path:
        """Compute and persist all analytics tables as Parquet.

        Writes to ``{output_dir}/analytics/``:
            ``score_distribution.parquet``
            ``signal_cooccurrence.parquet``
            ``per_scene_stats.parquet``
            ``category_breakdown.parquet``
            ``temporal_clustering.parquet``

        Args:
            output_dir: Override base output directory.

        Returns:
            Path to the analytics directory.
        """
        base = Path(output_dir) if output_dir else self._output_dir
        analytics_dir = base / "analytics"

        tables: dict[str, Any] = {
            "score_distribution": self.score_distribution(),
            "signal_cooccurrence": self.signal_cooccurrence(),
            "per_scene_stats": self.per_scene_stats(),
            "category_breakdown": self.category_breakdown(),
            "temporal_clustering": self.temporal_clustering(),
        }

        for name, df in tables.items():
            out_path = analytics_dir / f"{name}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.coalesce(1).write.mode("overwrite").parquet(str(out_path))
            logger.info("Saved %s → %s", name, out_path)

        return analytics_dir

    def print_summary_report(self) -> None:
        """Print a human-readable summary of rarity scores and signal prevalence."""
        total = self._df.count()
        rare = self._df.filter(F.col("rarity_score") >= 3).count()

        print("\n" + "=" * 60)
        print("  DriveSense Spark Analytics — Summary")
        print("=" * 60)
        print(f"  Total frames scored : {total}")
        print(f"  Rare frames (>=3)   : {rare}  ({100 * rare / max(total, 1):.1f}%)")
        print()
        print("  Score distribution:")
        dist_rows = (
            self._df.groupBy("rarity_score")
            .agg(F.count("*").alias("n"))
            .orderBy("rarity_score")
            .collect()
        )
        for row in dist_rows:
            bar = "#" * min(40, int(row["n"] * 40 / max(total, 1)))
            print(f"    {row['rarity_score']}  [{bar:<40}] {row['n']:>5}")
        print()
        print("  Signal prevalence (% of all frames):")
        for sig in self._SIGNAL_COLS:
            count = self._df.filter(F.col(sig) == 1).count()
            label = sig.replace("sig_", "")
            print(f"    {label:<25} {count:>5}  ({100 * count / max(total, 1):.1f}%)")
        print("=" * 60 + "\n")
