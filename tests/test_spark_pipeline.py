"""Tests for drivesense.data.spark_pipeline (Phase 1a-spark).

All tests use a module-scoped SparkSession with a minimal synthetic dataset
that matches FRAME_SCHEMA.  The nuScenes devkit and a real dataset are NOT
required — extraction tests are skipped when the devkit is absent.

Run with:
    python -m pytest tests/test_spark_pipeline.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable regardless of install state.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Java compatibility shim — must run before PySpark creates any SparkContext.
#
# PySpark 4.0 bundles Hadoop 3.4.1 which calls Subject.getSubject() (removed
# in JDK 23).  We patch java.base with a Subject.class where getSubject()
# returns null instead of throwing, so Hadoop falls through to getLoginUser().
# The patched class lives at compat/java_patch/javax/security/auth/Subject.class.
# ---------------------------------------------------------------------------
_PATCH_DIR = Path(__file__).resolve().parent.parent / "compat" / "java_patch"
if _PATCH_DIR.exists():
    _submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "pyspark-shell")
    _patch_opt = f"--patch-module java.base={_PATCH_DIR}"
    if _patch_opt not in _submit_args:
        # Inject the patch option into driver JVM args.
        _submit_args = (
            f'--driver-java-options "{_patch_opt}" '
            + _submit_args.replace("pyspark-shell", "").strip()
            + " pyspark-shell"
        ).strip()
        os.environ["PYSPARK_SUBMIT_ARGS"] = _submit_args

# Skip the entire module if PySpark is not installed.
pyspark = pytest.importorskip("pyspark", reason="pyspark not installed")

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402  # noqa: N812

from drivesense.data.spark_pipeline import (  # noqa: E402
    FRAME_SCHEMA,
    SparkAnalytics,
    SparkRarityScorer,
)

# ---------------------------------------------------------------------------
# Config fixture shared across all tests
# ---------------------------------------------------------------------------
_TEST_CONFIG: dict = {
    "nuscenes": {
        "version": "v1.0-mini",
        "rarity": {
            "proximity_threshold_m": 5.0,
            "occlusion_min_visibility": 0,
            "occlusion_max_visibility": 40,
            "min_agents_for_density": 3,
            "adverse_weather_keywords": ["rain", "night", "fog"],
            "intersection_keywords": ["intersection", "crossing"],
            "min_rarity_score": 3,
        },
    },
    "spark": {
        "app_name": "DriveSense-Test",
        "master": "local[2]",
        "driver_memory": "1g",
        "shuffle_partitions": 2,
        "output_dir": "outputs/test/spark",
        "log_level": "ERROR",
    },
    "paths": {
        "nuscenes_root": "~/data/nuscenes",
        "output_dir": "outputs/data",
        "cache_dir": ".cache/drivesense",
    },
}

# ---------------------------------------------------------------------------
# Module-scoped SparkSession (created once, shared by all tests in module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("DriveSense-Test")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Synthetic metadata DataFrame
# ---------------------------------------------------------------------------

def _make_annotation(
    token: str,
    category: str,
    distance: float,
    vis_level: int,
    tx: float = 0.0,
    ty: float = 0.0,
) -> dict:
    return {
        "token": token,
        "category_name": category,
        "translation_x": tx,
        "translation_y": ty,
        "translation_z": 0.0,
        "visibility_level": vis_level,
        "distance_to_ego": distance,
    }


@pytest.fixture(scope="module")
def sample_metadata_df(spark: SparkSession):
    """Synthetic DataFrame with 4 frames covering every signal path."""
    rows = [
        # sample_001: pedestrian within 5m + at intersection scene
        {
            "sample_token": "s001",
            "scene_token": "sc001",
            "timestamp": 1_000_000,
            "scene_description": "Night, rain near intersection",
            "ego_x": 0.0, "ego_y": 0.0, "ego_z": 0.0,
            "cam_front_path": "/data/img/s001.jpg",
            "cam_front_token": "ct001",
            "num_annotations": 4,
            "annotations": [
                _make_annotation("a1", "human.pedestrian.adult", 3.0, 1),
                _make_annotation("a2", "human.pedestrian.child", 2.5, 2),
                _make_annotation("a3", "vehicle.car", 10.0, 4),
                _make_annotation("a4", "vehicle.car", 8.0, 4),
            ],
        },
        # sample_002: cyclist + dense scene (3 annotations, threshold=3)
        {
            "sample_token": "s002",
            "scene_token": "sc001",
            "timestamp": 1_500_000,
            "scene_description": "Night, rain near intersection",
            "ego_x": 1.0, "ego_y": 1.0, "ego_z": 0.0,
            "cam_front_path": "/data/img/s002.jpg",
            "cam_front_token": "ct002",
            "num_annotations": 3,
            "annotations": [
                _make_annotation("b1", "vehicle.bicycle", 6.0, 2),
                _make_annotation("b2", "vehicle.motorcycle", 4.0, 1),
                _make_annotation("b3", "vehicle.car", 15.0, 4),
            ],
        },
        # sample_003: low rarity (no special signals fire)
        {
            "sample_token": "s003",
            "scene_token": "sc002",
            "timestamp": 2_000_000,
            "scene_description": "Clear sunny day in Singapore",
            "ego_x": 5.0, "ego_y": 5.0, "ego_z": 0.0,
            "cam_front_path": "/data/img/s003.jpg",
            "cam_front_token": "ct003",
            "num_annotations": 1,
            "annotations": [
                _make_annotation("c1", "vehicle.car", 20.0, 4),
            ],
        },
        # sample_004: occluded pedestrian + adverse weather (fog) — different scene
        {
            "sample_token": "s004",
            "scene_token": "sc002",
            "timestamp": 3_000_000,
            "scene_description": "Heavy fog, poor visibility",
            "ego_x": 10.0, "ego_y": 10.0, "ego_z": 0.0,
            "cam_front_path": "/data/img/s004.jpg",
            "cam_front_token": "ct004",
            "num_annotations": 2,
            "annotations": [
                _make_annotation("d1", "human.pedestrian.adult", 8.0, 1),
                _make_annotation("d2", "vehicle.bus.rigid", 12.0, 2),
            ],
        },
    ]
    return spark.createDataFrame(rows, schema=FRAME_SCHEMA)


# ===========================================================================
# 1. Schema
# ===========================================================================

class TestSchema:
    def test_frame_schema_has_required_fields(self) -> None:
        """FRAME_SCHEMA must include all 11 required top-level fields."""
        field_names = {f.name for f in FRAME_SCHEMA.fields}
        required = {
            "sample_token", "scene_token", "timestamp", "scene_description",
            "ego_x", "ego_y", "ego_z", "cam_front_path", "cam_front_token",
            "num_annotations", "annotations",
        }
        assert required <= field_names

    def test_annotations_array_schema(self) -> None:
        """Annotations field must be ArrayType with annotation sub-fields."""
        ann_field = next(f for f in FRAME_SCHEMA.fields if f.name == "annotations")
        from pyspark.sql.types import ArrayType
        assert isinstance(ann_field.dataType, ArrayType)
        sub_names = {sf.name for sf in ann_field.dataType.elementType.fields}
        assert {"token", "category_name", "distance_to_ego", "visibility_level"} <= sub_names


# ===========================================================================
# 2. SparkRarityScorer — compute_all_scores
# ===========================================================================

class TestSparkRarityScorer:
    @pytest.fixture(scope="class")
    def scorer(self, spark: SparkSession) -> SparkRarityScorer:
        s = SparkRarityScorer(_TEST_CONFIG)
        # Reuse the provided session instead of creating a new one.
        s.spark = spark
        return s

    def test_compute_all_scores_returns_dataframe(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        df = scorer.compute_all_scores(sample_metadata_df)
        assert df is not None
        assert df.count() == 4

    def test_signal_columns_present(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        df = scorer.compute_all_scores(sample_metadata_df)
        cols = set(df.columns)
        for sig in ("sig_proximity", "sig_occlusion", "sig_density",
                    "sig_weather", "sig_vru", "sig_cyclist"):
            assert sig in cols, f"Missing signal column: {sig}"

    def test_rarity_score_column_present(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        df = scorer.compute_all_scores(sample_metadata_df)
        assert "rarity_score" in df.columns

    def test_proximity_signal_fires_for_s001(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        df = scorer.compute_all_scores(sample_metadata_df)
        row = df.filter(F.col("sample_token") == "s001").select("sig_proximity").first()
        assert row is not None and row["sig_proximity"] == 1

    def test_proximity_signal_does_not_fire_for_s003(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        df = scorer.compute_all_scores(sample_metadata_df)
        row = df.filter(F.col("sample_token") == "s003").select("sig_proximity").first()
        assert row is not None and row["sig_proximity"] == 0

    def test_occlusion_signal_fires_when_vis_level_1(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        # s001 has vis_level=1 pedestrian, s002 has vis_level=1 cyclist
        df = scorer.compute_all_scores(sample_metadata_df)
        for token in ("s001", "s002"):
            row = df.filter(F.col("sample_token") == token).select("sig_occlusion").first()
            assert row is not None and row["sig_occlusion"] == 1, \
                f"occlusion should fire for {token}"

    def test_density_signal_fires_at_threshold(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        # threshold=3; s001 has 4, s002 has 3 → should fire
        df = scorer.compute_all_scores(sample_metadata_df)
        for token in ("s001", "s002"):
            row = df.filter(F.col("sample_token") == token).select("sig_density").first()
            assert row is not None and row["sig_density"] == 1, \
                f"density should fire for {token}"

    def test_density_signal_off_below_threshold(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        df = scorer.compute_all_scores(sample_metadata_df)
        row = df.filter(F.col("sample_token") == "s003").select("sig_density").first()
        assert row is not None and row["sig_density"] == 0

    def test_weather_signal_fires_for_adverse_keywords(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        # s001/s002 have "Night, rain" → should fire; s004 has "fog"
        df = scorer.compute_all_scores(sample_metadata_df)
        for token in ("s001", "s002", "s004"):
            row = df.filter(F.col("sample_token") == token).select("sig_weather").first()
            assert row is not None and row["sig_weather"] == 1, \
                f"weather should fire for {token}"

    def test_vru_signal_requires_ped_and_intersection(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        # s001: has pedestrians + "intersection" → fires
        # s003: no pedestrians → off
        # s004: has pedestrian but no intersection keyword → off
        df = scorer.compute_all_scores(sample_metadata_df)
        s001 = df.filter(F.col("sample_token") == "s001").select("sig_vru").first()
        assert s001 is not None and s001["sig_vru"] == 1
        s003 = df.filter(F.col("sample_token") == "s003").select("sig_vru").first()
        assert s003 is not None and s003["sig_vru"] == 0
        s004 = df.filter(F.col("sample_token") == "s004").select("sig_vru").first()
        assert s004 is not None and s004["sig_vru"] == 0

    def test_cyclist_signal_fires_for_bicycle_and_motorcycle(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        df = scorer.compute_all_scores(sample_metadata_df)
        row = df.filter(F.col("sample_token") == "s002").select("sig_cyclist").first()
        assert row is not None and row["sig_cyclist"] == 1

    def test_rarity_score_max_6(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        df = scorer.compute_all_scores(sample_metadata_df)
        max_score = df.agg(F.max("rarity_score")).first()[0]
        assert max_score is not None and max_score <= 6

    def test_filter_by_threshold_raises_before_scoring(
        self, spark: SparkSession
    ) -> None:
        # SparkSession.builder.getOrCreate() returns the shared module session.
        # Do NOT call scorer2.stop() — the session is owned by the module fixture.
        scorer2 = SparkRarityScorer(_TEST_CONFIG)
        scorer2.spark = spark  # point at shared session to prevent double-stop
        with pytest.raises(RuntimeError, match="compute_all_scores"):
            scorer2.filter_by_threshold()

    def test_filter_by_threshold_returns_subset(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        scored = scorer.compute_all_scores(sample_metadata_df)
        rare = scorer.filter_by_threshold(min_score=2)
        assert rare.count() <= scored.count()

    def test_filter_sorted_descending(
        self, scorer: SparkRarityScorer, sample_metadata_df
    ) -> None:
        scorer.compute_all_scores(sample_metadata_df)
        rare = scorer.filter_by_threshold(min_score=1)
        scores = [r["rarity_score"] for r in rare.select("rarity_score").collect()]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 3. SparkAnalytics
# ===========================================================================

class TestSparkAnalytics:
    @pytest.fixture(scope="class")
    def analytics(self, spark: SparkSession, sample_metadata_df) -> SparkAnalytics:
        scorer = SparkRarityScorer(_TEST_CONFIG)
        scorer.spark = spark
        scored_df = scorer.compute_all_scores(sample_metadata_df)
        return SparkAnalytics(scored_df, _TEST_CONFIG)

    def test_score_distribution_has_all_scores(
        self, analytics: SparkAnalytics
    ) -> None:
        dist = analytics.score_distribution()
        assert "rarity_score" in dist.columns
        assert "frame_count" in dist.columns
        assert "cumulative_pct" in dist.columns

    def test_signal_cooccurrence_shape(self, analytics: SparkAnalytics) -> None:
        coo = analytics.signal_cooccurrence()
        assert coo.count() == 36  # 6 × 6

    def test_per_scene_stats_columns(self, analytics: SparkAnalytics) -> None:
        stats = analytics.per_scene_stats()
        expected = {
            "scene_token", "total_frames", "rare_frames",
            "mean_rarity_score", "max_rarity_score",
        }
        assert expected <= set(stats.columns)

    def test_category_breakdown_sorted(self, analytics: SparkAnalytics) -> None:
        bd = analytics.category_breakdown()
        counts = [r["annotation_count"] for r in bd.select("annotation_count").collect()]
        assert counts == sorted(counts, reverse=True)

    def test_temporal_clustering_burst_id(self, analytics: SparkAnalytics) -> None:
        clusters = analytics.temporal_clustering()
        assert "burst_id" in clusters.columns
        assert "time_gap" in clusters.columns

    def test_save_all_analytics_creates_parquet(
        self, analytics: SparkAnalytics, tmp_path
    ) -> None:
        out = analytics.save_all_analytics(output_dir=tmp_path)
        expected_tables = [
            "score_distribution.parquet",
            "signal_cooccurrence.parquet",
            "per_scene_stats.parquet",
            "category_breakdown.parquet",
            "temporal_clustering.parquet",
        ]
        for table_name in expected_tables:
            assert (out / table_name).exists(), f"Missing analytics table: {table_name}"


# ===========================================================================
# 4. NuScenesMetadataExtractor
# ===========================================================================

class TestNuScenesMetadataExtractor:
    def test_extractor_raises_without_devkit(self) -> None:
        """ImportError if nuScenes devkit is absent."""
        from unittest.mock import patch

        from drivesense.data.spark_pipeline import NuScenesMetadataExtractor

        with patch("drivesense.data.spark_pipeline._NUSCENES_AVAILABLE", False):
            with pytest.raises(ImportError, match="nuScenes devkit"):
                NuScenesMetadataExtractor(
                    nuscenes_root=__import__("pathlib").Path("/tmp/fake"),
                    config=_TEST_CONFIG,
                )
