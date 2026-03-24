"""Tests for the nuScenes rarity filtering pipeline and related transforms.

All tests run on macOS WITHOUT nuScenes installed.  The mock infrastructure below
faithfully reproduces the nuScenes Python SDK interface so every public method of
NuScenesRarityFilter can be exercised in isolation.

Mock dataset design (3 samples across 2 scenes):
    scene_001  "Night, rain in Boston"   ← weather signal active
      sample_001  2 pedestrians (prox + occ + vru) → score 4  PASSES (≥3)
      sample_002  1 distant car                     → score 1  fails
    scene_002  "Clear day in Singapore"  ← no weather signal
      sample_003  15 cars + 1 nearby cyclist        → score 3  PASSES (≥3)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sys

# Ensure src/ on path for editable installs not yet installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data.transforms import normalize_bbox_to_1000, get_2d_bbox_from_3d
from drivesense.data.nuscenes_loader import NuScenesRarityFilter

# ---------------------------------------------------------------------------
# Mock nuScenes data definitions
# ---------------------------------------------------------------------------

_SCENES = [
    {
        "token": "scene_001",
        "description": "Night, rain in Boston, approaching intersection",
        "first_sample_token": "sample_001",
        "last_sample_token": "sample_002",
        "name": "scene-0001",
        "nbr_samples": 2,
        "log_token": "log_001",
    },
    {
        "token": "scene_002",
        "description": "Clear day in Singapore",
        "first_sample_token": "sample_003",
        "last_sample_token": "sample_003",
        "name": "scene-0002",
        "nbr_samples": 1,
        "log_token": "log_002",
    },
]

_SAMPLES = {
    "sample_001": {
        "token": "sample_001",
        "scene_token": "scene_001",
        "timestamp": 1_000_000,
        "anns": ["ann_ped_001", "ann_ped_002"],
        "data": {"CAM_FRONT": "camdata_001"},
        "next": "sample_002",
        "prev": "",
    },
    "sample_002": {
        "token": "sample_002",
        "scene_token": "scene_001",
        "timestamp": 1_500_000,
        "anns": ["ann_car_001"],
        "data": {"CAM_FRONT": "camdata_002"},
        "next": "",
        "prev": "sample_001",
    },
    "sample_003": {
        "token": "sample_003",
        "scene_token": "scene_002",
        "timestamp": 2_000_000,
        "anns": [f"ann_car_{i:03d}" for i in range(15)] + ["ann_cyc_001"],
        "data": {"CAM_FRONT": "camdata_003"},
        "next": "",
        "prev": "",
    },
}

# Annotations — pedestrian at ~0.7m and 3m from ego (ego_001 at origin).
_ANN_PED_001 = {
    "token": "ann_ped_001", "category_name": "human.pedestrian.adult",
    "translation": [0.5, 0.5, 0.0], "size": [0.7, 0.7, 1.7],
    "rotation": [1.0, 0.0, 0.0, 0.0], "visibility_token": "vis_1",
    "instance_token": "inst_001", "sample_token": "sample_001",
}
_ANN_PED_002 = {
    "token": "ann_ped_002", "category_name": "human.pedestrian.child",
    "translation": [3.0, 0.0, 0.0], "size": [0.5, 0.5, 1.2],
    "rotation": [1.0, 0.0, 0.0, 0.0], "visibility_token": "vis_4",
    "instance_token": "inst_002", "sample_token": "sample_001",
}
_ANN_CAR_001 = {
    "token": "ann_car_001", "category_name": "vehicle.car",
    "translation": [20.0, 5.0, 0.0], "size": [1.9, 4.5, 1.6],
    "rotation": [1.0, 0.0, 0.0, 0.0], "visibility_token": "vis_4",
    "instance_token": "inst_003", "sample_token": "sample_002",
}
_ANN_CYC_001 = {
    "token": "ann_cyc_001", "category_name": "vehicle.bicycle",
    "translation": [11.0, 10.0, 0.0], "size": [0.6, 1.5, 1.2],
    "rotation": [1.0, 0.0, 0.0, 0.0], "visibility_token": "vis_3",
    "instance_token": "inst_004", "sample_token": "sample_003",
}

_CAR_ANNS = {
    f"ann_car_{i:03d}": {
        "token": f"ann_car_{i:03d}", "category_name": "vehicle.car",
        "translation": [float(30 + i), float(i), 0.0],
        "size": [1.9, 4.5, 1.6], "rotation": [1.0, 0.0, 0.0, 0.0],
        "visibility_token": "vis_4",
        "instance_token": f"inst_car_{i:03d}", "sample_token": "sample_003",
    }
    for i in range(15)
}

_ANNOTATIONS = {
    "ann_ped_001": _ANN_PED_001,
    "ann_ped_002": _ANN_PED_002,
    "ann_car_001": _ANN_CAR_001,
    "ann_cyc_001": _ANN_CYC_001,
    **_CAR_ANNS,
}

_VISIBILITY = {
    "vis_1": {"token": "vis_1", "level": "1", "description": "0-40% visible"},
    "vis_3": {"token": "vis_3", "level": "3", "description": "60-80% visible"},
    "vis_4": {"token": "vis_4", "level": "4", "description": "80-100% visible"},
}

_EGO_POSES = {
    "ego_001": {"token": "ego_001", "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0], "timestamp": 1_000_000},
    "ego_002": {"token": "ego_002", "translation": [5.0, 5.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0], "timestamp": 1_500_000},
    "ego_003": {"token": "ego_003", "translation": [10.0, 10.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0], "timestamp": 2_000_000},
}

_CAM_DATA = {
    "camdata_001": {"token": "camdata_001", "ego_pose_token": "ego_001",
                    "calibrated_sensor_token": "cal_001", "width": 1600, "height": 900},
    "camdata_002": {"token": "camdata_002", "ego_pose_token": "ego_002",
                    "calibrated_sensor_token": "cal_001", "width": 1600, "height": 900},
    "camdata_003": {"token": "camdata_003", "ego_pose_token": "ego_003",
                    "calibrated_sensor_token": "cal_001", "width": 1600, "height": 900},
}


class MockNuScenes:
    """Minimal mock of the nuScenes NuScenes SDK for unit tests."""

    def __init__(self) -> None:
        self.scene = _SCENES
        self.sample = list(_SAMPLES.values())
        self._tables: dict[str, dict] = {
            "scene": {s["token"]: s for s in _SCENES},
            "sample": _SAMPLES,
            "sample_annotation": _ANNOTATIONS,
            "visibility": _VISIBILITY,
            "ego_pose": _EGO_POSES,
            "sample_data": _CAM_DATA,
        }

    def get(self, table: str, token: str) -> dict:  # noqa: D102
        return self._tables[table][token]

    def get_sample_data_path(self, token: str) -> str:  # noqa: D102
        return f"/mock/nuscenes/samples/CAM_FRONT/{token}.jpg"


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

_TEST_CONFIG: dict = {
    "nuscenes": {
        "version": "v1.0-mini",
        "cameras": ["CAM_FRONT"],
        "rarity": {
            "proximity_threshold_m": 5.0,
            "occlusion_min_visibility": 0,
            "occlusion_max_visibility": 40,
            "min_agents_for_density": 15,
            "adverse_weather_keywords": ["rain", "night", "fog", "storm"],
            "intersection_keywords": ["intersection", "crossing", "junction", "traffic light", "turn"],
            "min_rarity_score": 3,
        },
    },
    "paths": {"nuscenes_root": "/mock/nuscenes"},
}


@pytest.fixture
def mock_nusc() -> MockNuScenes:
    """Return a freshly constructed MockNuScenes instance."""
    return MockNuScenes()


@pytest.fixture
def rarity_filter(mock_nusc: MockNuScenes) -> NuScenesRarityFilter:
    """Return a NuScenesRarityFilter wired to the mock dataset."""
    with (
        patch("drivesense.data.nuscenes_loader._NUSCENES_AVAILABLE", True),
        patch("drivesense.data.nuscenes_loader.NuScenes", return_value=mock_nusc),
    ):
        filt = NuScenesRarityFilter(Path("/mock/nuscenes"), _TEST_CONFIG)
    return filt


# ---------------------------------------------------------------------------
# Tests — signal computation
# ---------------------------------------------------------------------------

class TestSignalComputation:
    """Verify each rarity signal produces the correct boolean + details."""

    def test_proximity_active_for_nearby_pedestrian(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Proximity signal fires when a pedestrian is within threshold distance."""
        active, details = rarity_filter._compute_proximity_score("sample_001")
        assert active is True
        distances = [d["distance_m"] for d in details]
        # ann_ped_001 at ~0.707m, ann_ped_002 at 3.0m — both within 5m.
        assert len(details) == 2
        assert all(d <= 5.0 for d in distances)

    def test_proximity_inactive_for_distant_car(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Proximity signal ignores non-VRU categories (cars) entirely."""
        active, details = rarity_filter._compute_proximity_score("sample_002")
        assert active is False
        assert details == []

    def test_occlusion_active_for_low_visibility(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Occlusion signal fires when visibility level is within [0, 40]%."""
        active, details = rarity_filter._compute_occlusion_score("sample_001")
        assert active is True
        tokens = [d["token"] for d in details]
        assert "ann_ped_001" in tokens  # vis_1 = 0-40%
        assert "ann_ped_002" not in tokens  # vis_4 = 80-100%

    def test_occlusion_inactive_when_all_visible(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Occlusion signal is quiet when all annotations are 80-100% visible."""
        active, details = rarity_filter._compute_occlusion_score("sample_002")
        assert active is False
        assert details == []

    def test_density_active_above_threshold(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Density signal fires when annotation count reaches the threshold."""
        active, count = rarity_filter._compute_density_score("sample_003")
        assert active is True
        assert count == 16  # 15 cars + 1 cyclist

    def test_density_inactive_below_threshold(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Density signal is quiet for sparse frames."""
        active, count = rarity_filter._compute_density_score("sample_001")
        assert active is False
        assert count == 2

    def test_weather_active_for_night_rain(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Weather signal fires when scene description contains a keyword."""
        active, kw = rarity_filter._compute_weather_score("scene_001")
        assert active is True
        assert kw in ("night", "rain")

    def test_weather_inactive_for_clear_day(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Weather signal is quiet for benign scene descriptions."""
        active, kw = rarity_filter._compute_weather_score("scene_002")
        assert active is False
        assert kw == ""

    def test_vru_active_with_pedestrians_at_intersection(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """VRU signal fires when pedestrian present AND scene contains intersection keyword."""
        # scene_001 = "Night, rain in Boston, approaching intersection" → keyword matches.
        active, count = rarity_filter._compute_vulnerable_road_user_score("sample_001")
        assert active is True
        assert count == 2

    def test_vru_requires_intersection_context(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """VRU signal is inactive when pedestrians present but no intersection keyword."""
        original = rarity_filter.nusc._tables["scene"]["scene_001"].copy()
        rarity_filter.nusc._tables["scene"]["scene_001"] = {
            **original, "description": "Night, rain in Boston"  # no intersection keyword
        }
        try:
            active, count = rarity_filter._compute_vulnerable_road_user_score("sample_001")
        finally:
            rarity_filter.nusc._tables["scene"]["scene_001"] = original

        assert active is False
        assert count == 2  # pedestrians counted for diagnostics even when signal inactive

    def test_cyclist_active(self, rarity_filter: NuScenesRarityFilter) -> None:
        """Cyclist signal fires when a bicycle annotation is present."""
        active, count = rarity_filter._compute_cyclist_score("sample_003")
        assert active is True
        assert count == 1

    def test_cyclist_inactive_when_none_present(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        """Cyclist signal is quiet for frames with no cyclist annotations."""
        active, count = rarity_filter._compute_cyclist_score("sample_001")
        assert active is False
        assert count == 0


# ---------------------------------------------------------------------------
# Tests — score aggregation
# ---------------------------------------------------------------------------

class TestScoreAggregation:
    """Verify composite score equals sum of active binary signals."""

    def test_score_is_sum_of_active_signals(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        for token, expected_score in [
            ("sample_001", 4),  # prox + occ + weather + vru (pedestrian + intersection kw)
            ("sample_002", 1),  # weather only
            ("sample_003", 3),  # prox + density + cyclist
        ]:
            result = rarity_filter.compute_rarity_score(token)
            manual_sum = sum(v["active"] for v in result["signals"].values())
            assert result["rarity_score"] == manual_sum, (
                f"{token}: rarity_score={result['rarity_score']} != manual_sum={manual_sum}"
            )
            assert result["rarity_score"] == expected_score

    def test_score_dict_has_required_keys(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        result = rarity_filter.compute_rarity_score("sample_001")
        required = {
            "sample_token", "scene_token", "rarity_score", "signals",
            "cam_front_path", "cam_front_token", "timestamp",
            "scene_description", "ego_pose", "num_annotations",
        }
        assert required.issubset(result.keys())
        assert set(result["signals"].keys()) == {
            "proximity", "occlusion", "density",
            "weather", "vulnerable_road_user", "cyclist",
        }


# ---------------------------------------------------------------------------
# Tests — filtering threshold
# ---------------------------------------------------------------------------

class TestFilteringThreshold:
    """Verify filter_rare_frames respects min_score and sorts results correctly."""

    def test_default_threshold_passes_expected_frames(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        filtered = rarity_filter.filter_rare_frames()
        tokens = [f["sample_token"] for f in filtered]
        assert "sample_001" in tokens  # score 4
        assert "sample_003" in tokens  # score 3
        assert "sample_002" not in tokens  # score 1

    def test_results_sorted_by_score_descending(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        filtered = rarity_filter.filter_rare_frames()
        scores = [f["rarity_score"] for f in filtered]
        assert scores == sorted(scores, reverse=True)

    def test_override_min_score(self, rarity_filter: NuScenesRarityFilter) -> None:
        strict = rarity_filter.filter_rare_frames(min_score=4)
        tokens = [f["sample_token"] for f in strict]
        assert "sample_001" in tokens
        assert "sample_003" not in tokens  # score 3 < 4
        assert "sample_002" not in tokens

    def test_all_frames_returned_at_zero_threshold(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        all_frames = rarity_filter.filter_rare_frames(min_score=0)
        assert len(all_frames) == 3

    def test_empty_result_at_impossible_threshold(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        empty = rarity_filter.filter_rare_frames(min_score=10)
        assert empty == []

    def test_score_distribution_populated_after_filter(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        rarity_filter.filter_rare_frames(min_score=10)  # empty result
        dist = rarity_filter.get_score_distribution()
        # All 3 samples still scored.
        assert sum(dist.values()) == 3

    def test_score_distribution_empty_before_filter(
        self, rarity_filter: NuScenesRarityFilter
    ) -> None:
        # Never called filter_rare_frames → empty distribution.
        dist = rarity_filter.get_score_distribution()
        assert dist == {}


# ---------------------------------------------------------------------------
# Tests — export
# ---------------------------------------------------------------------------

class TestExport:
    """Verify export_filtered_dataset creates expected files and structure."""

    def test_export_creates_expected_files(
        self, rarity_filter: NuScenesRarityFilter, tmp_path: Path
    ) -> None:
        rarity_filter.filter_rare_frames()
        with patch("shutil.copy2"):
            out = rarity_filter.export_filtered_dataset(tmp_path / "out")

        assert (out / "metadata.json").exists()
        assert (out / "score_distribution.json").exists()
        assert (out / "summary.json").exists()
        assert (out / "images").is_dir()

    def test_metadata_contains_exported_frames(
        self, rarity_filter: NuScenesRarityFilter, tmp_path: Path
    ) -> None:
        rarity_filter.filter_rare_frames()
        with patch("shutil.copy2"):
            out = rarity_filter.export_filtered_dataset(tmp_path / "out")

        metadata = json.loads((out / "metadata.json").read_text())
        assert len(metadata) == 2
        tokens = {m["sample_token"] for m in metadata}
        assert tokens == {"sample_001", "sample_003"}

    def test_summary_has_correct_counts(
        self, rarity_filter: NuScenesRarityFilter, tmp_path: Path
    ) -> None:
        rarity_filter.filter_rare_frames()
        with patch("shutil.copy2"):
            out = rarity_filter.export_filtered_dataset(tmp_path / "out")

        summary = json.loads((out / "summary.json").read_text())
        assert summary["total_frames_scanned"] == 3
        assert summary["frames_exported"] == 2

    def test_export_raises_if_filter_not_called(
        self, rarity_filter: NuScenesRarityFilter, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="filter_rare_frames"):
            rarity_filter.export_filtered_dataset(tmp_path / "out")

    def test_export_is_idempotent(
        self, rarity_filter: NuScenesRarityFilter, tmp_path: Path
    ) -> None:
        """Running export twice produces the same metadata.json."""
        rarity_filter.filter_rare_frames()
        out = tmp_path / "out"
        with patch("shutil.copy2"):
            rarity_filter.export_filtered_dataset(out)
            rarity_filter.export_filtered_dataset(out)

        metadata = json.loads((out / "metadata.json").read_text())
        assert len(metadata) == 2


# ---------------------------------------------------------------------------
# Tests — transforms
# ---------------------------------------------------------------------------

class TestBboxNormalization:
    """Verify normalize_bbox_to_1000 maps pixel coords to [0, 1000] correctly."""

    def test_full_image_maps_to_1000(self) -> None:
        assert normalize_bbox_to_1000([0, 0, 1600, 900], 1600, 900) == [0, 0, 1000, 1000]

    def test_center_quarter(self) -> None:
        result = normalize_bbox_to_1000([400, 225, 1200, 675], 1600, 900)
        assert result == [250, 250, 750, 750]

    def test_small_box_top_left(self) -> None:
        result = normalize_bbox_to_1000([0, 0, 160, 90], 1600, 900)
        assert result == [0, 0, 100, 100]

    def test_result_types_are_int(self) -> None:
        result = normalize_bbox_to_1000([10.5, 20.5, 100.5, 200.5], 1000, 1000)
        assert all(isinstance(v, int) for v in result)

    def test_rounding_behavior(self) -> None:
        # 1 pixel in 1600-wide image → 1000/1600 = 0.625 → rounds to 1
        result = normalize_bbox_to_1000([0, 0, 1, 1], 1600, 900)
        assert result[2] == round(1000 / 1600)


class TestProjectionBehindCamera:
    """Verify get_2d_bbox_from_3d returns None when all corners are behind camera."""

    def test_all_corners_behind_camera_returns_none(self) -> None:
        mock_nusc = MagicMock()
        mock_nusc.get.side_effect = lambda table, token: {
            ("sample_data", "cam_tok"): {
                "token": "cam_tok", "ego_pose_token": "ego_tok",
                "calibrated_sensor_token": "cal_tok", "width": 1600, "height": 900,
            },
            ("ego_pose", "ego_tok"): {
                "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            ("calibrated_sensor", "cal_tok"): {
                "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [[1000, 0, 800], [0, 1000, 450], [0, 0, 1]],
            },
        }[(table, token)]

        mock_box = MagicMock()
        # All 8 corners have z < 0 — entirely behind the camera plane.
        corners = np.zeros((3, 8))
        corners[2, :] = -5.0
        mock_box.corners.return_value = corners
        mock_nusc.get_box.return_value = mock_box

        with (
            patch("drivesense.data.transforms._NUSCENES_AVAILABLE", True),
            patch("drivesense.data.transforms.Quaternion") as MockQ,
        ):
            MockQ.return_value.inverse = MagicMock()
            result = get_2d_bbox_from_3d(mock_nusc, "ann_tok", "cam_tok")

        assert result is None

    def test_corners_in_front_returns_bbox(self) -> None:
        """Verify a valid projection returns [x1, y1, x2, y2] in [0, 1000] range."""
        mock_nusc = MagicMock()
        mock_nusc.get.side_effect = lambda table, token: {
            ("sample_data", "cam_tok"): {
                "token": "cam_tok", "ego_pose_token": "ego_tok",
                "calibrated_sensor_token": "cal_tok", "width": 1600, "height": 900,
            },
            ("ego_pose", "ego_tok"): {
                "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            ("calibrated_sensor", "cal_tok"): {
                "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]],
            },
        }[(table, token)]

        mock_box = MagicMock()
        # Corners in a neat cluster in front of the camera (z > 0).
        corners = np.array([
            [700.0, 750.0, 700.0, 750.0, 700.0, 750.0, 700.0, 750.0],  # x
            [350.0, 350.0, 400.0, 400.0, 350.0, 350.0, 400.0, 400.0],  # y
            [5.0,   5.0,   5.0,   5.0,   6.0,   6.0,   6.0,   6.0],   # z > 0
        ])
        mock_box.corners.return_value = corners
        mock_nusc.get_box.return_value = mock_box

        def mock_view_points(pts: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
            # Simple projection: divide x,y by z and apply intrinsic scale.
            result = pts.copy()
            if normalize:
                result[0] = pts[0] / pts[2] * 1000 + 800
                result[1] = pts[1] / pts[2] * 1000 + 450
            return result

        with (
            patch("drivesense.data.transforms._NUSCENES_AVAILABLE", True),
            patch("drivesense.data.transforms.view_points", side_effect=mock_view_points),
            patch("drivesense.data.transforms.Quaternion") as MockQ,
        ):
            MockQ.return_value.inverse = MagicMock()
            result = get_2d_bbox_from_3d(mock_nusc, "ann_tok", "cam_tok")

        assert result is not None
        assert len(result) == 4
        x1, y1, x2, y2 = result
        assert 0 <= x1 <= x2 <= 1000
        assert 0 <= y1 <= y2 <= 1000

    def test_raises_without_nuscenes(self) -> None:
        """ImportError is raised when nuScenes devkit is absent."""
        with patch("drivesense.data.transforms._NUSCENES_AVAILABLE", False):
            with pytest.raises(ImportError, match="nuScenes devkit"):
                get_2d_bbox_from_3d(MagicMock(), "ann", "cam")
