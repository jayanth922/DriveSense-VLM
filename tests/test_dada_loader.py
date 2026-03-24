"""Tests for the DADA-2000 dataset loader (Phase 1b).

All tests are fully mocked — no real DADA-2000 dataset is required.
A synthetic directory tree with minimal PNG files is created in a tmp_path
fixture for filesystem-level tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

# Ensure src/ on path for editable installs not yet installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data.dada_loader import (  # noqa: E402
    DADA2000Loader,
    _infer_road_type,
    _infer_time_of_day,
    _infer_weather,
    normalize_column_names,
)

# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

_BASE_CONFIG: dict = {
    "dada2000": {
        "frame_extraction": {"method": "critical_moment", "additional_context_frames": 2},
        "max_frames": 20,
    },
    "preprocessing": {"target_resolution": [672, 448]},
    "paths": {"dada2000_root": "~/data/dada2000"},
}


def _make_config(**overrides: object) -> dict:
    import copy
    cfg = copy.deepcopy(_BASE_CONFIG)
    for k, v in overrides.items():
        cfg["dada2000"][k] = v
    return cfg


# ---------------------------------------------------------------------------
# Fixtures: synthetic filesystem
# ---------------------------------------------------------------------------

def _write_small_png(path: Path) -> None:
    """Write a 10×10 black PNG to path."""
    img = PILImage.new("RGB", (10, 10), color=(0, 0, 0))
    img.save(path, format="PNG")


@pytest.fixture()
def dada_tree(tmp_path: Path) -> Path:
    """Create a minimal DADA-2000-style directory tree.

    Tree:
        <tmp>/DADA-2000/
            001/001/images/  → 5 frames (001.png .. 005.png)
            001/002/images/  → 3 frames
            002/001/images/  → 0 frames (empty → should be skipped)
    """
    root = tmp_path / "DADA-2000"
    for cat, seq, n_frames in [("001", "001", 5), ("001", "002", 3), ("002", "001", 0)]:
        img_dir = root / cat / seq / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, n_frames + 1):
            _write_small_png(img_dir / f"{i:03d}.png")
    return tmp_path


# ---------------------------------------------------------------------------
# TestInferHelpers
# ---------------------------------------------------------------------------

class TestInferHelpers:
    def test_infer_weather_rain(self) -> None:
        assert _infer_weather("Heavy rain on the motorway") == "rain"

    def test_infer_weather_fog(self) -> None:
        assert _infer_weather("Dense fog reduces visibility") == "fog"

    def test_infer_weather_night(self) -> None:
        assert _infer_weather("Dark night, city street") == "night"

    def test_infer_weather_clear(self) -> None:
        assert _infer_weather("Sunny day downtown") == "clear"

    def test_infer_time_of_day_night(self) -> None:
        assert _infer_time_of_day("Rainy night scene") == "night"

    def test_infer_time_of_day_day(self) -> None:
        assert _infer_time_of_day("Midday traffic") == "day"

    def test_infer_road_type_highway(self) -> None:
        assert _infer_road_type("Vehicle on the highway overtaking") == "highway"

    def test_infer_road_type_intersection(self) -> None:
        assert _infer_road_type("Collision at an intersection") == "intersection"

    def test_infer_road_type_default(self) -> None:
        assert _infer_road_type("Pedestrian on residential street") == "urban"


# ---------------------------------------------------------------------------
# TestNormalizeColumnNames
# ---------------------------------------------------------------------------

class TestNormalizeColumnNames:
    def test_exact_match(self) -> None:
        mock_df = MagicMock()
        mock_df.columns = ["category", "sequence", "accident_frame", "description"]
        mapping = normalize_column_names(mock_df)
        assert mapping["category"] == "category"
        assert mapping["accident_frame"] == "accident_frame"

    def test_fuzzy_match(self) -> None:
        mock_df = MagicMock()
        mock_df.columns = ["Cat", "Seq", "Accident Frame", "Scene Description"]
        mapping = normalize_column_names(mock_df)
        assert mapping.get("category") == "Cat"
        assert mapping.get("sequence") == "Seq"
        assert mapping.get("accident_frame") == "Accident Frame"

    def test_missing_column_omitted(self) -> None:
        mock_df = MagicMock()
        mock_df.columns = ["category"]
        mapping = normalize_column_names(mock_df)
        assert "sequence" not in mapping
        assert "category" in mapping


# ---------------------------------------------------------------------------
# TestDADA2000LoaderDiscovery
# ---------------------------------------------------------------------------

class TestDADA2000LoaderDiscovery:
    def test_discovers_sequences(self, dada_tree: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        # 3 sequences: 001/001 (5 frames), 001/002 (3 frames), 002/001 (0 frames)
        # All have an images/ dir that exists, so all are discovered.
        assert len(loader._sequences) == 3

    def test_sequence_fields(self, dada_tree: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        seq = next(s for s in loader._sequences if s["category"] == "001" and s["sequence"] == "001")
        assert seq["images_dir"].exists()
        assert seq["weather"] in ("clear", "rain", "fog", "snow", "night")

    def test_category_distribution(self, dada_tree: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        dist = loader.get_category_distribution()
        assert dist["001"] == 2

    def test_summary_statistics_keys(self, dada_tree: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        stats = loader.get_summary_statistics()
        assert "total_sequences" in stats
        assert "total_categories" in stats
        assert "max_frames_budget" in stats
        assert stats["target_resolution"] == [672, 448]

    def test_missing_dataset_dir(self, tmp_path: Path) -> None:
        """Loader should not raise when dataset dir is absent."""
        loader = DADA2000Loader(tmp_path / "nonexistent", _make_config())
        assert len(loader._sequences) == 0


# ---------------------------------------------------------------------------
# TestExtractKeyframes
# ---------------------------------------------------------------------------

class TestExtractKeyframes:
    def test_extract_returns_correct_types(self, dada_tree: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        seq = next(s for s in loader._sequences if s["sequence"] == "001")
        seq["accident_frame"] = 3  # frame 3 of 5
        frames = loader.extract_keyframes(seq)
        assert len(frames) > 0
        for f in frames:
            assert "frame_index" in f
            assert "frame_type" in f
            assert f["frame_type"] in ("pre_accident", "critical", "mid_accident")
            assert isinstance(f["image"], PILImage.Image)

    def test_critical_frame_present(self, dada_tree: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        seq = next(s for s in loader._sequences if s["sequence"] == "001")
        seq["accident_frame"] = 3
        frames = loader.extract_keyframes(seq)
        types = [f["frame_type"] for f in frames]
        assert "critical" in types

    def test_output_image_size(self, dada_tree: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        seq = loader._sequences[0]
        seq["accident_frame"] = 2
        frames = loader.extract_keyframes(seq)
        for f in frames:
            assert f["image"].size == (672, 448)

    def test_no_frames_empty_dir(self, dada_tree: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        empty_seq = {
            "category": "003",
            "sequence": "001",
            "images_dir": dada_tree / "nonexistent",
            "accident_frame": None,
            "description": "",
            "weather": "clear",
            "time_of_day": "day",
            "road_type": "urban",
        }
        frames = loader.extract_keyframes(empty_seq)
        assert frames == []

    def test_max_frames_respected(self, dada_tree: Path) -> None:
        cfg = _make_config()
        cfg["dada2000"]["max_frames"] = 2
        loader = DADA2000Loader(dada_tree, cfg)
        frames = loader.extract_all_keyframes()
        assert len(frames) <= 2


# ---------------------------------------------------------------------------
# TestExportKeyframes
# ---------------------------------------------------------------------------

class TestExportKeyframes:
    def test_export_writes_jsonl(self, dada_tree: Path, tmp_path: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        out = tmp_path / "export"
        jsonl_path = loader.export_keyframes(out)
        assert jsonl_path.exists()
        records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        assert len(records) > 0

    def test_export_record_schema(self, dada_tree: Path, tmp_path: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        out = tmp_path / "export2"
        jsonl_path = loader.export_keyframes(out)
        record = json.loads(jsonl_path.read_text().splitlines()[0])
        for key in ("frame_id", "source", "image_path", "category", "sequence", "frame_type"):
            assert key in record
        assert record["source"] == "dada2000"

    def test_export_images_exist(self, dada_tree: Path, tmp_path: Path) -> None:
        loader = DADA2000Loader(dada_tree, _make_config())
        out = tmp_path / "export3"
        jsonl_path = loader.export_keyframes(out)
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                assert Path(rec["image_path"]).exists()
