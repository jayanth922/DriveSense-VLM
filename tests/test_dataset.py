"""Tests for UnifiedDatasetBuilder and DriveSenseDataset (Phase 1b).

All tests are fully mocked — no real dataset files are required.
JSONL manifests are written to tmp_path fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image as PILImage

# Ensure src/ on path for editable installs not yet installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data.dataset import (  # noqa: E402
    DriveSenseDataset,
    UnifiedDatasetBuilder,
    _load_jsonl,
    _load_parquet,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_BASE_CONFIG: dict = {
    "splits": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
    "spark": {"output_dir": "outputs/data/spark_processed"},
    "unified": {"output_dir": "outputs/data/unified"},
}

_NUSCENES_RECORD = {
    "sample_token": "tok_001",
    "scene_token": "scene_001",
    "camera": "CAM_FRONT",
    "image_path": "/fake/path/001.png",
    "scene_description": "Night rain at intersection",
    "rarity_score": 4,
    "annotations": [{"category": "pedestrian", "token": "ann_001"}],
}

_DADA_RECORD = {
    "frame_id": "dada_cat01_seq001_frame0003_critical",
    "source": "dada2000",
    "image_path": "/fake/path/dada_001.png",
    "description": "Pedestrian crossing accident",
    "weather": "clear",
    "time_of_day": "day",
    "road_type": "urban",
    "category": "001",
    "sequence": "001",
    "frame_index": 3,
    "frame_type": "critical",
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (672, 448), color=(100, 100, 100)).save(path, format="PNG")


# ---------------------------------------------------------------------------
# TestLoadJSONL
# ---------------------------------------------------------------------------

class TestLoadJSONL:
    def test_loads_records(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, [{"a": 1}, {"b": 2}])
        records = _load_jsonl(p)
        assert len(records) == 2
        assert records[0]["a"] == 1

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        records = _load_jsonl(tmp_path / "nonexistent.jsonl")
        assert records == []

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text('{"ok": 1}\nnot-json\n{"ok": 2}\n')
        records = _load_jsonl(p)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# TestLoadParquet
# ---------------------------------------------------------------------------

class TestLoadParquet:
    def test_missing_path_returns_empty(self, tmp_path: Path) -> None:
        records = _load_parquet(tmp_path / "nonexistent.parquet")
        assert records == []

    def test_returns_empty_without_pandas(self, tmp_path: Path) -> None:
        import unittest.mock as mock
        with mock.patch("drivesense.data.dataset._PANDAS_AVAILABLE", False):
            records = _load_parquet(tmp_path)
            assert records == []


# ---------------------------------------------------------------------------
# TestUnifiedDatasetBuilderLoad
# ---------------------------------------------------------------------------

class TestUnifiedDatasetBuilderLoad:
    def test_load_nuscenes_from_jsonl(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "metadata.jsonl"
        _write_jsonl(jsonl, [_NUSCENES_RECORD] * 5)
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        n = builder.load_nuscenes_frames(tmp_path)
        assert n == 5
        assert len(builder._frames) == 5

    def test_load_nuscenes_normalises_source(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "metadata.jsonl"
        _write_jsonl(jsonl, [_NUSCENES_RECORD])
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        builder.load_nuscenes_frames(tmp_path)
        assert builder._frames[0]["source"] == "nuscenes"

    def test_load_dada_from_jsonl(self, tmp_path: Path) -> None:
        dada_dir = tmp_path / "dada"
        _write_jsonl(dada_dir / "metadata.jsonl", [_DADA_RECORD] * 3)
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        n = builder.load_dada2000_frames(dada_dir)
        assert n == 3
        assert builder._frames[0]["source"] == "dada2000"

    def test_load_missing_dada_returns_zero(self, tmp_path: Path) -> None:
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        n = builder.load_dada2000_frames(tmp_path / "empty")
        assert n == 0

    def test_combined_load(self, tmp_path: Path) -> None:
        _write_jsonl(tmp_path / "metadata.jsonl", [_NUSCENES_RECORD] * 4)
        dada_dir = tmp_path / "dada"
        _write_jsonl(dada_dir / "metadata.jsonl", [_DADA_RECORD] * 6)
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        builder.load_nuscenes_frames(tmp_path)
        builder.load_dada2000_frames(dada_dir)
        assert len(builder._frames) == 10


# ---------------------------------------------------------------------------
# TestUnifiedDatasetBuilderSplits
# ---------------------------------------------------------------------------

class TestUnifiedDatasetBuilderSplits:
    def _builder_with_frames(self, n: int) -> UnifiedDatasetBuilder:
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        for i in range(n):
            builder._frames.append({
                "frame_id": f"frame_{i}",
                "source": "nuscenes" if i % 2 == 0 else "dada2000",
                "image_path": "",
                "scene_description": "",
                "weather": "clear",
                "time_of_day": "day",
                "road_type": "urban",
                "category": "001",
                "rarity_score": 3,
                "source_metadata": {},
                "annotations": [],
                "split": "",
            })
        return builder

    def test_all_frames_assigned(self) -> None:
        builder = self._builder_with_frames(20)
        builder.assign_splits()
        assert all(f["split"] in ("train", "val", "test") for f in builder._frames)

    def test_train_is_largest(self) -> None:
        builder = self._builder_with_frames(30)
        builder.assign_splits()
        stats = builder.get_statistics()
        assert stats["train"] >= stats["val"]
        assert stats["train"] >= stats["test"]

    def test_no_overlap(self) -> None:
        builder = self._builder_with_frames(20)
        builder.assign_splits()
        ids_by_split: dict[str, set] = {"train": set(), "val": set(), "test": set()}
        for f in builder._frames:
            ids_by_split[f["split"]].add(f["frame_id"])
        for a in ("train", "val", "test"):
            for b in ("train", "val", "test"):
                if a != b:
                    assert ids_by_split[a].isdisjoint(ids_by_split[b])

    def test_small_dataset_sequential_fallback(self) -> None:
        """Fewer than 10 frames should fall back to sequential splitting."""
        builder = self._builder_with_frames(5)
        builder.assign_splits()
        assert all(f["split"] != "" for f in builder._frames)


# ---------------------------------------------------------------------------
# TestUnifiedDatasetBuilderBuild
# ---------------------------------------------------------------------------

class TestUnifiedDatasetBuilderBuild:
    def test_build_writes_manifest_files(self, tmp_path: Path) -> None:
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        for i in range(15):
            builder._frames.append({
                "frame_id": f"f{i}", "source": "nuscenes", "image_path": "",
                "scene_description": "", "weather": "clear", "time_of_day": "day",
                "road_type": "urban", "category": "001", "rarity_score": 3,
                "source_metadata": {}, "annotations": [], "split": "",
            })
        paths = builder.build(tmp_path / "out")
        for split in ("train", "val", "test"):
            assert paths[split].exists()

    def test_build_manifest_valid_jsonl(self, tmp_path: Path) -> None:
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        for i in range(10):
            builder._frames.append({
                "frame_id": f"f{i}", "source": "dada2000", "image_path": "",
                "scene_description": "", "weather": "rain", "time_of_day": "night",
                "road_type": "highway", "category": "002", "rarity_score": 0,
                "source_metadata": {}, "annotations": [], "split": "",
            })
        paths = builder.build(tmp_path / "out2")
        all_records = []
        for p in paths.values():
            all_records += _load_jsonl(p)
        assert len(all_records) == 10

    def test_get_statistics_keys(self, tmp_path: Path) -> None:
        builder = UnifiedDatasetBuilder(_BASE_CONFIG)
        for i in range(8):
            builder._frames.append({
                "frame_id": f"f{i}", "source": "nuscenes", "image_path": "",
                "scene_description": "", "weather": "clear", "time_of_day": "day",
                "road_type": "urban", "category": "001", "rarity_score": 0,
                "source_metadata": {}, "annotations": [], "split": "",
            })
        builder.assign_splits()
        stats = builder.get_statistics()
        assert "total" in stats
        assert "sources" in stats
        assert "categories" in stats
        assert stats["total"] == 8


# ---------------------------------------------------------------------------
# TestDriveSenseDataset
# ---------------------------------------------------------------------------

class TestDriveSenseDataset:
    def _make_manifest(self, tmp_path: Path, n: int, with_images: bool = False) -> Path:
        manifest = tmp_path / "train_manifest.jsonl"
        records = []
        for i in range(n):
            img_path = tmp_path / f"frame_{i:04d}.png"
            if with_images:
                _write_image(img_path)
            records.append({
                "frame_id": f"frame_{i}",
                "source": "nuscenes",
                "image_path": str(img_path),
                "scene_description": "Test scene",
                "weather": "clear",
                "time_of_day": "day",
                "road_type": "urban",
                "category": "pedestrian",
                "rarity_score": 3,
                "source_metadata": {},
                "annotations": [{"hazard_class": "pedestrian_in_path", "severity": 2}],
                "split": "train",
            })
        _write_jsonl(manifest, records)
        return manifest

    def test_len(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, 7)
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        assert len(ds) == 7

    def test_getitem_keys(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, 3, with_images=True)
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        item = ds[0]
        assert "image" in item
        assert "input_text" in item
        assert "target_text" in item
        assert "metadata" in item

    def test_getitem_image_loaded(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, 2, with_images=True)
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        item = ds[0]
        assert isinstance(item["image"], PILImage.Image)

    def test_getitem_missing_image_returns_none(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, 2, with_images=False)
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        item = ds[0]
        assert item["image"] is None

    def test_getitem_target_text_is_json(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, 2)
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        item = ds[0]
        parsed = json.loads(item["target_text"])
        assert isinstance(parsed, dict)

    def test_metadata_fields(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, 2)
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        meta = ds[0]["metadata"]
        for key in ("frame_id", "source", "split", "rarity_score"):
            assert key in meta

    def test_collate_fn_batching(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, 3)
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        batch = [ds[0], ds[1], ds[2]]
        collated = DriveSenseDataset.collate_fn(batch)
        assert len(collated["images"]) == 3
        assert len(collated["input_texts"]) == 3
        assert len(collated["target_texts"]) == 3
        assert len(collated["metadata"]) == 3

    def test_get_collate_fn_callable(self, tmp_path: Path) -> None:
        manifest = self._make_manifest(tmp_path, 2)
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        fn = ds.get_collate_fn()
        assert callable(fn)

    def test_empty_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("")
        ds = DriveSenseDataset(manifest, "train", _BASE_CONFIG)
        assert len(ds) == 0
