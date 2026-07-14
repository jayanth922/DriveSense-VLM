"""Tests for the streaming bounded-storage image miner (CPU-only, no network)."""

from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data import streaming_miner as sm


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ann(category: str, vis: int = 4) -> dict:
    return {"category_name": category, "visibility_level": vis}


def _record(basename: str, anns: list[dict], scene: str = "sc1") -> dict:
    return {
        "sample_token": basename.split("__")[0],
        "scene_token": scene,
        "cam_front_path": f"/data/nuscenes/samples/CAM_FRONT/{basename}",
        "num_annotations": len(anns),
        "annotations": anns,
    }


def _write_meta(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "metadata.jsonl"
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def _make_tar_gz(path: Path, members: dict[str, bytes]) -> None:
    """Write a gzipped tarball with the given {arcname: bytes} members."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    path.write_bytes(gzip.compress(raw.getvalue()))


# ---------------------------------------------------------------------------
# Hazard counting + band filtering
# ---------------------------------------------------------------------------


def test_hazard_count_hazard_class_mode_counts_only_mapped():
    rec = _record("a.jpg", [
        _ann("human.pedestrian.adult"),      # -> jaywalking (hazard)
        _ann("vehicle.bicycle"),             # -> cyclist_proximity (hazard)
        _ann("vehicle.car"),                 # -> None (not a hazard)
        _ann("movable_object.trafficcone"),  # -> construction_zone (hazard)
    ])
    assert sm.frame_hazard_count(rec, "hazard_class") == 3
    assert sm.frame_hazard_count(rec, "num_annotations") == 4


def test_occluded_pedestrian_still_counts_as_hazard():
    rec = _record("a.jpg", [_ann("human.pedestrian.adult", vis=1)])
    assert sm.frame_hazard_count(rec) == 1


def test_in_band_inclusive():
    assert sm.in_band(3, (3, 20))
    assert sm.in_band(20, (3, 20))
    assert not sm.in_band(2, (3, 20))
    assert not sm.in_band(21, (3, 20))


def test_band_frames_excludes_present_images(tmp_path):
    cam = tmp_path / "CAM_FRONT"
    cam.mkdir()
    (cam / "have.jpg").write_bytes(b"x")  # already local -> excluded
    hazards = [_ann("human.pedestrian.adult")] * 5
    meta = _write_meta(tmp_path, [
        _record("have.jpg", hazards),        # in band but present -> skip
        _record("want.jpg", hazards),        # in band, missing -> keep
        _record("sparse.jpg", [_ann("human.pedestrian.adult")]),  # count 1 -> out of band
    ])
    out = sm.band_frames_without_images(meta, cam, (3, 20))
    assert [f["basename"] for f in out] == ["want.jpg"]
    assert out[0]["hazard_count"] == 5


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _frames(counts: list[int]) -> list[dict]:
    return [{"basename": f"f{i}.jpg", "hazard_count": c} for i, c in enumerate(counts)]


def test_stratified_sample_hits_target_and_is_deterministic():
    frames = _frames([3] * 40 + [7] * 40 + [15] * 20)  # 100 frames across 3 strata
    a = sm.stratified_sample(frames, 30, [3, 6, 9, 13, 21], seed=42)
    b = sm.stratified_sample(frames, 30, [3, 6, 9, 13, 21], seed=42)
    assert len(a) == 30
    assert [f["basename"] for f in a] == [f["basename"] for f in b]  # deterministic


def test_stratified_sample_preserves_distribution_not_densest():
    frames = _frames([3] * 80 + [15] * 20)  # 80% low-density, 20% dense
    picked = sm.stratified_sample(frames, 50, [3, 6, 9, 13, 21], seed=1)
    dense = sum(1 for f in picked if f["hazard_count"] == 15)
    # proportional: ~20% dense, NOT skewed to the densest frames
    assert dense <= 15


def test_stratified_sample_returns_all_when_target_exceeds():
    frames = _frames([3, 7, 15])
    assert len(sm.stratified_sample(frames, 999, [3, 6, 9, 13, 21])) == 3
    assert len(sm.stratified_sample(frames, 0, [3, 6, 9, 13, 21])) == 3


# ---------------------------------------------------------------------------
# Streaming extraction — the bounded-storage core
# ---------------------------------------------------------------------------


def test_stream_extract_only_wanted_cam_front_keyframes(tmp_path):
    tar = tmp_path / "blob.tgz"
    _make_tar_gz(tar, {
        "samples/CAM_FRONT/want1.jpg": b"aaaa",
        "samples/CAM_FRONT/want2.jpg": b"bbbb",
        "samples/CAM_FRONT/unwanted.jpg": b"cccc",   # not on shopping list
        "sweeps/CAM_FRONT/nonkey.jpg": b"dddd",       # non-keyframe -> ignored
        "samples/CAM_BACK/other.jpg": b"eeee",        # wrong camera -> ignored
    })
    dest = tmp_path / "out"
    written, skipped = sm.stream_extract_blob(tar, {"want1.jpg", "want2.jpg"}, dest)
    assert sorted(written) == ["want1.jpg", "want2.jpg"]
    assert skipped == 0
    assert (dest / "want1.jpg").read_bytes() == b"aaaa"
    assert not (dest / "unwanted.jpg").exists()
    assert not (dest / "nonkey.jpg").exists()


def test_stream_extract_idempotent_skips_existing(tmp_path):
    tar = tmp_path / "blob.tgz"
    _make_tar_gz(tar, {"samples/CAM_FRONT/w.jpg": b"orig"})
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "w.jpg").write_bytes(b"orig")  # already fetched
    written, skipped = sm.stream_extract_blob(tar, {"w.jpg"}, dest)
    assert written == []
    assert skipped == 1


# ---------------------------------------------------------------------------
# Auth resolution — no guessed URLs
# ---------------------------------------------------------------------------


def test_resolve_prefers_local_then_url_then_token(tmp_path):
    blob = "v1.0-trainval02_blobs.tgz"
    (tmp_path / blob).write_bytes(b"x")
    assert sm.resolve_blob_source(blob, tmp_path, {}, None, "u")[0] == "local"
    assert sm.resolve_blob_source(blob, None, {blob: "http://s"}, None, "u") == ("url", "http://s")
    assert sm.resolve_blob_source(blob, None, {}, "tok", "http://b")[0] == "token"


def test_resolve_missing_when_no_auth():
    kind, ref = sm.resolve_blob_source("blob.tgz", None, {}, None, "http://b")
    assert kind == "missing" and ref == ""


def test_auth_instructions_are_concrete():
    txt = sm.auth_instructions()
    assert "nuscenes.org" in txt
    assert "--blob-urls-file" in txt and "--blob-dir" in txt


# ---------------------------------------------------------------------------
# Resume manifest
# ---------------------------------------------------------------------------


def test_manifest_persists_and_resumes(tmp_path):
    path = tmp_path / "m.json"
    m = sm.MiningManifest(path)
    assert not m.is_done("b1")
    m.mark_done("b1", matched=12, hwm_gb=24.5)
    m2 = sm.MiningManifest(path)  # reload from disk
    assert m2.is_done("b1")
    assert m2.per_blob["b1"] == 12
    assert m2.total_matched() == 12
    assert m2.disk_hwm_gb == 24.5


def test_manifest_hwm_is_max(tmp_path):
    m = sm.MiningManifest(tmp_path / "m.json")
    m.mark_done("b1", 1, 30.0)
    m.mark_done("b2", 1, 20.0)
    assert m.disk_hwm_gb == 30.0


# ---------------------------------------------------------------------------
# Shopping-list round-trip
# ---------------------------------------------------------------------------


def test_shoppinglist_roundtrip(tmp_path):
    frames = [{"basename": "a.jpg", "hazard_count": 5, "scene_token": "s"}]
    p = tmp_path / "sub" / "list.jsonl"
    sm.write_shoppinglist(frames, p)
    assert sm.load_shoppinglist(p) == frames


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
