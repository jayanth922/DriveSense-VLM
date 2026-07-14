"""Tests for scripts/merge_sft_v2.py (CPU-only, no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_sft_v2", _REPO / "scripts" / "merge_sft_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load_module()


def _rec(fid: str, scene: str) -> dict:
    return {"frame_id": fid, "scene_token": scene, "messages": [], "images": []}


def _write_split_dir(tmp_path: Path, name: str, recs_by_split: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    for sp in ("train", "val", "test"):
        (d / f"sft_{sp}_enriched.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs_by_split.get(sp, [])))
    return d


# ---------------------------------------------------------------------------


def test_dedup_first_occurrence_wins():
    a = {"frame_id": "f1", "scene_token": "s", "src": "old"}
    b = {"frame_id": "f1", "scene_token": "s", "src": "new"}
    c = {"frame_id": "f2", "scene_token": "s", "src": "new"}
    out = m.dedup_by_frame_id([a, b, c])
    assert [r["frame_id"] for r in out] == ["f1", "f2"]
    assert out[0]["src"] == "old"  # existing set wins on conflict


def test_assign_scene_split_no_leak_and_deterministic():
    recs = [_rec(f"f{i}", f"scene{i % 20}") for i in range(200)]
    m.assign_scene_split(recs, seed=42)
    m.verify_no_scene_leak(recs)  # must not raise
    # a scene's records all share one split
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene_token"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_scene.values())


def test_verify_no_scene_leak_raises_on_leak():
    recs = [{"scene_token": "s1", "split": "train"}, {"scene_token": "s1", "split": "test"}]
    with pytest.raises(AssertionError):
        m.verify_no_scene_leak(recs)


def test_require_keys_exits_on_missing(tmp_path):
    with pytest.raises(SystemExit):
        m.require_keys([{"frame_id": "f1"}])  # no scene_token


def test_full_merge_roundtrip(tmp_path):
    old = _write_split_dir(tmp_path, "old", {
        "train": [_rec("f1", "sA"), _rec("f2", "sB")],
        "test": [_rec("f3", "sC")],
    })
    new = _write_split_dir(tmp_path, "new", {
        "train": [_rec("f3", "sC"), _rec("f4", "sD")],  # f3 duplicates old → dropped
        "val": [_rec("f5", "sE")],
    })
    pool = m.load_split_records(old) + m.load_split_records(new)
    m.require_keys(pool)
    recs = m.dedup_by_frame_id(pool)
    assert sorted(r["frame_id"] for r in recs) == ["f1", "f2", "f3", "f4", "f5"]  # 6 → 5 unique
    m.assign_scene_split(recs, seed=1)
    m.verify_no_scene_leak(recs)
    out = tmp_path / "merged"
    paths = m.write_splits(recs, out)
    total = sum(len(p.read_text().splitlines()) for p in paths.values())
    assert total == 5  # every unique record written exactly once


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
