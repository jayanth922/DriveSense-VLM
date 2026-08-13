"""Tests for drivesense.data.mining_targets and scripts/select_mining_targets.py.
Pure data/logic — no GPU/torch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data.mining_targets import (  # noqa: E402
    clean_target_spec,
    hazard_annotations,
    infer_time_of_day,
    infer_weather,
    parse_bucket_key,
    proxied_size_tier,
    score_frame,
    score_histogram,
    select_targets,
    worst_bucket_to_target_spec,
)


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ann(category: str, distance: float = 15.0, vis: int = 4) -> dict:
    return {"category_name": category, "distance_to_ego": distance, "visibility_level": vis}


def _record(basename: str, anns: list[dict], scene_desc: str = "", scene: str = "sc1") -> dict:
    return {
        "sample_token": basename.split("__")[0], "scene_token": scene,
        "cam_front_path": f"/root/nuscenes/samples/CAM_FRONT/{basename}",
        "num_annotations": len(anns), "annotations": anns, "scene_description": scene_desc,
    }


# ---------------------------------------------------------------------------
# Proxies
# ---------------------------------------------------------------------------


class TestProxies:
    @pytest.mark.parametrize("dist,expected", [
        (0.0, "large"), (5.0, "large"), (7.999, "large"),
        (8.0, "medium"), (15.0, "medium"), (19.999, "medium"),
        (20.0, "small"), (35.0, "small"), (39.999, "small"),
        (40.0, "tiny"), (200.0, "tiny"),
    ])
    def test_proxied_size_tier_boundaries(self, dist, expected):
        assert proxied_size_tier(dist) == expected

    def test_infer_weather_keywords(self):
        assert infer_weather("Scene with heavy RAIN and traffic") == "rain"
        assert infer_weather("foggy morning near the storm front") == "fog"
        assert infer_weather("clear sunny afternoon") == "clear"

    def test_infer_time_of_day_keywords(self):
        assert infer_time_of_day("driving at night near downtown") == "night"
        assert infer_time_of_day("bright daytime scene") == "day"


class TestHazardAnnotations:
    def test_filters_to_hazard_relevant_only(self):
        anns = [_ann("human.pedestrian.adult"), _ann("vehicle.car"), _ann("vehicle.bicycle")]
        haz = hazard_annotations({"annotations": anns})
        assert len(haz) == 2  # pedestrian + bicycle; car excluded

    def test_empty_annotations(self):
        assert hazard_annotations({"annotations": []}) == []
        assert hazard_annotations({}) == []


# ---------------------------------------------------------------------------
# Target spec parsing
# ---------------------------------------------------------------------------


class TestParseBucketKey:
    def test_dimension_tier_value(self):
        assert parse_bucket_key("weather:tiny|rain") == {"size_tier": "tiny", "weather": "rain"}

    def test_all_on_either_side_dropped(self):
        assert parse_bucket_key("weather:tiny|ALL") == {"size_tier": "tiny"}
        assert parse_bucket_key("weather:ALL|rain") == {"weather": "rain"}
        assert parse_bucket_key("weather:ALL|ALL") == {}

    def test_bare_tier_value_no_dimension_prefix(self):
        # No "dimension:" prefix -> only size_tier is extractable (no dim to
        # attach the value to).
        assert parse_bucket_key("large|urban") == {"size_tier": "large"}


class TestCleanTargetSpec:
    def test_drops_location_keeps_supported(self):
        spec = {"size_tier": "large", "weather": "rain", "location": "urban"}
        cleaned = clean_target_spec(spec)
        assert cleaned == {"size_tier": "large", "weather": "rain"}
        assert "location" not in cleaned

    def test_all_supported_passes_through(self):
        spec = {"size_tier": "tiny", "weather": "rain", "time_of_day": "night"}
        assert clean_target_spec(spec) == spec


class TestWorstBucketToTargetSpec:
    def test_picks_first_worst_bucket(self):
        report = {"worst_buckets": [{"bucket": "weather:tiny|rain", "n": 10}]}
        assert worst_bucket_to_target_spec(report) == {"size_tier": "tiny", "weather": "rain"}

    def test_empty_worst_buckets_is_empty_spec(self):
        assert worst_bucket_to_target_spec({"worst_buckets": []}) == {}
        assert worst_bucket_to_target_spec({}) == {}

    def test_location_only_bucket_yields_empty_spec(self):
        report = {"worst_buckets": [{"bucket": "location:ALL|urban", "n": 10}]}
        assert worst_bucket_to_target_spec(report) == {}


# ---------------------------------------------------------------------------
# score_frame — the actual ranking logic; must produce a SENSIBLE ordering
# ---------------------------------------------------------------------------


class TestScoreFrame:
    def test_full_match_scores_higher_than_partial(self):
        target = {"size_tier": "large", "weather": "rain"}
        full = _record("a.jpg", [_ann("human.pedestrian.adult", distance=3.0)],
                       scene_desc="heavy rain tonight")
        partial = _record("b.jpg", [_ann("human.pedestrian.adult", distance=3.0)],
                          scene_desc="clear skies")  # size matches, weather doesn't
        none = _record("c.jpg", [_ann("human.pedestrian.adult", distance=100.0)],
                       scene_desc="clear skies")  # neither matches
        s_full, s_partial, s_none = (score_frame(r, target) for r in (full, partial, none))
        assert s_full > s_partial > s_none
        assert s_full == pytest.approx(1.0)
        assert s_none == pytest.approx(0.0)

    def test_no_hazard_annotations_scores_zero(self):
        target = {"size_tier": "large"}
        rec = _record("a.jpg", [_ann("vehicle.car", distance=3.0)])  # not hazard-relevant
        assert score_frame(rec, target) == 0.0

    def test_empty_target_scores_zero(self):
        rec = _record("a.jpg", [_ann("human.pedestrian.adult")])
        assert score_frame(rec, {}) == 0.0

    def test_mixed_annotations_partial_fraction(self):
        # 2 hazard-relevant anns, only 1 matches the target size tier -> 0.5
        target = {"size_tier": "large"}
        rec = _record("a.jpg", [
            _ann("human.pedestrian.adult", distance=3.0),   # large
            _ann("vehicle.bicycle", distance=100.0),        # tiny
        ])
        assert score_frame(rec, target) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# select_targets — end-to-end ranking + exclusion + band filter
# ---------------------------------------------------------------------------


class TestSelectTargets:
    def _write_metadata(self, tmp_path: Path, records: list[dict]) -> Path:
        p = tmp_path / "metadata.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records))
        return p

    def test_ranking_is_sensible_best_match_first(self, tmp_path):
        peds5 = [_ann("human.pedestrian.adult", distance=3.0) for _ in range(5)]
        records = [
            _record("best.jpg", peds5, scene_desc="heavy rain"),          # 5 hazards, large, rain
            _record("mid.jpg", peds5, scene_desc="clear"),                # 5 hazards, large, clear
            _record("worst.jpg", [_ann("human.pedestrian.adult", distance=100.0)] * 5,
                    scene_desc="clear"),                                   # tiny, clear
        ]
        meta = self._write_metadata(tmp_path, records)
        rows = select_targets(meta, {"size_tier": "large", "weather": "rain"},
                              band=(1, 20), target_count=10)
        basenames = [r["basename"] for r in rows]
        assert basenames[0] == "best.jpg"
        assert basenames[-1] == "worst.jpg"
        assert rows[0]["mining_score"] > rows[-1]["mining_score"]

    def test_already_mined_excluded_via_have_manifest(self, tmp_path):
        peds = [_ann("human.pedestrian.adult", distance=3.0)] * 5
        records = [_record("have.jpg", peds), _record("want.jpg", peds)]
        meta = self._write_metadata(tmp_path, records)
        rows = select_targets(meta, {"size_tier": "large"}, already_have={"have.jpg"},
                              band=(1, 20))
        assert [r["basename"] for r in rows] == ["want.jpg"]

    def test_band_filter_excludes_out_of_band_frames(self, tmp_path):
        many = [_ann("human.pedestrian.adult", distance=3.0)] * 25  # hazard_count 25, out of [3,20]
        few = [_ann("human.pedestrian.adult", distance=3.0)] * 5
        records = [_record("toomany.jpg", many), _record("inband.jpg", few)]
        meta = self._write_metadata(tmp_path, records)
        rows = select_targets(meta, {"size_tier": "large"}, band=(3, 20))
        assert [r["basename"] for r in rows] == ["inband.jpg"]

    def test_target_count_caps_output(self, tmp_path):
        records = [_record(f"f{i}.jpg", [_ann("human.pedestrian.adult")] * 5) for i in range(10)]
        meta = self._write_metadata(tmp_path, records)
        rows = select_targets(meta, {"size_tier": "medium"}, band=(1, 20), target_count=3)
        assert len(rows) == 3

    def test_min_score_filters_out_zero_matches(self, tmp_path):
        records = [
            _record("match.jpg", [_ann("human.pedestrian.adult", distance=3.0)] * 5),
            _record("nomatch.jpg", [_ann("vehicle.car")] * 5),  # no hazard anns -> score 0
        ]
        meta = self._write_metadata(tmp_path, records)
        rows = select_targets(meta, {"size_tier": "large"}, band=(1, 20), min_score=0.01)
        assert [r["basename"] for r in rows] == ["match.jpg"]

    def test_output_rows_carry_shopping_list_schema(self, tmp_path):
        records = [_record("f.jpg", [_ann("human.pedestrian.adult", distance=3.0)] * 5)]
        meta = self._write_metadata(tmp_path, records)
        rows = select_targets(meta, {"size_tier": "large"}, band=(1, 20))
        assert set(rows[0]) >= {"basename", "sample_token", "scene_token",
                                "hazard_count", "cam_front_path", "mining_score"}


class TestScoreHistogram:
    def test_buckets_by_score(self):
        rows = [{"mining_score": s} for s in (0.0, 0.3, 0.6, 0.9, 1.0)]
        hist = score_histogram(rows)
        assert sum(hist.values()) == 5


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestSelectMiningTargetsCLI:
    def test_runs_end_to_end(self, tmp_path, capsys):
        mod = _load_module("scripts/select_mining_targets.py", "select_cli_test")
        records = [_record(f"f{i}.jpg", [_ann("human.pedestrian.adult", distance=3.0)] * 5,
                           scene_desc="rain") for i in range(5)]
        meta_path = tmp_path / "metadata.jsonl"
        meta_path.write_text("\n".join(json.dumps(r) for r in records))
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(
            {"worst_buckets": [{"bucket": "weather:large|rain", "n": 20}]}))
        out_path = tmp_path / "shoppinglist.jsonl"
        sys.argv = ["select", "--report", str(report_path), "--metadata", str(meta_path),
                   "--output", str(out_path), "--target-count", "3"]
        mod.main()
        out = capsys.readouterr().out
        assert "Selected" in out
        assert json.loads(Path(out_path).read_text().splitlines()[0])["basename"]

    def test_missing_report_exits_two(self, tmp_path):
        mod = _load_module("scripts/select_mining_targets.py", "select_cli_test2")
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["select", "--report", str(tmp_path / "nope.json"),
                       "--metadata", str(tmp_path / "nope2.jsonl")]
            mod.main()
        assert exc.value.code == 2

    def test_location_only_worst_bucket_exits_two_with_guidance(self, tmp_path):
        mod = _load_module("scripts/select_mining_targets.py", "select_cli_test3")
        meta_path = tmp_path / "metadata.jsonl"
        meta_path.write_text("")
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(
            {"worst_buckets": [{"bucket": "location:ALL|urban", "n": 20}]}))
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["select", "--report", str(report_path), "--metadata", str(meta_path)]
            mod.main()
        assert exc.value.code == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
