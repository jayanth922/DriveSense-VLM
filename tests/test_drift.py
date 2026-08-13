"""Tests for drivesense.monitoring.drift (PSI-based drift detection) and the
demo script built on it. Pure data/logic — no GPU/torch.
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

from drivesense.monitoring.drift import (  # noqa: E402
    DEFAULT_DIMENSIONS,
    DriftMonitor,
    category_distribution,
    category_distribution_from_records,
    extract_dimension_values,
    population_stability_index,
    psi_severity,
)


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(weather="clear", time_of_day="day", location="urban", hazards=None) -> dict:
    hazards = [{"label": h} for h in (hazards if hazards is not None else ["jaywalking"])]
    return {
        "weather": weather, "time_of_day": time_of_day, "location": location,
        "messages": [{"role": "assistant", "content": json.dumps({"hazards": hazards})}],
    }


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------


class TestExtractDimensionValues:
    def test_simple_dimension_top_level(self):
        assert extract_dimension_values(_rec(weather="rain"), "weather") == ["rain"]

    def test_missing_dimension_is_unknown(self):
        assert extract_dimension_values({}, "weather") == ["unknown"]

    def test_metadata_and_ego_context_fallback(self):
        rec = {"metadata": {"weather": "fog"}}
        assert extract_dimension_values(rec, "weather") == ["fog"]
        rec2 = {"ego_context": {"weather": "rain"}}
        assert extract_dimension_values(rec2, "weather") == ["rain"]

    def test_hazard_class_multi_valued(self):
        rec = _rec(hazards=["jaywalking", "cyclist_proximity"])
        assert extract_dimension_values(rec, "hazard_class") == \
            ["jaywalking", "cyclist_proximity"]

    def test_hazard_class_empty_is_no_hazard(self):
        rec = _rec(hazards=[])
        assert extract_dimension_values(rec, "hazard_class") == ["no_hazard"]

    def test_hazard_class_direct_hazards_key(self):
        rec = {"hazards": [{"label": "debris"}]}
        assert extract_dimension_values(rec, "hazard_class") == ["debris"]


# ---------------------------------------------------------------------------
# category_distribution / category_distribution_from_records
# ---------------------------------------------------------------------------


class TestCategoryDistribution:
    def test_normalises_counts_to_proportions(self):
        dist = category_distribution(["a", "a", "b"])
        assert dist == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}

    def test_empty_is_empty_dict(self):
        assert category_distribution([]) == {}

    def test_from_records_flattens_multi_valued_dimension(self):
        records = [_rec(hazards=["a", "b"]), _rec(hazards=["a"])]
        dist = category_distribution_from_records(records, "hazard_class")
        assert dist == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------


class TestPSI:
    def test_identical_distributions_is_zero(self):
        d = {"a": 0.5, "b": 0.5}
        assert population_stability_index(d, d) == 0.0

    def test_no_drift_band_for_small_shift(self):
        ref = {"clear": 0.9, "rain": 0.1}
        inc = {"clear": 0.88, "rain": 0.12}
        psi = population_stability_index(ref, inc)
        assert psi < 0.10
        assert psi_severity(psi) == "none"

    def test_significant_drift_band_for_total_shift(self):
        ref = {"clear": 1.0}
        inc = {"rain": 1.0}
        psi = population_stability_index(ref, inc)
        assert psi > 0.20
        assert psi_severity(psi) == "significant"

    def test_new_category_in_incoming_contributes_to_psi(self):
        # A category never seen in the reference is itself a drift signal.
        ref = {"clear": 1.0}
        inc = {"clear": 0.5, "snow": 0.5}
        assert population_stability_index(ref, inc) > 0.20

    def test_severity_bands_are_monotonic(self):
        assert psi_severity(0.05) == "none"
        assert psi_severity(0.10) == "moderate"
        assert psi_severity(0.15) == "moderate"
        assert psi_severity(0.20) == "significant"
        assert psi_severity(1.0) == "significant"


# ---------------------------------------------------------------------------
# DriftMonitor — the actual class the demo script drives
# ---------------------------------------------------------------------------


class TestDriftMonitor:
    def test_from_records_builds_reference_per_dimension(self):
        records = [_rec(weather="clear") for _ in range(9)] + [_rec(weather="rain")]
        monitor = DriftMonitor.from_records(records, ["weather"])
        assert monitor.reference["weather"]["clear"] == pytest.approx(0.9)
        assert monitor.reference["weather"]["rain"] == pytest.approx(0.1)

    def test_same_distribution_incoming_batch_has_no_drift(self):
        # The exact demo scenario: reference and incoming from the SAME source
        # distribution must not trip any dimension.
        ref_records = [_rec(weather="clear", location=("urban" if i % 2 else "highway"))
                       for i in range(100)]
        inc_records = [_rec(weather="clear", location=("urban" if i % 2 else "highway"))
                       for i in range(100)]
        monitor = DriftMonitor.from_records(ref_records, DEFAULT_DIMENSIONS)
        report = monitor.check(inc_records)
        assert not DriftMonitor.any_drifted(report)
        assert all(r["severity"] == "none" for r in report.values())

    def test_skewed_incoming_batch_flags_only_that_dimension(self):
        # The exact demo scenario: force weather to a single value in the
        # incoming batch -> only 'weather' should flag, other dims stay clean.
        ref_records = [_rec(weather=("clear" if i % 10 else "rain"),
                            location=("urban" if i % 2 else "highway"))
                       for i in range(200)]
        incoming = [dict(r, weather="rain") for r in ref_records[:100]]
        monitor = DriftMonitor.from_records(ref_records, DEFAULT_DIMENSIONS)
        report = monitor.check(incoming)
        assert report["weather"]["drifted"] is True
        assert report["location"]["drifted"] is False
        assert report["time_of_day"]["drifted"] is False

    def test_custom_threshold_is_respected(self):
        ref_records = [_rec(weather="clear") for _ in range(90)] + \
                      [_rec(weather="rain") for _ in range(10)]
        incoming = [_rec(weather="clear") for _ in range(80)] + \
                   [_rec(weather="rain") for _ in range(20)]
        monitor_strict = DriftMonitor.from_records(ref_records, ["weather"], threshold=0.01)
        monitor_loose = DriftMonitor.from_records(ref_records, ["weather"], threshold=5.0)
        assert monitor_strict.check(incoming)["weather"]["drifted"] is True
        assert monitor_loose.check(incoming)["weather"]["drifted"] is False

    def test_any_drifted_false_when_report_empty(self):
        assert DriftMonitor.any_drifted({}) is False


# ---------------------------------------------------------------------------
# Demo script — runs standalone (synthetic fallback), and correctly
# discriminates the no-drift vs drift cases (its own built-in self-check).
# ---------------------------------------------------------------------------


class TestDemoScript:
    def test_demo_runs_standalone_and_passes_its_own_checks(self, capsys):
        mod = _load_module("scripts/demo_drift_monitor.py", "drift_demo_test")
        sys.argv = ["demo", "--n-synthetic", "300", "--seed", "7"]
        mod.main()  # must not raise / sys.exit(1) — both cases must PASS
        out = capsys.readouterr().out
        assert "Case A correctly showed no drift : PASS" in out
        assert "Case B correctly flagged weather  : PASS" in out

    def test_demo_falls_back_to_synthetic_when_labels_file_missing(self, tmp_path, capsys):
        mod = _load_module("scripts/demo_drift_monitor.py", "drift_demo_test2")
        sys.argv = ["demo", "--labels", str(tmp_path / "nope.jsonl"), "--n-synthetic", "200"]
        mod.main()
        assert "generating synthetic records instead" in capsys.readouterr().out

    def test_demo_uses_real_labels_file_when_present(self, tmp_path, capsys):
        mod = _load_module("scripts/demo_drift_monitor.py", "drift_demo_test3")
        labels = tmp_path / "labels.jsonl"
        records = [_rec(weather=("clear" if i % 5 else "rain")) for i in range(60)]
        labels.write_text("\n".join(json.dumps(r) for r in records))
        sys.argv = ["demo", "--labels", str(labels)]
        mod.main()
        out = capsys.readouterr().out
        assert "Loaded 60 real records" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
