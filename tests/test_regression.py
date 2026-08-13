"""Tests for drivesense.eval.regression and the two scripts built on it
(run_regression_gate.py, compare_eval_runs.py). Pure data/logic — no GPU/torch.
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

from drivesense.eval.regression import (  # noqa: E402
    DEFAULT_METRICS,
    build_comparison_rows,
    compare_summaries,
    evaluate_metric,
    get_metric,
    relative_change,
    verdict_label,
)


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary(det01=0.1, det03=0.05, det05=0.02, hdr=0.05, iou=0.1, parse=0.2, acc=0.3) -> dict:
    return {"level1": {
        "hazard_detection_rate": hdr,
        "detection_rate_by_iou": {"0.1": det01, "0.3": det03, "0.5": det05},
        "mean_best_pair_iou": iou,
        "parse_failure_rate": parse,
        "classification_accuracy": acc,
    }}


# ---------------------------------------------------------------------------
# get_metric — including the exact bug found via real execution: a dotted-path
# split on EVERY "." shreds the literal key "0.1" into ["0", "1"].
# ---------------------------------------------------------------------------


class TestGetMetric:
    def test_top_level_key(self):
        assert get_metric(_summary(hdr=0.42), "hazard_detection_rate") == 0.42

    def test_nested_key_containing_a_dot(self):
        # "0.1" is itself a dict key with a literal dot in it — must not be
        # split further. Regression test for the real bug found while testing.
        s = _summary(det01=0.33)
        assert get_metric(s, "detection_rate_by_iou.0.1") == 0.33

    def test_all_three_iou_thresholds_resolve(self):
        s = _summary(det01=0.1, det03=0.2, det05=0.3)
        assert get_metric(s, "detection_rate_by_iou.0.1") == 0.1
        assert get_metric(s, "detection_rate_by_iou.0.3") == 0.2
        assert get_metric(s, "detection_rate_by_iou.0.5") == 0.3

    def test_missing_key_is_none(self):
        assert get_metric(_summary(), "does_not_exist") is None
        assert get_metric(_summary(), "detection_rate_by_iou.0.9") is None

    def test_tolerates_bare_level1_dict(self):
        bare = _summary()["level1"]
        assert get_metric(bare, "hazard_detection_rate") == bare["hazard_detection_rate"]


# ---------------------------------------------------------------------------
# relative_change — direction + zero-baseline handling
# ---------------------------------------------------------------------------


class TestRelativeChange:
    def test_higher_better_drop_is_positive(self):
        assert relative_change("higher_better", 0.50, 0.25) == pytest.approx(0.5)

    def test_higher_better_improvement_is_negative(self):
        assert relative_change("higher_better", 0.25, 0.50) == pytest.approx(-1.0)

    def test_lower_better_increase_is_positive(self):
        # parse_failure_rate going UP is bad.
        assert relative_change("lower_better", 0.20, 0.40) == pytest.approx(1.0)

    def test_lower_better_decrease_is_negative(self):
        assert relative_change("lower_better", 0.40, 0.20) == pytest.approx(-0.5)

    def test_zero_baseline_higher_better_no_regression_possible(self):
        assert relative_change("higher_better", 0.0, 0.0) == 0.0
        assert relative_change("higher_better", 0.0, 0.5) == 0.0  # can't regress below 0

    def test_zero_baseline_lower_better_flags_new_failures(self):
        # A perfect (0% parse-failure) baseline that starts failing must be
        # flagged, even though 0.05/0 is mathematically undefined.
        assert relative_change("lower_better", 0.0, 0.05) == 1.0
        assert relative_change("lower_better", 0.0, 0.0) == 0.0

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError):
            relative_change("sideways", 0.1, 0.2)


# ---------------------------------------------------------------------------
# evaluate_metric — tolerance-gated pass/fail (the CI gate's core decision)
# ---------------------------------------------------------------------------


class TestEvaluateMetric:
    def test_small_drop_within_tolerance_is_flat_not_regressed(self):
        spec = {"path": "x", "direction": "higher_better", "tolerance": 0.10}
        r = evaluate_metric("x", spec, 0.50, 0.48)  # 4% drop, tolerance 10%
        assert r["status"] == "FLAT"
        assert not r["regressed"]

    def test_large_drop_beyond_tolerance_regresses(self):
        spec = {"path": "x", "direction": "higher_better", "tolerance": 0.10}
        r = evaluate_metric("x", spec, 0.50, 0.30)  # 40% drop
        assert r["status"] == "REGRESSED"
        assert r["regressed"]

    def test_missing_value_is_missing_not_regressed(self):
        spec = {"path": "x", "direction": "higher_better", "tolerance": 0.10}
        r = evaluate_metric("x", spec, None, 0.5)
        assert r["status"] == "MISSING"
        assert not r["regressed"]

    def test_lower_better_metric_regression_direction(self):
        spec = {"path": "parse_failure_rate", "direction": "lower_better", "tolerance": 0.10}
        r = evaluate_metric("parse_failure_rate", spec, 0.24, 0.55)
        assert r["status"] == "REGRESSED"


# ---------------------------------------------------------------------------
# compare_summaries — the exact reported v2->v3 regression scenario
# ---------------------------------------------------------------------------


class TestCompareSummaries:
    def test_catches_iou_and_parse_failure_regression(self):
        # Mirrors the real fixture that reproduced the reported v2->v3 bug:
        # detection_rate_by_iou@0.1 drops hard, parse_failure_rate spikes.
        baseline = _summary(det01=0.33, det03=0.20, det05=0.05, hdr=0.06,
                            iou=0.12, parse=0.24, acc=0.35)
        new = _summary(det01=0.15, det03=0.20, det05=0.05, hdr=0.06,
                       iou=0.12, parse=0.55, acc=0.35)
        results = compare_summaries(baseline, new)
        regressed = {r["metric"] for r in results if r["regressed"]}
        assert regressed == {"detection_rate_by_iou@0.1", "parse_failure_rate"}

    def test_all_improved_passes_clean(self):
        baseline = _summary(det01=0.10, det03=0.05, det05=0.02, hdr=0.03,
                            iou=0.08, parse=0.40, acc=0.30)
        new = _summary(det01=0.33, det03=0.20, det05=0.05, hdr=0.06,
                       iou=0.12, parse=0.24, acc=0.35)
        results = compare_summaries(baseline, new)
        assert not any(r["regressed"] for r in results)
        assert all(r["status"] == "IMPROVED" for r in results)

    def test_uses_default_metrics_by_default(self):
        results = compare_summaries(_summary(), _summary())
        assert {r["metric"] for r in results} == set(DEFAULT_METRICS)


# ---------------------------------------------------------------------------
# verdict_label + build_comparison_rows — the human comparison report
# ---------------------------------------------------------------------------


class TestVerdictLabel:
    def test_first_run_has_no_verdict(self):
        assert verdict_label("higher_better", None, 0.5) == "n/a"

    def test_improved_regressed_flat(self):
        assert verdict_label("higher_better", 0.10, 0.20) == "↑ improved"
        assert verdict_label("higher_better", 0.20, 0.10) == "↓ regressed"
        assert verdict_label("higher_better", 0.20, 0.20) == "→ flat"

    def test_tiny_change_within_eps_is_flat(self):
        assert verdict_label("higher_better", 0.500, 0.501, eps=0.01) == "→ flat"


class TestBuildComparisonRows:
    def test_three_run_sequence_verdicts_vs_previous(self):
        v1 = _summary(det01=0.10, hdr=0.03, parse=0.40)
        v2 = _summary(det01=0.33, hdr=0.06, parse=0.24)
        v3 = _summary(det01=0.15, hdr=0.06, parse=0.55)
        rows = build_comparison_rows([("v1", v1), ("v2", v2), ("v3", v3)])
        by_metric = {r["metric"]: r["cells"] for r in rows}

        det = by_metric["detection_rate_by_iou@0.1"]
        assert det[0]["verdict"] == "n/a"          # first run: nothing to compare
        assert det[1]["verdict"] == "↑ improved"    # v1 -> v2
        assert det[2]["verdict"] == "↓ regressed"   # v2 -> v3

        parse = by_metric["parse_failure_rate"]
        assert parse[1]["verdict"] == "↑ improved"
        assert parse[2]["verdict"] == "↓ regressed"


# ---------------------------------------------------------------------------
# CLI smoke tests — real files on disk, real exit codes
# ---------------------------------------------------------------------------


class TestRegressionGateCLI:
    def test_pass_does_not_exit_nonzero(self, tmp_path, capsys):
        # main() has no sys.exit() call at all on the success path — it just
        # returns, so no SystemExit is raised (that's the "pass" signal).
        mod = _load_module("scripts/run_regression_gate.py", "gate_test_pass")
        baseline = tmp_path / "b.json"
        new = tmp_path / "n.json"
        baseline.write_text(json.dumps(_summary(det01=0.10, parse=0.40)))
        new.write_text(json.dumps(_summary(det01=0.33, parse=0.20)))
        sys.argv = ["gate", "--baseline", str(baseline), "--new", str(new)]
        mod.main()  # must NOT raise
        assert "GATE PASSED" in capsys.readouterr().out

    def test_regression_exits_one(self, tmp_path, capsys):
        mod = _load_module("scripts/run_regression_gate.py", "gate_test_fail")
        baseline = tmp_path / "b.json"
        new = tmp_path / "n.json"
        baseline.write_text(json.dumps(_summary(det01=0.33, parse=0.24)))
        new.write_text(json.dumps(_summary(det01=0.15, parse=0.55)))
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["gate", "--baseline", str(baseline), "--new", str(new)]
            mod.main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "GATE FAILED" in out
        assert "detection_rate_by_iou@0.1" in out

    def test_missing_file_exits_two(self, tmp_path):
        mod = _load_module("scripts/run_regression_gate.py", "gate_test_missing")
        new = tmp_path / "n.json"
        new.write_text(json.dumps(_summary()))
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["gate", "--baseline", str(tmp_path / "nope.json"), "--new", str(new)]
            mod.main()
        assert exc.value.code == 2

    def test_tolerance_override_relaxes_a_metric(self, tmp_path, capsys):
        mod = _load_module("scripts/run_regression_gate.py", "gate_test_tol")
        baseline = tmp_path / "b.json"
        new = tmp_path / "n.json"
        baseline.write_text(json.dumps(_summary(parse=0.24)))
        new.write_text(json.dumps(_summary(parse=0.55)))  # +129%, fails at 10% tol
        sys.argv = ["gate", "--baseline", str(baseline), "--new", str(new),
                   "--tolerance", "parse_failure_rate=2.0"]
        mod.main()  # relaxed tolerance -> must NOT raise
        assert "GATE PASSED" in capsys.readouterr().out

    def test_unknown_tolerance_metric_exits_two(self, tmp_path):
        mod = _load_module("scripts/run_regression_gate.py", "gate_test_badtol")
        baseline = tmp_path / "b.json"
        new = tmp_path / "n.json"
        baseline.write_text(json.dumps(_summary()))
        new.write_text(json.dumps(_summary()))
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["gate", "--baseline", str(baseline), "--new", str(new),
                       "--tolerance", "not_a_real_metric=0.5"]
            mod.main()
        assert exc.value.code == 2

    def test_output_report_written(self, tmp_path):
        mod = _load_module("scripts/run_regression_gate.py", "gate_test_output")
        baseline = tmp_path / "b.json"
        new = tmp_path / "n.json"
        report = tmp_path / "report.json"
        baseline.write_text(json.dumps(_summary()))
        new.write_text(json.dumps(_summary()))
        sys.argv = ["gate", "--baseline", str(baseline), "--new", str(new),
                   "--output", str(report)]
        mod.main()
        assert json.loads(report.read_text())


class TestCompareEvalRunsCLI:
    def test_console_output_columns_dont_collide(self, tmp_path, capsys):
        mod = _load_module("scripts/compare_eval_runs.py", "compare_test_console")
        v1 = tmp_path / "v1.json"
        v2 = tmp_path / "v2.json"
        v1.write_text(json.dumps(_summary(det01=0.10)))
        v2.write_text(json.dumps(_summary(det01=0.33)))
        sys.argv = ["compare", "--run", f"v1={v1}", "--run", f"v2={v2}"]
        mod.main()
        out = capsys.readouterr().out
        # Regression test for the real bug found via execution: value+verdict
        # cells running together with no separating whitespace.
        assert "improved0." not in out
        assert "0.3300 ↑ improved" in out

    def test_markdown_output_has_table(self, tmp_path, capsys):
        mod = _load_module("scripts/compare_eval_runs.py", "compare_test_md")
        v1 = tmp_path / "v1.json"
        v2 = tmp_path / "v2.json"
        v1.write_text(json.dumps(_summary(det01=0.10)))
        v2.write_text(json.dumps(_summary(det01=0.33)))
        sys.argv = ["compare", "--run", f"v1={v1}", "--run", f"v2={v2}", "--format", "markdown"]
        mod.main()
        out = capsys.readouterr().out
        assert "| Metric | v1 | v2 |" in out
        assert "0.3300 (↑ improved)" in out

    def test_missing_file_exits_two(self, tmp_path):
        mod = _load_module("scripts/compare_eval_runs.py", "compare_test_missing")
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["compare", "--run", f"v1={tmp_path / 'nope.json'}"]
            mod.main()
        assert exc.value.code == 2

    def test_bad_run_spec_exits_two(self, tmp_path):
        mod = _load_module("scripts/compare_eval_runs.py", "compare_test_badspec")
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["compare", "--run", "no-equals-sign-here"]
            mod.main()
        assert exc.value.code == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
