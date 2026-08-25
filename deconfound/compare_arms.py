#!/usr/bin/env python3
"""FM-vs-GT comparison, read from the repo's actual eval output.

run_full_evaluation --level 1 4 writes, per arm, a
results_<arm>/level4_robustness/robustness_metrics.json. That file already holds
both the overall metrics (under by_location / by_source — a single bucket that
covers the whole test set) and the condition-stratified detection rates
(by_time_of_day, by_weather). That is everything the de-confound headline needs,
so we read it directly rather than the human-readable Level-1 report.

    python deconfound/compare_arms.py --fm results_fm --gt results_gt \
        --out deconfound_result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(results_dir: str) -> dict:
    p = Path(results_dir) / "level4_robustness" / "robustness_metrics.json"
    return json.loads(p.read_text())


def overall(rm: dict) -> dict:
    # by_location has a single bucket covering the full test set; use it as "overall"
    m = next(iter(rm["by_location"].values()))
    return {
        "recall": m["hazard_detection_rate"],
        "precision": m["precision"],
        "f1": m["f1_score"],
        "false_positive_rate": m["false_positive_rate"],
        "mean_iou_matched": m["mean_iou"],
        "mean_best_pair_iou": m["mean_best_pair_iou"],
        "frame_detect_at_0.5": m["detection_rate_by_iou"].get("0.5"),
        "no_hazard_accuracy": m["no_hazard_accuracy"],
        "classification_accuracy": m["classification_accuracy"],
    }


def stratified(rm: dict) -> dict:
    tod, wx = rm.get("by_time_of_day", {}), rm.get("by_weather", {})
    dr = lambda b: b.get("hazard_detection_rate")
    return {
        "day": dr(tod.get("day", {})), "night": dr(tod.get("night", {})),
        "clear": dr(wx.get("clear", {})), "rain": dr(wx.get("rain", {})),
        "day_night_gap": rm.get("gaps", {}).get("day_night_detection_rate_gap"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True, help="arm_fm eval output dir (results_fm)")
    ap.add_argument("--gt", required=True, help="arm_gt eval output dir (results_gt)")
    ap.add_argument("--out", default="deconfound_result.json")
    a = ap.parse_args()

    fm, gt = load_metrics(a.fm), load_metrics(a.gt)
    fo, go, fs, gs = overall(fm), overall(gt), stratified(fm), stratified(gt)

    rows = []
    def add(label, f, g, better="high"):
        delta = None if (f is None or g is None) else round(g - f, 4)
        rows.append({"metric": label, "fm": f, "gt": g, "delta_gt_minus_fm": delta, "better": better})

    add("Recall (detection rate)", fo["recall"], go["recall"])
    add("Precision", fo["precision"], go["precision"])
    add("F1", fo["f1"], go["f1"])
    add("False-positive rate", fo["false_positive_rate"], go["false_positive_rate"], "low")
    add("Mean best-pair IoU", fo["mean_best_pair_iou"], go["mean_best_pair_iou"])
    add("Frame detect @IoU 0.5", fo["frame_detect_at_0.5"], go["frame_detect_at_0.5"])
    add("No-hazard accuracy", fo["no_hazard_accuracy"], go["no_hazard_accuracy"])
    add("Mean IoU (matched)", fo["mean_iou_matched"], go["mean_iou_matched"])
    add("Label accuracy (matched)", fo["classification_accuracy"], go["classification_accuracy"])
    for key, label in [("day", "Detect [day]"), ("night", "Detect [night]"),
                       ("clear", "Detect [clear]"), ("rain", "Detect [rain]")]:
        add(label, fs[key], gs[key])

    w = max(len(r["metric"]) for r in rows)
    print(f"\n{'metric':{w}}  {'FM':>8} {'GT':>8} {'Δ(GT-FM)':>10}")
    print("-" * (w + 30))
    for r in rows:
        f = "n/a" if r["fm"] is None else f"{r['fm']:.4f}"
        g = "n/a" if r["gt"] is None else f"{r['gt']:.4f}"
        d = "n/a" if r["delta_gt_minus_fm"] is None else f"{r['delta_gt_minus_fm']:+.4f}"
        print(f"{r['metric']:{w}}  {f:>8} {g:>8} {d:>10}")

    result = {"overall_fm": fo, "overall_gt": go, "stratified_fm": fs, "stratified_gt": gs, "table": rows}
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
