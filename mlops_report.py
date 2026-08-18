#!/usr/bin/env python3
"""Self-contained MLOps report for DriveSense-VLM: model comparison + regression gate + drift.

Reads results/metrics_registry.json (the versioned source of truth) and emits:
  1. eval_loss trajectory (v2 -> v3 -> v4)
  2. L1 grounding comparison (production vs candidate)
  3. L4 stratified bucket comparison, with deltas
  4. REGRESSION GATE verdict (BLOCK/PASS) on the weak buckets
  5. training-data DRIFT (total-variation distance of label distributions)
Writes a markdown report and (with --gate) exits non-zero if the gate BLOCKs — so CI can
fail a PR that regresses the weak buckets. No dependencies beyond the standard library.

Usage:
  python mlops_report.py                         # print report, always exit 0
  python mlops_report.py --gate                  # exit 1 if the candidate regresses (for CI)
  python mlops_report.py --candidate v4 --baseline v3 --out mlops_report.md
"""
import argparse, json, sys
from pathlib import Path


def tv_distance(a: dict, b: dict) -> float:
    """Total-variation distance between two label-count distributions (0..1)."""
    keys = set(a) | set(b)
    sa, sb = sum(a.values()) or 1, sum(b.values()) or 1
    return 0.5 * sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in keys)


def fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="results/metrics_registry.json")
    ap.add_argument("--candidate", default="v4")
    ap.add_argument("--baseline", default=None, help="default: registry 'production'")
    ap.add_argument("--gate", action="store_true", help="exit 1 if candidate regresses")
    ap.add_argument("--out", default="results/mlops_report.md")
    a = ap.parse_args()

    reg = json.loads(Path(a.registry).read_text())
    M = reg["models"]
    base = a.baseline or reg.get("production", "v3")
    cand = a.candidate
    pol = reg["gate_policy"]
    L = []  # report lines

    L.append(f"# MLOps report — {cand} vs {base} (baseline = production `{base}`)\n")
    L.append(f"Test set: {reg['test_set']['frames']} frames, fixed across {base}/{cand}.\n")

    # 1. eval_loss trajectory
    L.append("## 1. eval_loss trajectory")
    L.append("| model | train ex. | epochs | eval_loss |")
    L.append("|---|---|---|---|")
    for v in ("v2", "v3", "v4"):
        m = M.get(v, {})
        L.append(f"| {v} | {m.get('train_examples','—')} | {m.get('epochs','—')} | {fmt(m.get('eval_loss'))} |")
    L.append("_v2→v3: naive scale-up doubled eval_loss (generalization hurt). v4: clean, no overfit._\n")

    # 2. L1
    b1, c1 = M[base].get("l1"), M[cand].get("l1")
    L.append("## 2. L1 grounding (IoU>=0.5)")
    if b1 and c1:
        L.append(f"| metric | {base} | {cand} | Δ |")
        L.append("|---|---|---|---|")
        for k in ("precision", "recall", "f1", "mean_iou", "class_acc", "parse_rate"):
            d = c1[k] - b1[k]
            L.append(f"| {k} | {fmt(b1[k])} | {fmt(c1[k])} | {d:+.3f} |")
    L.append("")

    # 3. + 4. L4 buckets + gate
    b4, c4 = M[base]["l4_det_at_0.5"], M[cand]["l4_det_at_0.5"]
    L.append("## 3. L4 stratified detection@0.5")
    L.append(f"| bucket | {base} | {cand} | Δ |")
    L.append("|---|---|---|---|")
    for k in b4:
        if k in c4:
            L.append(f"| {k} | {fmt(b4[k])} | {fmt(c4[k])} | {c4[k]-b4[k]:+.3f} |")
    L.append("")

    weak = pol["weak_buckets"]; tol = pol["tolerance"]
    regressions = [(k, b4[k], c4[k]) for k in weak if k in b4 and k in c4 and c4[k] < b4[k] - tol]
    blocked = bool(regressions)
    L.append("## 4. Regression gate")
    L.append(f"Policy: BLOCK if any of {weak} drops below (production − {tol}) on `{pol['metric']}`.")
    if blocked:
        L.append(f"\n**VERDICT: 🚫 BLOCK — `{cand}` must not replace `{base}`.** Regressed buckets:")
        for k, bo, co in regressions:
            L.append(f"- {k}: {bo:.3f} → {co:.3f}  ({(co-bo)/bo*100:+.0f}% rel)")
    else:
        L.append(f"\n**VERDICT: ✅ PASS — `{cand}` may be promoted.**")
    L.append("")

    # 5. drift
    da, db = M[base].get("train_label_dist"), M[cand].get("train_label_dist")
    L.append("## 5. Training-data drift (label distribution)")
    if da and db:
        tv = tv_distance(da, db)
        L.append(f"Total-variation distance {base}→{cand}: **{tv:.3f}** "
                 f"(0 = identical, 1 = disjoint).")
        keys = sorted(set(da) | set(db))
        sa, sb = sum(da.values()), sum(db.values())
        L.append(f"| class | {base} % | {cand} % | Δpp |")
        L.append("|---|---|---|---|")
        for k in keys:
            pa, pb = 100*da.get(k,0)/sa, 100*db.get(k,0)/sb
            L.append(f"| {k} | {pa:.1f} | {pb:.1f} | {pb-pa:+.1f} |")
        L.append(f"_Biggest shift: `no_hazard` 0.0% → {100*db.get('no_hazard',0)/sb:.1f}% "
                 f"(the recall-suppressing negatives introduced in {cand})._")
    L.append("")

    report = "\n".join(L)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(report)
    print(report)
    print(f"\n[wrote {a.out}]")

    if a.gate and blocked:
        print(f"\n::error:: regression gate BLOCKED {cand} vs {base}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
