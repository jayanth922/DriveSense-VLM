# MLOps report — v4 vs v3 (baseline = production `v3`)

Test set: 1041 frames, fixed across v3/v4.

## 1. eval_loss trajectory
| model | train ex. | epochs | eval_loss |
|---|---|---|---|
| v2 | 2754 | 5 | 0.310 |
| v3 | 7228 | 5 | 0.660 |
| v4 | 8670 | 3 | 0.694 |
_v2→v3: naive scale-up doubled eval_loss (generalization hurt). v4: clean, no overfit._

## 2. L1 grounding (IoU>=0.5)
| metric | v3 | v4 | Δ |
|---|---|---|---|
| precision | 0.400 | 0.370 | -0.030 |
| recall | 0.240 | 0.190 | -0.050 |
| f1 | 0.300 | 0.250 | -0.050 |
| mean_iou | 0.670 | 0.656 | -0.014 |
| class_acc | 0.940 | 0.946 | +0.006 |
| parse_rate | 0.987 | 0.974 | -0.013 |

## 3. L4 stratified detection@0.5
| bucket | v3 | v4 | Δ |
|---|---|---|---|
| overall | 0.280 | 0.230 | -0.050 |
| tiny | 0.228 | 0.172 | -0.056 |
| small | 0.464 | 0.429 | -0.035 |
| medium | 0.526 | 0.526 | +0.000 |
| rain | 0.125 | 0.074 | -0.051 |
| night_tiny | 0.127 | 0.107 | -0.020 |
| clear_medium | 0.690 | 0.690 | +0.000 |

## 4. Regression gate
Policy: BLOCK if any of ['rain', 'night_tiny', 'tiny'] drops below (production − 0.0) on `l4_det_at_0.5`.

**VERDICT: 🚫 BLOCK — `v4` must not replace `v3`.** Regressed buckets:
- rain: 0.125 → 0.074  (-41% rel)
- night_tiny: 0.127 → 0.107  (-16% rel)
- tiny: 0.228 → 0.172  (-25% rel)

## 5. Training-data drift (label distribution)
Total-variation distance v3→v4: **0.033** (0 = identical, 1 = disjoint).
| class | v3 % | v4 % | Δpp |
|---|---|---|---|
| construction_zone | 25.3 | 25.8 | +0.5 |
| cyclist_proximity | 6.3 | 6.1 | -0.2 |
| high_density | 31.6 | 30.3 | -1.3 |
| jaywalking | 27.9 | 26.1 | -1.8 |
| no_hazard | 0.0 | 1.0 | +1.0 |
| occluded_pedestrian | 8.4 | 9.7 | +1.4 |
| unusual_object | 0.6 | 0.9 | +0.4 |
_Biggest shift: `no_hazard` 0.0% → 1.0% (the recall-suppressing negatives introduced in v4)._
