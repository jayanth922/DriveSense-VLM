# Task 3 — box-provenance de-confound

The v3→v4 flywheel turn (see [`FLYWHEEL_V4_FINDINGS.md`](FLYWHEEL_V4_FINDINGS.md)) changed two
things about the training data at once: it added 1,442 targeted rain/night frames, and those
frames carried foundation-model-emitted boxes instead of the nuScenes GT-projected boxes used
everywhere else. Detection regressed on exactly the buckets the turn targeted, and there was no
way to tell how much of that came from the targeting itself versus the box-provenance shift
riding along with it (see [`FLYWHEEL_V4_FINDINGS.md` § Label-provenance
confound](FLYWHEEL_V4_FINDINGS.md#label-provenance-confound-in-the-v4-experiment)). Task 3
isolates the second variable with a controlled A/B: two LoRA arms identical in base data,
validation data, test data, and the exact targeted frame ids, differing only in whether the
targeted boxes are FM-emitted or GT-projected.

## Honesty note before the numbers

The original per-frame v3/v4 training data did not survive (the RunPod volume it lived on was
reclaimed). Task 3's data is rebuilt directly from the surviving nuScenes trainval tables and
CAM_FRONT images — a **faithful reconstruction of the experiment design at reduced scale**, not
a byte-for-byte replay of the published v3/v4 rows. The base set is 2,652 frames against the
original 7,228 target, and the evaluated test set is 402 of the 1,041 held-out frames (the
subset with images still available locally). Absolute recall numbers below are correspondingly
lower than v3/v4's; the number that matters here is the FM-vs-GT delta, not the absolute level.
The pipeline that does the rebuild is [`deconfound/`](../deconfound/), see
[`deconfound/RUNBOOK.md`](../deconfound/RUNBOOK.md) for the full reconstruction and run procedure.

## Setup

Qwen2.5-VL-3B + LoRA (rank 32, alpha 64, 3 epochs, effective batch 16 — the same recipe as
`configs/training.yaml`, retuned for a single H100). Both arms share:

- base: 2,652 frames
- val: 889 frames
- targeted: 1,162 frames, identical ids in both arms — only the box source differs
- test: 402 frames (image-available subset of the fixed 1,041-frame held-out set)

Zero train↔test and train↔val leakage (asserted by `deconfound/build_arms.py assemble`, not
just claimed). API spend for the describe + FM-label passes was approximately $22.

## Result

Overall, on the 402-frame test set:

| metric | FM | GT |
|---|---|---|
| Recall (detection rate) | 0.101 | 0.167 |
| Precision | 0.176 | 0.330 |
| F1 | 0.128 | 0.222 |
| False-positive rate (lower is better) | 0.488 | 0.169 |
| Mean best-pair IoU | 0.244 | 0.458 |
| Frame detect @ IoU 0.5 | 0.266 | 0.566 |
| No-hazard accuracy | 0.512 | 0.831 |
| Mean IoU (matched) | 0.632 | 0.641 |
| Label accuracy (matched) | 0.962 | 0.954 |

GT wins on every axis except matched-class label accuracy, where the two arms are within a
point of each other. That last row is the reason the rest of the table isn't a wash: when the
FM arm does produce a matched box, it names the hazard about as well as the GT arm does. The gap
is entirely in whether and where it draws a box, not in what it calls the box once drawn — a
calibration failure, not a classification one.

Detection rate by condition:

| condition | n | FM | GT |
|---|---|---|---|
| Day | 283 | 0.130 | 0.171 |
| Night | 119 | 0.000 | 0.151 |
| Clear | 298 | 0.127 | 0.197 |
| Rain | 104 | 0.000 | 0.050 |

The FM arm detects nothing at all at night or in rain. Its day/night detection gap is 0.130;
the GT arm's is 0.020. In rain, every box the FM arm does emit is wrong — a 1.00 false-positive
rate on that bucket, against 0.069 for the GT arm. The GT arm is also weak in rain (0.050 recall
on 104 frames), but it at least degrades rather than collapsing.

## Reading this result

GT-projected boxes were a large, previously-confounded driver of reliability in the v3→v4
comparison, not a minor implementation detail. Holding everything else fixed, switching the
targeted frames from GT to FM-emitted boxes roughly halved recall and F1, tripled the
false-positive rate, and cut localization quality (mean best-pair IoU) by nearly half — and it
is specifically the reason the model has any working detection at night or in rain rather than
none. The v4 regression on rain/night/tiny buckets is now attributable, at least in significant
part, to box-provenance quality on the added data, not solely to the targeting strategy itself.

## Limitations

- **Reconstruction, not replay.** The original v3/v4 per-frame data is gone; these are freshly
  rebuilt splits following the same design. Treat this as a faithful-family repeat of the
  experiment, not a re-measurement of the exact published v3/v4 rows.
- **Reduced scale.** Base is 2,652 frames against the original 7,228 target, and the test set is
  402 of the fixed 1,041 (the subset with locally available images). Absolute recall is low as a
  result — read the FM-vs-GT delta, not the absolute numbers, as the finding.
- **Small rain bucket.** 104 frames. The rain row is directionally consistent with the rest of
  the table but shouldn't be over-read on its own.
- **Single seed.** No confidence intervals yet; the result is a single run per arm.

## Reproducing this

The raw eval output backing every number above is committed at
[`results/task3_deconfound/`](../results/task3_deconfound/): `results_fm/` and `results_gt/`
(each with `level1_grounding/` and `level4_robustness/robustness_metrics.json`, produced by
`scripts/run_full_evaluation.py --level 1 4`) and `deconfound_result.json` (produced by
`deconfound/compare_arms.py` from those two).

The full phase-by-phase procedure, with cost gates before any spend, is in
[`deconfound/RUNBOOK.md`](../deconfound/RUNBOOK.md). In short: `reconstruct.py` rebuilds the
manifests from nuScenes and gates on cost; `describe_manifest.py` fills in
severity/reasoning/action on GT boxes via the Batch API; `scripts/v4/v4_batch_label.py` FM-labels
the targeted frames; `build_arms.py` assembles the two arms and asserts zero leakage; both arms
train with `scripts/run_training.py --config deconfound/training_h100.yaml`; predictions come
from `scripts/run_generate_predictions.py` against each trained arm; and `run_full_evaluation.py`
+ `deconfound/compare_arms.py` produce the tables above.
